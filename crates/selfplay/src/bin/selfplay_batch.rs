//! Self-play game generation. Plays N games with batched, NN-driven MCTS and
//! writes parquet shards to an output directory.
//!
//! Run:
//!   selfplay-batch --model checkpoints/gen_001.onnx \
//!                  --out-dir runs/foo/shards/gen_002/ \
//!                  --games 200 --sims-fast 200 --sims-full 800 \
//!                  --full-search-rate 0.25 --batch-size 32 \
//!                  --opening belgian-daisy --handicap-rate 0.7 \
//!                  --shard-games 8 --threads 4
//!
//! **There is one evaluator: the network.** The hand-written positional
//! evaluator is retired from the training loop (`docs/MODEL.md`) — capture
//! handicap seeding is what solves the cold start now, and it is a pure
//! data-distribution intervention with nothing to unlearn. The heuristic
//! survives only as a fixed benchmark opponent in `eval-match`.
//!
//! # Output
//!
//! Everything goes to stderr via `eprintln!`:
//!   - one start banner line
//!   - one line per completed game:
//!     `  [tN] game M: P plies, final=Wins(Black), handicap=b/w, score_diff=D`
//!   - one end summary line
//!
//! The Python driver redirects this stream to `runs/<id>/logs/
//! gen_NNN_selfplay.log` and polls the file every progress tick to
//! count completed games. The format above is the contract — keep the
//! `] game M: P plies, final=` substring stable.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Instant;

use abalone_game::{Opening, Side, DEFAULT_MAX_PLIES, NO_PROGRESS_DISABLED};
use abalone_selfplay::{
    ort_eval::OrtEvaluator, play_game, shard::ShardWriter, SelfPlayConfig,
};

fn parse_opening(s: &str) -> Opening {
    match s {
        "standard" => Opening::Standard,
        "belgian-daisy" | "belgian_daisy" | "belgian" => Opening::BelgianDaisy,
        other => panic!("unknown --opening '{other}' (expected 'standard' or 'belgian-daisy')"),
    }
}

#[derive(Clone, Debug)]
struct Args {
    model: PathBuf,
    out_dir: PathBuf,
    games: u32,
    cfg: SelfPlayConfig,
    shard_games: u32,
    threads: usize,
    seed: u64,
}

impl Args {
    fn parse() -> Self {
        let mut args = std::env::args().skip(1);
        let mut model: Option<PathBuf> = None;
        let mut out_dir = PathBuf::new();
        let mut games = 200u32;
        let mut shard_games = 8u32;
        let mut threads = num_cpus_or(8);
        let mut seed = 0u64;
        let mut cfg = SelfPlayConfig::default();
        // `--simulations N` is the pre-playout-cap flag the Python driver still
        // passes. It means "no cap randomisation": every move runs N sims and
        // every position carries a policy target. Explicit --sims-* flags
        // override it regardless of argument order.
        let mut legacy_simulations: Option<u32> = None;
        let mut sims_fast: Option<u32> = None;
        let mut sims_full: Option<u32> = None;
        let mut full_search_rate: Option<f32> = None;

        while let Some(k) = args.next() {
            let mut nxt = || args.next().expect("missing value");
            match k.as_str() {
                "--evaluator" => {
                    let v = nxt();
                    assert_eq!(
                        v, "model",
                        "--evaluator only accepts 'model'. The heuristic evaluator is \
                         retired from the training loop (docs/MODEL.md); it survives as a \
                         benchmark opponent in eval-match."
                    );
                }
                "--model" => model = Some(PathBuf::from(nxt())),
                "--out-dir" => out_dir = PathBuf::from(nxt()),
                "--games" => games = nxt().parse().unwrap(),
                "--simulations" => legacy_simulations = Some(nxt().parse().unwrap()),
                "--sims-fast" => sims_fast = Some(nxt().parse().unwrap()),
                "--sims-full" => sims_full = Some(nxt().parse().unwrap()),
                "--full-search-rate" => full_search_rate = Some(nxt().parse().unwrap()),
                "--batch-size" => cfg.batch_size = nxt().parse().unwrap(),
                "--virtual-loss" => cfg.virtual_loss = nxt().parse().unwrap(),
                "--fpu-reduction" => cfg.fpu_reduction = nxt().parse().unwrap(),
                "--c-puct" => cfg.c_puct = nxt().parse().unwrap(),
                "--temperature-plies" => cfg.temperature_plies = nxt().parse().unwrap(),
                "--temperature" => cfg.temperature = nxt().parse().unwrap(),
                "--dirichlet-alpha" => cfg.dirichlet_alpha = nxt().parse().unwrap(),
                "--dirichlet-eps" => cfg.dirichlet_eps = nxt().parse().unwrap(),
                "--opening" => cfg.opening = parse_opening(&nxt()),
                "--handicap-rate" => cfg.handicap_rate = nxt().parse().unwrap(),
                "--handicap-max" => cfg.handicap_max = nxt().parse().unwrap(),
                "--random-opening-plies" => cfg.random_opening_plies = nxt().parse().unwrap(),
                "--max-plies" => cfg.max_plies = nxt().parse().unwrap(),
                "--no-progress-plies" => {
                    // 0 means "off", spelled as the sentinel internally.
                    let v: u32 = nxt().parse().unwrap();
                    cfg.no_progress_plies = if v == 0 { NO_PROGRESS_DISABLED } else { v };
                }
                "--gamma" | "--capture-gamma" => cfg.capture_gamma = nxt().parse().unwrap(),
                "--shard-games" => shard_games = nxt().parse().unwrap(),
                "--threads" => threads = nxt().parse().unwrap(),
                "--seed" => seed = nxt().parse().unwrap(),
                _ => panic!("unknown arg: {k}"),
            }
        }

        if let Some(n) = legacy_simulations {
            cfg.sims_fast = n;
            cfg.sims_full = n;
            // Only force "every move is a full search" when the caller has not
            // asked for a fast/full split at all.
            if sims_fast.is_none() && sims_full.is_none() {
                cfg.full_search_rate = 1.0;
            }
        }
        if let Some(n) = sims_fast {
            cfg.sims_fast = n;
        }
        if let Some(n) = sims_full {
            cfg.sims_full = n;
        }
        if let Some(r) = full_search_rate {
            cfg.full_search_rate = r;
        }
        assert!(
            cfg.sims_full >= cfg.sims_fast,
            "--sims-full ({}) must be >= --sims-fast ({})",
            cfg.sims_full,
            cfg.sims_fast
        );
        assert!(!out_dir.as_os_str().is_empty(), "--out-dir required");
        Self {
            model: model.expect("--model required"),
            out_dir,
            games,
            cfg,
            shard_games,
            threads,
            seed,
        }
    }
}

fn num_cpus_or(default: usize) -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get().saturating_sub(1).max(1))
        .unwrap_or(default)
}

fn main() {
    let args = Args::parse();
    let c = &args.cfg;
    eprintln!(
        "selfplay-batch: model={} out={} games={} threads={} seed={}",
        args.model.display(),
        args.out_dir.display(),
        args.games,
        args.threads,
        args.seed
    );
    eprintln!(
        "  search: sims {}/{} @ rate {:.2}, batch {}, c_puct {}, vloss {}, fpu {}, \
         dirichlet a={} eps={}",
        c.sims_fast,
        c.sims_full,
        c.full_search_rate,
        c.batch_size,
        c.c_puct,
        c.virtual_loss,
        c.fpu_reduction,
        c.dirichlet_alpha,
        c.dirichlet_eps
    );
    eprintln!(
        "  games:  opening={:?}, handicap {:.2} of games over 0..={}, random opening plies {}, \
         max_plies {}, no_progress {}, temp {} for {} plies, capture gamma {}",
        c.opening,
        c.handicap_rate,
        c.handicap_max,
        c.random_opening_plies,
        c.max_plies,
        if c.no_progress_plies == NO_PROGRESS_DISABLED {
            "off".to_string()
        } else {
            c.no_progress_plies.to_string()
        },
        c.temperature,
        c.temperature_plies,
        c.capture_gamma
    );
    if c.max_plies != DEFAULT_MAX_PLIES {
        eprintln!("  note: ply cap {} differs from the default {DEFAULT_MAX_PLIES}; the ply input plane is normalised by it", c.max_plies);
    }
    std::fs::create_dir_all(&args.out_dir).expect("create out-dir");

    let next_game = Arc::new(AtomicU32::new(0));
    let total_t = Instant::now();

    thread::scope(|s| {
        for tid in 0..args.threads {
            let args = &args;
            let next_game = Arc::clone(&next_game);
            s.spawn(move || run_worker(tid, args, next_game));
        }
    });

    eprintln!(
        "selfplay-batch: {} games done in {:?}",
        args.games,
        total_t.elapsed()
    );
}

fn run_worker(tid: usize, args: &Args, next_game: Arc<AtomicU32>) {
    // Per-thread session: each worker loads its own ONNX. ORT 2.x's
    // Session::run takes &mut self, and a shared Mutex<Session> would
    // bottleneck all inference through one thread. ~30 MB × threads of
    // extra memory is cheap compared to the throughput we recover.
    let mut evaluator = OrtEvaluator::from_onnx(&args.model).expect("load onnx model");
    // CoreML compiles one model per input shape, so a search whose last batch
    // is partial would thrash between shapes; padding to a single width is
    // worth ~10x there. On CPU the cost is linear in width, so don't.
    if abalone_selfplay::ort_eval::use_coreml() {
        evaluator.set_fixed_batch(Some(args.cfg.batch_size.max(1)));
    }
    let mut shard_idx = 0u32;
    let mut writer: Option<ShardWriter> = None;
    let mut games_in_shard = 0u32;

    loop {
        let game_id = next_game.fetch_add(1, Ordering::Relaxed);
        if game_id >= args.games {
            break;
        }
        let seed = args.seed.wrapping_add(u64::from(game_id));
        let outcome = play_game(&args.cfg, game_id, seed, |batch| {
            evaluator.evaluate_batch(batch).expect("ort evaluate")
        });

        // Lazily create a shard writer; rotate every `shard_games`.
        if writer.is_none() {
            let path = args
                .out_dir
                .join(format!("shard_t{tid:02}_{shard_idx:04}.parquet"));
            writer = Some(ShardWriter::create(&path).expect("create shard"));
        }
        writer
            .as_mut()
            .unwrap()
            .write_game(&outcome)
            .expect("write game");
        games_in_shard += 1;

        if games_in_shard >= args.shard_games {
            writer.take().unwrap().finish().expect("finish shard");
            shard_idx += 1;
            games_in_shard = 0;
        }

        let plies = outcome.trajectory.len();
        // This line is the contract with the Python driver. It parses
        // "] game M: P plies, final=..." substrings out of the log file
        // to count completed games. Keep that part of the format stable;
        // trailing fields may be added.
        eprintln!(
            "  [t{}] game {}: {} plies, final={:?}, handicap={}/{}, score_diff={}",
            tid,
            game_id,
            plies,
            outcome.final_state.state(),
            outcome.handicap_black,
            outcome.handicap_white,
            outcome.final_score_diff(Side::Black),
        );
    }

    if let Some(w) = writer {
        w.finish().expect("finish trailing shard");
    }
}
