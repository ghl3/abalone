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
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
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
    let stats = Arc::new(BatchStats::default());
    let total_t = Instant::now();

    thread::scope(|s| {
        for tid in 0..args.threads {
            let args = &args;
            let next_game = Arc::clone(&next_game);
            let stats = Arc::clone(&stats);
            s.spawn(move || run_worker(tid, args, next_game, stats));
        }
    });

    eprintln!(
        "selfplay-batch: {} games done in {:?}",
        args.games,
        total_t.elapsed()
    );
    // Parsed by `model/train_loop.py::_parse_batch_fill`. Keep the key=value
    // shape stable; trailing fields may be added.
    let (raw, net) = stats.means();
    let width = args.cfg.batch_size.max(1) as f64;
    eprintln!(
        "selfplay-batch: nn_calls={} leaves={} plies={} mean_fill={:.2} \
         mean_fill_net={:.2} width={} fill_frac={:.3}",
        stats.calls.load(Ordering::Relaxed),
        stats.leaves.load(Ordering::Relaxed),
        stats.plies.load(Ordering::Relaxed),
        raw,
        net,
        args.cfg.batch_size.max(1),
        net / width,
    );
}

/// How full the evaluator's batches actually were.
///
/// Throughput here is *not* CPU-bound — `selfplay-batch` sits near 23% of a
/// 10-core machine while CoreML works — so positions/second is set by the
/// number of evaluator calls, and an under-filled batch costs the same wall
/// time as a full one: `set_fixed_batch` pads every call to one width because
/// CoreML compiles a separate model per input shape.
///
/// That makes fill the quantity to watch. `Search::next_batch` collects
/// `batch_size` *distinct* non-terminal leaves, and both of those words are
/// ways to come up short: a sharpening policy sends repeated descents down the
/// same branch (virtual loss only spreads them so far), and a search near the
/// capture threshold resolves leaves terminally, which need no evaluation. So
/// the fill is expected to fall exactly as the network gets better — and
/// generations 6 and 7 of run `ruby-panther` halved in positions/second with a
/// byte-identical config while the policy entropy gap rose 0.959 → 1.191.
/// This counter is what tells us whether those are the same fact.
#[derive(Default)]
struct BatchStats {
    /// Calls to the evaluator, including one root expansion per search.
    calls: AtomicU64,
    /// Leaves summed over those calls.
    leaves: AtomicU64,
    /// Plies played, which is also the number of searches — and therefore the
    /// number of size-1 root-expansion calls to discount.
    plies: AtomicU64,
}

impl BatchStats {
    /// `(mean fill, mean fill excluding root expansions)`, in leaves per call.
    fn means(&self) -> (f64, f64) {
        let calls = self.calls.load(Ordering::Relaxed) as f64;
        let leaves = self.leaves.load(Ordering::Relaxed) as f64;
        let plies = self.plies.load(Ordering::Relaxed) as f64;
        let raw = if calls > 0.0 { leaves / calls } else { 0.0 };
        // One root expansion per search, always a batch of exactly one. Left in
        // the raw figure it drags the mean down by a fixed amount that has
        // nothing to do with how well the search is batching.
        let net_calls = calls - plies;
        let net = if net_calls > 0.0 {
            (leaves - plies) / net_calls
        } else {
            0.0
        };
        (raw, net)
    }
}

fn run_worker(tid: usize, args: &Args, next_game: Arc<AtomicU32>, stats: Arc<BatchStats>) {
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
        let mut calls = 0u64;
        let mut leaves = 0u64;
        let outcome = play_game(&args.cfg, game_id, seed, |batch| {
            // Counted locally and flushed once per game: nine workers hammering
            // one atomic per evaluator call would be contention added by the
            // instrument measuring the contention.
            calls += 1;
            leaves += batch.len() as u64;
            evaluator.evaluate_batch(batch).expect("ort evaluate")
        });
        stats.calls.fetch_add(calls, Ordering::Relaxed);
        stats.leaves.fetch_add(leaves, Ordering::Relaxed);
        stats
            .plies
            .fetch_add(outcome.trajectory.len() as u64, Ordering::Relaxed);

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
