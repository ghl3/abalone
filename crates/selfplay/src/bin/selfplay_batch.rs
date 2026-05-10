//! ONNX-driven self-play. Plays N games using the supplied model and
//! writes parquet shards to an output directory.
//!
//! Run:
//!   selfplay-batch --model checkpoints/gen_001.onnx \
//!                  --out-dir runs/foo/shards/gen_002/ \
//!                  --games 200 --simulations 200 \
//!                  --shard-games 8 --threads 4
//!
//! Each thread plays games independently and rotates shard files every
//! `shard-games` completed games. Threads share one ort Session via Arc.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Instant;

use abalone_mcts::LeafEval;
use abalone_selfplay::{
    ort_eval::OrtEvaluator, play_game, shard::ShardWriter, SelfPlayConfig,
};
use rand::rngs::SmallRng;
use rand::SeedableRng;

#[derive(Clone, Debug)]
struct Args {
    model: PathBuf,
    out_dir: PathBuf,
    games: u32,
    simulations: u32,
    c_puct: f32,
    temperature_plies: u32,
    temperature: f32,
    dirichlet_alpha: f32,
    dirichlet_eps: f32,
    shard_games: u32,
    threads: usize,
    seed: u64,
}

impl Args {
    fn parse() -> Self {
        let mut args = std::env::args().skip(1);
        let mut a = Args {
            model: PathBuf::new(),
            out_dir: PathBuf::new(),
            games: 200,
            simulations: 200,
            c_puct: 1.4,
            temperature_plies: 50,
            temperature: 1.0,
            dirichlet_alpha: 0.3,
            dirichlet_eps: 0.25,
            shard_games: 8,
            threads: num_cpus_or(8),
            seed: 0,
        };
        while let Some(k) = args.next() {
            let mut nxt = || args.next().expect("missing value");
            match k.as_str() {
                "--model" => a.model = PathBuf::from(nxt()),
                "--out-dir" => a.out_dir = PathBuf::from(nxt()),
                "--games" => a.games = nxt().parse().unwrap(),
                "--simulations" => a.simulations = nxt().parse().unwrap(),
                "--c-puct" => a.c_puct = nxt().parse().unwrap(),
                "--temperature-plies" => a.temperature_plies = nxt().parse().unwrap(),
                "--temperature" => a.temperature = nxt().parse().unwrap(),
                "--dirichlet-alpha" => a.dirichlet_alpha = nxt().parse().unwrap(),
                "--dirichlet-eps" => a.dirichlet_eps = nxt().parse().unwrap(),
                "--shard-games" => a.shard_games = nxt().parse().unwrap(),
                "--threads" => a.threads = nxt().parse().unwrap(),
                "--seed" => a.seed = nxt().parse().unwrap(),
                _ => panic!("unknown arg: {}", k),
            }
        }
        assert!(!a.model.as_os_str().is_empty(), "--model required");
        assert!(!a.out_dir.as_os_str().is_empty(), "--out-dir required");
        a
    }
}

fn num_cpus_or(default: usize) -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get().saturating_sub(1).max(1))
        .unwrap_or(default)
}

fn main() {
    let args = Args::parse();
    eprintln!("selfplay-batch: {:?}", args);
    std::fs::create_dir_all(&args.out_dir).expect("create out-dir");

    // Per-thread sessions: each worker loads its own ONNX. ORT 2.x's
    // Session::run takes &mut self, and a shared Mutex<Session> would
    // bottleneck all inference through one thread. ~30 MB × threads of
    // extra memory is cheap compared to the throughput we recover.
    let cfg = SelfPlayConfig {
        simulations: args.simulations,
        c_puct: args.c_puct,
        temperature_plies: args.temperature_plies,
        temperature: args.temperature,
        dirichlet_alpha: args.dirichlet_alpha,
        dirichlet_eps: args.dirichlet_eps,
    };

    let next_game = Arc::new(AtomicU32::new(0));
    let games = args.games;
    let shard_games = args.shard_games;
    let total_t = Instant::now();

    thread::scope(|s| {
        for tid in 0..args.threads {
            let model_path = args.model.clone();
            let cfg = cfg.clone();
            let out_dir = args.out_dir.clone();
            let next_game = Arc::clone(&next_game);
            s.spawn(move || {
                let mut evaluator =
                    OrtEvaluator::from_onnx(&model_path).expect("load onnx model");
                run_worker(
                    tid,
                    &mut evaluator,
                    cfg,
                    out_dir,
                    games,
                    shard_games,
                    next_game,
                    args.seed,
                );
            });
        }
    });

    eprintln!(
        "selfplay-batch: {} games done in {:?}",
        games,
        total_t.elapsed()
    );
}

#[allow(clippy::too_many_arguments)]
fn run_worker(
    tid: usize,
    evaluator: &mut OrtEvaluator,
    cfg: SelfPlayConfig,
    out_dir: PathBuf,
    games: u32,
    shard_games: u32,
    next_game: Arc<AtomicU32>,
    seed: u64,
) {
    let mut shard_idx = 0u32;
    let mut writer: Option<ShardWriter> = None;
    let mut games_in_shard = 0u32;

    loop {
        let game_id = next_game.fetch_add(1, Ordering::Relaxed);
        if game_id >= games {
            break;
        }
        let mut rng = SmallRng::seed_from_u64(seed.wrapping_add(game_id as u64));

        let outcome = play_game(&cfg, &mut rng, |g, _rng| -> LeafEval {
            evaluator.evaluate(g).expect("ort evaluate")
        });

        // Lazily create a shard writer; rotate every `shard_games`.
        if writer.is_none() {
            let path = out_dir.join(format!("shard_t{:02}_{:04}.parquet", tid, shard_idx));
            writer = Some(ShardWriter::create(&path).expect("create shard"));
        }
        writer
            .as_mut()
            .unwrap()
            .write_game(&outcome)
            .expect("write game");
        games_in_shard += 1;

        if games_in_shard >= shard_games {
            writer.take().unwrap().finish().expect("finish shard");
            shard_idx += 1;
            games_in_shard = 0;
        }

        let plies = outcome.trajectory.len();
        let final_state = outcome.final_state.state();
        eprintln!(
            "  [t{}] game {}: {} plies, final={:?}",
            tid, game_id, plies, final_state
        );
    }

    if let Some(w) = writer {
        w.finish().expect("finish trailing shard");
    }
}
