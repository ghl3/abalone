//! Eval match between two players. Each player is one of:
//!   - `model:<path.onnx>` — ONNX-driven MCTS
//!   - `heuristic`          — heuristic-MCTS with default weights
//!   - `random`             — uniform-random move selection
//!
//! Plays N games (alternating colors), writes a JSON summary to
//! `--out-json`. The training loop's gating and heuristic-anchor
//! evaluations both invoke this binary with different player configs.
//! Games are independent and run in parallel across `--threads` workers
//! (default: cores-1). For model-vs-model gates the two ONNX sessions
//! are shared via `Arc<Mutex<Session>>` — concurrent `.evaluate()` calls
//! serialize on the mutex, but with two players we have two sessions so
//! contention is bounded.
//!
//! Run:
//!   eval-match --player-a model:checkpoints/gen_002.onnx \
//!              --player-b model:checkpoints/best.onnx \
//!              --games 21 --simulations 200 --threads 8 \
//!              --out-json runs/foo/eval/gen_002_gate.json

use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Instant;

use abalone_game::{Game, GameState, Move, Side};
use abalone_mcts::{heuristic, search, LeafEval, SearchConfig};
use abalone_selfplay::ort_eval::OrtEvaluator;
use rand::rngs::SmallRng;
use rand::Rng;
use rand::SeedableRng;
use serde::Serialize;

// Per-thread Player: each worker holds its own `OrtEvaluator` so all
// inference runs in parallel (vs. one shared `Arc<Mutex<Session>>`
// that would serialize). Cost is one ONNX load per thread; throughput
// gain is ~num-threads×.

#[derive(Clone, Debug)]
enum PlayerSpec {
    Model(PathBuf),
    Heuristic,
    Random,
}

impl PlayerSpec {
    fn parse(s: &str) -> Self {
        if let Some(rest) = s.strip_prefix("model:") {
            PlayerSpec::Model(PathBuf::from(rest))
        } else if s == "heuristic" {
            PlayerSpec::Heuristic
        } else if s == "random" {
            PlayerSpec::Random
        } else {
            panic!("unknown player spec: {s}")
        }
    }

    fn label(&self) -> String {
        match self {
            PlayerSpec::Model(p) => format!("model:{}", p.display()),
            PlayerSpec::Heuristic => "heuristic".to_string(),
            PlayerSpec::Random => "random".to_string(),
        }
    }
}

#[derive(Clone, Debug)]
struct Args {
    a: PlayerSpec,
    b: PlayerSpec,
    games: u32,
    simulations: u32,
    c_puct: f32,
    out_json: PathBuf,
    seed: u64,
    threads: usize,
}

impl Args {
    fn parse() -> Self {
        let mut args = std::env::args().skip(1);
        let mut a_spec: Option<PlayerSpec> = None;
        let mut b_spec: Option<PlayerSpec> = None;
        let mut a = Args {
            a: PlayerSpec::Random,
            b: PlayerSpec::Random,
            games: 21,
            simulations: 200,
            c_puct: 1.4,
            out_json: PathBuf::from("/tmp/eval-match.json"),
            seed: 0,
            threads: num_cpus_or(8),
        };
        while let Some(k) = args.next() {
            let mut nxt = || args.next().expect("missing value");
            match k.as_str() {
                "--player-a" => a_spec = Some(PlayerSpec::parse(&nxt())),
                "--player-b" => b_spec = Some(PlayerSpec::parse(&nxt())),
                "--games" => a.games = nxt().parse().unwrap(),
                "--simulations" => a.simulations = nxt().parse().unwrap(),
                "--c-puct" => a.c_puct = nxt().parse().unwrap(),
                "--out-json" => a.out_json = PathBuf::from(nxt()),
                "--seed" => a.seed = nxt().parse().unwrap(),
                "--threads" => a.threads = nxt().parse().unwrap(),
                _ => panic!("unknown arg: {}", k),
            }
        }
        a.a = a_spec.expect("--player-a required");
        a.b = b_spec.expect("--player-b required");
        a
    }
}

fn num_cpus_or(default: usize) -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get().saturating_sub(1).max(1))
        .unwrap_or(default)
}

enum Player {
    Model(OrtEvaluator),
    Heuristic,
    Random,
}

impl Player {
    fn from_spec(spec: &PlayerSpec) -> Self {
        match spec {
            PlayerSpec::Model(p) => {
                Player::Model(OrtEvaluator::from_onnx(p).expect("load onnx model"))
            }
            PlayerSpec::Heuristic => Player::Heuristic,
            PlayerSpec::Random => Player::Random,
        }
    }

    fn pick_move(&mut self, g: &Game, cfg: &SearchConfig, rng: &mut SmallRng) -> Move {
        match self {
            Player::Model(eval) => {
                let res = search(g, cfg, rng, |state, _| -> LeafEval {
                    eval.evaluate(state).expect("ort evaluate")
                })
                .expect("non-terminal => some move");
                res.best
            }
            Player::Heuristic => search(g, cfg, rng, heuristic).expect("non-terminal").best,
            Player::Random => {
                let moves = g.legal_moves();
                moves[rng.gen_range(0..moves.len())]
            }
        }
    }
}

#[derive(Serialize)]
struct MatchResult {
    player_a: String,
    player_b: String,
    games: u32,
    simulations: u32,
    wins_a: u32,
    wins_b: u32,
    draws: u32,
    winrate_a: f32, // wins_a / games
    winrate_a_excluding_draws: f32, // wins_a / (wins_a + wins_b)
    elapsed_seconds: f32,
}

fn main() {
    let args = Args::parse();
    eprintln!("eval-match: {:?}", args);

    let cfg = SearchConfig {
        simulations: args.simulations,
        c_puct: args.c_puct,
    };

    // Atomic counters: workers race to claim the next game id and bump
    // the per-result tally on completion. Using `Relaxed` is fine — we
    // only read the counters after `thread::scope` has joined.
    let next_game = Arc::new(AtomicU32::new(0));
    let wins_a = Arc::new(AtomicU32::new(0));
    let wins_b = Arc::new(AtomicU32::new(0));
    let draws = Arc::new(AtomicU32::new(0));
    let games = args.games;
    let seed = args.seed;
    let t = Instant::now();

    thread::scope(|s| {
        for _tid in 0..args.threads {
            // Each thread constructs its own Player(s), which for Model
            // means loading a fresh ort Session. This eliminates the
            // global inference Mutex.
            let a_spec = args.a.clone();
            let b_spec = args.b.clone();
            let cfg = cfg.clone();
            let next_game = Arc::clone(&next_game);
            let wins_a = Arc::clone(&wins_a);
            let wins_b = Arc::clone(&wins_b);
            let draws = Arc::clone(&draws);
            s.spawn(move || {
                let mut player_a = Player::from_spec(&a_spec);
                let mut player_b = Player::from_spec(&b_spec);
                loop {
                    let i = next_game.fetch_add(1, Ordering::Relaxed);
                    if i >= games {
                        break;
                    }
                    let a_is_black = i % 2 == 0;
                    let mut rng_a = SmallRng::seed_from_u64(seed.wrapping_add((i as u64) * 2));
                    let mut rng_b = SmallRng::seed_from_u64(seed.wrapping_add((i as u64) * 2 + 1));

                    let mut g = Game::new_standard();
                    while !g.is_terminal() {
                        let mv = if (g.turn == Side::Black) == a_is_black {
                            player_a.pick_move(&g, &cfg, &mut rng_a)
                        } else {
                            player_b.pick_move(&g, &cfg, &mut rng_b)
                        };
                        g.apply(mv);
                    }

                    match g.state() {
                        GameState::Wins(side) => {
                            let a_won = (side == Side::Black) == a_is_black;
                            if a_won {
                                wins_a.fetch_add(1, Ordering::Relaxed);
                            } else {
                                wins_b.fetch_add(1, Ordering::Relaxed);
                            }
                        }
                        GameState::Draw => {
                            draws.fetch_add(1, Ordering::Relaxed);
                        }
                        GameState::InProgress => unreachable!(),
                    }

                    eprintln!(
                        "  game {}: a_is_black={}, final={:?}",
                        i,
                        a_is_black,
                        g.state()
                    );
                }
            });
        }
    });

    let wins_a = wins_a.load(Ordering::Relaxed);
    let wins_b = wins_b.load(Ordering::Relaxed);
    let draws = draws.load(Ordering::Relaxed);

    let result = MatchResult {
        player_a: args.a.label(),
        player_b: args.b.label(),
        games: args.games,
        simulations: args.simulations,
        wins_a,
        wins_b,
        draws,
        winrate_a: wins_a as f32 / args.games as f32,
        winrate_a_excluding_draws: if wins_a + wins_b == 0 {
            0.5
        } else {
            wins_a as f32 / (wins_a + wins_b) as f32
        },
        elapsed_seconds: t.elapsed().as_secs_f32(),
    };

    if let Some(parent) = args.out_json.parent() {
        std::fs::create_dir_all(parent).expect("create out-json parent");
    }
    let s = serde_json::to_string_pretty(&result).expect("serialize");
    std::fs::write(&args.out_json, s).expect("write out-json");
    eprintln!("wrote {}", args.out_json.display());
    eprintln!(
        "  wins_a={}, wins_b={}, draws={} (winrate_a={:.3})",
        wins_a, wins_b, draws, result.winrate_a
    );
}
