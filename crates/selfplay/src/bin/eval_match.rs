//! Eval match between two players. Each player is one of:
//!   - `model:<path.onnx>` — ONNX-driven MCTS
//!   - `heuristic`          — heuristic-MCTS with default weights
//!   - `random`             — uniform-random move selection
//!
//! Plays N games (alternating colors), writes a JSON summary to
//! `--out-json`. The training loop's gating and heuristic-anchor
//! evaluations both invoke this binary with different player configs.
//!
//! Run:
//!   eval-match --player-a model:checkpoints/gen_002.onnx \
//!              --player-b model:checkpoints/best.onnx \
//!              --games 21 --simulations 200 \
//!              --out-json runs/foo/eval/gen_002_gate.json

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use abalone_game::{Game, GameState, Move, Side};
use abalone_mcts::{heuristic, search, LeafEval, SearchConfig};
use abalone_selfplay::ort_eval::OrtEvaluator;
use rand::rngs::SmallRng;
use rand::Rng;
use rand::SeedableRng;
use serde::Serialize;

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
                _ => panic!("unknown arg: {}", k),
            }
        }
        a.a = a_spec.expect("--player-a required");
        a.b = b_spec.expect("--player-b required");
        a
    }
}

enum Player {
    Model(Arc<OrtEvaluator>),
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

    fn pick_move(&self, g: &Game, cfg: &SearchConfig, rng: &mut SmallRng) -> Move {
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

    let player_a = Player::from_spec(&args.a);
    let player_b = Player::from_spec(&args.b);
    let cfg = SearchConfig {
        simulations: args.simulations,
        c_puct: args.c_puct,
    };

    let mut wins_a = 0u32;
    let mut wins_b = 0u32;
    let mut draws = 0u32;
    let t = Instant::now();

    for i in 0..args.games {
        let a_is_black = i % 2 == 0;
        let mut rng_a = SmallRng::seed_from_u64(args.seed.wrapping_add((i as u64) * 2));
        let mut rng_b = SmallRng::seed_from_u64(args.seed.wrapping_add((i as u64) * 2 + 1));

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
            GameState::Wins(s) => {
                let a_won = (s == Side::Black) == a_is_black;
                if a_won {
                    wins_a += 1;
                } else {
                    wins_b += 1;
                }
            }
            GameState::Draw => draws += 1,
            GameState::InProgress => unreachable!(),
        }

        eprintln!(
            "  game {}: a_is_black={}, final={:?}",
            i,
            a_is_black,
            g.state()
        );
    }

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
