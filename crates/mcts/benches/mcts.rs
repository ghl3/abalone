//! Manual MCTS bench. Run with:
//!     cargo run --release -p abalone-mcts --bin mcts-bench
//!
//! Sections:
//!   1. Single-search timing on the standard opening (sims/sec) for both
//!      random-rollout and heuristic evaluators.
//!   2. heuristic-MCTS vs uniform-random match.
//!   3. heuristic-MCTS vs random-rollout-MCTS match (same sim count).
//!   4. Round-robin weight tune over a few candidate `Weights`.

use std::time::Instant;

use abalone_game::{Game, GameState, Move, Side};
use abalone_mcts::eval::Weights;
use abalone_mcts::{eval, heuristic, random_rollout, search, SearchConfig, SearchResult};
use rand::rngs::SmallRng;
use rand::Rng;
use rand::SeedableRng;

fn pick_random<R: Rng + ?Sized>(g: &Game, rng: &mut R) -> Move {
    let moves = g.legal_moves();
    moves[rng.gen_range(0..moves.len())]
}

fn pick_mcts_random_rollout(g: &Game, cfg: &SearchConfig, rng: &mut SmallRng) -> Move {
    search(g, cfg, rng, random_rollout)
        .expect("non-terminal => some move")
        .best
}

fn pick_mcts_heuristic(g: &Game, cfg: &SearchConfig, rng: &mut SmallRng) -> Move {
    search(g, cfg, rng, heuristic)
        .expect("non-terminal => some move")
        .best
}

fn pick_mcts_with_weights(
    g: &Game,
    cfg: &SearchConfig,
    rng: &mut SmallRng,
    weights: &Weights,
) -> Move {
    search(g, cfg, rng, |gg, _| eval::evaluate(&gg.board, gg.turn, weights))
        .expect("non-terminal => some move")
        .best
}

#[derive(Default, Debug)]
struct MatchStats {
    wins_a: u32,
    wins_b: u32,
    draws: u32,
}

impl MatchStats {
    fn total(&self) -> u32 {
        self.wins_a + self.wins_b + self.draws
    }
    fn print(&self, label_a: &str, label_b: &str) {
        let total = self.total() as f32;
        let pa = self.wins_a as f32 / total * 100.0;
        let pb = self.wins_b as f32 / total * 100.0;
        let pd = self.draws as f32 / total * 100.0;
        println!(
            "  {} {} ({:.1}%)  ·  {} {} ({:.1}%)  ·  draws {} ({:.1}%)  · n={}",
            label_a,
            self.wins_a,
            pa,
            label_b,
            self.wins_b,
            pb,
            self.draws,
            pd,
            self.total()
        );
    }
}

fn play_one_game<FA, FB>(
    a_is_black: bool,
    mut pick_a: FA,
    mut pick_b: FB,
) -> GameState
where
    FA: FnMut(&Game) -> Move,
    FB: FnMut(&Game) -> Move,
{
    let mut g = Game::new_standard();
    while !g.is_terminal() {
        let mv = if (g.turn == Side::Black) == a_is_black {
            pick_a(&g)
        } else {
            pick_b(&g)
        };
        g.apply(mv);
    }
    g.state()
}

fn run_match<FA, FB>(games: u32, mut maker_a: FA, mut maker_b: FB) -> MatchStats
where
    FA: FnMut(u64) -> Box<dyn FnMut(&Game) -> Move>,
    FB: FnMut(u64) -> Box<dyn FnMut(&Game) -> Move>,
{
    let mut stats = MatchStats::default();
    for i in 0..games {
        let a_is_black = i % 2 == 0;
        let mut a = maker_a((i as u64) * 2);
        let mut b = maker_b((i as u64) * 2 + 1);
        let state = play_one_game(a_is_black, |g| a(g), |g| b(g));
        match state {
            GameState::Wins(s) => {
                let a_won = (s == Side::Black) == a_is_black;
                if a_won {
                    stats.wins_a += 1;
                } else {
                    stats.wins_b += 1;
                }
            }
            GameState::Draw => stats.draws += 1,
            GameState::InProgress => unreachable!(),
        }
    }
    stats
}

fn time_search<F>(label: &str, eval_fn: F)
where
    F: FnMut(&Game, &mut SmallRng) -> f32,
{
    let g = Game::new_standard();
    let cfg = SearchConfig {
        simulations: 800,
        c_puct: 1.4,
    };
    let mut rng = SmallRng::seed_from_u64(0);
    let t = Instant::now();
    let r: SearchResult = search(&g, &cfg, &mut rng, eval_fn).unwrap();
    let dt = t.elapsed();
    let sims_per_sec = cfg.simulations as f64 / dt.as_secs_f64();
    println!(
        "search-{} (standard, sims={}): {:?} ({:.0} sims/sec, best={:?})",
        label, cfg.simulations, dt, sims_per_sec, r.best
    );
}

fn main() {
    println!("== abalone-mcts bench ==\n");

    // 1) Single-search timing
    time_search("random-rollout", random_rollout);
    time_search("heuristic", heuristic);

    let cfg = SearchConfig {
        simulations: 200,
        c_puct: 1.4,
    };

    // 2) heuristic-MCTS vs random. Heuristic eval is fast enough (~0.5M
    //    sims/sec) that we can afford a higher sim count here than in the
    //    cross-match below.
    {
        let cfg_h = SearchConfig {
            simulations: 800,
            c_puct: 1.4,
        };
        let games = 30u32;
        println!(
            "\nheuristic-MCTS-{} vs uniform-random ({} games):",
            cfg_h.simulations, games
        );
        let t = Instant::now();
        let stats = run_match(
            games,
            |seed| {
                let cfg = cfg_h.clone();
                let mut rng = SmallRng::seed_from_u64(seed);
                Box::new(move |g: &Game| pick_mcts_heuristic(g, &cfg, &mut rng))
            },
            |seed| {
                let mut rng = SmallRng::seed_from_u64(seed);
                Box::new(move |g: &Game| pick_random(g, &mut rng))
            },
        );
        stats.print("heuristic", "random");
        println!("  match took {:?}", t.elapsed());
    }

    // 3) heuristic-MCTS vs random-rollout-MCTS (same sim count)
    {
        println!(
            "\nheuristic-MCTS-{} vs random-rollout-MCTS-{} (20 games):",
            cfg.simulations, cfg.simulations
        );
        let t = Instant::now();
        let stats = run_match(
            20,
            |seed| {
                let cfg = cfg.clone();
                let mut rng = SmallRng::seed_from_u64(seed);
                Box::new(move |g: &Game| pick_mcts_heuristic(g, &cfg, &mut rng))
            },
            |seed| {
                let cfg = cfg.clone();
                let mut rng = SmallRng::seed_from_u64(seed);
                Box::new(move |g: &Game| pick_mcts_random_rollout(g, &cfg, &mut rng))
            },
        );
        stats.print("heuristic", "rollout");
        println!("  match took {:?}", t.elapsed());
    }

    // 4) Weight tuning
    {
        println!("\nweight tune (round-robin, 4 games per ordered pair):");
        // After the first round of tuning identified that tripling the
        // centrality weight wins decisively, the new `default` is that
        // configuration `(6.0, 0.15, 0.10)`. The other candidates probe
        // around it.
        let candidates: &[(&str, Weights)] = &[
            ("default", Weights::default()),
            (
                "more-center",
                Weights {
                    w_capture: 6.0,
                    w_center: 0.30,
                    w_cohesion: 0.10,
                },
            ),
            (
                "more-cohesion",
                Weights {
                    w_capture: 6.0,
                    w_center: 0.15,
                    w_cohesion: 0.25,
                },
            ),
            (
                "high-cap",
                Weights {
                    w_capture: 10.0,
                    w_center: 0.15,
                    w_cohesion: 0.10,
                },
            ),
        ];
        let cfg_tune = SearchConfig {
            simulations: 100,
            c_puct: 1.4,
        };
        let games_per_pair = 4u32;
        let n = candidates.len();
        let mut wins = vec![0u32; n];
        let mut draws_total = 0u32;
        let t = Instant::now();
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                let stats = run_match(
                    games_per_pair,
                    |seed| {
                        let cfg = cfg_tune.clone();
                        let weights = candidates[i].1.clone();
                        let mut rng = SmallRng::seed_from_u64(
                            seed ^ ((i as u64).wrapping_mul(1_000_003)),
                        );
                        Box::new(move |g: &Game| {
                            pick_mcts_with_weights(g, &cfg, &mut rng, &weights)
                        })
                    },
                    |seed| {
                        let cfg = cfg_tune.clone();
                        let weights = candidates[j].1.clone();
                        let mut rng = SmallRng::seed_from_u64(
                            seed ^ ((j as u64).wrapping_mul(2_000_003)),
                        );
                        Box::new(move |g: &Game| {
                            pick_mcts_with_weights(g, &cfg, &mut rng, &weights)
                        })
                    },
                );
                wins[i] += stats.wins_a;
                wins[j] += stats.wins_b;
                draws_total += stats.draws;
            }
        }
        let total_games = (n * (n - 1)) as u32 * games_per_pair;
        println!(
            "  {} ordered pairs · {} games each · {} total games · took {:?}",
            n * (n - 1),
            games_per_pair,
            total_games,
            t.elapsed()
        );
        println!("  draws: {}", draws_total);
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by_key(|&i| std::cmp::Reverse(wins[i]));
        for i in order {
            println!("    {:14} wins {}", candidates[i].0, wins[i]);
        }
    }
}
