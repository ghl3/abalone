//! Manual MCTS bench. Run with:
//!     cargo run --release -p abalone-mcts --bin mcts-bench
//!
//! Reports:
//!   * single-search timing on the standard opening (sims/sec)
//!   * MCTS-200 vs uniform-random match winrate
//!   * MCTS-200 vs MCTS-200 mirror-match winrate (sanity check; should be
//!     ~50% with no first-move advantage signal)

use std::time::Instant;

use abalone_engine::{Game, GameState, Move, Side};
use abalone_mcts::{search, SearchConfig};
use rand::rngs::SmallRng;
use rand::Rng;
use rand::SeedableRng;

fn pick_random<R: Rng + ?Sized>(g: &Game, rng: &mut R) -> Move {
    let moves = g.legal_moves();
    moves[rng.gen_range(0..moves.len())]
}

fn pick_mcts(g: &Game, cfg: &SearchConfig, rng: &mut SmallRng) -> Move {
    search(g, cfg, rng).expect("non-terminal => some move").best
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
) -> (GameState, u32)
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
    (g.state(), g.ply)
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
        let (state, _ply) = play_one_game(
            a_is_black,
            |g| a(g),
            |g| b(g),
        );
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

fn main() {
    println!("== abalone-mcts bench ==");

    // 1) Single-search timing.
    {
        let g = Game::new_standard();
        let cfg = SearchConfig {
            simulations: 800,
            c_puct: 1.4,
        };
        let mut rng = SmallRng::seed_from_u64(0);
        let t = Instant::now();
        let r = search(&g, &cfg, &mut rng).unwrap();
        let dt = t.elapsed();
        let sims_per_sec = cfg.simulations as f64 / dt.as_secs_f64();
        println!(
            "search(standard, sims={}): {:?} ({:.0} sims/sec, best={:?})",
            cfg.simulations, dt, sims_per_sec, r.best
        );
    }

    // 2) MCTS-200 vs random
    {
        let cfg = SearchConfig {
            simulations: 200,
            c_puct: 1.4,
        };
        let games = 30;
        println!("\nMCTS-{} vs random ({} games):", cfg.simulations, games);
        let t = Instant::now();
        let stats = run_match(
            games,
            |seed| {
                let cfg = cfg.clone();
                let mut rng = SmallRng::seed_from_u64(seed);
                Box::new(move |g: &Game| pick_mcts(g, &cfg, &mut rng))
            },
            |seed| {
                let mut rng = SmallRng::seed_from_u64(seed);
                Box::new(move |g: &Game| pick_random(g, &mut rng))
            },
        );
        stats.print("MCTS", "random");
        println!("  match took {:?}", t.elapsed());
    }

    // 3) MCTS-200 mirror match
    {
        let cfg = SearchConfig {
            simulations: 200,
            c_puct: 1.4,
        };
        let games = 12;
        println!("\nMCTS-{} mirror match ({} games):", cfg.simulations, games);
        let t = Instant::now();
        let stats = run_match(
            games,
            |seed| {
                let cfg = cfg.clone();
                let mut rng = SmallRng::seed_from_u64(seed);
                Box::new(move |g: &Game| pick_mcts(g, &cfg, &mut rng))
            },
            |seed| {
                let cfg = cfg.clone();
                let mut rng = SmallRng::seed_from_u64(seed.wrapping_add(1_000_000));
                Box::new(move |g: &Game| pick_mcts(g, &cfg, &mut rng))
            },
        );
        stats.print("A", "B");
        println!("  match took {:?}", t.elapsed());
    }
}
