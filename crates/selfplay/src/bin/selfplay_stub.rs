//! End-to-end self-play smoke test: heuristic-MCTS as the leaf evaluator,
//! writing a binary shard. Run with:
//!
//!     cargo run --release -p abalone-selfplay --bin selfplay-stub -- \
//!         --games 4 --simulations 100 --out /tmp/abalone_shard_stub.bin
//!
//! The output shard is then consumable by the Python training-side
//! reader (to be written) and lets us verify the format independently
//! of the NN bridge.

use std::path::PathBuf;
use std::time::Instant;

use abalone_mcts::heuristic;
use abalone_selfplay::{play_game, shard::ShardWriter, SelfPlayConfig};
use rand::rngs::SmallRng;
use rand::SeedableRng;

fn parse_args() -> (u32, u32, PathBuf, u64) {
    let mut games = 4u32;
    let mut simulations = 100u32;
    let mut out = PathBuf::from("/tmp/abalone_shard_stub.bin");
    let mut seed = 0u64;

    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--games" => games = args.next().expect("--games <N>").parse().unwrap(),
            "--simulations" => {
                simulations = args.next().expect("--simulations <N>").parse().unwrap()
            }
            "--out" => out = PathBuf::from(args.next().expect("--out <path>")),
            "--seed" => seed = args.next().expect("--seed <N>").parse().unwrap(),
            other => eprintln!("unknown arg: {}", other),
        }
    }
    (games, simulations, out, seed)
}

fn main() {
    let (games, simulations, out, seed) = parse_args();
    println!(
        "selfplay-stub: games={} simulations={} out={} seed={}",
        games,
        simulations,
        out.display(),
        seed
    );

    let cfg = SelfPlayConfig {
        simulations,
        c_puct: 1.4,
        temperature_plies: 50,
        temperature: 1.0,
    };

    let mut writer = ShardWriter::create(&out, simulations).expect("create shard");
    let t = Instant::now();

    for i in 0..games {
        let mut rng = SmallRng::seed_from_u64(seed.wrapping_add(i as u64));
        let game_t = Instant::now();
        let outcome = play_game(&cfg, &mut rng, heuristic);
        let plies = outcome.trajectory.len();
        let final_state = outcome.final_state.state();
        writer.write_game(&outcome).expect("write_game");
        println!(
            "  game {}: {} plies, final={:?}, took {:?}",
            i,
            plies,
            final_state,
            game_t.elapsed()
        );
    }

    let entries = writer.entries_written;
    let bytes = writer.bytes_written;
    writer.finish().expect("finish");

    let dt = t.elapsed();
    println!(
        "wrote {} entries / {} bytes in {:?}  ({} games)",
        entries, bytes, dt, games
    );
}
