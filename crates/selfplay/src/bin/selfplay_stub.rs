//! End-to-end self-play smoke test and **shard fixture generator**: plays a few
//! games with the heuristic evaluator standing in for the network and writes a
//! parquet shard. Run with:
//!
//!     cargo run --release -p abalone-selfplay --bin selfplay-stub -- \
//!         --games 4 --simulations 100 --out /tmp/abalone_shard_stub.parquet
//!
//! The output shard is consumable by `pyarrow.parquet.read_table` on the Python
//! side (`tests/test_replay_buffer.py` builds this binary to get a fixture) and
//! lets us verify the format independently of the NN bridge.
//!
//! This is the ONLY place the hand-written evaluator still appears on the
//! generation side, and only because a fixture needs no network. The training
//! loop's `selfplay-batch` is network-only — see `docs/MODEL.md`.

use std::path::PathBuf;
use std::time::Instant;

use abalone_game::{Game, Opening};
use abalone_mcts::{heuristic, LeafEval};
use abalone_selfplay::{play_game, shard::ShardWriter, SelfPlayConfig};
use rand::rngs::SmallRng;
use rand::SeedableRng;

fn parse_args() -> (u32, u32, PathBuf, u64) {
    let mut games = 4u32;
    let mut simulations = 100u32;
    let mut out = PathBuf::from("/tmp/abalone_shard_stub.parquet");
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
            other => eprintln!("unknown arg: {other}"),
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
        // One budget for everything: the fixture wants a policy target on every
        // row, so no playout cap randomisation here.
        sims_fast: simulations,
        sims_full: simulations,
        full_search_rate: 1.0,
        batch_size: 8,
        temperature_plies: 50,
        // The heuristic's priors are uniform, so noise would add nothing.
        dirichlet_eps: 0.0,
        opening: Opening::BelgianDaisy,
        handicap_rate: 0.7,
        ..Default::default()
    };

    let mut writer = ShardWriter::create(&out).expect("create shard");
    let t = Instant::now();

    let mut eval = |batch: &[Game]| -> Vec<LeafEval> {
        let mut rng = SmallRng::seed_from_u64(0);
        batch.iter().map(|g| heuristic(g, &mut rng)).collect()
    };

    for i in 0..games {
        let game_t = Instant::now();
        let outcome = play_game(&cfg, i, seed.wrapping_add(u64::from(i)), &mut eval);
        let plies = outcome.trajectory.len();
        let final_state = outcome.final_state.state();
        writer.write_game(&outcome).expect("write_game");
        println!(
            "  game {}: {} plies, final={:?}, handicap={}/{}, took {:?}",
            i,
            plies,
            final_state,
            outcome.handicap_black,
            outcome.handicap_white,
            game_t.elapsed()
        );
    }

    let entries = writer.finish().expect("finish");
    let on_disk = std::fs::metadata(&out).map(|m| m.len()).unwrap_or(0);

    let dt = t.elapsed();
    println!("wrote {entries} entries / {on_disk} bytes in {dt:?}  ({games} games)");
}
