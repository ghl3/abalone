//! Tiny black-box benchmark for legal-move generation and apply.
//! Run with `cargo run --release --bin movegen-bench`.

use std::hint::black_box;
use std::time::Instant;

use abalone_game::{
    Board, Game, GameState, Side,
    moves::legal_moves,
};

fn bench_legal_moves(board: &Board, side: Side, n: u64) -> f64 {
    // Warm.
    for _ in 0..1000 {
        let _ = black_box(legal_moves(black_box(board), black_box(side)));
    }
    let t = Instant::now();
    for _ in 0..n {
        let _ = black_box(legal_moves(black_box(board), black_box(side)));
    }
    let s = t.elapsed().as_secs_f64();
    n as f64 / s
}

fn bench_apply(n: u64) -> f64 {
    // Each iter: take the standard board, apply the first legal move, undo by re-cloning.
    let base = Board::standard();
    let m = legal_moves(&base, Side::Black)[0];
    // Warm.
    for _ in 0..1000 {
        let mut b = black_box(base);
        b.apply(black_box(m), Side::Black);
        let _ = black_box(b);
    }
    let t = Instant::now();
    for _ in 0..n {
        let mut b = black_box(base);
        b.apply(black_box(m), Side::Black);
        let _ = black_box(b);
    }
    let s = t.elapsed().as_secs_f64();
    n as f64 / s
}

fn bench_random_playouts(n: u64) -> (f64, u64) {
    // Run n full random self-plays, return (playouts/sec, mean ply).
    let mut total_ply: u64 = 0;
    let mut rng: u64 = 0xa5a5_a5a5_a5a5_a5a5;
    let t = Instant::now();
    for _ in 0..n {
        let mut g = Game::new_standard();
        while !g.is_terminal() {
            let moves = g.legal_moves();
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let pick = (rng as usize) % moves.len();
            g.apply(moves[pick]);
        }
        total_ply += g.ply as u64;
        black_box(&g);
    }
    let s = t.elapsed().as_secs_f64();
    (n as f64 / s, total_ply / n)
}

fn fmt_rate(per_sec: f64) -> String {
    if per_sec > 1e6 {
        format!("{:.2}M/s", per_sec / 1e6)
    } else if per_sec > 1e3 {
        format!("{:.2}K/s", per_sec / 1e3)
    } else {
        format!("{:.0}/s", per_sec)
    }
}

fn main() {
    println!("== abalone-game micro-bench ==");

    let standard = Board::standard();
    let belgian = Board::belgian_daisy();

    let n_count_std = legal_moves(&standard, Side::Black).len();
    let n_count_bel = legal_moves(&belgian, Side::Black).len();
    println!("legal moves at standard opening: {} (Black)", n_count_std);
    println!("legal moves at Belgian Daisy:    {} (Black)", n_count_bel);

    let n = 5_000_000u64;
    let r1 = bench_legal_moves(&standard, Side::Black, n);
    println!("legal_moves (standard, Black): {} ({} iters)", fmt_rate(r1), n);

    let r2 = bench_legal_moves(&belgian, Side::Black, n);
    println!("legal_moves (Belgian,  Black): {} ({} iters)", fmt_rate(r2), n);

    let r3 = bench_apply(20_000_000);
    println!("apply (clone+inline-1):        {}", fmt_rate(r3));

    let n_play = 5_000u64;
    let (r4, mean_ply) = bench_random_playouts(n_play);
    let nodes_per_sec = r4 * mean_ply as f64;
    println!(
        "random self-play:              {} playouts/sec (~{} ply each, ~{} positions/sec)",
        fmt_rate(r4),
        mean_ply,
        fmt_rate(nodes_per_sec),
    );

    // Sanity: a fresh game, run-through, must end.
    let (_, _) = (GameState::InProgress, ());
}
