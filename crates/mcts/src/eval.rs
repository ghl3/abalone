//! Hand-crafted Abalone evaluator. Linear combination of three positional
//! features, squashed by `tanh` into `[-1, 1]`.
//!
//! **RETIRED FROM THE TRAINING LOOP — DO NOT WIRE THIS BACK INTO SELF-PLAY.**
//! Per `docs/MODEL.md`, the network learns from self-play outcomes alone; a
//! hand-written evaluator as teacher or bootstrap would inject exactly the
//! human strategy priors the project exists to avoid. It is kept for two
//! purposes only: as a *fixed* benchmark opponent on the Elo ladder (a
//! yardstick that never moves), and as a cheap deterministic test fixture for
//! the search code. Neither use touches training data.
//!
//! From the side-to-move's perspective:
//! 1. **Capture differential** — `(opp marbles we pushed off) - (own marbles
//!    they pushed off)`. The win condition is 6 of either side, so this
//!    feature dominates as the game progresses.
//! 2. **Centrality** — each cell scores `4 - hex_distance(E5)`; centred
//!    marbles attack better and get pushed off less. Summed over own
//!    marbles minus opponent's.
//! 3. **Cohesion** — number of own-own adjacent pairs minus opp-opp pairs,
//!    counted along the three positive hex axes (so each pair counted
//!    once). Cohesive groups resist push attacks.

use abalone_game::bitboard::{popcount, shift, BB};
use abalone_game::{Board, Side, POSITIVE_DIRS};

#[derive(Clone, Debug)]
pub struct Weights {
    pub w_capture: f32,
    pub w_center: f32,
    pub w_cohesion: f32,
}

impl Default for Weights {
    fn default() -> Self {
        // Picked by the round-robin tune in `mcts-bench`: tripling the
        // centrality weight beat the original `(6.0, 0.05, 0.10)` default
        // 12-4 over 24 games each, with the other two candidates
        // (strong-cap, cohesion-heavy) ranking lower. Captures still
        // dominate end-game eval (tanh(6 * 1) = 0.99), but mid-game now
        // has stronger positional pull toward the centre.
        Self {
            w_capture: 6.0,
            w_center: 0.15,
            w_cohesion: 0.10,
        }
    }
}

const fn abs_i32(x: i32) -> i32 {
    if x < 0 {
        -x
    } else {
        x
    }
}

const fn max3(a: i32, b: i32, c: i32) -> i32 {
    let ab = if a > b { a } else { b };
    if ab > c {
        ab
    } else {
        c
    }
}

const fn build_centerness() -> [u8; 81] {
    let mut t = [0u8; 81];
    let mut c = 0u8;
    while c < 81 {
        let r = c / 9;
        let q = c % 9;
        let u = q as i32 - 4;
        let v = r as i32 - 4;
        let d = max3(abs_i32(u), abs_i32(v), abs_i32(u - v));
        if d <= 4 {
            t[c as usize] = (4 - d) as u8;
        }
        c += 1;
    }
    t
}

/// `CENTERNESS[c]` ∈ `0..=4`, with `4` at E5 and `0` at the corner ring.
/// Off-board cells are `0` (and never queried in practice).
pub const CENTERNESS: [u8; 81] = build_centerness();

/// Sum of `CENTERNESS` over the set bits of `bb`.
pub fn centrality(bb: BB) -> i32 {
    let mut sum = 0i32;
    let mut x = bb;
    while x != 0 {
        let c = x.trailing_zeros() as usize;
        sum += CENTERNESS[c] as i32;
        x &= x - 1;
    }
    sum
}

/// Number of adjacent same-colour pairs in `bb`, counting along the three
/// positive hex axes (E, NE, NW) so each pair is counted once.
pub fn cohesion(bb: BB) -> i32 {
    let mut total = 0;
    for d in POSITIVE_DIRS {
        total += popcount(bb & shift(bb, d)) as i32;
    }
    total
}

/// Evaluate `board` from `to_move`'s perspective. Result is in `[-1, 1]`
/// (closed at the asymptotes via `tanh`).
pub fn evaluate(board: &Board, to_move: Side, w: &Weights) -> f32 {
    let own = board.bb(to_move);
    let opp = board.bb(to_move.other());

    let cap_for_us = board.lost(to_move.other()) as i32;
    let cap_for_them = board.lost(to_move) as i32;
    let cap_diff = (cap_for_us - cap_for_them) as f32;

    let center_diff = (centrality(own) - centrality(opp)) as f32;
    let cohesion_diff = (cohesion(own) - cohesion(opp)) as f32;

    (w.w_capture * cap_diff + w.w_center * center_diff + w.w_cohesion * cohesion_diff).tanh()
}

#[cfg(test)]
mod tests {
    use super::*;
    use abalone_game::cell::parse;

    #[test]
    fn centerness_is_4_at_e5() {
        assert_eq!(CENTERNESS[parse("E5").unwrap() as usize], 4);
    }

    #[test]
    fn centerness_is_0_at_corners() {
        for n in &["A1", "A5", "E1", "E9", "I5", "I9"] {
            let c = parse(n).unwrap();
            assert_eq!(CENTERNESS[c as usize], 0, "{}", n);
        }
    }

    #[test]
    fn centerness_intermediate_ring() {
        // B2 is one step in from A1 along the NE diagonal: distance 3 from E5,
        // so centerness = 4 - 3 = 1.
        assert_eq!(CENTERNESS[parse("B2").unwrap() as usize], 1);
        // D4 is two steps in along the same diagonal: distance 1, centerness 3.
        assert_eq!(CENTERNESS[parse("D4").unwrap() as usize], 3);
    }

    #[test]
    fn cohesion_pair_is_one() {
        let mut b = Board::empty();
        b.set(parse("C3").unwrap(), Some(Side::White));
        b.set(parse("C4").unwrap(), Some(Side::White));
        assert_eq!(cohesion(b.bb(Side::White)), 1);
    }

    #[test]
    fn cohesion_inline_triple_is_two() {
        let mut b = Board::empty();
        b.set(parse("C3").unwrap(), Some(Side::White));
        b.set(parse("C4").unwrap(), Some(Side::White));
        b.set(parse("C5").unwrap(), Some(Side::White));
        assert_eq!(cohesion(b.bb(Side::White)), 2);
    }

    #[test]
    fn cohesion_tight_triangle_is_three() {
        // C3, C4, D4: three pairs (C3-C4 along E, C3-D4 along NE, C4-D4 along NW).
        let mut b = Board::empty();
        b.set(parse("C3").unwrap(), Some(Side::White));
        b.set(parse("C4").unwrap(), Some(Side::White));
        b.set(parse("D4").unwrap(), Some(Side::White));
        assert_eq!(cohesion(b.bb(Side::White)), 3);
    }

    #[test]
    fn standard_opening_evaluates_to_zero() {
        let b = Board::standard();
        let w = Weights::default();
        assert!(evaluate(&b, Side::Black, &w).abs() < 1e-6);
        assert!(evaluate(&b, Side::White, &w).abs() < 1e-6);
    }

    #[test]
    fn near_capture_loss_is_strongly_negative() {
        // White has pushed off 5 of black's marbles -> black is one capture
        // away from losing.
        let mut b = Board::standard();
        b.pushed_off[Side::White.idx()] = 5;
        let w = Weights::default();
        let v_black = evaluate(&b, Side::Black, &w);
        let v_white = evaluate(&b, Side::White, &w);
        assert!(v_black < -0.99, "expected black eval near -1, got {}", v_black);
        assert!(v_white > 0.99, "expected white eval near +1, got {}", v_white);
    }

    #[test]
    fn centred_marbles_beat_corner_marbles() {
        // White has its 14 marbles in the centre; black has its 14 along the
        // outer ring (well, as many as fit). White should evaluate strongly
        // positive even at zero captures.
        let w = Weights::default();
        let mut b = Board::empty();
        // Pack white into the inner positions: D4, D5, E4, E5, E6, F5, F6, ...
        for n in &[
            "D4", "D5", "D6", "E4", "E5", "E6", "F4", "F5", "F6",
            "C4", "C5", "G5", "G6", "G7",
        ] {
            b.set(parse(n).unwrap(), Some(Side::White));
        }
        // Spread black around the outer ring.
        for n in &[
            "A1", "A2", "A4", "A5", "B1", "B6", "I5", "I6", "I8", "I9", "H4", "H9", "E1", "E9",
        ] {
            b.set(parse(n).unwrap(), Some(Side::Black));
        }
        let v_white = evaluate(&b, Side::White, &w);
        assert!(v_white > 0.0, "centred white should be > 0, got {}", v_white);
    }
}
