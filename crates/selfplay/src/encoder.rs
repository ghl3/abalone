//! Position → `(14, 9, 9)` plane encoding for the NN evaluator.
//!
//! Mirrors `model/encoder.py::encode_position`. The two implementations must
//! agree **exactly** — not approximately — and that agreement is enforced by
//! `tests/test_conformance.py`, which replays the golden fixtures emitted by
//! `bin/dump_golden.rs` through the Python encoder and asserts bitwise float
//! equality. A silent divergence here (swapped capture planes) was live for
//! three generations, so the conformance test is part of the contract, not a
//! nicety. See `docs/ARCHITECTURE.md` §5.2 and `docs/MODEL.md` §5.
//!
//! Layout (matches Python):
//!
//! | plane | contents                                                       |
//! |-------|----------------------------------------------------------------|
//! | 0     | own marbles (binary) — the side to move                        |
//! | 1     | opponent marbles (binary)                                      |
//! | 2..7  | own-losses thermometer: plane `2 + k` is 1 iff `own >= k + 1`   |
//! | 7..12 | opponent-losses thermometer: plane `7 + k` is 1 iff `opp >= k+1`|
//! | 12    | `ply / max_plies`, clamped to `[0, 1]`                          |
//! | 13    | valid-cell mask (1 on the 61 on-board cells)                    |
//!
//! **Side-to-move relative.** Planes 0/1 are own/opponent, never black/white,
//! so one network plays both colours with no side-to-move embedding.
//!
//! **Thermometer, not a `count / 6` scalar.** Losses are ordinal with a hard
//! threshold at 6; `>= 1, >= 2, ...` lets a single linear layer read off both
//! the magnitude and "one away from losing".
//!
//! **The constant planes (2..14) are filled across all 81 cells**, including
//! the 20 off-board slots. The network masks off-board activations internally,
//! so the encoder must *not* pre-mask — and both implementations must agree on
//! that, which the golden fixtures enforce.
//!
//! **Hex geometry needs no special handling.** In axial coordinates all six
//! neighbours land inside a plain 3×3 kernel, so `Conv2d` is correct here.

use abalone_game::bitboard::{BitIter, VALID_MASK};
use abalone_game::Game;

pub const NUM_INPUT_CHANNELS: usize = 14;
pub const BOARD_H: usize = 9;
pub const BOARD_W: usize = 9;
/// Cells in one plane, including the 20 off-board slots.
pub const CELLS_PER_PLANE: usize = BOARD_H * BOARD_W;
pub const PLANE_SIZE: usize = NUM_INPUT_CHANNELS * CELLS_PER_PLANE;

/// Plane index of the side-to-move's marbles.
pub const OWN_MARBLES_PLANE: usize = 0;
/// Plane index of the opponent's marbles.
pub const OPP_MARBLES_PLANE: usize = 1;
/// First plane of the own-losses thermometer.
pub const OWN_LOSSES_PLANE: usize = 2;
/// First plane of the opponent-losses thermometer.
pub const OPP_LOSSES_PLANE: usize = 7;
/// Thresholds per thermometer: `>= 1 ..= >= 5`. Six losses is terminal, so a
/// `>= 6` plane would only ever be set in a position that is already over.
pub const LOSS_THERMOMETER_PLANES: usize = 5;
/// Plane index of the normalised ply counter.
pub const PLY_PLANE: usize = 12;
/// Plane index of the valid-cell mask.
pub const VALID_MASK_PLANE: usize = 13;

const _: () = assert!(OPP_LOSSES_PLANE == OWN_LOSSES_PLANE + LOSS_THERMOMETER_PLANES);
const _: () = assert!(PLY_PLANE == OPP_LOSSES_PLANE + LOSS_THERMOMETER_PLANES);
const _: () = assert!(NUM_INPUT_CHANNELS == VALID_MASK_PLANE + 1);

/// Number of marbles `side` has **lost** (had pushed off the board).
///
/// The naming trap that produced the original plane-swap bug:
/// `Board::pushed_off[s]` counts the marbles side `s` has pushed off *the
/// opponent*, so `s`'s own losses live in `pushed_off[s.other()]`. The field
/// name reads naturally as the opposite of what it holds. `Board::lost` does
/// the flip for us; never index `pushed_off` directly here.
#[inline]
fn losses(game: &Game, side: abalone_game::Side) -> u8 {
    game.board.lost(side)
}

/// Normalised ply, `ply / max_plies` clamped to `[0, 1]`.
///
/// Computed in `f64` and narrowed once, so it is bit-identical to Python's
/// `min(ply / max_plies, 1.0)` cast to `float32`. `max_plies == 0` would be a
/// degenerate configuration; it yields 0 rather than a NaN, and Python agrees.
#[inline]
fn ply_fraction(ply: u32, max_plies: u32) -> f32 {
    if max_plies == 0 {
        return 0.0;
    }
    (f64::from(ply) / f64::from(max_plies)).clamp(0.0, 1.0) as f32
}

/// Fill `out` with the encoded planes. `out.len()` must be [`PLANE_SIZE`].
pub fn encode_planes(game: &Game, out: &mut [f32]) {
    debug_assert_eq!(out.len(), PLANE_SIZE);
    out.fill(0.0);

    let plane = |i: usize| i * CELLS_PER_PLANE;

    // --- planes 0/1: marbles, side-to-move relative ---
    let own = game.board.bb(game.turn);
    let opp = game.board.bb(game.turn.other());
    for c in BitIter(own) {
        out[plane(OWN_MARBLES_PLANE) + c as usize] = 1.0;
    }
    for c in BitIter(opp) {
        out[plane(OPP_MARBLES_PLANE) + c as usize] = 1.0;
    }

    // --- planes 2..12: loss thermometers ---
    // `own_losses` = marbles the SIDE TO MOVE has had pushed off.
    let own_losses = losses(game, game.turn);
    let opp_losses = losses(game, game.turn.other());
    for k in 0..LOSS_THERMOMETER_PLANES {
        let threshold = (k + 1) as u8;
        if own_losses >= threshold {
            let base = plane(OWN_LOSSES_PLANE + k);
            out[base..base + CELLS_PER_PLANE].fill(1.0);
        }
        if opp_losses >= threshold {
            let base = plane(OPP_LOSSES_PLANE + k);
            out[base..base + CELLS_PER_PLANE].fill(1.0);
        }
    }

    // --- plane 12: normalised ply ---
    // Normalised by the *game's* cap, not a hardcoded constant, so changing
    // the cap moves both implementations in lockstep.
    let ply_norm = ply_fraction(game.ply, game.max_plies);
    let base = plane(PLY_PLANE);
    out[base..base + CELLS_PER_PLANE].fill(ply_norm);

    // --- plane 13: valid-cell mask ---
    for c in BitIter(VALID_MASK) {
        out[plane(VALID_MASK_PLANE) + c as usize] = 1.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use abalone_game::game::{DEFAULT_MAX_PLIES, NO_PROGRESS_DISABLED};
    use abalone_game::{Opening, Side};
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    fn encode(g: &Game) -> Vec<f32> {
        let mut buf = vec![0f32; PLANE_SIZE];
        encode_planes(g, &mut buf);
        buf
    }

    fn plane_of(buf: &[f32], i: usize) -> &[f32] {
        &buf[i * CELLS_PER_PLANE..(i + 1) * CELLS_PER_PLANE]
    }

    fn plane_sum(buf: &[f32], i: usize) -> f32 {
        plane_of(buf, i).iter().sum()
    }

    /// The thermometer planes a given loss count should light up, as a bitmask
    /// over `k in 0..5`.
    fn thermometer(buf: &[f32], first_plane: usize) -> Vec<f32> {
        (0..LOSS_THERMOMETER_PLANES)
            .map(|k| {
                let p = plane_of(buf, first_plane + k);
                // Constant planes must be constant across all 81 cells.
                assert!(
                    p.iter().all(|&v| v == p[0]),
                    "plane {} is not constant",
                    first_plane + k
                );
                p[0]
            })
            .collect()
    }

    #[test]
    fn plane_count_and_size() {
        assert_eq!(NUM_INPUT_CHANNELS, 14);
        assert_eq!(CELLS_PER_PLANE, 81);
        assert_eq!(PLANE_SIZE, 14 * 81);
    }

    #[test]
    fn standard_opening_planes_are_well_formed() {
        let g = Game::new_standard();
        let buf = encode(&g);

        // Marbles: 14 each, and they never overlap.
        assert_eq!(plane_sum(&buf, OWN_MARBLES_PLANE), 14.0);
        assert_eq!(plane_sum(&buf, OPP_MARBLES_PLANE), 14.0);
        let overlap = plane_of(&buf, OWN_MARBLES_PLANE)
            .iter()
            .zip(plane_of(&buf, OPP_MARBLES_PLANE))
            .position(|(&a, &b)| a * b != 0.0);
        assert_eq!(overlap, None, "own and opp marbles overlap");

        // No losses yet: every thermometer plane is cold.
        for k in 0..LOSS_THERMOMETER_PLANES {
            assert_eq!(plane_sum(&buf, OWN_LOSSES_PLANE + k), 0.0);
            assert_eq!(plane_sum(&buf, OPP_LOSSES_PLANE + k), 0.0);
        }

        // Ply 0 of 200.
        assert_eq!(plane_sum(&buf, PLY_PLANE), 0.0);

        // Valid mask covers exactly the 61 on-board cells.
        assert_eq!(plane_sum(&buf, VALID_MASK_PLANE), 61.0);
    }

    #[test]
    fn marble_planes_are_side_to_move_relative() {
        let mut g = Game::new_standard();
        let black_view = encode(&g);
        let mv = g.legal_moves()[0];
        g.apply(mv);
        assert_eq!(g.turn, Side::White);
        let white_view = encode(&g);

        // White to move: plane 0 now holds white's marbles, which under the
        // black view were plane 1 (black's move did not touch them).
        assert_eq!(
            plane_of(&white_view, OWN_MARBLES_PLANE),
            plane_of(&black_view, OPP_MARBLES_PLANE),
            "plane 0 must follow the side to move"
        );
    }

    #[test]
    fn marble_planes_place_marbles_at_the_right_cells() {
        let g = Game::new_standard();
        let buf = encode(&g);
        let own = plane_of(&buf, OWN_MARBLES_PLANE);
        let opp = plane_of(&buf, OPP_MARBLES_PLANE);
        // Black (to move) holds I5 (r=8, q=4) and G5 (r=6, q=4).
        assert_eq!(own[8 * 9 + 4], 1.0);
        assert_eq!(own[6 * 9 + 4], 1.0);
        // White holds A1 (r=0, q=0) and C3 (r=2, q=2).
        assert_eq!(opp[0], 1.0);
        assert_eq!(opp[2 * 9 + 2], 1.0);
        // E5 (r=4, q=4) is empty in both.
        assert_eq!(own[4 * 9 + 4], 0.0);
        assert_eq!(opp[4 * 9 + 4], 0.0);
    }

    #[test]
    fn thermometer_is_monotone_for_every_loss_count() {
        let mut rng = SmallRng::seed_from_u64(1);
        for n in 0..=5u8 {
            let g = Game::with_handicap(
                Opening::Standard,
                DEFAULT_MAX_PLIES,
                NO_PROGRESS_DISABLED,
                n,
                0,
                &mut rng,
            );
            assert_eq!(g.turn, Side::Black);
            let buf = encode(&g);
            let own = thermometer(&buf, OWN_LOSSES_PLANE);
            let expected: Vec<f32> = (0..5)
                .map(|k| if n >= (k + 1) as u8 { 1.0 } else { 0.0 })
                .collect();
            assert_eq!(own, expected, "own thermometer wrong for {n} losses");
            // Monotone non-increasing: `>= k` implies `>= k-1`.
            for k in 1..LOSS_THERMOMETER_PLANES {
                assert!(own[k] <= own[k - 1]);
            }
            // The opponent lost nothing.
            assert_eq!(thermometer(&buf, OPP_LOSSES_PLANE), vec![0.0; 5]);
        }
    }

    /// The regression test for the bug that motivated the conformance suite:
    /// own and opponent losses must land in *their own* plane group. Any swap
    /// (here or in Python) flips these assertions.
    #[test]
    fn asymmetric_handicap_lands_in_the_right_plane_group() {
        let mut rng = SmallRng::seed_from_u64(2);
        // Black has lost 4, White has lost 1. Black to move, so
        // own_losses = 4, opp_losses = 1.
        let g = Game::with_handicap(
            Opening::BelgianDaisy,
            DEFAULT_MAX_PLIES,
            NO_PROGRESS_DISABLED,
            4,
            1,
            &mut rng,
        );
        assert_eq!(g.turn, Side::Black);
        assert_eq!(g.board.lost(Side::Black), 4);
        assert_eq!(g.board.lost(Side::White), 1);
        // ...and the counters they were credited to are the *other* side's,
        // which is exactly the naming trap.
        assert_eq!(g.board.pushed_off[Side::White.idx()], 4);
        assert_eq!(g.board.pushed_off[Side::Black.idx()], 1);

        let buf = encode(&g);
        assert_eq!(
            thermometer(&buf, OWN_LOSSES_PLANE),
            vec![1.0, 1.0, 1.0, 1.0, 0.0],
            "planes 2-6 must hold the SIDE TO MOVE's losses (4)"
        );
        assert_eq!(
            thermometer(&buf, OPP_LOSSES_PLANE),
            vec![1.0, 0.0, 0.0, 0.0, 0.0],
            "planes 7-11 must hold the opponent's losses (1)"
        );

        // Board counts corroborate: Black is down 4 marbles, White down 1.
        assert_eq!(plane_sum(&buf, OWN_MARBLES_PLANE), 10.0);
        assert_eq!(plane_sum(&buf, OPP_MARBLES_PLANE), 13.0);
    }

    #[test]
    fn thermometer_groups_swap_with_the_side_to_move() {
        let mut rng = SmallRng::seed_from_u64(3);
        let mut g = Game::with_handicap(
            Opening::BelgianDaisy,
            DEFAULT_MAX_PLIES,
            NO_PROGRESS_DISABLED,
            5,
            1,
            &mut rng,
        );
        let black_view = encode(&g);
        g.apply(g.legal_moves()[0]);
        assert_eq!(g.turn, Side::White);
        let white_view = encode(&g);

        assert_eq!(
            thermometer(&black_view, OWN_LOSSES_PLANE),
            thermometer(&white_view, OPP_LOSSES_PLANE),
            "black's losses become the opponent group once white is to move"
        );
        assert_eq!(
            thermometer(&black_view, OPP_LOSSES_PLANE),
            thermometer(&white_view, OWN_LOSSES_PLANE),
        );
    }

    #[test]
    fn ply_plane_uses_the_games_own_cap() {
        // A 100-ply cap must read twice as hot as the 200-ply default at the
        // same ply. This is the plumbing that used to be a hardcoded 400.
        let mut short = Game::new(Opening::Standard, 100, NO_PROGRESS_DISABLED);
        let mut long = Game::new(Opening::Standard, 200, NO_PROGRESS_DISABLED);
        for _ in 0..10 {
            short.apply(short.legal_moves()[0]);
            long.apply(long.legal_moves()[0]);
        }
        assert_eq!(plane_of(&encode(&short), PLY_PLANE)[0], 0.10);
        assert_eq!(plane_of(&encode(&long), PLY_PLANE)[0], 0.05);
    }

    #[test]
    fn ply_plane_clamps_and_survives_a_zero_cap() {
        let mut g = Game::new(Opening::Standard, 8, NO_PROGRESS_DISABLED);
        g.ply = 40; // past the cap; the plane saturates rather than exceeding 1
        assert_eq!(plane_of(&encode(&g), PLY_PLANE)[0], 1.0);

        g.max_plies = 0;
        let v = plane_of(&encode(&g), PLY_PLANE)[0];
        assert!(v.is_finite() && v == 0.0, "degenerate cap must not produce NaN");
    }

    #[test]
    fn constant_planes_cover_all_81_cells_including_off_board() {
        // Deliberate: the encoder does NOT mask the constant planes. The
        // network masks internally, and the golden fixtures pin this down, so
        // "helpfully" masking here silently breaks conformance.
        let mut rng = SmallRng::seed_from_u64(4);
        let mut g = Game::with_handicap(
            Opening::Standard,
            DEFAULT_MAX_PLIES,
            NO_PROGRESS_DISABLED,
            3,
            2,
            &mut rng,
        );
        g.ply = 50;
        let buf = encode(&g);

        // Every set thermometer plane sums to 81, not 61.
        for k in 0..3 {
            assert_eq!(plane_sum(&buf, OWN_LOSSES_PLANE + k), 81.0);
        }
        for k in 0..2 {
            assert_eq!(plane_sum(&buf, OPP_LOSSES_PLANE + k), 81.0);
        }
        assert_eq!(plane_sum(&buf, PLY_PLANE), 81.0 * 0.25);

        // The mask plane is the one plane that IS restricted to 61 cells.
        let mask = plane_of(&buf, VALID_MASK_PLANE);
        assert_eq!(mask.iter().sum::<f32>(), 61.0);
        // (r=1, q=6): |q - r| = 5, off-board.
        assert_eq!(mask[9 + 6], 0.0);
        assert_eq!(mask[4 * 9 + 4], 1.0);
    }

    #[test]
    fn marbles_never_appear_off_board() {
        let mut rng = SmallRng::seed_from_u64(5);
        let mut g = Game::new_belgian_daisy();
        for _ in 0..60 {
            if g.is_terminal() {
                break;
            }
            let moves = g.legal_moves();
            let pick = rng.gen_range(0..moves.len());
            g.apply(moves[pick]);
            let buf = encode(&g);
            let mask = plane_of(&buf, VALID_MASK_PLANE);
            for p in [OWN_MARBLES_PLANE, OPP_MARBLES_PLANE] {
                for (i, (&m, &v)) in mask.iter().zip(plane_of(&buf, p)).enumerate() {
                    assert!(m != 0.0 || v == 0.0, "marble on off-board cell {i} of plane {p}");
                }
            }
        }
    }

    #[test]
    fn every_value_is_zero_or_one_except_the_ply_plane() {
        let mut rng = SmallRng::seed_from_u64(6);
        let mut g = Game::with_handicap(
            Opening::BelgianDaisy,
            DEFAULT_MAX_PLIES,
            NO_PROGRESS_DISABLED,
            2,
            5,
            &mut rng,
        );
        g.ply = 77;
        let buf = encode(&g);
        for p in 0..NUM_INPUT_CHANNELS {
            if p == PLY_PLANE {
                continue;
            }
            for (i, &v) in plane_of(&buf, p).iter().enumerate() {
                assert!(v == 0.0 || v == 1.0, "plane {p} cell {i} = {v}");
            }
        }
    }
}
