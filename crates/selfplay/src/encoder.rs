//! Position → 6×9×9 plane encoding for the NN evaluator.
//!
//! Mirrors `model/encoder.py::encode_position`. The two implementations
//! must agree byte-for-byte; the canonical spec lives in Python (where
//! it's exercised by the test suite). This Rust copy exists because
//! self-play needs to encode positions on the hot path.
//!
//! Layout (matches Python):
//!   plane 0: own marbles (binary)
//!   plane 1: opp marbles (binary)
//!   plane 2: own captures broadcast = `lost(turn) / 6`
//!   plane 3: opp captures broadcast = `lost(turn.other()) / 6`
//!   plane 4: ply / 400 broadcast
//!   plane 5: valid mask (1 for the 61 on-board cells)

use abalone_game::bitboard::{BitIter, VALID_MASK};
use abalone_game::Game;

pub const NUM_INPUT_CHANNELS: usize = 6;
pub const BOARD_H: usize = 9;
pub const BOARD_W: usize = 9;
pub const PLANE_SIZE: usize = NUM_INPUT_CHANNELS * BOARD_H * BOARD_W;

/// Fill `out` with the encoded planes. `out.len()` must be `PLANE_SIZE`.
pub fn encode_planes(game: &Game, out: &mut [f32]) {
    debug_assert_eq!(out.len(), PLANE_SIZE);
    out.fill(0.0);

    let cell_count = (BOARD_H * BOARD_W) as usize;
    let own_off = 0;
    let opp_off = cell_count;
    let own_cap_off = cell_count * 2;
    let opp_cap_off = cell_count * 3;
    let ply_off = cell_count * 4;
    let mask_off = cell_count * 5;

    let own = game.board.bb(game.turn);
    let opp = game.board.bb(game.turn.other());

    for c in BitIter(own) {
        out[own_off + c as usize] = 1.0;
    }
    for c in BitIter(opp) {
        out[opp_off + c as usize] = 1.0;
    }

    let own_lost = game.board.lost(game.turn) as f32 / 6.0;
    let opp_lost = game.board.lost(game.turn.other()) as f32 / 6.0;
    let ply_norm = (game.ply.min(400) as f32) / 400.0;

    for i in 0..cell_count {
        out[own_cap_off + i] = own_lost;
        out[opp_cap_off + i] = opp_lost;
        out[ply_off + i] = ply_norm;
    }

    for c in BitIter(VALID_MASK) {
        out[mask_off + c as usize] = 1.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_opening_planes_are_well_formed() {
        let g = Game::new_standard();
        let mut buf = vec![0f32; PLANE_SIZE];
        encode_planes(&g, &mut buf);

        // Plane 0 (own = black) and plane 1 (opp = white) sum to 14 each.
        let p0_sum: f32 = buf[..81].iter().sum();
        let p1_sum: f32 = buf[81..162].iter().sum();
        assert_eq!(p0_sum, 14.0);
        assert_eq!(p1_sum, 14.0);

        // Captures planes are 0 (no marbles pushed off yet).
        let p2_sum: f32 = buf[162..243].iter().sum();
        let p3_sum: f32 = buf[243..324].iter().sum();
        assert_eq!(p2_sum, 0.0);
        assert_eq!(p3_sum, 0.0);

        // Ply plane is 0/400 = 0.
        let p4_sum: f32 = buf[324..405].iter().sum();
        assert_eq!(p4_sum, 0.0);

        // Valid-mask plane sums to 61.
        let p5_sum: f32 = buf[405..486].iter().sum();
        assert_eq!(p5_sum, 61.0);
    }
}
