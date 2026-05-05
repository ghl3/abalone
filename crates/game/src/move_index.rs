//! Flat enumeration of all (anchor × direction × size) move possibilities.
//!
//! 1098 inline + 1464 broadside = 2562 total indices. Many encode positions
//! that are structurally illegal (e.g. a size-3 inline whose anchor leaves
//! the board) — these never come back from [`crate::moves::legal_moves`],
//! and the policy network masks them at sample time.
//!
//! Layout:
//!   `[0 .. 1098)` inline:    `idx = anchor_compact * 18 + dir.idx * 3 + (size - 1)`
//!   `[1098 .. 2562)` broadside:
//!     `j = idx - 1098 = anchor_compact * 24 + gi * 8 + mi * 2 + (size - 2)`
//!     where `gi ∈ 0..3` indexes [`POSITIVE_DIRS`] and
//!           `mi ∈ 0..4` indexes [`BROADSIDE_DIRS`]`[gi]`.
//!
//! `anchor_compact` is the anchor's index into the 61 valid cells (the cell
//! itself is `0..81`; we collapse out the 20 off-board slots).

use crate::cell::{in_board, Cell, Dir, BROADSIDE_DIRS, N_CELLS, N_VALID, POSITIVE_DIRS};
use crate::moves::Move;

pub const MOVE_SPACE: usize = 2562;
pub const INLINE_OFFSET: usize = 0;
pub const BROADSIDE_OFFSET: usize = 1098;

const N_INLINE: usize = N_VALID * 6 * 3; // 1098
const N_BROADSIDE: usize = N_VALID * 3 * 4 * 2; // 1464

const _: () = assert!(N_INLINE + N_BROADSIDE == MOVE_SPACE);
const _: () = assert!(BROADSIDE_OFFSET == N_INLINE);

const INVALID_COMPACT: u8 = 0xFF;

const fn build_cell_to_compact() -> [u8; N_CELLS] {
    let mut out = [INVALID_COMPACT; N_CELLS];
    let mut idx: u8 = 0;
    let mut c: u8 = 0;
    while (c as usize) < N_CELLS {
        let r = c / 9;
        let q = c % 9;
        if in_board(r as u32, q as u32) {
            out[c as usize] = idx;
            idx += 1;
        }
        c += 1;
    }
    out
}

const fn build_compact_to_cell() -> [Cell; N_VALID] {
    let mut out = [0u8; N_VALID];
    let mut idx: usize = 0;
    let mut c: u8 = 0;
    while (c as usize) < N_CELLS {
        let r = c / 9;
        let q = c % 9;
        if in_board(r as u32, q as u32) {
            out[idx] = c;
            idx += 1;
        }
        c += 1;
    }
    out
}

static CELL_TO_COMPACT: [u8; N_CELLS] = build_cell_to_compact();
static COMPACT_TO_CELL: [Cell; N_VALID] = build_compact_to_cell();

const fn positive_dir_idx(d: Dir) -> usize {
    match d {
        Dir::E => 0,
        Dir::NE => 1,
        Dir::NW => 2,
        _ => panic!("group_dir not in POSITIVE_DIRS"),
    }
}

fn broadside_dir_idx(gi: usize, d: Dir) -> usize {
    BROADSIDE_DIRS[gi]
        .iter()
        .position(|&x| x as u8 == d as u8)
        .expect("move_dir not in BROADSIDE_DIRS[gi]")
}

pub fn encode(mv: Move) -> u16 {
    match mv {
        Move::Inline { anchor, dir, size } => {
            debug_assert!((1..=3).contains(&size));
            let a = CELL_TO_COMPACT[anchor as usize];
            debug_assert!(a != INVALID_COMPACT, "encode: anchor not on board");
            (INLINE_OFFSET + a as usize * 18 + dir.idx() * 3 + (size as usize - 1)) as u16
        }
        Move::Broadside {
            anchor,
            group_dir,
            move_dir,
            size,
        } => {
            debug_assert!((2..=3).contains(&size));
            let a = CELL_TO_COMPACT[anchor as usize];
            debug_assert!(a != INVALID_COMPACT, "encode: anchor not on board");
            let gi = positive_dir_idx(group_dir);
            let mi = broadside_dir_idx(gi, move_dir);
            (BROADSIDE_OFFSET + a as usize * 24 + gi * 8 + mi * 2 + (size as usize - 2)) as u16
        }
    }
}

pub fn decode(idx: u16) -> Move {
    let i = idx as usize;
    debug_assert!(i < MOVE_SPACE);
    if i < BROADSIDE_OFFSET {
        let a = i / 18;
        let rem = i % 18;
        let d = rem / 3;
        let size = (rem % 3) + 1;
        Move::Inline {
            anchor: COMPACT_TO_CELL[a],
            dir: Dir::from_idx(d as u8),
            size: size as u8,
        }
    } else {
        let j = i - BROADSIDE_OFFSET;
        let a = j / 24;
        let rem = j % 24;
        let gi = rem / 8;
        let rem2 = rem % 8;
        let mi = rem2 / 2;
        let size = (rem2 % 2) + 2;
        Move::Broadside {
            anchor: COMPACT_TO_CELL[a],
            group_dir: POSITIVE_DIRS[gi],
            move_dir: BROADSIDE_DIRS[gi][mi],
            size: size as u8,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;
    use crate::cell::Side;
    use crate::moves::legal_moves;
    use std::collections::HashSet;

    #[test]
    fn encode_decode_round_trip_for_all_indices() {
        for i in 0..MOVE_SPACE {
            let mv = decode(i as u16);
            let back = encode(mv);
            assert_eq!(back, i as u16, "round-trip failed at idx {}", i);
        }
    }

    #[test]
    fn legal_moves_round_trip() {
        for &b in &[Board::standard(), Board::belgian_daisy()] {
            for side in [Side::Black, Side::White] {
                for m in legal_moves(&b, side) {
                    let idx = encode(m);
                    assert!((idx as usize) < MOVE_SPACE);
                    let back = decode(idx);
                    assert_eq!(back, m);
                }
            }
        }
    }

    #[test]
    fn legal_moves_have_distinct_indices() {
        let b = Board::standard();
        let moves = legal_moves(&b, Side::White);
        let n = moves.len();
        let unique: HashSet<u16> = moves.iter().map(|&m| encode(m)).collect();
        assert_eq!(unique.len(), n, "two legal moves collide on an index");
    }

    #[test]
    fn inline_and_broadside_offsets_disjoint() {
        // Pure structural check: any inline move encodes to < BROADSIDE_OFFSET.
        for &anchor in COMPACT_TO_CELL.iter() {
            for d in 0..6 {
                for size in 1..=3u8 {
                    let m = Move::Inline {
                        anchor,
                        dir: Dir::from_idx(d as u8),
                        size,
                    };
                    let idx = encode(m) as usize;
                    assert!(idx < BROADSIDE_OFFSET);
                }
            }
        }
    }

    #[test]
    fn move_space_size_matches_constants() {
        assert_eq!(MOVE_SPACE, 2562);
        assert_eq!(N_INLINE, 1098);
        assert_eq!(N_BROADSIDE, 1464);
    }
}
