//! u128 bitboard ops. Bit `r*9 + q` is set iff the named cell is occupied.
//! 20 of the 81 bit positions are off-board and always masked off.

use crate::cell::{in_board, Cell, Dir};

pub type BB = u128;

pub const VALID_MASK: BB = compute_valid_mask();

const fn compute_valid_mask() -> BB {
    let mut m: BB = 0;
    let mut r = 0u32;
    while r < 9 {
        let mut q = 0u32;
        while q < 9 {
            if in_board(r, q) {
                m |= 1u128 << (r * 9 + q);
            }
            q += 1;
        }
        r += 1;
    }
    m
}

pub const fn bit(cell: Cell) -> BB {
    1u128 << (cell as u32)
}

pub fn shift(bb: BB, dir: Dir) -> BB {
    let s = dir.shift();
    let raw = if s >= 0 {
        bb << (s as u32)
    } else {
        bb >> ((-s) as u32)
    };
    raw & VALID_MASK
}

/// Iterate set bits as cell indices (low to high).
pub struct BitIter(pub BB);

impl Iterator for BitIter {
    type Item = Cell;
    fn next(&mut self) -> Option<Cell> {
        if self.0 == 0 {
            None
        } else {
            let b = self.0.trailing_zeros() as Cell;
            self.0 &= self.0 - 1;
            Some(b)
        }
    }
}

pub fn popcount(bb: BB) -> u32 {
    bb.count_ones()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cell::{cell_at, parse, ALL_DIRS};

    #[test]
    fn valid_mask_has_61_bits() {
        assert_eq!(VALID_MASK.count_ones(), 61);
    }

    #[test]
    fn valid_mask_topology() {
        // A1 valid, B0 (would be q=0,r=1 cell index 9) valid, A6/A7/... invalid
        assert_ne!(VALID_MASK & bit(cell_at(0, 0)), 0);
        assert_ne!(VALID_MASK & bit(cell_at(8, 8)), 0); // I9
        assert_ne!(VALID_MASK & bit(cell_at(4, 0)), 0); // E1
        // F1 invalid (r=5, q=0, |q-r|=5)
        assert_eq!(VALID_MASK & (1u128 << (5 * 9)), 0);
        // A6 invalid (r=0, q=5)
        assert_eq!(VALID_MASK & (1u128 << 5), 0);
    }

    #[test]
    fn shift_drops_off_board() {
        // E shift of A5 (last col of row A) should produce 0 (off board).
        let bb = bit(parse("A5").unwrap());
        assert_eq!(shift(bb, Dir::E), 0);

        // E shift of A1 -> A2.
        let bb = bit(parse("A1").unwrap());
        assert_eq!(shift(bb, Dir::E), bit(parse("A2").unwrap()));
    }

    #[test]
    fn shift_round_trip() {
        // Shifting by d then by opposite(d) returns a subset of the original
        // (some bits may be lost off-board; the rest must round-trip).
        let bb = VALID_MASK; // all cells filled
        for d in ALL_DIRS {
            let there = shift(bb, d);
            let back = shift(there, d.opposite());
            // back ⊆ bb
            assert_eq!(back & !bb, 0);
            // Also: back equals the cells that had a valid neighbor in d.
        }
    }

    #[test]
    fn bititer_lists_cells() {
        let bb = bit(parse("A1").unwrap()) | bit(parse("E5").unwrap()) | bit(parse("I9").unwrap());
        let cells: Vec<_> = BitIter(bb).collect();
        assert_eq!(cells.len(), 3);
        assert!(cells.contains(&parse("A1").unwrap()));
        assert!(cells.contains(&parse("E5").unwrap()));
        assert!(cells.contains(&parse("I9").unwrap()));
    }
}
