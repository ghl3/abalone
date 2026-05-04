//! `Board`: two bitboards (one per side) plus the count of opponent
//! marbles each side has pushed off. Cheap to copy.
//!
//! Game-level state (whose turn, terminal check, ply count) lives in
//! [`crate::game::Game`]; `Board` is a pure position.

use crate::bitboard::{bit, BB, VALID_MASK};
use crate::cell::{cell_at, parse, Cell, Side};
use std::fmt;

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct Board {
    /// `marbles[Side::Black as usize]` = bitboard of black marbles, etc.
    pub marbles: [BB; 2],
    /// Count of *opponent* marbles each side has pushed off.
    /// `pushed_off[Side::Black]` = number of white marbles black has pushed.
    pub pushed_off: [u8; 2],
}

impl Board {
    pub const fn empty() -> Self {
        Self {
            marbles: [0, 0],
            pushed_off: [0, 0],
        }
    }

    /// Standard Abalone starting position.
    /// Black (north): rows H/I full, plus three on G (G5, G6, G7).
    /// White (south): rows A/B full, plus three on C (C3, C4, C5).
    pub fn standard() -> Self {
        let mut b = Self::empty();
        // White (south): row A all 5, row B all 6, C3 C4 C5.
        for q in 0..=4 {
            b.set(cell_at(0, q), Some(Side::White));
        }
        for q in 0..=5 {
            b.set(cell_at(1, q), Some(Side::White));
        }
        for q in 2..=4 {
            b.set(cell_at(2, q), Some(Side::White));
        }
        // Black (north): row I all 5, row H all 6, G5 G6 G7.
        for q in 4..=8 {
            b.set(cell_at(8, q), Some(Side::Black));
        }
        for q in 3..=8 {
            b.set(cell_at(7, q), Some(Side::Black));
        }
        for q in 4..=6 {
            b.set(cell_at(6, q), Some(Side::Black));
        }
        b
    }

    /// Belgian Daisy opening: two clusters of each color in opposite corners.
    /// Common tournament opening that produces livelier games than the
    /// standard layout.
    pub fn belgian_daisy() -> Self {
        let mut b = Self::empty();
        let names = [
            // White cluster south-west: A1 A2 B1 B2 B3 C2 C3
            ("A1", Side::White), ("A2", Side::White),
            ("B1", Side::White), ("B2", Side::White), ("B3", Side::White),
            ("C2", Side::White), ("C3", Side::White),
            // Black cluster south-east
            ("A4", Side::Black), ("A5", Side::Black),
            ("B4", Side::Black), ("B5", Side::Black), ("B6", Side::Black),
            ("C5", Side::Black), ("C6", Side::Black),
            // White cluster north-east
            ("I8", Side::White), ("I9", Side::White),
            ("H7", Side::White), ("H8", Side::White), ("H9", Side::White),
            ("G7", Side::White), ("G8", Side::White),
            // Black cluster north-west
            ("I5", Side::Black), ("I6", Side::Black),
            ("H4", Side::Black), ("H5", Side::Black), ("H6", Side::Black),
            ("G4", Side::Black), ("G5", Side::Black),
        ];
        for (n, s) in names {
            b.set(parse(n).unwrap(), Some(s));
        }
        b
    }

    pub const fn bb(&self, side: Side) -> BB {
        self.marbles[side.idx()]
    }

    pub fn bb_mut(&mut self, side: Side) -> &mut BB {
        &mut self.marbles[side.idx()]
    }

    pub fn occupied(&self) -> BB {
        self.marbles[0] | self.marbles[1]
    }

    pub fn empty_cells(&self) -> BB {
        VALID_MASK & !self.occupied()
    }

    pub fn at(&self, cell: Cell) -> Option<Side> {
        let m = bit(cell);
        if self.marbles[Side::Black.idx()] & m != 0 {
            Some(Side::Black)
        } else if self.marbles[Side::White.idx()] & m != 0 {
            Some(Side::White)
        } else {
            None
        }
    }

    pub fn set(&mut self, cell: Cell, side: Option<Side>) {
        let m = bit(cell);
        self.marbles[0] &= !m;
        self.marbles[1] &= !m;
        if let Some(s) = side {
            self.marbles[s.idx()] |= m;
        }
    }

    pub fn count(&self, side: Side) -> u32 {
        self.marbles[side.idx()].count_ones()
    }

    /// Number of `side`'s marbles pushed off the board (= number of marbles
    /// the opponent has gained as captures).
    pub fn lost(&self, side: Side) -> u8 {
        self.pushed_off[side.other().idx()]
    }
}

impl fmt::Display for Board {
    // Pretty-print the board, north at top, hex layout.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for r in (0..9).rev() {
            // Indent so rows form a hexagon.
            let indent = if r >= 4 { 8 - r } else { r } as usize;
            for _ in 0..indent {
                write!(f, " ")?;
            }
            write!(f, "{} ", (b'A' + r as u8) as char)?;
            let q_min = if r >= 4 { (r - 4) as u32 } else { 0 };
            let q_max = if r <= 4 { (r + 4) as u32 } else { 8 };
            for q in q_min..=q_max {
                let c = cell_at(r as u32, q);
                let ch = match self.at(c) {
                    None => '.',
                    Some(Side::Black) => '@',
                    Some(Side::White) => 'O',
                };
                write!(f, "{} ", ch)?;
            }
            writeln!(f)?;
        }
        write!(f, "  pushed off: B={}, W={}", self.pushed_off[0], self.pushed_off[1])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_starts_empty() {
        let b = Board::empty();
        assert_eq!(b.count(Side::Black), 0);
        assert_eq!(b.count(Side::White), 0);
    }

    #[test]
    fn standard_has_14_per_side() {
        let b = Board::standard();
        assert_eq!(b.count(Side::Black), 14);
        assert_eq!(b.count(Side::White), 14);
        // No overlap.
        assert_eq!(b.marbles[0] & b.marbles[1], 0);
        // Every occupied cell is in VALID_MASK.
        assert_eq!(b.occupied() & !VALID_MASK, 0);
    }

    #[test]
    fn standard_corners() {
        let b = Board::standard();
        assert_eq!(b.at(parse("A1").unwrap()), Some(Side::White));
        assert_eq!(b.at(parse("A5").unwrap()), Some(Side::White));
        assert_eq!(b.at(parse("C3").unwrap()), Some(Side::White));
        assert_eq!(b.at(parse("C5").unwrap()), Some(Side::White));
        assert_eq!(b.at(parse("I5").unwrap()), Some(Side::Black));
        assert_eq!(b.at(parse("I9").unwrap()), Some(Side::Black));
        assert_eq!(b.at(parse("G5").unwrap()), Some(Side::Black));
        assert_eq!(b.at(parse("G7").unwrap()), Some(Side::Black));
        // Empty middle.
        assert_eq!(b.at(parse("E5").unwrap()), None);
        assert_eq!(b.at(parse("D5").unwrap()), None);
    }

    #[test]
    fn belgian_daisy_has_14_per_side() {
        let b = Board::belgian_daisy();
        assert_eq!(b.count(Side::Black), 14);
        assert_eq!(b.count(Side::White), 14);
        assert_eq!(b.marbles[0] & b.marbles[1], 0);
    }

    #[test]
    fn set_and_at_roundtrip() {
        let mut b = Board::empty();
        let c = parse("E5").unwrap();
        b.set(c, Some(Side::Black));
        assert_eq!(b.at(c), Some(Side::Black));
        b.set(c, Some(Side::White));
        assert_eq!(b.at(c), Some(Side::White));
        b.set(c, None);
        assert_eq!(b.at(c), None);
    }

}
