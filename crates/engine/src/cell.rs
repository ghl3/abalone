//! Cell coordinates and the six hex directions.
//!
//! The board is 9 rows × up-to-9 columns laid out in axial coords:
//!   r is the row index (A=0, I=8), q is the diagonal index (1=0, 9=8).
//! A cell is valid iff 0..=8 for both and `|q - r| <= 4` (= 61 cells).
//!
//! We pack each cell into a single u128 bit at index `r*9 + q`.
//! The 20 invalid (q, r) slots are wasted bits; we mask them off via
//! [`bitboard::VALID_MASK`].
//!
//! Direction shifts (in bit-index deltas) work uniformly on bitboards
//! via shl/shr + mask, and on individual cells via i32 arithmetic.

use std::fmt;

/// Cell index 0..81. Only 61 are valid; the rest are masked off.
pub type Cell = u8;

pub const N_CELLS: usize = 81;
pub const N_VALID: usize = 61;

#[repr(u8)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum Side {
    Black = 0,
    White = 1,
}

impl Side {
    pub const fn other(self) -> Side {
        match self {
            Side::Black => Side::White,
            Side::White => Side::Black,
        }
    }

    pub const fn idx(self) -> usize {
        self as usize
    }
}

#[repr(u8)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum Dir {
    E = 0,
    NE = 1,
    NW = 2,
    W = 3,
    SW = 4,
    SE = 5,
}

impl Dir {
    pub const fn shift(self) -> i32 {
        match self {
            Dir::E => 1,
            Dir::NE => 10,
            Dir::NW => 9,
            Dir::W => -1,
            Dir::SW => -10,
            Dir::SE => -9,
        }
    }

    pub const fn opposite(self) -> Dir {
        match self {
            Dir::E => Dir::W,
            Dir::NE => Dir::SW,
            Dir::NW => Dir::SE,
            Dir::W => Dir::E,
            Dir::SW => Dir::NE,
            Dir::SE => Dir::NW,
        }
    }

    pub const fn idx(self) -> usize {
        self as usize
    }

    pub const fn from_idx(i: u8) -> Dir {
        match i {
            0 => Dir::E,
            1 => Dir::NE,
            2 => Dir::NW,
            3 => Dir::W,
            4 => Dir::SW,
            5 => Dir::SE,
            _ => panic!("bad dir idx"),
        }
    }
}

pub const ALL_DIRS: [Dir; 6] = [Dir::E, Dir::NE, Dir::NW, Dir::W, Dir::SW, Dir::SE];

/// The three "positive-shift" directions; we anchor broadside groups on these
/// to canonicalize the (start, end)/(end, start) ambiguity.
pub const POSITIVE_DIRS: [Dir; 3] = [Dir::E, Dir::NE, Dir::NW];

/// The four sideways directions for each `POSITIVE_DIRS` choice.
/// Sideways = not the group direction itself, not its opposite.
pub const BROADSIDE_DIRS: [[Dir; 4]; 3] = [
    // E (opposite W) -> NE, NW, SE, SW
    [Dir::NE, Dir::NW, Dir::SE, Dir::SW],
    // NE (opposite SW) -> E, W, NW, SE
    [Dir::E, Dir::W, Dir::NW, Dir::SE],
    // NW (opposite SE) -> E, W, NE, SW
    [Dir::E, Dir::W, Dir::NE, Dir::SW],
];

pub const fn cell_at(r: u32, q: u32) -> Cell {
    debug_assert!(in_board(r, q));
    (r * 9 + q) as Cell
}

pub const fn in_board(r: u32, q: u32) -> bool {
    if r >= 9 || q >= 9 {
        return false;
    }
    let dq = q as i32 - r as i32;
    dq >= -4 && dq <= 4
}

pub const fn cell_in_board(c: i32) -> bool {
    if c < 0 || c >= N_CELLS as i32 {
        return false;
    }
    let r = (c as u32) / 9;
    let q = (c as u32) % 9;
    in_board(r, q)
}

/// Step `cell` one hex in `dir`. Returns `None` if it leaves the board.
pub fn step(cell: Cell, dir: Dir) -> Option<Cell> {
    let new = cell as i32 + dir.shift();
    if cell_in_board(new) {
        Some(new as Cell)
    } else {
        None
    }
}

/// Step `cell` `n` hexes in `dir`. Returns `None` if any intermediate or
/// final step leaves the board.
pub fn step_n(cell: Cell, dir: Dir, n: i32) -> Option<Cell> {
    let mut c = cell as i32;
    for _ in 0..n {
        c += dir.shift();
        if !cell_in_board(c) {
            return None;
        }
    }
    Some(c as Cell)
}

/// (row, q) for a cell. Row 0 = "A", q 0 = diagonal "1".
pub const fn rq(cell: Cell) -> (u8, u8) {
    (cell / 9, cell % 9)
}

/// Pretty name like "A1" .. "I9".
pub fn name(cell: Cell) -> String {
    let (r, q) = rq(cell);
    let row = (b'A' + r) as char;
    let diag = (b'1' + q) as char;
    format!("{}{}", row, diag)
}

/// Parse a name like "A1" .. "I9".
pub fn parse(s: &str) -> Option<Cell> {
    let bytes = s.as_bytes();
    if bytes.len() != 2 {
        return None;
    }
    let r = bytes[0].checked_sub(b'A')?;
    let q = bytes[1].checked_sub(b'1')?;
    if !in_board(r as u32, q as u32) {
        return None;
    }
    Some(cell_at(r as u32, q as u32))
}

impl fmt::Display for Side {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Side::Black => f.write_str("Black"),
            Side::White => f.write_str("White"),
        }
    }
}

impl fmt::Display for Dir {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Dir::E => "E",
            Dir::NE => "NE",
            Dir::NW => "NW",
            Dir::W => "W",
            Dir::SW => "SW",
            Dir::SE => "SE",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn count_valid_cells() {
        let mut n = 0;
        for r in 0..9 {
            for q in 0..9 {
                if in_board(r, q) {
                    n += 1;
                }
            }
        }
        assert_eq!(n, N_VALID);
    }

    #[test]
    fn parse_roundtrip() {
        for r in 0..9 {
            for q in 0..9 {
                if in_board(r, q) {
                    let c = cell_at(r, q);
                    let s = name(c);
                    assert_eq!(parse(&s), Some(c));
                }
            }
        }
    }

    #[test]
    fn step_basics() {
        // A1 (r=0, q=0) east -> A2 (r=0, q=1)
        assert_eq!(step(parse("A1").unwrap(), Dir::E), Some(parse("A2").unwrap()));
        // A5 east -> off board
        assert_eq!(step(parse("A5").unwrap(), Dir::E), None);
        // C3 NE -> D4
        assert_eq!(step(parse("C3").unwrap(), Dir::NE), Some(parse("D4").unwrap()));
        // C3 SE -> B3
        assert_eq!(step(parse("C3").unwrap(), Dir::SE), Some(parse("B3").unwrap()));
        // I9 NE -> off
        assert_eq!(step(parse("I9").unwrap(), Dir::NE), None);
    }

    #[test]
    fn opposite_is_involution() {
        for d in ALL_DIRS {
            assert_eq!(d.opposite().opposite(), d);
        }
    }
}
