//! Standard Abalone move notation. Two forms:
//!
//! - Inline: `<rear-cell><dir>` e.g. `A1E` (move the trailing marble of the
//!   line east; size is implied by the longest-line rule -- but for
//!   round-tripping we use an explicit form `<anchor><dir>:<size>`).
//! - Broadside: `<anchor>-<front>:<dir>` e.g. `A1-A3:NE`.
//!
//! We keep this minimal and unambiguous for now (engine-internal). A
//! human-friendly parser can be layered on later.

use std::fmt;

use crate::cell::{name, parse, Cell, Dir};
#[allow(unused_imports)]
use crate::cell::POSITIVE_DIRS;
use crate::moves::Move;

impl fmt::Display for Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Move::Inline { anchor, dir, size } => {
                write!(f, "{}{}:{}", name(*anchor), dir, size)
            }
            Move::Broadside {
                anchor,
                group_dir,
                move_dir,
                size,
            } => {
                let front_i = *anchor as i32 + (*size as i32 - 1) * group_dir.shift();
                write!(
                    f,
                    "{}-{}:{}",
                    name(*anchor),
                    name(front_i as Cell),
                    move_dir
                )
            }
        }
    }
}

/// Parse a move in the format produced by [`Move::fmt`]. Returns `None` on
/// any malformed input. Caller is responsible for confirming legality.
pub fn parse_move(s: &str) -> Option<Move> {
    let (left, right) = s.rsplit_once(':')?;
    if let Some((a, b)) = left.split_once('-') {
        // Broadside: "<cell>-<cell>:<dir>"
        let anchor = parse(a)?;
        let front = parse(b)?;
        let move_dir = parse_dir(right)?;
        let (group_dir, size) = recover_group(anchor, front)?;
        Some(Move::Broadside {
            anchor,
            group_dir,
            move_dir,
            size,
        })
    } else {
        // Inline: "<cell><dir>:<size>" (cell is 2 chars, dir is 1 or 2)
        if left.len() < 3 {
            return None;
        }
        let cell_str = &left[..2];
        let dir_str = &left[2..];
        let anchor = parse(cell_str)?;
        let dir = parse_dir(dir_str)?;
        let size: u8 = right.parse().ok()?;
        Some(Move::Inline { anchor, dir, size })
    }
}

fn parse_dir(s: &str) -> Option<Dir> {
    Some(match s {
        "E" => Dir::E,
        "NE" => Dir::NE,
        "NW" => Dir::NW,
        "W" => Dir::W,
        "SW" => Dir::SW,
        "SE" => Dir::SE,
        _ => return None,
    })
}

fn recover_group(anchor: Cell, front: Cell) -> Option<(Dir, u8)> {
    use crate::cell::POSITIVE_DIRS;
    for d in POSITIVE_DIRS {
        let s = d.shift();
        let delta = front as i32 - anchor as i32;
        if delta == s {
            return Some((d, 2));
        }
        if delta == 2 * s {
            return Some((d, 3));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cell::Side;
    use crate::moves::legal_moves;
    use crate::board::Board;

    #[test]
    fn roundtrip_all_legal_moves_standard() {
        let b = Board::standard();
        for m in legal_moves(&b, Side::Black) {
            let s = format!("{}", m);
            let parsed = parse_move(&s).unwrap_or_else(|| panic!("failed to parse {s}"));
            assert_eq!(parsed, m, "roundtrip mismatch for {s}");
        }
        for m in legal_moves(&b, Side::White) {
            let s = format!("{}", m);
            let parsed = parse_move(&s).unwrap_or_else(|| panic!("failed to parse {s}"));
            assert_eq!(parsed, m, "roundtrip mismatch for {s}");
        }
    }
}
