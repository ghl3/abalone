//! Move type, legal-move generation, and `Board::apply`.
//!
//! Moves are anchored at the "low" end of a group (the cell with the
//! smallest bit index when the group is along a positive-shift direction),
//! which canonicalises away the (start, end) / (end, start) ambiguity
//! that broadside notation usually has.
//!
//! Inline moves: column of `size` ∈ {1, 2, 3} own marbles slides one cell
//! along `dir`. May push 0, 1, or 2 opponent marbles (1-vs-1 not allowed,
//! attacker must strictly outnumber defender).
//!
//! Broadside moves: row of `size` ∈ {2, 3} own marbles aligned along
//! `group_dir` ∈ {E, NE, NW} slides one cell along `move_dir` (which is
//! neither parallel nor antiparallel to `group_dir`). All destination cells
//! must be empty.

use arrayvec::ArrayVec;

use crate::bitboard::{bit, shift, BitIter, BB, VALID_MASK};
use crate::board::Board;
use crate::cell::{
    cell_in_board, Cell, Dir, Side, ALL_DIRS, BROADSIDE_DIRS, POSITIVE_DIRS,
};

/// Upper bound on legal moves in any Abalone position. Empirically max ~140;
/// 256 leaves plenty of headroom and keeps the structure cache-aligned.
pub const MAX_LEGAL: usize = 256;

pub type MoveList = ArrayVec<Move, MAX_LEGAL>;

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum Move {
    Inline {
        anchor: Cell,
        dir: Dir,
        size: u8, // 1..=3
    },
    Broadside {
        anchor: Cell,
        group_dir: Dir, // ∈ POSITIVE_DIRS
        move_dir: Dir,  // ∈ BROADSIDE_DIRS[group_dir]
        size: u8,       // 2..=3
    },
}

pub fn legal_moves(board: &Board, side: Side) -> MoveList {
    let mut moves = MoveList::new();
    let own = board.bb(side);
    let opp = board.bb(side.other());
    let empty = VALID_MASK & !(own | opp);

    // ===== Inline =====
    for dir in ALL_DIRS {
        let d = dir.shift();

        // 1-marble: own with empty in front.
        let dests = shift(own, dir) & empty;
        for dest in BitIter(dests) {
            let anchor = (dest as i32 - d) as Cell;
            moves.push(Move::Inline {
                anchor,
                dir,
                size: 1,
            });
        }

        // pair_fronts: bit p set iff own at p and own at p-d (a pair whose
        // *front* is p, *anchor* is p-d).
        let pair_fronts = own & shift(own, dir);

        // 2-marble inline.
        for front in BitIter(pair_fronts) {
            let anchor = (front as i32 - d) as Cell;
            let ahead1_i = front as i32 + d;
            if !cell_in_board(ahead1_i) {
                continue; // would push own off
            }
            let ahead1_bb = bit(ahead1_i as Cell);
            if empty & ahead1_bb != 0 {
                moves.push(Move::Inline {
                    anchor,
                    dir,
                    size: 2,
                });
            } else if opp & ahead1_bb != 0 {
                // 2-vs-1: cell at front+2d must be empty or off-board, and
                // not own/another-opp (that'd be 2-vs-2 or blocked).
                let ahead2_i = front as i32 + 2 * d;
                if !cell_in_board(ahead2_i) {
                    moves.push(Move::Inline {
                        anchor,
                        dir,
                        size: 2,
                    });
                } else {
                    let ahead2_bb = bit(ahead2_i as Cell);
                    if empty & ahead2_bb != 0 {
                        moves.push(Move::Inline {
                            anchor,
                            dir,
                            size: 2,
                        });
                    }
                }
            }
        }

        // triple_fronts: bit p set iff own at p, p-d, p-2d.
        let triple_fronts = pair_fronts & shift(pair_fronts, dir);

        // 3-marble inline.
        for front in BitIter(triple_fronts) {
            let anchor = (front as i32 - 2 * d) as Cell;
            let ahead1_i = front as i32 + d;
            if !cell_in_board(ahead1_i) {
                continue;
            }
            let ahead1_bb = bit(ahead1_i as Cell);
            if empty & ahead1_bb != 0 {
                moves.push(Move::Inline {
                    anchor,
                    dir,
                    size: 3,
                });
                continue;
            }
            if opp & ahead1_bb == 0 {
                continue; // own at ahead1 -> would be a 4-line; blocked
            }
            // ≥1 opp.
            let ahead2_i = front as i32 + 2 * d;
            if !cell_in_board(ahead2_i) {
                moves.push(Move::Inline {
                    anchor,
                    dir,
                    size: 3,
                });
                continue;
            }
            let ahead2_bb = bit(ahead2_i as Cell);
            if empty & ahead2_bb != 0 {
                // 3-vs-1 slide
                moves.push(Move::Inline {
                    anchor,
                    dir,
                    size: 3,
                });
            } else if opp & ahead2_bb != 0 {
                // 3-vs-2: ahead3 must be empty or off-board.
                let ahead3_i = front as i32 + 3 * d;
                if !cell_in_board(ahead3_i) {
                    moves.push(Move::Inline {
                        anchor,
                        dir,
                        size: 3,
                    });
                } else {
                    let ahead3_bb = bit(ahead3_i as Cell);
                    if empty & ahead3_bb != 0 {
                        moves.push(Move::Inline {
                            anchor,
                            dir,
                            size: 3,
                        });
                    }
                }
            }
            // own at ahead2 -> blocked
        }
    }

    // ===== Broadside =====
    for (gi, &group_dir) in POSITIVE_DIRS.iter().enumerate() {
        let dg = group_dir.shift();
        let pair_fronts = own & shift(own, group_dir);
        let triple_fronts = pair_fronts & shift(pair_fronts, group_dir);

        for &move_dir in &BROADSIDE_DIRS[gi] {
            let dm = move_dir.shift();

            // 2-marble groups
            for front in BitIter(pair_fronts) {
                let anchor = (front as i32 - dg) as Cell;
                let dest_a = anchor as i32 + dm;
                let dest_f = front as i32 + dm;
                if !cell_in_board(dest_a) || !cell_in_board(dest_f) {
                    continue;
                }
                let needed = bit(dest_a as Cell) | bit(dest_f as Cell);
                if empty & needed == needed {
                    moves.push(Move::Broadside {
                        anchor,
                        group_dir,
                        move_dir,
                        size: 2,
                    });
                }
            }

            // 3-marble groups
            for front in BitIter(triple_fronts) {
                let mid_i = front as i32 - dg;
                let anchor_i = front as i32 - 2 * dg;
                let dest_a = anchor_i + dm;
                let dest_m = mid_i + dm;
                let dest_f = front as i32 + dm;
                if !cell_in_board(dest_a)
                    || !cell_in_board(dest_m)
                    || !cell_in_board(dest_f)
                {
                    continue;
                }
                let needed =
                    bit(dest_a as Cell) | bit(dest_m as Cell) | bit(dest_f as Cell);
                if empty & needed == needed {
                    moves.push(Move::Broadside {
                        anchor: anchor_i as Cell,
                        group_dir,
                        move_dir,
                        size: 3,
                    });
                }
            }
        }
    }

    moves
}

impl Board {
    /// Apply `mv` for `side`. `mv` MUST be legal for `side` in this position
    /// (use [`legal_moves`] first). Updates `pushed_off` if marbles fall off.
    pub fn apply(&mut self, mv: Move, side: Side) {
        match mv {
            Move::Inline { anchor, dir, size } => {
                self.apply_inline(anchor, dir, size, side);
            }
            Move::Broadside {
                anchor,
                group_dir,
                move_dir,
                size,
            } => {
                self.apply_broadside(anchor, group_dir, move_dir, size, side);
            }
        }
    }

    fn apply_inline(&mut self, anchor: Cell, dir: Dir, size: u8, side: Side) {
        let d = dir.shift();
        let mut own_clear: BB = 0;
        let mut own_set: BB = 0;
        for i in 0..size as i32 {
            let from = anchor as i32 + i * d;
            let to = from + d;
            own_clear |= bit(from as Cell);
            own_set |= bit(to as Cell);
        }

        let front = anchor as i32 + (size as i32 - 1) * d;
        let ahead1 = front + d;
        let opp_bb = self.bb(side.other());

        let mut opp_clear: BB = 0;
        let mut opp_set: BB = 0;
        let mut drops: u8 = 0;

        if cell_in_board(ahead1) && opp_bb & bit(ahead1 as Cell) != 0 {
            opp_clear |= bit(ahead1 as Cell);
            let ahead2 = ahead1 + d;
            if cell_in_board(ahead2) {
                let ahead2_bb = bit(ahead2 as Cell);
                if opp_bb & ahead2_bb != 0 {
                    // 2 opps in line: front opp leaves first.
                    opp_clear |= ahead2_bb;
                    let ahead3 = ahead2 + d;
                    if cell_in_board(ahead3) {
                        opp_set |= bit(ahead3 as Cell);
                    } else {
                        drops += 1;
                    }
                    opp_set |= ahead2_bb; // rear opp moves into ahead2
                } else {
                    opp_set |= ahead2_bb;
                }
            } else {
                drops += 1;
            }
        }

        let s = side.idx();
        let o = side.other().idx();
        self.marbles[s] = (self.marbles[s] & !own_clear) | own_set;
        self.marbles[o] = (self.marbles[o] & !opp_clear) | opp_set;
        self.pushed_off[s] += drops;
    }

    fn apply_broadside(
        &mut self,
        anchor: Cell,
        group_dir: Dir,
        move_dir: Dir,
        size: u8,
        side: Side,
    ) {
        let dg = group_dir.shift();
        let dm = move_dir.shift();
        let mut own_clear: BB = 0;
        let mut own_set: BB = 0;
        for i in 0..size as i32 {
            let from = anchor as i32 + i * dg;
            let to = from + dm;
            own_clear |= bit(from as Cell);
            own_set |= bit(to as Cell);
        }
        let s = side.idx();
        self.marbles[s] = (self.marbles[s] & !own_clear) | own_set;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cell::parse;

    fn count(board: &Board, side: Side) -> usize {
        legal_moves(board, side).len()
    }

    #[test]
    fn standard_opening_move_count() {
        let b = Board::standard();
        // The standard opening has the same legal-move count for both sides
        // by symmetry; literature reports 44 moves for the standard opening.
        let mw = count(&b, Side::White);
        let mb = count(&b, Side::Black);
        assert_eq!(mw, mb, "white/black move counts must match by symmetry");
        assert!(
            (30..=60).contains(&mw),
            "expected ~44 moves at standard opening, got {}",
            mw
        );
    }

    #[test]
    fn no_self_push_off_edge() {
        let b = Board::standard();
        let moves = legal_moves(&b, Side::White);
        // No move should push a white marble off-board (standard opening has
        // no opponents adjacent, so there can't be).
        let mut b2 = b;
        for m in &moves {
            let mut t = b2;
            t.apply(*m, Side::White);
            assert_eq!(t.count(Side::White), 14);
            assert_eq!(t.count(Side::Black), 14);
        }
        // Suppress warning.
        let _ = &mut b2;
    }

    #[test]
    fn single_move_slides_one_marble() {
        let mut b = Board::empty();
        b.set(parse("E5").unwrap(), Some(Side::White));
        let moves = legal_moves(&b, Side::White);
        assert_eq!(moves.len(), 6, "lone marble has 6 dirs to move");
        for m in moves {
            let mut t = b;
            t.apply(m, Side::White);
            assert_eq!(t.count(Side::White), 1);
        }
    }

    #[test]
    fn push_one_off_edge() {
        // White 2 vs Black 1, black at edge -> push off.
        let mut b = Board::empty();
        b.set(parse("A1").unwrap(), Some(Side::White));
        b.set(parse("A2").unwrap(), Some(Side::White));
        b.set(parse("A3").unwrap(), Some(Side::Black));
        // ... then a void to E side, and A5 at edge. Let's set A4 black so we test the push.
        // Actually pushing toward east: A1, A2 (own pair), A3 black, A4 empty, A5 empty.
        // 2v1 push slides black to A4. Not a drop.
        let mw = legal_moves(&b, Side::White);
        let push = mw
            .iter()
            .find(|m| {
                matches!(m, Move::Inline {
                    anchor, dir: Dir::E, size: 2
                } if *anchor == parse("A1").unwrap())
            })
            .copied();
        assert!(push.is_some(), "expected a 2-marble East push starting at A1");
        let mut t = b;
        t.apply(push.unwrap(), Side::White);
        assert_eq!(t.at(parse("A1").unwrap()), None);
        assert_eq!(t.at(parse("A2").unwrap()), Some(Side::White));
        assert_eq!(t.at(parse("A3").unwrap()), Some(Side::White));
        assert_eq!(t.at(parse("A4").unwrap()), Some(Side::Black));
        assert_eq!(t.pushed_off[Side::White.idx()], 0);
    }

    #[test]
    fn push_one_off_with_drop() {
        // White 2 at A4 A5 ... wait A5 is east-most. Try West: A5 (W) A4 (W) A3 (B), going W pushes B to A2 (empty). Not a drop.
        // For a drop: black at edge gets pushed off. A1 black at west edge, A2 white, A3 white. Going W: A3, A2 own, then A1 opp, then off-board.
        let mut b = Board::empty();
        b.set(parse("A1").unwrap(), Some(Side::Black));
        b.set(parse("A2").unwrap(), Some(Side::White));
        b.set(parse("A3").unwrap(), Some(Side::White));
        let mw = legal_moves(&b, Side::White);
        let push = mw
            .iter()
            .find(|m| {
                matches!(m, Move::Inline {
                    anchor, dir: Dir::W, size: 2
                } if *anchor == parse("A3").unwrap())
            })
            .copied();
        assert!(push.is_some(), "expected a 2-marble West push starting at A3");
        let mut t = b;
        t.apply(push.unwrap(), Side::White);
        assert_eq!(t.at(parse("A3").unwrap()), None);
        assert_eq!(t.at(parse("A2").unwrap()), Some(Side::White));
        assert_eq!(t.at(parse("A1").unwrap()), Some(Side::White));
        assert_eq!(t.count(Side::Black), 0);
        assert_eq!(t.pushed_off[Side::White.idx()], 1);
    }

    #[test]
    fn cannot_push_equal_numbers() {
        // 2 vs 2 must be blocked.
        let mut b = Board::empty();
        b.set(parse("A1").unwrap(), Some(Side::White));
        b.set(parse("A2").unwrap(), Some(Side::White));
        b.set(parse("A3").unwrap(), Some(Side::Black));
        b.set(parse("A4").unwrap(), Some(Side::Black));
        let mw = legal_moves(&b, Side::White);
        let blocked = mw.iter().find(|m| {
            matches!(m, Move::Inline { anchor, dir: Dir::E, size: 2 }
                if *anchor == parse("A1").unwrap())
        });
        assert!(blocked.is_none(), "2-vs-2 push must be illegal");
    }

    #[test]
    fn three_pushes_two() {
        // 3 vs 2, with empty behind: legal.
        // White A1 A2 A3 (own), Black A4 A5, A5 is east edge so push drops one.
        let mut b = Board::empty();
        b.set(parse("A1").unwrap(), Some(Side::White));
        b.set(parse("A2").unwrap(), Some(Side::White));
        b.set(parse("A3").unwrap(), Some(Side::White));
        b.set(parse("A4").unwrap(), Some(Side::Black));
        b.set(parse("A5").unwrap(), Some(Side::Black));
        let mw = legal_moves(&b, Side::White);
        let push = mw
            .iter()
            .find(|m| {
                matches!(m, Move::Inline {
                    anchor, dir: Dir::E, size: 3
                } if *anchor == parse("A1").unwrap())
            })
            .copied();
        assert!(push.is_some(), "3-vs-2 push to edge expected");
        let mut t = b;
        t.apply(push.unwrap(), Side::White);
        assert_eq!(t.at(parse("A2").unwrap()), Some(Side::White));
        assert_eq!(t.at(parse("A3").unwrap()), Some(Side::White));
        assert_eq!(t.at(parse("A4").unwrap()), Some(Side::White));
        assert_eq!(t.at(parse("A5").unwrap()), Some(Side::Black));
        assert_eq!(t.count(Side::Black), 1);
        assert_eq!(t.pushed_off[Side::White.idx()], 1);
    }

    #[test]
    fn broadside_two_moves_sideways() {
        // White at C3, C4 (in a row), broadside-move them NE (sideways) to D4, D5.
        let mut b = Board::empty();
        b.set(parse("C3").unwrap(), Some(Side::White));
        b.set(parse("C4").unwrap(), Some(Side::White));
        let mw = legal_moves(&b, Side::White);
        // C3-C4 is a row in direction E. Group_dir = E, anchor = C3.
        let bs = mw
            .iter()
            .find(|m| {
                matches!(m,
                    Move::Broadside { anchor, group_dir: Dir::E, move_dir: Dir::NE, size: 2 }
                    if *anchor == parse("C3").unwrap()
                )
            })
            .copied();
        assert!(bs.is_some(), "expected a broadside NE move of C3-C4");
        let mut t = b;
        t.apply(bs.unwrap(), Side::White);
        assert_eq!(t.at(parse("C3").unwrap()), None);
        assert_eq!(t.at(parse("C4").unwrap()), None);
        assert_eq!(t.at(parse("D4").unwrap()), Some(Side::White));
        assert_eq!(t.at(parse("D5").unwrap()), Some(Side::White));
    }

    #[test]
    fn broadside_blocked_by_own() {
        // Group at C3-C4, but D4 already occupied -> cannot broadside NE.
        let mut b = Board::empty();
        b.set(parse("C3").unwrap(), Some(Side::White));
        b.set(parse("C4").unwrap(), Some(Side::White));
        b.set(parse("D4").unwrap(), Some(Side::White));
        let mw = legal_moves(&b, Side::White);
        let bs = mw.iter().find(|m| {
            matches!(m,
                Move::Broadside { anchor, group_dir: Dir::E, move_dir: Dir::NE, size: 2 }
                if *anchor == parse("C3").unwrap()
            )
        });
        assert!(bs.is_none(), "broadside blocked by own marble");
    }

    #[test]
    fn no_duplicate_moves() {
        let b = Board::standard();
        let moves = legal_moves(&b, Side::White);
        let n = moves.len();
        let unique: std::collections::HashSet<_> = moves.into_iter().collect();
        assert_eq!(unique.len(), n, "duplicate move emitted");
    }
}
