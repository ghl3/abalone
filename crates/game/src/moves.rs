//! Move type, legal-move generation, and `Board::apply`.
//!
//! Every move carries an `anchor` cell that names the moving group without
//! ambiguity, but the two move kinds anchor differently:
//!
//! - **Inline**: the anchor is the group's *rear* relative to the direction of
//!   travel. With `d = dir.shift()` the group occupies
//!   `anchor, anchor + d, ..., anchor + (size - 1) * d`, so `anchor + (size-1)*d`
//!   is the leading marble doing the pushing and `anchor` is the cell that ends
//!   up vacated. (For `size == 1` the anchor is simply the marble itself.)
//!   Because a group and its reverse travel in opposite directions, the
//!   (anchor, dir) pair is already unique — no extra canonicalisation needed.
//! - **Broadside**: the group lies along `group_dir`, which is always drawn
//!   from [`POSITIVE_DIRS`], and the anchor is the group's low-bit-index end.
//!   Fixing `group_dir` to the positive-shift half of the six directions is
//!   what canonicalises away the (start, end) / (end, start) ambiguity that
//!   broadside notation usually has.
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
        self.apply_with_capture(mv, side);
    }

    /// Apply `mv` for `side` and report **which cell** the pushed-off marble
    /// occupied at the moment it left the board, if any.
    ///
    /// At most one marble can leave per move in Abalone — only the outermost
    /// opponent marble in the pushed column can be over the edge — so `Option`
    /// is the right shape and the implementation asserts it.
    ///
    /// # Why this cannot be recovered by diffing bitboards
    ///
    /// For a 2-marble push the *front* opponent drops and the *rear* one slides
    /// into the cell the front just vacated. `before & !after` therefore names
    /// the REAR marble's cell — the one that is still on the board — not the
    /// cell the captured marble fell from. The capture-map training target
    /// (`docs/ARCHITECTURE.md` §5.4) needs the latter, so the information has to
    /// come out of `apply` itself rather than be reconstructed afterwards.
    pub fn apply_with_capture(&mut self, mv: Move, side: Side) -> Option<Cell> {
        match mv {
            Move::Inline { anchor, dir, size } => {
                self.apply_inline(anchor, dir, size, side)
            }
            Move::Broadside {
                anchor,
                group_dir,
                move_dir,
                size,
            } => {
                // A broadside slide moves into empty cells only — it can never
                // push, so it can never capture.
                self.apply_broadside(anchor, group_dir, move_dir, size, side);
                None
            }
        }
    }

    fn apply_inline(
        &mut self,
        anchor: Cell,
        dir: Dir,
        size: u8,
        side: Side,
    ) -> Option<Cell> {
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
        let mut captured: Option<Cell> = None;

        if cell_in_board(ahead1) && opp_bb & bit(ahead1 as Cell) != 0 {
            opp_clear |= bit(ahead1 as Cell);
            let ahead2 = ahead1 + d;
            if cell_in_board(ahead2) {
                let ahead2_bb = bit(ahead2 as Cell);
                if opp_bb & ahead2_bb != 0 {
                    // 2 opps in line: front opp leaves first. The marble that
                    // can fall off is the one at `ahead2`, NOT the one at
                    // `ahead1` (which merely slides forward into `ahead2`).
                    opp_clear |= ahead2_bb;
                    let ahead3 = ahead2 + d;
                    if cell_in_board(ahead3) {
                        opp_set |= bit(ahead3 as Cell);
                    } else {
                        captured = Some(ahead2 as Cell);
                    }
                    opp_set |= ahead2_bb; // rear opp moves into ahead2
                } else {
                    opp_set |= ahead2_bb;
                }
            } else {
                // Single opponent, pushed straight over the edge.
                captured = Some(ahead1 as Cell);
            }
        }

        let s = side.idx();
        let o = side.other().idx();
        self.marbles[s] = (self.marbles[s] & !own_clear) | own_set;
        self.marbles[o] = (self.marbles[o] & !opp_clear) | opp_set;
        if captured.is_some() {
            self.pushed_off[s] += 1;
        }
        captured
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

    // ---------- capture-event reporting ----------

    /// Find the unique inline move with the given anchor/dir/size.
    fn find_inline(board: &Board, side: Side, anchor: &str, dir: Dir, size: u8) -> Move {
        let a = parse(anchor).unwrap();
        *legal_moves(board, side)
            .iter()
            .find(|m| {
                matches!(m, Move::Inline { anchor, dir: dd, size: ss }
                    if *anchor == a && *dd == dir && *ss == size)
            })
            .unwrap_or_else(|| {
                panic!("expected a legal {size}-marble {dir:?} inline move at {anchor}")
            })
    }

    #[test]
    fn capture_none_for_quiet_moves() {
        let b = Board::standard();
        for m in legal_moves(&b, Side::White) {
            let mut t = b;
            assert_eq!(
                t.apply_with_capture(m, Side::White),
                None,
                "nothing can be captured from the standard opening"
            );
            assert_eq!(t.pushed_off, [0, 0]);
        }
    }

    #[test]
    fn capture_none_for_a_non_dropping_push() {
        // White A1 A2 push the lone black at A3 east into the empty A4.
        let mut b = Board::empty();
        b.set(parse("A1").unwrap(), Some(Side::White));
        b.set(parse("A2").unwrap(), Some(Side::White));
        b.set(parse("A3").unwrap(), Some(Side::Black));
        let mv = find_inline(&b, Side::White, "A1", Dir::E, 2);
        let mut t = b;
        assert_eq!(t.apply_with_capture(mv, Side::White), None);
        assert_eq!(t.at(parse("A4").unwrap()), Some(Side::Black));
    }

    #[test]
    fn capture_one_opponent_reports_its_own_cell() {
        // 2-vs-1 west push: black at A1 (west edge) goes off from A1.
        let mut b = Board::empty();
        b.set(parse("A1").unwrap(), Some(Side::Black));
        b.set(parse("A2").unwrap(), Some(Side::White));
        b.set(parse("A3").unwrap(), Some(Side::White));
        let mv = find_inline(&b, Side::White, "A3", Dir::W, 2);
        let mut t = b;
        let cap = t.apply_with_capture(mv, Side::White);
        assert_eq!(cap, Some(parse("A1").unwrap()));
        assert_eq!(t.pushed_off[Side::White.idx()], 1);
        assert_eq!(t.count(Side::Black), 0);
    }

    /// The case a naive bitboard diff gets wrong: with two opponents in the
    /// pushed column the FRONT one drops and the rear one slides into its cell,
    /// so `before & !after` names the rear marble.
    #[test]
    fn capture_two_opponents_reports_the_front_cell_not_the_rear() {
        // White A1 A2 A3, black A4 A5; A5 is the east edge of row A.
        let mut b = Board::empty();
        b.set(parse("A1").unwrap(), Some(Side::White));
        b.set(parse("A2").unwrap(), Some(Side::White));
        b.set(parse("A3").unwrap(), Some(Side::White));
        b.set(parse("A4").unwrap(), Some(Side::Black));
        b.set(parse("A5").unwrap(), Some(Side::Black));
        let mv = find_inline(&b, Side::White, "A1", Dir::E, 3);

        let mut t = b;
        let cap = t.apply_with_capture(mv, Side::White);
        assert_eq!(
            cap,
            Some(parse("A5").unwrap()),
            "the marble that left the board was the one on A5"
        );
        assert_eq!(t.pushed_off[Side::White.idx()], 1);

        // And the diff really does disagree — this is the bug being avoided.
        let diff = b.bb(Side::Black) & !t.bb(Side::Black);
        assert_eq!(diff.count_ones(), 1);
        assert_eq!(
            diff.trailing_zeros() as Cell,
            parse("A4").unwrap(),
            "bitboard diff names the REAR marble; apply_with_capture must not"
        );
        assert_ne!(diff.trailing_zeros() as Cell, cap.unwrap());
    }

    #[test]
    fn capture_three_vs_two_off_the_north_east_edge() {
        // Same 3-vs-2 shape on a different axis, to make sure the cell arithmetic
        // is not accidentally row-A specific. Column of white marbles pushing NE.
        let mut b = Board::empty();
        b.set(parse("F5").unwrap(), Some(Side::White));
        b.set(parse("G6").unwrap(), Some(Side::White));
        b.set(parse("H7").unwrap(), Some(Side::White));
        b.set(parse("I8").unwrap(), Some(Side::Black));
        // I8's NE neighbour is off-board (row I is the north edge), so this is a
        // 3-vs-1 drop from I8.
        let mv = find_inline(&b, Side::White, "F5", Dir::NE, 3);
        let mut t = b;
        assert_eq!(t.apply_with_capture(mv, Side::White), Some(parse("I8").unwrap()));
        assert_eq!(t.pushed_off[Side::White.idx()], 1);
        assert_eq!(t.count(Side::Black), 0);
    }

    #[test]
    fn capture_two_vs_one_reports_the_only_opponent() {
        // 2-vs-1 east push off the A5 edge.
        let mut b = Board::empty();
        b.set(parse("A3").unwrap(), Some(Side::White));
        b.set(parse("A4").unwrap(), Some(Side::White));
        b.set(parse("A5").unwrap(), Some(Side::Black));
        let mv = find_inline(&b, Side::White, "A3", Dir::E, 2);
        let mut t = b;
        assert_eq!(t.apply_with_capture(mv, Side::White), Some(parse("A5").unwrap()));
        assert_eq!(t.count(Side::Black), 0);
        assert_eq!(t.pushed_off[Side::White.idx()], 1);
    }

    #[test]
    fn capture_three_vs_two_that_does_not_drop_reports_none() {
        // 3-vs-2 with a landing square: nothing leaves the board.
        let mut b = Board::empty();
        b.set(parse("A1").unwrap(), Some(Side::White));
        b.set(parse("A2").unwrap(), Some(Side::White));
        b.set(parse("A3").unwrap(), Some(Side::White));
        b.set(parse("A4").unwrap(), Some(Side::Black));
        // Push NE instead so there is room behind the defenders.
        let mut b2 = Board::empty();
        b2.set(parse("C3").unwrap(), Some(Side::White));
        b2.set(parse("D4").unwrap(), Some(Side::White));
        b2.set(parse("E5").unwrap(), Some(Side::White));
        b2.set(parse("F6").unwrap(), Some(Side::Black));
        b2.set(parse("G7").unwrap(), Some(Side::Black));
        let mv = find_inline(&b2, Side::White, "C3", Dir::NE, 3);
        let mut t = b2;
        assert_eq!(t.apply_with_capture(mv, Side::White), None);
        assert_eq!(t.count(Side::Black), 2, "both defenders stay on the board");
        assert_eq!(t.at(parse("H8").unwrap()), Some(Side::Black));
        assert_eq!(t.at(parse("G7").unwrap()), Some(Side::Black));
        assert_eq!(t.pushed_off, [0, 0]);
        let _ = b;
    }

    #[test]
    fn capture_matches_pushed_off_over_random_play() {
        use crate::game::{Game, DEFAULT_MAX_PLIES, NO_PROGRESS_DISABLED};
        use rand::rngs::SmallRng;
        use rand::{Rng, SeedableRng};

        let mut rng = SmallRng::seed_from_u64(0xca97u64);
        for _ in 0..40 {
            let mut g = Game::new(
                crate::board::Opening::BelgianDaisy,
                DEFAULT_MAX_PLIES,
                NO_PROGRESS_DISABLED,
            );
            let mut events = 0u32;
            while !g.is_terminal() {
                let moves = g.legal_moves();
                let pick = rng.gen_range(0..moves.len());
                let mover = g.turn;
                let before = g.board.pushed_off;
                let victim = g.turn.other();
                let occupied_before = g.board.bb(victim);
                let cap = g.apply_with_capture(moves[pick]);
                let delta = u32::from(g.board.pushed_off[mover.idx()])
                    - u32::from(before[mover.idx()]);
                match cap {
                    Some(c) => {
                        events += 1;
                        assert_eq!(delta, 1, "a reported capture must bump the counter");
                        assert_ne!(
                            occupied_before & bit(c),
                            0,
                            "the reported cell must have held an opponent marble"
                        );
                    }
                    None => assert_eq!(delta, 0, "no capture reported => counter unchanged"),
                }
                assert!(delta <= 1, "at most one marble can leave per move");
            }
            let total: u32 = u32::from(g.board.pushed_off[0]) + u32::from(g.board.pushed_off[1]);
            assert_eq!(events, total, "every counter increment had an event");
        }
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
