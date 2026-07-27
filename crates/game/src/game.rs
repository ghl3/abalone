//! `Game`: position + turn + ply count + terminal check.
//!
//! Terminal, in precedence order (see `docs/MODEL.md` §3):
//! - `pushed_off[side]` reaching [`WIN_THRESHOLD`] (= 6) means `side` wins
//!   (they have pushed 6 of the opponent's marbles off).
//! - Otherwise, when the ply cap or the optional no-progress limit is hit, the
//!   game is *adjudicated on capture differential*: whoever has pushed more of
//!   the opponent off wins; an equal count is a genuine draw.
//!
//! Adjudication uses the capture count and nothing else. Capture count is the
//! game's own scoring quantity — the thing the win condition counts — so
//! resolving a truncated game with it is objective, in the same way Go's area
//! scoring is. A positional tiebreak (centre control, cohesion, edge distance)
//! would be a hand-written strategy claim and is deliberately absent.

use rand::Rng;

use crate::board::{Board, Opening};
use crate::cell::Side;
use crate::moves::{legal_moves, Move, MoveList};

/// Standard win condition: push 6 of the opponent's marbles off the board.
pub const WIN_THRESHOLD: u8 = 6;

/// Default hard cap on plies. Real Abalone runs 60-100 moves; capping at 200
/// plies keeps the capture differential a tight signal instead of spending half
/// of every game on a tail that is pure noise under weak play.
pub const DEFAULT_MAX_PLIES: u32 = 200;

/// Sentinel for "no no-progress rule". Unreachable in practice given any sane
/// ply cap.
pub const NO_PROGRESS_DISABLED: u32 = u32::MAX;

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum GameState {
    InProgress,
    Wins(Side),
    Draw,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct Game {
    pub board: Board,
    pub turn: Side,
    pub ply: u32,
    /// Adjudicate on capture differential once `ply` reaches this.
    pub max_plies: u32,
    /// Plies elapsed since the last ply on which a marble was pushed off.
    pub plies_since_capture: u32,
    /// Adjudicate once `plies_since_capture` reaches this.
    /// [`NO_PROGRESS_DISABLED`] (the default) switches the rule off.
    pub no_progress_plies: u32,
}

impl Default for Game {
    fn default() -> Self {
        Self::new_standard()
    }
}

impl Game {
    /// Fresh game from `opening` with an explicit ply cap and no-progress limit.
    /// Pass [`NO_PROGRESS_DISABLED`] to switch the no-progress rule off.
    pub fn new(opening: Opening, max_plies: u32, no_progress_plies: u32) -> Self {
        Self {
            board: Board::from_opening(opening),
            // Black moves first by tournament convention.
            turn: Side::Black,
            ply: 0,
            max_plies,
            plies_since_capture: 0,
            no_progress_plies,
        }
    }

    /// Fresh game from `opening` with a capture handicap already applied
    /// (`docs/MODEL.md` §4).
    ///
    /// `black_conceded` is the number of marbles **Black has lost**: that many
    /// black marbles are removed at random and White's capture counter is
    /// raised by the same amount. Likewise `white_conceded` for White.
    ///
    /// # Panics
    /// If either handicap is not in `0..=5` terms — see [`Board::concede`].
    pub fn with_handicap<R: Rng>(
        opening: Opening,
        max_plies: u32,
        no_progress_plies: u32,
        black_conceded: u8,
        white_conceded: u8,
        rng: &mut R,
    ) -> Self {
        let mut g = Self::new(opening, max_plies, no_progress_plies);
        g.board.concede(Side::Black, black_conceded, rng);
        g.board.concede(Side::White, white_conceded, rng);
        g
    }

    pub fn new_standard() -> Self {
        Self::new(Opening::Standard, DEFAULT_MAX_PLIES, NO_PROGRESS_DISABLED)
    }

    pub fn new_belgian_daisy() -> Self {
        Self::new(
            Opening::BelgianDaisy,
            DEFAULT_MAX_PLIES,
            NO_PROGRESS_DISABLED,
        )
    }

    pub fn legal_moves(&self) -> MoveList {
        legal_moves(&self.board, self.turn)
    }

    pub fn apply(&mut self, mv: Move) {
        self.apply_with_capture(mv);
    }

    /// [`apply`](Self::apply), additionally reporting the cell a pushed-off
    /// marble occupied when it left the board (see
    /// [`Board::apply_with_capture`]). The marble belongs to the side that was
    /// *not* to move, i.e. `self.turn.other()` as of before the call.
    ///
    /// Self-play records these events to build the capture-map training target;
    /// diffing bitboards would name the wrong cell for multi-marble pushes.
    pub fn apply_with_capture(&mut self, mv: Move) -> Option<crate::cell::Cell> {
        let captured = self.board.apply_with_capture(mv, self.turn);
        if captured.is_none() {
            self.plies_since_capture = self.plies_since_capture.saturating_add(1);
        } else {
            self.plies_since_capture = 0;
        }
        self.turn = self.turn.other();
        self.ply += 1;
        captured
    }

    /// Captures made minus captures conceded, from `side`'s point of view,
    /// clamped to `[-6, 6]`. This is the game's own score, not an evaluation.
    pub fn score_diff(&self, side: Side) -> i32 {
        let made = i32::from(self.board.pushed_off[side.idx()]);
        let conceded = i32::from(self.board.pushed_off[side.other().idx()]);
        (made - conceded).clamp(-i32::from(WIN_THRESHOLD), i32::from(WIN_THRESHOLD))
    }

    /// Resolve a truncated game on capture differential alone.
    fn adjudicate(&self) -> GameState {
        let d = i32::from(self.board.pushed_off[Side::Black.idx()])
            - i32::from(self.board.pushed_off[Side::White.idx()]);
        match d.signum() {
            1 => GameState::Wins(Side::Black),
            -1 => GameState::Wins(Side::White),
            _ => GameState::Draw,
        }
    }

    pub fn state(&self) -> GameState {
        if self.board.pushed_off[Side::Black.idx()] >= WIN_THRESHOLD {
            // Black has pushed 6 white marbles off.
            GameState::Wins(Side::Black)
        } else if self.board.pushed_off[Side::White.idx()] >= WIN_THRESHOLD {
            GameState::Wins(Side::White)
        } else if self.ply >= self.max_plies
            || self.plies_since_capture >= self.no_progress_plies
        {
            self.adjudicate()
        } else {
            GameState::InProgress
        }
    }

    pub fn is_terminal(&self) -> bool {
        !matches!(self.state(), GameState::InProgress)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::SmallRng;
    use rand::SeedableRng;

    /// Play uniformly random moves to termination. Returns the finished game.
    fn random_playout(mut g: Game, rng: &mut SmallRng) -> Game {
        while !g.is_terminal() {
            let moves = g.legal_moves();
            assert!(
                !moves.is_empty(),
                "no legal moves but not terminal at ply {}",
                g.ply
            );
            let pick = rng.gen_range(0..moves.len());
            g.apply(moves[pick]);
        }
        g
    }

    #[test]
    fn standard_starts_in_progress() {
        let g = Game::new_standard();
        assert_eq!(g.state(), GameState::InProgress);
        assert_eq!(g.turn, Side::Black);
        assert_eq!(g.max_plies, DEFAULT_MAX_PLIES);
        assert_eq!(g.no_progress_plies, NO_PROGRESS_DISABLED);
        assert_eq!(g.plies_since_capture, 0);
    }

    #[test]
    fn constructors_agree_with_new() {
        assert_eq!(
            Game::new_standard(),
            Game::new(Opening::Standard, DEFAULT_MAX_PLIES, NO_PROGRESS_DISABLED)
        );
        assert_eq!(
            Game::new_belgian_daisy(),
            Game::new(
                Opening::BelgianDaisy,
                DEFAULT_MAX_PLIES,
                NO_PROGRESS_DISABLED
            )
        );
        assert_eq!(Game::default(), Game::new_standard());
    }

    #[test]
    fn first_move_swaps_turn() {
        let mut g = Game::new_standard();
        let mv = g.legal_moves()[0];
        g.apply(mv);
        assert_eq!(g.turn, Side::White);
        assert_eq!(g.ply, 1);
        assert_eq!(g.plies_since_capture, 1, "a non-capturing ply increments");
    }

    #[test]
    fn six_pushed_off_wins() {
        let mut g = Game::new_standard();
        g.board.pushed_off[Side::Black.idx()] = 6;
        assert_eq!(g.state(), GameState::Wins(Side::Black));

        let mut g = Game::new_standard();
        g.board.pushed_off[Side::White.idx()] = 6;
        assert_eq!(g.state(), GameState::Wins(Side::White));
    }

    #[test]
    fn win_threshold_takes_precedence_over_cap() {
        // Contrived 6-6: the differential is zero, so adjudication would say
        // Draw. The win threshold must fire first.
        let mut g = Game::new_standard();
        g.board.pushed_off = [6, 6];
        g.ply = g.max_plies;
        assert_eq!(g.state(), GameState::Wins(Side::Black));
        // ...and it fires even without the cap being reached.
        g.ply = 0;
        assert_eq!(g.state(), GameState::Wins(Side::Black));
    }

    #[test]
    fn cap_adjudicates_positive_differential_to_black() {
        let mut g = Game::new_standard();
        g.board.pushed_off = [3, 1];
        assert_eq!(g.state(), GameState::InProgress);
        g.ply = g.max_plies;
        assert_eq!(g.state(), GameState::Wins(Side::Black));
    }

    #[test]
    fn cap_adjudicates_negative_differential_to_white() {
        let mut g = Game::new_standard();
        g.board.pushed_off = [1, 5];
        g.ply = g.max_plies;
        assert_eq!(g.state(), GameState::Wins(Side::White));
    }

    #[test]
    fn cap_adjudicates_equal_differential_to_draw() {
        let mut g = Game::new_standard();
        g.board.pushed_off = [0, 0];
        g.ply = g.max_plies;
        assert_eq!(g.state(), GameState::Draw);
        g.board.pushed_off = [4, 4];
        assert_eq!(g.state(), GameState::Draw);
    }

    #[test]
    fn cap_is_configurable() {
        let mut g = Game::new(Opening::Standard, 12, NO_PROGRESS_DISABLED);
        g.ply = 11;
        assert_eq!(g.state(), GameState::InProgress);
        g.ply = 12;
        assert_eq!(g.state(), GameState::Draw);
        g.board.pushed_off = [1, 0];
        assert_eq!(g.state(), GameState::Wins(Side::Black));
    }

    #[test]
    fn no_progress_rule_disabled_by_default() {
        let mut g = Game::new_standard();
        assert_eq!(g.no_progress_plies, NO_PROGRESS_DISABLED);
        g.plies_since_capture = 100_000;
        assert_eq!(g.state(), GameState::InProgress);
    }

    #[test]
    fn no_progress_rule_fires_and_adjudicates() {
        let mut g = Game::new(Opening::Standard, DEFAULT_MAX_PLIES, 10);
        let mut rng = SmallRng::seed_from_u64(3);
        for _ in 0..10 {
            assert_eq!(g.state(), GameState::InProgress);
            let moves = g.legal_moves();
            let pick = rng.gen_range(0..moves.len());
            g.apply(moves[pick]);
        }
        // No capture is reachable in 10 plies from the standard opening.
        assert_eq!(g.plies_since_capture, 10);
        assert!(g.ply < g.max_plies, "the ply cap must not be what fired");
        assert_eq!(g.state(), GameState::Draw);

        // Same rule, decisive differentials.
        g.board.pushed_off = [2, 0];
        assert_eq!(g.state(), GameState::Wins(Side::Black));
        g.board.pushed_off = [2, 4];
        assert_eq!(g.state(), GameState::Wins(Side::White));
    }

    #[test]
    fn plies_since_capture_resets_only_on_capture() {
        // White A3+A2 push the lone black marble at A1 off the west edge.
        let mut b = Board::empty();
        b.set(crate::cell::parse("A1").unwrap(), Some(Side::Black));
        b.set(crate::cell::parse("A2").unwrap(), Some(Side::White));
        b.set(crate::cell::parse("A3").unwrap(), Some(Side::White));
        // A spare black marble far away, so Black still has a quiet reply.
        b.set(crate::cell::parse("E5").unwrap(), Some(Side::Black));
        let mut g = Game {
            board: b,
            turn: Side::White,
            ply: 0,
            max_plies: DEFAULT_MAX_PLIES,
            plies_since_capture: 37,
            no_progress_plies: NO_PROGRESS_DISABLED,
        };
        let push = *g
            .legal_moves()
            .iter()
            .find(|m| {
                matches!(m, Move::Inline { anchor, dir: crate::cell::Dir::W, size: 2 }
                    if *anchor == crate::cell::parse("A3").unwrap())
            })
            .expect("2-vs-1 west push off the edge should be legal");
        g.apply(push);
        assert_eq!(g.board.pushed_off[Side::White.idx()], 1);
        assert_eq!(g.plies_since_capture, 0, "capture must reset the counter");

        // A quiet ply increments again.
        let quiet = g.legal_moves()[0];
        g.apply(quiet);
        assert_eq!(g.plies_since_capture, 1);
    }

    #[test]
    fn plies_since_capture_invariant_over_random_playouts() {
        let mut rng = SmallRng::seed_from_u64(0xabc);
        for _ in 0..20 {
            let mut g = Game::new(Opening::BelgianDaisy, DEFAULT_MAX_PLIES, NO_PROGRESS_DISABLED);
            while !g.is_terminal() {
                let before_counters = g.board.pushed_off;
                let before_counter = g.plies_since_capture;
                let moves = g.legal_moves();
                let pick = rng.gen_range(0..moves.len());
                g.apply(moves[pick]);
                if g.board.pushed_off == before_counters {
                    assert_eq!(g.plies_since_capture, before_counter + 1);
                } else {
                    assert_eq!(g.plies_since_capture, 0);
                }
            }
        }
    }

    #[test]
    fn score_diff_is_symmetric_and_clamped() {
        let mut g = Game::new_standard();
        assert_eq!(g.score_diff(Side::Black), 0);
        assert_eq!(g.score_diff(Side::White), 0);

        g.board.pushed_off = [4, 1];
        assert_eq!(g.score_diff(Side::Black), 3);
        assert_eq!(g.score_diff(Side::White), -3);

        g.board.pushed_off = [6, 0];
        assert_eq!(g.score_diff(Side::Black), 6);
        assert_eq!(g.score_diff(Side::White), -6);

        // Out-of-range counters clamp rather than overflow the target range.
        g.board.pushed_off = [14, 0];
        assert_eq!(g.score_diff(Side::Black), 6);
        assert_eq!(g.score_diff(Side::White), -6);
    }

    #[test]
    fn handicap_direction_and_consistency() {
        let mut rng = SmallRng::seed_from_u64(11);
        // Black has lost 2, White has lost 4.
        let g = Game::with_handicap(
            Opening::BelgianDaisy,
            DEFAULT_MAX_PLIES,
            NO_PROGRESS_DISABLED,
            2,
            4,
            &mut rng,
        );
        assert_eq!(g.board.count(Side::Black), 12);
        assert_eq!(g.board.count(Side::White), 10);
        assert_eq!(g.board.pushed_off[Side::White.idx()], 2, "White is credited Black's losses");
        assert_eq!(g.board.pushed_off[Side::Black.idx()], 4, "Black is credited White's losses");
        assert_eq!(g.score_diff(Side::Black), 2);
        assert_eq!(g.state(), GameState::InProgress);
    }

    #[test]
    fn handicap_zero_zero_is_the_plain_opening() {
        let mut rng = SmallRng::seed_from_u64(1);
        let g = Game::with_handicap(
            Opening::Standard,
            DEFAULT_MAX_PLIES,
            NO_PROGRESS_DISABLED,
            0,
            0,
            &mut rng,
        );
        assert_eq!(g, Game::new_standard());
    }

    #[test]
    fn handicap_five_five_ends_on_the_next_capture() {
        let mut rng = SmallRng::seed_from_u64(7);
        let mut saw_capture = false;
        for _ in 0..40 {
            let mut g = Game::with_handicap(
                Opening::BelgianDaisy,
                DEFAULT_MAX_PLIES,
                NO_PROGRESS_DISABLED,
                5,
                5,
                &mut rng,
            );
            assert_eq!(g.board.pushed_off, [5, 5]);
            assert_eq!(g.board.count(Side::Black), 9);
            assert_eq!(g.board.count(Side::White), 9);
            assert_eq!(g.state(), GameState::InProgress);

            while !g.is_terminal() {
                let before = g.board.pushed_off;
                let moves = g.legal_moves();
                let pick = rng.gen_range(0..moves.len());
                g.apply(moves[pick]);
                if g.board.pushed_off != before {
                    saw_capture = true;
                    let winner = if g.board.pushed_off[Side::Black.idx()] > before[Side::Black.idx()]
                    {
                        Side::Black
                    } else {
                        Side::White
                    };
                    assert_eq!(
                        g.state(),
                        GameState::Wins(winner),
                        "at 5-5 the very next capture must end the game"
                    );
                    break;
                }
            }
        }
        assert!(saw_capture, "expected at least one capture across 40 playouts");
    }

    #[test]
    fn random_playout_terminates() {
        let mut rng = SmallRng::seed_from_u64(0xdeadbeef);
        let g = random_playout(Game::new_standard(), &mut rng);
        assert!(g.ply <= g.max_plies);
        assert!(g.is_terminal());
    }

    #[test]
    fn seeded_belgian_playouts_terminate_with_mixed_outcomes() {
        let mut rng = SmallRng::seed_from_u64(0x1234_5678);
        let mut black = 0;
        let mut white = 0;
        let mut draws = 0;
        for _ in 0..100 {
            let a = rng.gen_range(0..=5u8);
            let b = rng.gen_range(0..=5u8);
            let g = Game::with_handicap(
                Opening::BelgianDaisy,
                DEFAULT_MAX_PLIES,
                NO_PROGRESS_DISABLED,
                a,
                b,
                &mut rng,
            );
            let g = random_playout(g, &mut rng);
            assert!(g.ply <= g.max_plies);
            match g.state() {
                GameState::Wins(Side::Black) => black += 1,
                GameState::Wins(Side::White) => white += 1,
                GameState::Draw => draws += 1,
                GameState::InProgress => unreachable!(),
            }
        }
        assert_eq!(black + white + draws, 100);
        assert!(black > 0 && white > 0, "expected both sides to win some games: B={black} W={white} D={draws}");
        assert!(
            black + white >= 70,
            "handicap seeding should make most games decisive: B={black} W={white} D={draws}"
        );
    }

    /// Decisive rate (fraction of games not ending in `Draw`) and mean plies
    /// under uniformly-random play. Run with:
    /// `cargo test -p abalone-game -- --ignored --nocapture measure_decisive_rates`
    #[test]
    #[ignore = "measurement; run explicitly with --ignored --nocapture"]
    fn measure_decisive_rates() {
        const N: u32 = 200;

        fn measure<F: FnMut(&mut SmallRng) -> Game>(
            label: &str,
            n: u32,
            rng: &mut SmallRng,
            mut make: F,
            legacy_draw_at_cap: bool,
        ) {
            let mut decisive = 0u32;
            let mut plies = 0u64;
            let mut abs_diff = 0u64;
            let mut in_play_captures = 0u64;
            for _ in 0..n {
                let mut g = make(rng);
                // Captures already on the board from handicap seeding do not
                // count as captures *earned in play*.
                let seeded = u64::from(g.board.pushed_off[0]) + u64::from(g.board.pushed_off[1]);
                while !g.is_terminal() {
                    let moves = g.legal_moves();
                    let pick = rng.gen_range(0..moves.len());
                    g.apply(moves[pick]);
                }
                plies += u64::from(g.ply);
                in_play_captures += u64::from(g.board.pushed_off[0])
                    + u64::from(g.board.pushed_off[1])
                    - seeded;
                abs_diff += u64::from(g.score_diff(Side::Black).unsigned_abs());
                // Old semantics: anything short of six captures was a draw.
                let reached_threshold = g.board.pushed_off[0] >= WIN_THRESHOLD
                    || g.board.pushed_off[1] >= WIN_THRESHOLD;
                let decisive_here = if legacy_draw_at_cap {
                    reached_threshold
                } else {
                    !matches!(g.state(), GameState::Draw)
                };
                if decisive_here {
                    decisive += 1;
                }
            }
            let n = f64::from(n);
            println!(
                "{label:<56} decisive {:>5.1}%   mean plies {:>5.1}   mean |d| {:>4.2}   \
                 mean captures in play {:>4.2}",
                100.0 * f64::from(decisive) / n,
                plies as f64 / n,
                abs_diff as f64 / n,
                in_play_captures as f64 / n,
            );
        }

        let mut rng = SmallRng::seed_from_u64(0x5eed_5eed);
        println!("\n{N} uniformly-random playouts per condition:");
        measure(
            "(a) standard, 400-ply cap, draw-at-cap (old baseline)",
            N,
            &mut rng,
            |_| Game::new(Opening::Standard, 400, NO_PROGRESS_DISABLED),
            true,
        );
        measure(
            "(b) standard, 200-ply cap, score adjudication",
            N,
            &mut rng,
            |_| Game::new(Opening::Standard, DEFAULT_MAX_PLIES, NO_PROGRESS_DISABLED),
            false,
        );
        measure(
            "(c) belgian daisy, 200-ply cap, score adjudication",
            N,
            &mut rng,
            |_| Game::new(Opening::BelgianDaisy, DEFAULT_MAX_PLIES, NO_PROGRESS_DISABLED),
            false,
        );
        measure(
            "(d) belgian daisy, 200-ply cap, adjudication, handicap 0..=5",
            N,
            &mut rng,
            |r| {
                let a = r.gen_range(0..=5u8);
                let b = r.gen_range(0..=5u8);
                Game::with_handicap(
                    Opening::BelgianDaisy,
                    DEFAULT_MAX_PLIES,
                    NO_PROGRESS_DISABLED,
                    a,
                    b,
                    r,
                )
            },
            false,
        );
        println!();
    }
}
