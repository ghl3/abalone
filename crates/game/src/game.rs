//! `Game`: position + turn + ply count + terminal check.
//!
//! Terminal:
//! - `pushed_off[side]` reaching `WIN_THRESHOLD` (= 6) means `side` wins
//!   (they have pushed 6 of the opponent's marbles off).
//! - Or `ply_count >= MAX_PLIES` -> draw.

use crate::board::Board;
use crate::cell::Side;
use crate::moves::{legal_moves, Move, MoveList};

/// Standard win condition: push 6 of the opponent's marbles off the board.
pub const WIN_THRESHOLD: u8 = 6;

/// Hard cap on plies to prevent infinite drift in self-play. Tournament
/// games often cap around 200; we set it slightly higher to be safe.
pub const MAX_PLIES: u32 = 400;

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
}

impl Default for Game {
    fn default() -> Self {
        Self::new_standard()
    }
}

impl Game {
    pub fn new_standard() -> Self {
        Self {
            board: Board::standard(),
            // Black moves first by tournament convention.
            turn: Side::Black,
            ply: 0,
        }
    }

    pub fn new_belgian_daisy() -> Self {
        Self {
            board: Board::belgian_daisy(),
            turn: Side::Black,
            ply: 0,
        }
    }

    pub fn legal_moves(&self) -> MoveList {
        legal_moves(&self.board, self.turn)
    }

    pub fn apply(&mut self, mv: Move) {
        self.board.apply(mv, self.turn);
        self.turn = self.turn.other();
        self.ply += 1;
    }

    pub fn state(&self) -> GameState {
        if self.board.pushed_off[Side::Black.idx()] >= WIN_THRESHOLD {
            // Black has pushed 6 white marbles off.
            GameState::Wins(Side::Black)
        } else if self.board.pushed_off[Side::White.idx()] >= WIN_THRESHOLD {
            GameState::Wins(Side::White)
        } else if self.ply >= MAX_PLIES {
            GameState::Draw
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

    #[test]
    fn standard_starts_in_progress() {
        let g = Game::new_standard();
        assert_eq!(g.state(), GameState::InProgress);
        assert_eq!(g.turn, Side::Black);
    }

    #[test]
    fn first_move_swaps_turn() {
        let mut g = Game::new_standard();
        let mv = g.legal_moves()[0];
        g.apply(mv);
        assert_eq!(g.turn, Side::White);
        assert_eq!(g.ply, 1);
    }

    #[test]
    fn six_pushed_off_wins() {
        let mut g = Game::new_standard();
        g.board.pushed_off[Side::Black.idx()] = 6;
        assert_eq!(g.state(), GameState::Wins(Side::Black));
    }

    #[test]
    fn ply_cap_draws() {
        let mut g = Game::new_standard();
        g.ply = MAX_PLIES;
        assert_eq!(g.state(), GameState::Draw);
    }

    #[test]
    fn random_playout_terminates() {
        // Self-play with simple pseudo-random move pick (deterministic seed)
        // must terminate within MAX_PLIES.
        let mut g = Game::new_standard();
        let mut rng_state: u64 = 0xdeadbeef;
        while !g.is_terminal() {
            let moves = g.legal_moves();
            assert!(!moves.is_empty(), "no legal moves but not terminal at ply {}", g.ply);
            // xorshift
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            let pick = (rng_state as usize) % moves.len();
            g.apply(moves[pick]);
        }
        assert!(g.ply <= MAX_PLIES);
    }
}
