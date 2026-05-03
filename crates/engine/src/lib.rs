//! Core Abalone engine: bitboard-backed board, legal-move generation,
//! and game-state machine. No I/O, no allocation in hot paths.

pub mod bitboard;
pub mod board;
pub mod cell;
pub mod game;
pub mod moves;
pub mod notation;

pub use board::Board;
pub use cell::{Cell, Dir, Side, ALL_DIRS, POSITIVE_DIRS};
pub use game::{Game, GameState};
pub use moves::Move;
