//! Core Abalone engine: bitboard-backed board, legal-move generation,
//! and game-state machine. No I/O, no allocation in hot paths.
//!
//! D6 hex symmetry is intentionally not implemented here — it lives in
//! `model/encoder.py` instead, where it's applied as data augmentation
//! during training. The encoding here (cells, directions, move-index
//! space) is the canonical specification; Python builds its symmetry
//! tables from these conventions and tests them independently.

pub mod bitboard;
pub mod board;
pub mod cell;
pub mod game;
pub mod move_index;
pub mod moves;
pub mod notation;

pub use board::Board;
pub use cell::{Cell, Dir, Side, ALL_DIRS, POSITIVE_DIRS};
pub use game::{Game, GameState};
pub use move_index::{decode, encode, MOVE_SPACE};
pub use moves::Move;
