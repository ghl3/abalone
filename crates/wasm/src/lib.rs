//! WebAssembly wrapper around `abalone-engine` for the React frontend.
//!
//! The boundary is kept narrow on purpose:
//!   * `Game` is opaque to JS (a handle wrapping a Rust struct).
//!   * Moves cross as `u16` indices, not enum variants. JS asks for the legal
//!     index list, picks one, and applies it back. No serde, no JSON.
//!   * Cells cross as `u8` indices `0..81`; JS computes (q, r) by `(c % 9, c / 9)`.

use abalone_engine::{decode, encode, Game, GameState, Move, Side};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
#[derive(Copy, Clone)]
pub enum WasmSide {
    Black = 0,
    White = 1,
}

impl From<WasmSide> for Side {
    fn from(s: WasmSide) -> Side {
        match s {
            WasmSide::Black => Side::Black,
            WasmSide::White => Side::White,
        }
    }
}

#[wasm_bindgen]
#[derive(Copy, Clone)]
pub enum WasmGameState {
    InProgress = 0,
    BlackWins = 1,
    WhiteWins = 2,
    Draw = 3,
}

impl From<GameState> for WasmGameState {
    fn from(g: GameState) -> WasmGameState {
        match g {
            GameState::InProgress => WasmGameState::InProgress,
            GameState::Wins(Side::Black) => WasmGameState::BlackWins,
            GameState::Wins(Side::White) => WasmGameState::WhiteWins,
            GameState::Draw => WasmGameState::Draw,
        }
    }
}

#[wasm_bindgen]
pub struct WasmGame {
    inner: Game,
}

#[wasm_bindgen]
impl WasmGame {
    /// Standard tournament starting position (Black to move).
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmGame {
        WasmGame {
            inner: Game::new_standard(),
        }
    }

    pub fn belgian_daisy() -> WasmGame {
        WasmGame {
            inner: Game::new_belgian_daisy(),
        }
    }

    pub fn turn(&self) -> WasmSide {
        match self.inner.turn {
            Side::Black => WasmSide::Black,
            Side::White => WasmSide::White,
        }
    }

    pub fn ply(&self) -> u32 {
        self.inner.ply
    }

    pub fn state(&self) -> WasmGameState {
        self.inner.state().into()
    }

    /// Marble at cell `c` (0..81): -1 empty, 0 Black, 1 White.
    /// Off-board cells return -1.
    pub fn cell(&self, c: u8) -> i8 {
        match self.inner.board.at(c) {
            None => -1,
            Some(Side::Black) => 0,
            Some(Side::White) => 1,
        }
    }

    /// Number of marbles `side` has lost (i.e. been pushed off).
    pub fn lost(&self, side: WasmSide) -> u8 {
        self.inner.board.lost(side.into())
    }

    /// Legal move indices for the side to move.
    pub fn legal_indices(&self) -> Vec<u16> {
        self.inner.legal_moves().iter().map(|&m| encode(m)).collect()
    }

    /// Cells occupied by the moving group of `idx`, as `[c0, c1, c2, c3]`
    /// padded with `0xFF` for unused slots. JS uses this to highlight the
    /// source cells when previewing or executing a move.
    pub fn move_source_cells(&self, idx: u16) -> Vec<u8> {
        let m = decode(idx);
        let mut out = vec![0xFFu8; 4];
        match m {
            Move::Inline { anchor, dir, size } => {
                let d = dir.shift();
                for i in 0..size as i32 {
                    out[i as usize] = (anchor as i32 + i * d) as u8;
                }
            }
            Move::Broadside {
                anchor,
                group_dir,
                size,
                ..
            } => {
                let dg = group_dir.shift();
                for i in 0..size as i32 {
                    out[i as usize] = (anchor as i32 + i * dg) as u8;
                }
            }
        }
        out
    }

    /// Apply a legal move (by index). Caller must ensure `idx` came from
    /// `legal_indices()`; behaviour on illegal indices is debug-asserted.
    pub fn apply_index(&mut self, idx: u16) {
        let m = decode(idx);
        self.inner.apply(m);
    }

    pub fn debug_render(&self) -> String {
        format!("{}", self.inner.board)
    }
}

/// Standalone helper: human-readable notation for a move index.
#[wasm_bindgen]
pub fn move_notation(idx: u16) -> String {
    format!("{}", decode(idx))
}

impl Default for WasmGame {
    fn default() -> Self {
        Self::new()
    }
}
