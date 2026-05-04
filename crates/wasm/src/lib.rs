//! WebAssembly wrapper around `abalone-game` for the React frontend.
//!
//! The boundary is kept narrow on purpose:
//!   * `Game` is opaque to JS (a handle wrapping a Rust struct).
//!   * Moves cross as `u16` indices, not enum variants. JS asks for the legal
//!     index list, picks one, and applies it back. No serde, no JSON.
//!   * Cells cross as `u8` indices `0..81`; JS computes (q, r) by `(c % 9, c / 9)`.

use abalone_game::{decode, encode, Game, GameState, Move, Side};
use abalone_mcts::eval::{evaluate, Weights};
use abalone_mcts::{heuristic, search, SearchConfig};
use rand::rngs::SmallRng;
use rand::SeedableRng;
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

    /// Hypothetically apply `idx` to a clone of the current position and
    /// return the resulting cell array (parallel to [`Self::cell`]). Does not
    /// mutate `self`. JS uses this to render a full preview (own destinations,
    /// pushed opponents, captures) while the user is mid-drag.
    pub fn move_preview(&self, idx: u16) -> Vec<i8> {
        let m = decode(idx);
        let mut tmp = self.inner;
        tmp.apply(m);
        let mut out = vec![-1i8; 81];
        for c in 0..81u8 {
            match tmp.board.at(c) {
                None => out[c as usize] = -1,
                Some(Side::Black) => out[c as usize] = 0,
                Some(Side::White) => out[c as usize] = 1,
            }
        }
        out
    }

    /// Find the legal move whose source cells equal `cells` (any order) and
    /// whose direction of motion is `dir_idx` (0..6, matching `Dir`). Returns
    /// the move index, or -1 if no such move exists.
    ///
    /// "Direction of motion" is the inline direction for inline moves, or
    /// the move_dir for broadside moves.
    pub fn find_move(&self, cells: Vec<u8>, dir_idx: u8) -> i32 {
        if dir_idx >= 6 {
            return -1;
        }
        let want_dir = abalone_game::Dir::from_idx(dir_idx);

        let mut want_set: u128 = 0;
        for &c in &cells {
            if (c as usize) >= 81 {
                return -1;
            }
            want_set |= 1u128 << c;
        }

        for m in self.inner.legal_moves() {
            let motion = match m {
                Move::Inline { dir, .. } => dir,
                Move::Broadside { move_dir, .. } => move_dir,
            };
            if motion as u8 != want_dir as u8 {
                continue;
            }
            // Build the source-cell bitset for this move and compare.
            let src = self.move_source_cells(encode(m));
            let mut got_set: u128 = 0;
            for &c in &src {
                if c != 0xFF {
                    got_set |= 1u128 << c;
                }
            }
            if got_set == want_set {
                return encode(m) as i32;
            }
        }
        -1
    }

    pub fn debug_render(&self) -> String {
        format!("{}", self.inner.board)
    }

    /// Heuristic eval of the current position from White's POV (positive
    /// = White advantage, negative = Black advantage). Range [-1, 1].
    /// O(1); useful as the eval-bar value when the analysis panel is off
    /// or while MCTS is computing.
    pub fn eval_white_pov(&self) -> f32 {
        let pov_to_move = evaluate(&self.inner.board, self.inner.turn, &Weights::default());
        match self.inner.turn {
            Side::White => pov_to_move,
            Side::Black => -pov_to_move,
        }
    }

    /// Run heuristic-MCTS for `simulations` iterations from the current
    /// position and return ranked children with their Q-values (white POV)
    /// and visit counts. Returns `None` if the game is terminal.
    pub fn analyze(&self, simulations: u32) -> Option<AnalysisResult> {
        if self.inner.is_terminal() {
            return None;
        }
        let cfg = SearchConfig {
            simulations: simulations.max(1),
            c_puct: 1.4,
        };
        // Heuristic eval is deterministic given a position; rng is only
        // needed to satisfy the search signature.
        let mut rng = SmallRng::seed_from_u64(0);
        let res = search(&self.inner, &cfg, &mut rng, heuristic)?;

        let to_white_sign = if self.inner.turn == Side::White {
            1.0f32
        } else {
            -1.0f32
        };

        let n = res.visits.len();
        let mut indices = Vec::with_capacity(n);
        let mut evals = Vec::with_capacity(n);
        let mut visits = Vec::with_capacity(n);
        for (&(mv, v), &q) in res.visits.iter().zip(res.q_parent_pov.iter()) {
            indices.push(encode(mv));
            evals.push(q * to_white_sign);
            visits.push(v);
        }

        // Root eval: the Q-value of the most-visited child (the engine's
        // recommended line) from white POV. This gives the eval bar a
        // search-derived value, not just a static heuristic snapshot.
        let root_eval = res
            .visits
            .iter()
            .zip(res.q_parent_pov.iter())
            .max_by_key(|((_, v), _)| *v)
            .map(|(_, q)| *q * to_white_sign)
            .unwrap_or(0.0);

        Some(AnalysisResult {
            indices,
            evals,
            visits,
            root_eval,
        })
    }
}

/// Result of a single MCTS analysis pass. Fields are exposed via getters
/// so JS can pull out parallel arrays in one allocation each.
#[wasm_bindgen]
pub struct AnalysisResult {
    indices: Vec<u16>,
    evals: Vec<f32>,
    visits: Vec<u32>,
    root_eval: f32,
}

#[wasm_bindgen]
impl AnalysisResult {
    /// Move indices for every legal move, in the same order as
    /// [`evals`](Self::evals) and [`visits`](Self::visits).
    pub fn indices(&self) -> Vec<u16> {
        self.indices.clone()
    }
    /// Q-value of each move from White's POV (positive = White advantage).
    pub fn evals(&self) -> Vec<f32> {
        self.evals.clone()
    }
    /// MCTS visit count of each move.
    pub fn visits(&self) -> Vec<u32> {
        self.visits.clone()
    }
    /// Engine's evaluation of the current root position (white POV) — the
    /// Q-value of the most-visited child.
    pub fn root_eval(&self) -> f32 {
        self.root_eval
    }
}

/// Standalone helper: human-readable notation for a move index.
#[wasm_bindgen]
pub fn move_notation(idx: u16) -> String {
    format!("{}", decode(idx))
}

/// Direction of motion for a move index, returned as the engine `Dir`
/// index `0..6` (matches `DIR_SHIFTS` and `DIR_PIXEL` on the JS side).
/// For inline moves, this is the inline direction; for broadside, it's
/// the move_dir (perpendicular slide).
#[wasm_bindgen]
pub fn move_motion_dir(idx: u16) -> u8 {
    match decode(idx) {
        Move::Inline { dir, .. } => dir as u8,
        Move::Broadside { move_dir, .. } => move_dir as u8,
    }
}

impl Default for WasmGame {
    fn default() -> Self {
        Self::new()
    }
}
