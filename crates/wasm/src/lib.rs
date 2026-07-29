//! WebAssembly wrapper around `abalone-game` for the React frontend.
//!
//! The boundary is kept narrow on purpose:
//!   * `Game` is opaque to JS (a handle wrapping a Rust struct).
//!   * Moves cross as `u16` indices, not enum variants. JS asks for the legal
//!     index list, picks one, and applies it back. No serde, no JSON.
//!   * Cells cross as `u8` indices `0..81`; JS computes (q, r) by `(c % 9, c / 9)`.

use abalone_encoder::{encode_planes, PLANE_SIZE};
use abalone_game::{decode, encode, Game, GameState, Move, Side, MOVE_SPACE};
use abalone_mcts::{LeafEval, Search, SearchConfig};
use wasm_bindgen::prelude::*;

/// Class order of the 3-way value head, matching `model/batch.py` and
/// `ort_eval.rs`. The browser must collapse it the same way the trainer does
/// or the eval bar and the search disagree with self-play.
const VALUE_WIN: usize = 0;
const VALUE_DRAW: usize = 1;
const VALUE_LOSS: usize = 2;
const VALUE_CLASSES: usize = 3;
/// The score head is a softmax over a capture differential of −6..+6.
const SCORE_CLASSES: usize = 13;
const SCORE_OFFSET: usize = 6;

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
        self.inner
            .legal_moves()
            .iter()
            .map(|&m| encode(m))
            .collect()
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

    /// Start a network-guided search from the current position, to be driven
    /// from JS as a coroutine (see [`WasmSearch`]). Nothing is evaluated here:
    /// the caller alternates [`WasmSearch::next_batch`] and
    /// [`WasmSearch::submit`] until [`WasmSearch::is_done`].
    pub fn begin_search(
        &self,
        simulations: u32,
        batch_size: usize,
        c_puct: f32,
        seed: u32,
    ) -> WasmSearch {
        let cfg = SearchConfig {
            simulations: simulations.max(1),
            c_puct,
            batch_size: batch_size.max(1),
            // Exploration noise is a self-play device; a browser opponent
            // should play its best move, not a deliberately noised one.
            dirichlet_eps: 0.0,
            // The browser is the analysis client, and the only caller that
            // wants the readout. Self-play leaves this off and runs the
            // AlphaZero search unchanged.
            track_outcome_stats: true,
            ..Default::default()
        };
        WasmSearch {
            inner: Search::begin(&self.inner, &cfg, u64::from(seed)),
            batch: Vec::with_capacity(cfg.batch_size),
            planes: Vec::new(),
            root_turn: self.inner.turn,
            score_available: false,
        }
    }
}

/// A network-guided MCTS driven from JavaScript, because `onnxruntime-web`'s
/// `run()` is async and WASM cannot await it. Selection, virtual loss, backup
/// and the node arena all stay in Rust — the same [`Search`] self-play uses —
/// and only the forward pass crosses back to JS:
///
/// ```js
/// const s = game.begin_search(400, 16, 1.4, seed);
/// for (;;) {
///   const planes = s.next_batch();          // (n, 14, 9, 9) flattened
///   if (planes.length === 0) break;
///   const out = await session.run({ planes: tensor(planes) });
///   s.submit(out.policy_logits.data, out.value.data);
/// }
/// const r = s.result();
/// ```
#[wasm_bindgen]
pub struct WasmSearch {
    inner: Search,
    /// The positions behind the batch last handed to JS. Kept because
    /// `submit` must gather each leaf's policy logits in that leaf's own
    /// `legal_moves()` order — the order [`Search`] expects its priors in.
    batch: Vec<Game>,
    /// Reused staging buffer for the encoded batch.
    planes: Vec<f32>,
    root_turn: Side,
    /// Whether the last `submit` carried a usable `score` head. A model
    /// exported without one is still perfectly playable; it just has no margin
    /// to report, and reporting zeroes would be worse than reporting nothing.
    score_available: bool,
}

#[wasm_bindgen]
impl WasmSearch {
    /// Positions awaiting evaluation, encoded as a flattened `(n, 14, 9, 9)`
    /// float tensor. An empty result means the search is finished.
    ///
    /// Calling this twice without an intervening [`submit`](Self::submit)
    /// discards the un-submitted batch, reverting its virtual loss.
    pub fn next_batch(&mut self) -> Vec<f32> {
        self.batch.clear();
        self.batch.extend_from_slice(self.inner.next_batch());
        self.planes.clear();
        self.planes.resize(self.batch.len() * PLANE_SIZE, 0.0);
        for (i, g) in self.batch.iter().enumerate() {
            encode_planes(
                g,
                &mut self.planes[i * PLANE_SIZE..(i + 1) * PLANE_SIZE],
            );
        }
        self.planes.clone()
    }

    /// Positions in the batch last returned by [`next_batch`](Self::next_batch).
    pub fn batch_len(&self) -> usize {
        self.batch.len()
    }

    /// Hand back the network's raw output for the last batch: `policy_logits`
    /// of `n * 2562` and `value` of `n * 3` (win, draw, loss), both exactly as
    /// the ONNX graph emits them. Masking to legal moves, the softmax over
    /// them, and the `P(win) - P(loss)` collapse all happen here, so the
    /// browser applies the identical arithmetic to `ort_eval.rs`.
    pub fn submit(
        &mut self,
        policy_logits: &[f32],
        value: &[f32],
        score: &[f32],
    ) -> Result<(), JsError> {
        let n = self.batch.len();
        if n == 0 {
            return Ok(());
        }
        if policy_logits.len() != n * MOVE_SPACE {
            return Err(JsError::new(&format!(
                "policy_logits has {} elements, expected {} ({n}x{MOVE_SPACE})",
                policy_logits.len(),
                n * MOVE_SPACE,
            )));
        }
        if !value.len().is_multiple_of(n) {
            return Err(JsError::new(&format!(
                "value has {} elements, not divisible by batch {n}",
                value.len(),
            )));
        }
        let value_dim = value.len() / n;

        // A `score` row per leaf is optional: a model without the head passes
        // an empty array and the margin column simply has nothing to show.
        let has_score = score.len() == n * SCORE_CLASSES;
        self.score_available = has_score;

        let mut evals = Vec::with_capacity(n);
        let mut wdl = Vec::with_capacity(n);
        let mut margins = Vec::with_capacity(n);
        for (i, g) in self.batch.iter().enumerate() {
            let row_value = &value[i * value_dim..(i + 1) * value_dim];
            let v = collapse_value(row_value)?;
            let row = &policy_logits[i * MOVE_SPACE..(i + 1) * MOVE_SPACE];
            let legal = g.legal_moves();
            let logits: Vec<f32> =
                legal.iter().map(|&m| row[encode(m) as usize]).collect();
            evals.push(LeafEval {
                value: v,
                priors: Some(softmax(logits)),
            });
            wdl.push(distribute_value(row_value, v));
            margins.push(if has_score {
                expected_margin(&score[i * SCORE_CLASSES..(i + 1) * SCORE_CLASSES])
            } else {
                0.0
            });
        }
        // The scalar in `evals` is the only thing that steers the search; the
        // two extra arrays are backed up alongside it purely so the panel can
        // report probabilities and a margin that came out of the tree rather
        // than off a single forward pass.
        self.inner.submit_with_stats(&evals, &wdl, &margins);
        Ok(())
    }

    pub fn is_done(&self) -> bool {
        self.inner.is_done()
    }

    /// Simulations backed up into the root so far — a progress readout while
    /// the search is still running.
    pub fn root_visits(&self) -> u32 {
        self.inner.root_visits()
    }

    /// Ranked root children with their visit counts and Q-values.
    /// `None` until the first [`submit`](Self::submit) has expanded the root,
    /// and for a terminal position.
    ///
    /// Safe to call *during* a search, not only at the end: it reads the tree
    /// as it stands, which is what lets the panel refine its rows as visits
    /// accumulate instead of blanking until the budget is spent.
    pub fn result(&self) -> Option<AnalysisResult> {
        let res = self.inner.result()?;
        let to_white_sign = if self.root_turn == Side::White {
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

        // Restated for White, like the evals. Swapping win and loss is the
        // POV flip for a distribution; negating is the flip for a margin.
        let white = to_white_sign > 0.0;
        let mut wdl = Vec::with_capacity(n * 3);
        for w in &res.wdl_parent_pov {
            if white {
                wdl.extend_from_slice(&[w[0], w[1], w[2]]);
            } else {
                wdl.extend_from_slice(&[w[2], w[1], w[0]]);
            }
        }
        let margins: Vec<f32> = if self.score_available {
            res.score_parent_pov
                .iter()
                .map(|s| s * to_white_sign)
                .collect()
        } else {
            Vec::new()
        };

        // The engine's own reading of the position is the line it intends to
        // play, so every root-level number is taken from the most-visited
        // child rather than averaged over children it rejected.
        let best_slot = res
            .visits
            .iter()
            .enumerate()
            .max_by_key(|(_, (_, v))| *v)
            .map(|(i, _)| i);
        let root_eval = best_slot.map_or(0.0, |i| res.q_parent_pov[i] * to_white_sign);
        let root_wdl = best_slot
            .filter(|_| !wdl.is_empty())
            .map(|i| vec![wdl[i * 3], wdl[i * 3 + 1], wdl[i * 3 + 2]])
            .unwrap_or_default();
        let root_margin = best_slot.and_then(|i| margins.get(i).copied());

        Some(AnalysisResult {
            indices,
            evals,
            visits,
            wdl,
            margins,
            root_eval,
            root_wdl,
            root_margin,
        })
    }

    /// The line search explored under root move `idx`: move indices along the
    /// most-visited path, starting with `idx` itself, capped at `max_len`.
    /// Empty if `idx` is not a root child of this search.
    ///
    /// This is read off the tree the search already built — no extra
    /// inference — so a panel can show *why* a move is ranked where it is and
    /// not just that it was.
    pub fn principal_variation(&self, idx: u16, max_len: usize) -> Vec<u16> {
        self.inner
            .principal_variation(decode(idx), max_len)
            .into_iter()
            .map(encode)
            .collect()
    }

    /// Move index the search would play: the most-visited root child.
    /// `-1` if the root has no children.
    pub fn best_index(&self) -> i32 {
        match self.inner.result() {
            Some(r) => encode(r.best) as i32,
            None => -1,
        }
    }
}

/// Collapse one row of the value head to the scalar MCTS backs up. Mirrors
/// `ort_eval::collapse_value`: 3-way softmax then `P(win) - P(loss)`, with the
/// draw class contributing nothing. A width-1 head passes straight through.
fn collapse_value(row: &[f32]) -> Result<f32, JsError> {
    match row.len() {
        VALUE_CLASSES => {
            let p = softmax(row.to_vec());
            Ok(p[VALUE_WIN] - p[VALUE_LOSS])
        }
        1 => Ok(row[0]),
        other => Err(JsError::new(&format!(
            "value head has width {other}, expected 3 (win, draw, loss) or 1 (scalar)"
        ))),
    }
}

/// The value head as three probabilities, in the same POV `collapse_value`
/// returns its scalar in.
///
/// `collapsed` is passed in rather than recomputed so the two cannot drift: the
/// search backs up that exact number, and the distribution has to decompose
/// *it*. A width-1 head carries no draw information at all, so it is spread
/// across win and loss — which keeps `P(win) - P(loss) == collapsed` true, and
/// that identity is what the whole readout is checked against.
fn distribute_value(row: &[f32], collapsed: f32) -> [f32; 3] {
    if row.len() == VALUE_CLASSES {
        let p = softmax(row.to_vec());
        [p[VALUE_WIN], p[VALUE_DRAW], p[VALUE_LOSS]]
    } else {
        let v = collapsed.clamp(-1.0, 1.0);
        [(v + 1.0) / 2.0, 0.0, (1.0 - v) / 2.0]
    }
}

/// Expected final capture differential from the score head: a softmax over
/// `-6..=6`, reduced to its mean. Side-to-move POV, like the value head.
fn expected_margin(row: &[f32]) -> f32 {
    let p = softmax(row.to_vec());
    p.iter()
        .enumerate()
        .map(|(i, q)| q * (i as f32 - SCORE_OFFSET as f32))
        .sum()
}

fn softmax(mut logits: Vec<f32>) -> Vec<f32> {
    if logits.is_empty() {
        return logits;
    }
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for x in logits.iter_mut() {
        *x = (*x - max).exp();
        sum += *x;
    }
    if sum > 0.0 {
        for x in logits.iter_mut() {
            *x /= sum;
        }
    }
    logits
}

/// Result of a single MCTS analysis pass. Fields are exposed via getters
/// so JS can pull out parallel arrays in one allocation each.
#[wasm_bindgen]
pub struct AnalysisResult {
    indices: Vec<u16>,
    evals: Vec<f32>,
    visits: Vec<u32>,
    /// Win/draw/loss per move, White's POV, flattened three-per-move.
    wdl: Vec<f32>,
    /// Expected capture differential per move, White's POV. Empty if the
    /// model has no `score` head.
    margins: Vec<f32>,
    root_eval: f32,
    root_wdl: Vec<f32>,
    root_margin: Option<f32>,
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
    /// Searched win/draw/loss for each move, White's POV, flattened three per
    /// move in the same order as [`indices`](Self::indices). Empty if the
    /// search was not tracking outcome statistics.
    ///
    /// These come out of the tree, not off a forward pass: every leaf the
    /// search reached contributed its distribution, backed up and visit-
    /// weighted exactly as the scalar eval is. `wdl[0] - wdl[2]` reproduces
    /// [`evals`](Self::evals) — the identity the Rust tests assert.
    pub fn wdl(&self) -> Vec<f32> {
        self.wdl.clone()
    }

    /// Searched expected capture differential for each move, White's POV.
    /// Empty if the model has no `score` head.
    pub fn margins(&self) -> Vec<f32> {
        self.margins.clone()
    }

    /// Engine's evaluation of the current root position (white POV) — the
    /// Q-value of the most-visited child.
    pub fn root_eval(&self) -> f32 {
        self.root_eval
    }

    /// Win/draw/loss of the position, White's POV: the most-visited child's,
    /// so it describes the line the engine intends rather than an average over
    /// moves it has already rejected.
    pub fn root_wdl(&self) -> Vec<f32> {
        self.root_wdl.clone()
    }

    /// Expected capture differential of the position, White's POV. `None` if
    /// the model has no `score` head.
    pub fn root_margin(&self) -> Option<f32> {
        self.root_margin
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
