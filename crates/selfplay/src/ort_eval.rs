//! ONNX Runtime-backed leaf evaluator for MCTS.
//!
//! Holds an `ort::Session` loaded from a .onnx file produced by
//! `model/export_onnx.py`. The graph signature is
//! [ARCHITECTURE §5.3](../../../docs/ARCHITECTURE.md):
//!
//! ```text
//! input   planes         (B, 14, 9, 9)
//! output  policy_logits  (B, 2562)   unmasked, unnormalised
//!         value          (B, 3)      logits over (win, draw, loss)
//!         score          (B, 13)     logits over score_diff in [-6, +6]
//!         capture_map    (B, 2, 9, 9)
//! ```
//!
//! `score` and `capture_map` are **training-only auxiliaries**: they are read
//! from the shard, not from the network, so self-play never touches them. We
//! simply do not fetch them, which also means a graph exported without them
//! still works here.
//!
//! # The 3-way value head
//!
//! MCTS backs up a scalar. The collapse happens here and only here:
//!
//! ```text
//! p = softmax(value_logits)          // (win, draw, loss)
//! value = p[win] - p[loss]           // = E[outcome] in [-1, 1]
//! ```
//!
//! Search never sees the 3-way form (`abalone_mcts::LeafEval` is scalar), and a
//! draw contributes 0 exactly as a tanh head would. A scalar (B, 1) value output
//! is also accepted and passed through untouched, so debug exports keep working.
//!
//! # Batching
//!
//! [`OrtEvaluator::evaluate_batch`] encodes N positions into one
//! `(N, 14, 9, 9)` tensor and issues a **single** `session.run`. This is the
//! whole point of the batched MCTS coroutine: at 800 simulations and batch 32 a
//! move costs ~25 network calls instead of 800. `evaluate` is the N = 1 case and
//! exists for callers (eval-match, tests) that are not batching.
//!
//! # Threading model
//!
//! We construct one `OrtEvaluator` per worker thread (NOT one shared
//! evaluator across threads). `Session::run` requires `&mut self`; a
//! shared evaluator with `Mutex<Session>` was the obvious-but-wrong
//! design and serialized all inference through a single mutex,
//! capping throughput at ~1 effective core regardless of `--threads`.
//!
//! With per-thread sessions each worker runs inference in parallel.
//! Memory cost is ~30 MB × N threads, which is fine.
//!
//! We also force `intra_op_num_threads=1` so ORT's internal thread
//! pool doesn't over-subscribe — with N worker threads each spawning
//! its own pool we'd otherwise have N × cores fighting for time slices.

use std::path::Path;

use abalone_game::{encode, Game};
use abalone_mcts::LeafEval;
use ndarray::Array4;
use ort::execution_providers::CoreMLExecutionProvider;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::TensorRef;

use crate::encoder::{encode_planes, BOARD_H, BOARD_W, NUM_INPUT_CHANNELS, PLANE_SIZE};

/// Flat policy-logit width; must match `abalone_game::MOVE_SPACE`.
const POLICY_WIDTH: usize = abalone_game::MOVE_SPACE;

/// Class order of the 3-way value head, matching `model/batch.py`.
const VALUE_WIN: usize = 0;
const VALUE_LOSS: usize = 2;
/// Width of the 3-way value head.
const VALUE_CLASSES: usize = 3;

#[derive(Debug, thiserror::Error)]
pub enum OrtEvalError {
    #[error("ort session error: {0}")]
    Ort(#[from] ort::Error),
    #[error("ort tensor extraction error: {0}")]
    Shape(String),
}

/// Whether the CoreML execution provider is requested, via `ABALONE_USE_COREML`
/// (`1`/`true`/`yes`). Single source of truth: `train_loop.py` sets the env var
/// from the YAML `use_coreml` knob, and callers use this to decide whether
/// fixed-width batching is worth it (see
/// [`OrtEvaluator::set_fixed_batch`]).
pub fn use_coreml() -> bool {
    std::env::var("ABALONE_USE_COREML")
        .map(|v| matches!(v.as_str(), "1" | "true" | "yes"))
        .unwrap_or(false)
}

pub struct OrtEvaluator {
    session: Session,
    /// Reused input staging buffer so a batched evaluate does not allocate a
    /// fresh `N * 1134` float vector per call.
    planes: Vec<f32>,
    /// If set, every batch is zero-padded up to this width so the graph only
    /// ever sees one input shape.
    fixed_batch: Option<usize>,
}

impl OrtEvaluator {
    pub fn from_onnx(path: impl AsRef<Path>) -> Result<Self, OrtEvalError> {
        // Optional CoreML execution provider, controlled by the
        // `ABALONE_USE_COREML` env var. Set by `train_loop.py` from the YAML
        // `use_coreml` knob when spawning subprocesses; can also be flipped
        // manually for ad-hoc runs.
        //
        // Batched evaluation inverted the old ranking exactly as MODEL.md §7.1
        // predicted. One worker, 3.0M-param `base` preset, self-play at
        // sims 100/400 (positions/sec and network evals/sec):
        //
        //   provider  batch 1              batch 32
        //   CPU        0.9 pos/s,  130 e/s  0.9 pos/s,  144 e/s
        //   CoreML    14.1 pos/s, 2013 e/s  29.5 pos/s, 4539 e/s  ← with padding
        //   CoreML                          1.5 pos/s,  236 e/s   ← without
        //
        // On CPU with one intra-op thread the forward pass is compute-bound, so
        // batching buys almost nothing (~10%, from de-duplicated leaves) — the
        // win there is still parallel workers. On the ANE, batching is worth
        // 2× and, together with fixed-shape padding, 20×. See
        // [`OrtEvaluator::set_fixed_batch`] for why padding is not optional
        // there.
        //
        // `intra_threads(1)` is critical for CPU: ORT defaults to N
        // internal threads per session, and N workers × N internal ≈
        // N² fight for 10 cores → measured 1,263 sims/sec (worse than
        // 4 threads). Capping to 1 internal thread per session gives the
        // OS a clean N-thread workload.
        let mut builder = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?;
        if use_coreml() {
            builder =
                builder.with_execution_providers([CoreMLExecutionProvider::default().build()])?;
        }
        let session = builder.commit_from_file(path.as_ref())?;
        Ok(Self {
            session,
            planes: Vec::new(),
            fixed_batch: None,
        })
    }

    /// Zero-pad every batch out to `n` positions, so the graph only ever sees
    /// the input shape `(n, 14, 9, 9)`. Padding rows are evaluated and thrown
    /// away.
    ///
    /// **Only worth it on CoreML, where it is worth 20x.** ORT's CoreML
    /// provider compiles a shape-specialised model per input shape and runs on
    /// the CPU fallback until that finishes. A search emits a partial batch
    /// whenever the remaining simulations or the distinct leaves run short, so
    /// an unpadded self-play run cycles through every width from 1 to
    /// `batch_size` and spends its life recompiling. Measured on the `base`
    /// preset, batch 32, one worker: 236 evals/sec unpadded, 4539 padded.
    ///
    /// On the CPU provider the forward pass is compute-bound and linear in
    /// batch width, so padding is pure waste — leave it off.
    pub fn set_fixed_batch(&mut self, n: Option<usize>) {
        self.fixed_batch = n.filter(|&n| n > 0);
    }

    /// Evaluate one position. Equivalent to a one-element
    /// [`evaluate_batch`](Self::evaluate_batch).
    pub fn evaluate(&mut self, game: &Game) -> Result<LeafEval, OrtEvalError> {
        let mut out = self.evaluate_batch(std::slice::from_ref(game))?;
        Ok(out.pop().expect("one position in, one eval out"))
    }

    /// Evaluate `games` in a **single** session call.
    ///
    /// Returns one [`LeafEval`] per input position, in order. Each eval's
    /// priors are the softmax over that position's *legal* moves only, parallel
    /// to `game.legal_moves()` — the ordering `abalone_mcts` expects.
    pub fn evaluate_batch(&mut self, games: &[Game]) -> Result<Vec<LeafEval>, OrtEvalError> {
        if games.is_empty() {
            return Ok(Vec::new());
        }
        let n = games.len();
        // Rows `n..width` stay zeroed; their outputs are discarded.
        let width = self.fixed_batch.filter(|&w| w >= n).unwrap_or(n);
        self.planes.clear();
        self.planes.resize(width * PLANE_SIZE, 0.0);
        for (i, g) in games.iter().enumerate() {
            encode_planes(g, &mut self.planes[i * PLANE_SIZE..(i + 1) * PLANE_SIZE]);
        }
        let input = Array4::from_shape_vec(
            (width, NUM_INPUT_CHANNELS, BOARD_H, BOARD_W),
            std::mem::take(&mut self.planes),
        )
        .map_err(|e| OrtEvalError::Shape(e.to_string()))?;

        let outputs = self.session.run(ort::inputs![
            "planes" => TensorRef::from_array_view(&input)?,
        ])?;

        let policy_view = outputs
            .get("policy_logits")
            .ok_or_else(|| OrtEvalError::Shape("missing 'policy_logits' output".into()))?
            .try_extract_array::<f32>()?;
        let value_view = outputs
            .get("value")
            .ok_or_else(|| OrtEvalError::Shape("missing 'value' output".into()))?
            .try_extract_array::<f32>()?;

        let policy = policy_view
            .as_slice()
            .ok_or_else(|| OrtEvalError::Shape("policy not contiguous".into()))?;
        let value = value_view
            .as_slice()
            .ok_or_else(|| OrtEvalError::Shape("value not contiguous".into()))?;

        if policy.len() != width * POLICY_WIDTH {
            return Err(OrtEvalError::Shape(format!(
                "policy_logits has {} elements, expected {} ({}x{})",
                policy.len(),
                width * POLICY_WIDTH,
                width,
                POLICY_WIDTH
            )));
        }
        if value.len() % width != 0 {
            return Err(OrtEvalError::Shape(format!(
                "value has {} elements, not divisible by batch {}",
                value.len(),
                width
            )));
        }
        let value_dim = value.len() / width;

        let mut evals = Vec::with_capacity(n);
        for (i, g) in games.iter().enumerate() {
            let v = collapse_value(&value[i * value_dim..(i + 1) * value_dim])?;
            let row = &policy[i * POLICY_WIDTH..(i + 1) * POLICY_WIDTH];
            let legal_moves = g.legal_moves();
            let mut legal_logits: Vec<f32> = Vec::with_capacity(legal_moves.len());
            for &m in legal_moves.iter() {
                legal_logits.push(row[encode(m) as usize]);
            }
            evals.push(LeafEval {
                value: v,
                priors: Some(softmax_in_place(legal_logits)),
            });
        }

        // Reclaim the staging buffer for the next call.
        self.planes = input.into_raw_vec_and_offset().0;
        Ok(evals)
    }
}

/// Collapse one row of the value head to the scalar MCTS backs up.
///
/// 3-way head: `softmax` then `P(win) - P(loss)`; the draw class contributes
/// nothing, which is exactly right. A width-1 head is passed straight through.
fn collapse_value(row: &[f32]) -> Result<f32, OrtEvalError> {
    match row.len() {
        VALUE_CLASSES => {
            let p = softmax_in_place(row.to_vec());
            Ok(p[VALUE_WIN] - p[VALUE_LOSS])
        }
        1 => Ok(row[0]),
        other => Err(OrtEvalError::Shape(format!(
            "value head has width {other}, expected 3 (win, draw, loss) or 1 (scalar)"
        ))),
    }
}

fn softmax_in_place(mut logits: Vec<f32>) -> Vec<f32> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn value_collapse_is_win_minus_loss() {
        // Equal logits -> uniform -> 1/3 - 1/3 = 0.
        let v = collapse_value(&[0.0, 0.0, 0.0]).unwrap();
        assert!(v.abs() < 1e-6, "uniform 3-way must collapse to 0, got {v}");

        // Certain win.
        let v = collapse_value(&[30.0, 0.0, 0.0]).unwrap();
        assert!((v - 1.0).abs() < 1e-5, "certain win must collapse to +1, got {v}");

        // Certain loss.
        let v = collapse_value(&[0.0, 0.0, 30.0]).unwrap();
        assert!((v + 1.0).abs() < 1e-5, "certain loss must collapse to -1, got {v}");

        // Certain draw.
        let v = collapse_value(&[0.0, 30.0, 0.0]).unwrap();
        assert!(v.abs() < 1e-5, "certain draw must collapse to 0, got {v}");
    }

    #[test]
    fn value_collapse_is_antisymmetric_and_bounded() {
        for &(w, d, l) in &[(1.0f32, 0.5, -2.0), (-3.0, 0.0, 1.0), (0.2, 0.2, 0.2)] {
            let a = collapse_value(&[w, d, l]).unwrap();
            let b = collapse_value(&[l, d, w]).unwrap();
            assert!((a + b).abs() < 1e-6, "swapping win/loss must negate: {a} vs {b}");
            assert!((-1.0..=1.0).contains(&a));
        }
    }

    #[test]
    fn scalar_value_head_passes_through() {
        assert_eq!(collapse_value(&[0.42]).unwrap(), 0.42);
    }

    #[test]
    fn wrong_width_value_head_is_an_error() {
        assert!(collapse_value(&[0.0, 1.0]).is_err());
        assert!(collapse_value(&[]).is_err());
    }

    /// Raw batched-inference throughput, isolated from search.
    ///
    /// ```text
    /// ABALONE_BENCH_MODEL=path.onnx cargo test --release -p abalone-selfplay \
    ///     -- --ignored --nocapture batched_inference
    /// ```
    ///
    /// `ABALONE_USE_COREML=1` switches execution provider. This is the
    /// measurement that says whether batching is worth anything on a given
    /// backend: sessions are built with `intra_threads(1)` (one worker per
    /// core), so on CPU a batch of N costs ~N times a batch of 1 and only the
    /// per-call overhead is amortised. Accelerators, where per-call overhead
    /// dominates, are the case batching is really for.
    #[test]
    #[ignore = "measurement; run explicitly with --ignored --nocapture"]
    fn measure_batched_inference() {
        use abalone_game::Opening;
        use std::time::Instant;

        let Ok(path) = std::env::var("ABALONE_BENCH_MODEL") else {
            println!("\nABALONE_BENCH_MODEL unset; skipping inference benchmark\n");
            return;
        };
        let mut e = OrtEvaluator::from_onnx(&path).expect("load ABALONE_BENCH_MODEL");

        // A spread of real positions rather than 64 copies of one board.
        let mut positions = Vec::new();
        let mut g = Game::new(Opening::BelgianDaisy, 200, u32::MAX);
        while positions.len() < 64 && !g.is_terminal() {
            positions.push(g);
            let mv = g.legal_moves()[positions.len() % g.legal_moves().len()];
            g.apply(mv);
        }

        println!(
            "\nbatched inference ({}, coreml={})",
            path,
            std::env::var("ABALONE_USE_COREML").unwrap_or_else(|_| "0".into())
        );
        for &bs in &[1usize, 8, 32, 64] {
            let batch = &positions[..bs.min(positions.len())];
            // Warm up properly. CoreML compiles a shape-specialised model on
            // first use and falls back to CPU until it is ready, so a two-call
            // warmup reports the fallback path and understates the ANE by ~10x.
            let warm = Instant::now();
            while warm.elapsed().as_secs_f64() < 1.0 {
                e.evaluate_batch(batch).expect("evaluate");
            }
            let target_positions = 2048;
            let iters = (target_positions / batch.len()).max(4);
            let t = Instant::now();
            for _ in 0..iters {
                e.evaluate_batch(batch).expect("evaluate");
            }
            let secs = t.elapsed().as_secs_f64();
            let evals = (iters * batch.len()) as f64;
            println!(
                "  batch {bs:>3}: {:>8.1} positions/sec  {:>7.1} calls/sec  {:>6.2} ms/call",
                evals / secs,
                iters as f64 / secs,
                1000.0 * secs / iters as f64,
            );
        }
        println!();
    }

    #[test]
    fn softmax_normalises() {
        let p = softmax_in_place(vec![1.0, 2.0, 3.0]);
        let sum: f32 = p.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(p[2] > p[1] && p[1] > p[0]);
    }
}
