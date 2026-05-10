//! ONNX Runtime-backed leaf evaluator for MCTS.
//!
//! Holds an `ort::Session` loaded from a .onnx file produced by
//! `model/export_onnx.py`. Each `evaluate` call:
//!   1. Encodes the position to a (1, 6, 9, 9) plane tensor.
//!   2. Runs the session (returns `policy_logits[2562]` and `value`).
//!   3. Extracts the legal-move logits, applies softmax → priors.
//!   4. Returns `LeafEval { value, priors: Some(legal_priors) }`.
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

#[derive(Debug, thiserror::Error)]
pub enum OrtEvalError {
    #[error("ort session error: {0}")]
    Ort(#[from] ort::Error),
    #[error("ort tensor extraction error: {0}")]
    Shape(String),
}

pub struct OrtEvaluator {
    session: Session,
}

impl OrtEvaluator {
    pub fn from_onnx(path: impl AsRef<Path>) -> Result<Self, OrtEvalError> {
        // Optional CoreML execution provider, controlled by the
        // `ABALONE_USE_COREML` env var (set to "1" or "true"). Set by
        // `train_loop.py` from the YAML `use_coreml` knob when spawning
        // subprocesses; can also be flipped manually for ad-hoc runs.
        //
        // Benchmarks on the current 4×64 / 7M-param model on M1 Pro:
        //   - CPU 1-thread:        702 sims/sec
        //   - CPU 9-thread:      3,089 sims/sec  ← default; 4.4× scaling
        //   - CoreML 1-thread:   1,538 sims/sec  (2.2× single-thread vs CPU)
        //   - CoreML 9-thread:   1,894 sims/sec  (plateaus; ANE serializes)
        //
        // So CoreML loses for our small model + parallel-workers setup,
        // but should win once the model grows past ANE's call-overhead
        // crossover — keep the option here for that.
        //
        // `intra_threads(1)` is critical for CPU: ORT defaults to N
        // internal threads per session, and N workers × N internal ≈
        // N² fight for 10 cores → measured 1,263 sims/sec (worse than
        // 4 threads). Capping to 1 internal thread per session gives the
        // OS a clean N-thread workload.
        let use_coreml = std::env::var("ABALONE_USE_COREML")
            .map(|v| matches!(v.as_str(), "1" | "true" | "yes"))
            .unwrap_or(false);
        let mut builder = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?;
        if use_coreml {
            builder = builder
                .with_execution_providers([CoreMLExecutionProvider::default().build()])?;
        }
        let session = builder.commit_from_file(path.as_ref())?;
        Ok(Self { session })
    }

    pub fn evaluate(&mut self, game: &Game) -> Result<LeafEval, OrtEvalError> {
        let mut buf = vec![0f32; PLANE_SIZE];
        encode_planes(game, &mut buf);
        let input = Array4::from_shape_vec((1, NUM_INPUT_CHANNELS, BOARD_H, BOARD_W), buf)
            .map_err(|e| OrtEvalError::Shape(e.to_string()))?;

        let outputs = self.session.run(ort::inputs![
            "planes" => TensorRef::from_array_view(&input)?,
        ])?;

        let policy_view = outputs["policy_logits"]
            .try_extract_array::<f32>()?;
        let value_view = outputs["value"]
            .try_extract_array::<f32>()?;

        let policy_slice = policy_view
            .as_slice()
            .ok_or_else(|| OrtEvalError::Shape("policy not contiguous".into()))?;
        let value_slice = value_view
            .as_slice()
            .ok_or_else(|| OrtEvalError::Shape("value not contiguous".into()))?;
        let v = value_slice[0];

        // Extract the legal-move logits and softmax them into priors.
        let legal_moves = game.legal_moves();
        let mut legal_logits: Vec<f32> = Vec::with_capacity(legal_moves.len());
        for &m in legal_moves.iter() {
            legal_logits.push(policy_slice[encode(m) as usize]);
        }
        let priors = softmax_in_place(legal_logits);

        Ok(LeafEval {
            value: v,
            priors: Some(priors),
        })
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
