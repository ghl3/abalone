//! ONNX Runtime-backed leaf evaluator for MCTS.
//!
//! Holds an `ort::Session` loaded from a .onnx file produced by
//! `model/export_onnx.py`. Each `evaluate` call:
//!   1. Encodes the position to a (1, 6, 9, 9) plane tensor.
//!   2. Runs the session (returns `policy_logits[2562]` and `value`).
//!   3. Extracts the legal-move logits, applies softmax → priors.
//!   4. Returns `LeafEval { value, priors: Some(legal_priors) }`.
//!
//! Sessions are thread-safe in `ort` 2.x; multiple workers can share an
//! `Arc<OrtEvaluator>` and call `evaluate` concurrently.

use std::path::Path;
use std::sync::{Arc, Mutex};

use abalone_game::{encode, Game};
use abalone_mcts::LeafEval;
use ndarray::Array4;
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
    // ort 2.x's Session::run requires &mut self, so we serialize
    // inference behind a Mutex. With ~2ms inference per call and 8
    // workers, contention is bounded; if it ever becomes a
    // bottleneck we'd switch to one Session per worker thread.
    session: Mutex<Session>,
}

impl OrtEvaluator {
    pub fn from_onnx(path: impl AsRef<Path>) -> Result<Arc<Self>, OrtEvalError> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_file(path.as_ref())?;
        Ok(Arc::new(Self {
            session: Mutex::new(session),
        }))
    }

    pub fn evaluate(&self, game: &Game) -> Result<LeafEval, OrtEvalError> {
        let mut buf = vec![0f32; PLANE_SIZE];
        encode_planes(game, &mut buf);
        let input = Array4::from_shape_vec((1, NUM_INPUT_CHANNELS, BOARD_H, BOARD_W), buf)
            .map_err(|e| OrtEvalError::Shape(e.to_string()))?;

        let mut session = self.session.lock().expect("OrtEvaluator session poisoned");
        let outputs = session.run(ort::inputs![
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
