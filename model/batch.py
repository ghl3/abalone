"""The training batch — the contract between `replay_buffer` and `train_step`.

This lives in its own module because two independent consumers depend on its
exact shape: `replay_buffer.py` produces it and `train_step.py` / `validate.py`
consume it. Defining it here keeps neither of them owning the other's schema.

Canonical spec: `docs/ARCHITECTURE.md` section 5.5.

Field notes that are easy to get wrong:

* `value` and `score` are INTEGER CLASS INDICES, not scalars. The network's value
  head is 3-way and its score head is 13-way, so both train under cross-entropy.
      value: 0 = win, 1 = draw, 2 = loss   (from this position's to-move POV)
      score: score_diff + 6, i.e. [-6, +6] mapped to [0, 12]

* `policy_weight` implements playout cap randomisation. Positions searched at the
  *fast* simulation count still carry usable value, score and capture-map
  targets, but their visit distribution is too noisy to be a policy target. Those
  rows get weight 0.0 and must not contribute to the policy loss.

* `q` is carried for diagnostics and the review UI only. It is deliberately NOT a
  loss term: the value head trains on the game outcome alone. See
  `docs/ARCHITECTURE.md` section 5.5 for why the old z/q blend was dropped.

* `capture_map` targets are in [0, 1]. Channel 0 = own losses, channel 1 =
  opponent losses. Only the channel is POV-relative; cells are absolute board
  indices, matching the own/opp convention of the input planes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Shape constants, duplicated deliberately so a consumer can validate without
# importing the network or the encoder.
INPUT_PLANES = 14
BOARD_H = 9
BOARD_W = 9
MOVE_SPACE = 2562
VALUE_CLASSES = 3   # win, draw, loss
SCORE_CLASSES = 13  # score_diff in [-6, +6]
CAPTURE_MAP_CHANNELS = 2

# Class indices for the value head.
VALUE_WIN = 0
VALUE_DRAW = 1
VALUE_LOSS = 2

#: Offset applied to `score_diff` to get the score-head class index.
SCORE_OFFSET = 6


def value_class(z: int) -> int:
    """Map a terminal outcome in {+1, 0, -1} to a value-head class index."""
    if z > 0:
        return VALUE_WIN
    if z < 0:
        return VALUE_LOSS
    return VALUE_DRAW


def score_class(score_diff: int) -> int:
    """Map a capture differential in [-6, +6] to a score-head class index."""
    return int(np.clip(score_diff, -6, 6)) + SCORE_OFFSET


@dataclass
class Batch:
    """One dense training batch. See module docstring for field semantics."""

    planes: np.ndarray        # (B, 14, 9, 9) float32
    policy: np.ndarray        # (B, 2562)     float32, normalised over legal moves
    legal_mask: np.ndarray    # (B, 2562)     float32, 1.0 on legal moves
    policy_weight: np.ndarray # (B,)          float32, 1.0 iff a usable policy target
    value: np.ndarray         # (B,)          int64 class index
    score: np.ndarray         # (B,)          int64 class index
    capture_map: np.ndarray   # (B, 2, 9, 9)  float32 in [0, 1]
    q: np.ndarray             # (B,)          float32, diagnostics only

    @property
    def size(self) -> int:
        return int(self.planes.shape[0])

    def validate(self) -> None:
        """Assert every field's shape and dtype. Cheap; call it in tests and at
        the head of a training run rather than debugging a silent broadcast."""
        b = self.size
        expect = {
            "planes": ((b, INPUT_PLANES, BOARD_H, BOARD_W), np.float32),
            "policy": ((b, MOVE_SPACE), np.float32),
            "legal_mask": ((b, MOVE_SPACE), np.float32),
            "policy_weight": ((b,), np.float32),
            "value": ((b,), np.int64),
            "score": ((b,), np.int64),
            "capture_map": ((b, CAPTURE_MAP_CHANNELS, BOARD_H, BOARD_W), np.float32),
            "q": ((b,), np.float32),
        }
        for name, (shape, dtype) in expect.items():
            arr = getattr(self, name)
            if arr.shape != shape:
                raise ValueError(f"Batch.{name}: expected shape {shape}, got {arr.shape}")
            if arr.dtype != dtype:
                raise ValueError(f"Batch.{name}: expected dtype {dtype}, got {arr.dtype}")
        if b and (self.value.min() < 0 or self.value.max() >= VALUE_CLASSES):
            raise ValueError(f"Batch.value out of range [0, {VALUE_CLASSES})")
        if b and (self.score.min() < 0 or self.score.max() >= SCORE_CLASSES):
            raise ValueError(f"Batch.score out of range [0, {SCORE_CLASSES})")
