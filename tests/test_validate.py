"""Tests for `model.validate` — the held-out feedback loop.

Each test uses synthetic data with a *known* answer rather than a real network,
because the point of this module is that its numbers can be trusted when a run
looks wrong. In particular:

  * **`policy_target_entropy` on a uniform target must come out at exactly
    `ln(n)`.** That is the precise pathology of the failed run — MCTS returning
    a flat visit distribution, policy loss pinned at `ln(branching)` — and this
    test is what stops the metric from quietly under-reporting it.
  * **Calibration on perfectly-calibrated synthetic data must give ~0 ECE**, and
    must give a large one on a deliberately overconfident model.
  * **AUC without sklearn**: perfect separation gives 1.0, constant predictions
    (an all-ties input) give exactly 0.5.
  * **No side effects**: the model's train/eval mode survives, and no gradients
    are accumulated.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch
import torch.nn as nn

from model.abalone_net import build
from model.batch import (
    BOARD_H,
    BOARD_W,
    CAPTURE_MAP_CHANNELS,
    INPUT_PLANES,
    MOVE_SPACE,
    SCORE_CLASSES,
    SCORE_OFFSET,
    VALUE_CLASSES,
    VALUE_DRAW,
    VALUE_LOSS,
    VALUE_WIN,
    Batch,
)
from model.encoder import PLY_PLANE, VALID_CELL_MASK
from model.train_step import batch_to_tensors, capture_map_loss
from model.validate import (
    DATASET_KEY_PREFIX,
    DATASET_METRIC_KEYS,
    METRIC_KEYS,
    NON_SCALAR_KEYS,
    TOP_K,
    VAL_FROZEN_PREFIX,
    VAL_ROLLING_PREFIX,
    _auc,
    scalar_metrics,
    validate,
)
from tests.test_train_step import make_batch

CPU = torch.device("cpu")


# ----- a network whose outputs the test chooses ------------------------------


class ScriptedNet(nn.Module):
    """Returns pre-set logits, consuming them in order across calls.

    A real net's outputs are unknown, which is exactly wrong for testing a
    metric. This lets each test state the model's answer and the expected
    metric in the same breath.
    """

    def __init__(
        self,
        policy: torch.Tensor,
        value: torch.Tensor,
        score: torch.Tensor,
        capture: torch.Tensor,
    ) -> None:
        super().__init__()
        self.policy, self.value, self.score, self.capture = policy, value, score, capture
        self.cursor = 0

    def forward(self, x: torch.Tensor):
        n = x.shape[0]
        if self.cursor >= self.policy.shape[0]:
            self.cursor = 0
        s = slice(self.cursor, self.cursor + n)
        self.cursor += n
        return self.policy[s], self.value[s], self.score[s], self.capture[s]


def blank_batch(size: int, *, seed: int = 0) -> Batch:
    """A minimal contract-valid batch: zeros everywhere, no policy target.
    Tests then overwrite only the fields the metric under test reads."""
    rng = np.random.default_rng(seed)
    return Batch(
        planes=np.zeros((size, INPUT_PLANES, BOARD_H, BOARD_W), dtype=np.float32),
        policy=np.zeros((size, MOVE_SPACE), dtype=np.float32),
        legal_mask=np.zeros((size, MOVE_SPACE), dtype=np.float32),
        policy_weight=np.zeros(size, dtype=np.float32),
        value=np.zeros(size, dtype=np.int64),
        score=np.full(size, SCORE_OFFSET, dtype=np.int64),
        capture_map=np.zeros((size, CAPTURE_MAP_CHANNELS, BOARD_H, BOARD_W), dtype=np.float32),
        q=rng.random(size, dtype=np.float32),
    )


def zeros_like_outputs(size: int) -> tuple[torch.Tensor, ...]:
    return (
        torch.zeros(size, MOVE_SPACE),
        torch.zeros(size, VALUE_CLASSES),
        torch.zeros(size, SCORE_CLASSES),
        torch.zeros(size, CAPTURE_MAP_CHANNELS, BOARD_H, BOARD_W),
    )


# ----- policy ----------------------------------------------------------------


def test_top1_agreement_is_one_for_a_model_rigged_to_agree() -> None:
    batch = make_batch(size=6, seed=1, legal_per_row=9)
    p, v, s, c = zeros_like_outputs(batch.size)
    # Rig the policy head to reproduce the target distribution exactly.
    p = torch.from_numpy(batch.policy).clone()
    net = ScriptedNet(p, v, s, c)

    out = validate(net, [batch], CPU)
    assert out["policy_top1_agreement"] == 1.0
    assert out["policy_rows"] == 6


def test_top1_agreement_is_zero_when_the_model_always_picks_another_legal_move() -> None:
    batch = make_batch(size=4, seed=2, legal_per_row=5)
    p, v, s, c = zeros_like_outputs(batch.size)
    for i in range(batch.size):
        legal = np.flatnonzero(batch.legal_mask[i])
        best = int(batch.policy[i].argmax())
        other = int(next(j for j in legal if j != best))
        p[i, other] = 5.0
    out = validate(ScriptedNet(p, v, s, c), [batch], CPU)
    assert out["policy_top1_agreement"] == 0.0


def test_top1_agreement_ignores_illegal_moves_the_model_prefers() -> None:
    """The argmax is over legal moves only. A model whose global argmax is an
    illegal move still agrees if its best *legal* move is the MCTS choice."""
    batch = make_batch(size=3, seed=3, legal_per_row=4)
    p, v, s, c = zeros_like_outputs(batch.size)
    p = torch.from_numpy(batch.policy).clone()
    illegal = np.flatnonzero(batch.legal_mask[0] == 0)[:1]
    p[:, illegal] = 99.0
    out = validate(ScriptedNet(p, v, s, c), [batch], CPU)
    assert out["policy_top1_agreement"] == 1.0


def test_uniform_target_entropy_is_log_n() -> None:
    """The exact pathology of the failed run: a flat visit distribution over
    `n` legal moves has entropy `ln(n)`, and `policy_entropy_ratio` is 1.0.
    If this ever reads low, the metric is lying and the collapse is invisible.
    """
    n = 37
    size = 5
    batch = blank_batch(size, seed=4)
    rng = np.random.default_rng(0)
    for i in range(size):
        idx = rng.choice(MOVE_SPACE, size=n, replace=False)
        batch.legal_mask[i, idx] = 1.0
        batch.policy[i, idx] = 1.0 / n
    batch.policy_weight[:] = 1.0
    batch.validate()

    out = validate(ScriptedNet(*zeros_like_outputs(size)), [batch], CPU)
    assert out["data_policy_target_entropy"] == pytest.approx(math.log(n), abs=1e-5)
    assert out["data_mean_legal_moves"] == pytest.approx(float(n))
    assert out["data_policy_uniform_entropy"] == pytest.approx(math.log(n), abs=1e-9)
    assert out["data_policy_entropy_ratio"] == pytest.approx(1.0, abs=1e-5)
    # A uniform model against a uniform target: CE == the target entropy.
    assert out["policy_ce"] == pytest.approx(math.log(n), abs=1e-5)


def test_peaked_target_entropy_is_far_below_log_n() -> None:
    """The healthy case, for contrast: a target concentrated on two moves has
    entropy near `ln(2)` regardless of how many moves are legal."""
    n = 40
    batch = blank_batch(2, seed=5)
    rng = np.random.default_rng(1)
    for i in range(2):
        idx = rng.choice(MOVE_SPACE, size=n, replace=False)
        batch.legal_mask[i, idx] = 1.0
        batch.policy[i, idx[:2]] = 0.5
    batch.policy_weight[:] = 1.0

    out = validate(ScriptedNet(*zeros_like_outputs(2)), [batch], CPU)
    assert out["data_policy_target_entropy"] == pytest.approx(math.log(2.0), abs=1e-5)
    assert out["data_policy_entropy_ratio"] < 0.2


def test_policy_metrics_use_only_weighted_rows() -> None:
    """Zero-weight rows carry a garbage visit distribution by construction;
    including them in top-1 agreement would understate the model."""
    batch = make_batch(size=4, seed=6, legal_per_row=6, policy_weight=np.array([1, 0, 1, 0]))
    p, v, s, c = zeros_like_outputs(batch.size)
    p = torch.from_numpy(batch.policy).clone()
    # Break the two zero-weight rows completely.
    for i in (1, 3):
        p[i] = torch.zeros(MOVE_SPACE)
        p[i, int(np.flatnonzero(batch.legal_mask[i])[-1])] = 9.0
    out = validate(ScriptedNet(p, v, s, c), [batch], CPU)
    assert out["policy_top1_agreement"] == 1.0
    assert out["policy_rows"] == 2
    assert out["data_policy_target_rate"] == pytest.approx(0.5)


def test_no_weighted_rows_gives_nan_not_zero() -> None:
    batch = blank_batch(3, seed=7)
    out = validate(ScriptedNet(*zeros_like_outputs(3)), [batch], CPU)
    assert math.isnan(out["policy_top1_agreement"])
    assert math.isnan(out["policy_ce"])
    assert out["policy_rows"] == 0


def test_top5_agreement_is_looser_than_top1() -> None:
    """Top-1 against a ~1/60 chance level is nearly all noise. Top-5 must
    credit a model that ranks the MCTS choice second."""
    n = 20
    size = 4
    batch = blank_batch(size, seed=40)
    rng = np.random.default_rng(11)
    p, v, s, c = zeros_like_outputs(size)
    for i in range(size):
        idx = rng.choice(MOVE_SPACE, size=n, replace=False)
        batch.legal_mask[i, idx] = 1.0
        batch.policy[i, idx[0]] = 1.0  # MCTS picks idx[0]
        # The model ranks idx[1] first and the MCTS choice second.
        p[i, idx[1]] = 9.0
        p[i, idx[0]] = 8.0
    batch.policy_weight[:] = 1.0

    out = validate(ScriptedNet(p, v, s, c), [batch], CPU)
    assert out["policy_top1_agreement"] == 0.0
    assert out["policy_top5_agreement"] == 1.0


def test_top5_agreement_is_zero_when_the_choice_is_ranked_sixth() -> None:
    n = 20
    batch = blank_batch(1, seed=41)
    p, v, s, c = zeros_like_outputs(1)
    idx = np.random.default_rng(12).choice(MOVE_SPACE, size=n, replace=False)
    batch.legal_mask[0, idx] = 1.0
    batch.policy[0, idx[0]] = 1.0
    # Five legal moves score strictly above the MCTS choice: rank 5, so out.
    for k in range(1, TOP_K + 1):
        p[0, idx[k]] = 10.0 + k
    p[0, idx[0]] = 1.0
    batch.policy_weight[:] = 1.0

    out = validate(ScriptedNet(p, v, s, c), [batch], CPU)
    assert out["policy_top5_agreement"] == 0.0


def test_top5_agreement_is_not_inflated_by_illegal_moves() -> None:
    """A row with three legal moves has only three candidates. `topk` over the
    2562-wide head would pad the other two slots with illegal moves and hand
    out agreement for free."""
    batch = blank_batch(1, seed=42)
    p, v, s, c = zeros_like_outputs(1)
    idx = np.array([100, 200, 300])
    batch.legal_mask[0, idx] = 1.0
    batch.policy[0, idx[2]] = 1.0  # MCTS picks the model's *worst* legal move
    p[0, idx[0]] = 3.0
    p[0, idx[1]] = 2.0
    p[0, idx[2]] = 1.0
    batch.policy_weight[:] = 1.0

    out = validate(ScriptedNet(p, v, s, c), [batch], CPU)
    # Two legal moves beat it, so rank 2 < 5: still agreement, correctly.
    assert out["policy_top5_agreement"] == 1.0
    assert out["policy_top1_agreement"] == 0.0


# ----- KL(visits || prior) ---------------------------------------------------


def _kl_batch(size: int, n_legal: int, *, seed: int) -> tuple[Batch, np.ndarray]:
    """A batch whose every legal move carries visits, plus the legal indices."""
    batch = blank_batch(size, seed=seed)
    idx = np.random.default_rng(seed).choice(MOVE_SPACE, size=n_legal, replace=False)
    for i in range(size):
        batch.legal_mask[i, idx] = 1.0
        batch.policy[i, idx] = 1.0 / n_legal
    batch.policy_weight[:] = 1.0
    return batch, idx


def test_kl_visits_from_prior_is_zero_when_the_prior_equals_the_visits() -> None:
    """The quantity the policy head is trying to drive to zero. If it does not
    read 0 on an exact match, every reading of it is wrong by an offset."""
    batch, idx = _kl_batch(3, 25, seed=43)
    p, v, s, c = zeros_like_outputs(3)
    # Uniform logits over the legal moves reproduce the uniform target exactly.
    out = validate(ScriptedNet(p, v, s, c), [batch], CPU)
    assert out["policy_kl_visits_from_prior"] == pytest.approx(0.0, abs=1e-6)
    # ...and the identity KL = CE − H holds.
    assert out["policy_ce"] == pytest.approx(out["data_policy_target_entropy"], abs=1e-5)
    assert idx.size == 25


def test_kl_visits_from_prior_is_positive_and_hand_checkable() -> None:
    """Peaked visits against a uniform prior: KL = ln(n) − H(visits)."""
    n = 30
    batch = blank_batch(1, seed=44)
    idx = np.random.default_rng(14).choice(MOVE_SPACE, size=n, replace=False)
    batch.legal_mask[0, idx] = 1.0
    batch.policy[0, idx[:2]] = 0.5  # H = ln 2
    batch.policy_weight[:] = 1.0

    out = validate(ScriptedNet(*zeros_like_outputs(1)), [batch], CPU)
    assert out["policy_kl_visits_from_prior"] == pytest.approx(
        math.log(n) - math.log(2.0), abs=1e-5
    )
    assert out["policy_kl_visits_from_prior"] > 0.0


def test_kl_visits_from_prior_uses_only_full_search_rows() -> None:
    """A `policy_weight == 0` row's visit distribution is noise from the fast
    search. Averaging its KL in would make the metric track the playout-cap
    rate rather than the network."""
    n = 16
    batch = blank_batch(2, seed=45)
    idx = np.random.default_rng(15).choice(MOVE_SPACE, size=n, replace=False)
    for i in range(2):
        batch.legal_mask[i, idx] = 1.0
    batch.policy[0, idx] = 1.0 / n  # full search: matches a uniform prior
    batch.policy[1, idx[0]] = 1.0  # fast search: maximally far from uniform
    batch.policy_weight[:] = np.array([1.0, 0.0], dtype=np.float32)

    out = validate(ScriptedNet(*zeros_like_outputs(2)), [batch], CPU)
    assert out["policy_rows"] == 1
    assert out["policy_kl_visits_from_prior"] == pytest.approx(0.0, abs=1e-6)


def test_kl_visits_from_prior_is_nan_without_any_policy_target() -> None:
    out = validate(ScriptedNet(*zeros_like_outputs(3)), [blank_batch(3, seed=46)], CPU)
    assert math.isnan(out["policy_kl_visits_from_prior"])


# ----- value -----------------------------------------------------------------


def test_value_ce_and_accuracy_are_hand_checkable() -> None:
    batch = blank_batch(4, seed=8)
    batch.value[:] = np.array([VALUE_WIN, VALUE_DRAW, VALUE_LOSS, VALUE_WIN])
    p, v, s, c = zeros_like_outputs(4)
    v[0, VALUE_WIN] = 10.0  # confident and right
    v[1, VALUE_WIN] = 10.0  # confident and wrong (target is draw)
    # rows 2, 3 stay uniform -> CE = ln 3, argmax = class 0 -> wrong, right
    out = validate(ScriptedNet(p, v, s, c), [batch], CPU)

    hi = math.exp(10.0)
    ce_confident_right = math.log((hi + 2.0) / hi)
    ce_confident_wrong = math.log(hi + 2.0)
    expected = (ce_confident_right + ce_confident_wrong + 2 * math.log(3.0)) / 4.0
    assert out["value_ce"] == pytest.approx(expected, abs=1e-5)
    assert out["value_accuracy"] == pytest.approx(0.5)


def test_value_brier_is_hand_workable() -> None:
    """Multiclass Brier: `Σ_c (p_c − y_c)²`.

    Row 0 is a uniform head on a WIN: `(1/3−1)² + (1/3)² + (1/3)² = 2/3`.
    Row 1 is confidently right: ≈ 0. Mean = 1/3.
    """
    batch = blank_batch(2, seed=47)
    batch.value[:] = np.array([VALUE_WIN, VALUE_LOSS])
    p, v, s, c = zeros_like_outputs(2)
    v[1, VALUE_LOSS] = 40.0
    out = validate(ScriptedNet(p, v, s, c), [batch], CPU)
    assert out["value_brier"] == pytest.approx((2.0 / 3.0 + 0.0) / 2.0, abs=1e-5)


def test_value_brier_is_bounded_where_cross_entropy_is_not() -> None:
    """A confidently wrong row: Brier saturates at 2, CE runs away. That bound
    is the whole point — it is why the pair separates 'generally
    miscalibrated' from 'a few confident errors'."""
    batch = blank_batch(1, seed=48)
    batch.value[:] = VALUE_WIN
    p, v, s, c = zeros_like_outputs(1)
    v[0, VALUE_LOSS] = 60.0
    out = validate(ScriptedNet(p, v, s, c), [batch], CPU)
    assert out["value_brier"] == pytest.approx(2.0, abs=1e-6)
    assert out["value_ce"] > 50.0


def test_value_brier_is_zero_for_a_perfect_confident_model() -> None:
    batch = blank_batch(3, seed=49)
    batch.value[:] = np.array([VALUE_WIN, VALUE_DRAW, VALUE_LOSS])
    p, v, s, c = zeros_like_outputs(3)
    for i, cls in enumerate((VALUE_WIN, VALUE_DRAW, VALUE_LOSS)):
        v[i, cls] = 60.0
    out = validate(ScriptedNet(p, v, s, c), [batch], CPU)
    assert out["value_brier"] == pytest.approx(0.0, abs=1e-9)


def _calibration_batch(p_win: np.ndarray, wins: np.ndarray) -> tuple[Batch, torch.Tensor]:
    """A batch whose labels realise `wins`, with value logits that decode back to
    exactly `p_win` under softmax (the draw logit is pushed to ~0 probability)."""
    n = p_win.size
    batch = blank_batch(n, seed=9)
    batch.value[:] = np.where(wins, VALUE_WIN, VALUE_LOSS).astype(np.int64)
    logits = torch.zeros(n, VALUE_CLASSES)
    logits[:, VALUE_WIN] = torch.from_numpy(np.log(np.clip(p_win, 1e-9, None))).float()
    logits[:, VALUE_DRAW] = -60.0
    logits[:, VALUE_LOSS] = torch.from_numpy(np.log(np.clip(1.0 - p_win, 1e-9, None))).float()
    return batch, logits


def test_calibration_is_near_zero_on_perfectly_calibrated_data() -> None:
    rng = np.random.default_rng(0)
    n = 2500
    p_win = rng.random(n)
    wins = rng.random(n) < p_win  # win frequency equals the stated probability
    batch, v = _calibration_batch(p_win, wins)
    p, _, s, c = zeros_like_outputs(n)

    out = validate(ScriptedNet(p, v, s, c), [batch], CPU)
    assert out["value_ece"] < 0.05

    curve = out["value_calibration"]
    assert len(curve) == 10
    assert sum(b["count"] for b in curve) == n
    for b in curve:
        if b["count"] > 100:
            assert b["observed_win_rate"] == pytest.approx(b["mean_predicted"], abs=0.12)


def test_calibration_catches_an_overconfident_model() -> None:
    """Same labels, but the model always claims 99% — ECE must be large. Without
    this the previous test would pass on a metric hard-wired to 0."""
    rng = np.random.default_rng(0)
    n = 2000
    truth = rng.random(n)
    wins = rng.random(n) < truth
    batch, _ = _calibration_batch(truth, wins)
    _, v = _calibration_batch(np.full(n, 0.99), wins)
    p, _, s, c = zeros_like_outputs(n)

    out = validate(ScriptedNet(p, v, s, c), [batch], CPU)
    assert out["value_ece"] > 0.3


def test_empty_calibration_bins_report_nan_not_zero() -> None:
    batch, v = _calibration_batch(np.full(20, 0.5), np.ones(20, dtype=bool))
    p, _, s, c = zeros_like_outputs(20)
    out = validate(ScriptedNet(p, v, s, c), [batch], CPU)
    filled = [b for b in out["value_calibration"] if b["count"]]
    assert len(filled) == 1
    for b in out["value_calibration"]:
        if not b["count"]:
            assert math.isnan(b["mean_predicted"])
            assert math.isnan(b["observed_win_rate"])


# ----- score -----------------------------------------------------------------


def test_score_mae_is_in_marbles() -> None:
    """MAE between the head's expected differential and the true one. Row 0 is
    exactly right, row 1 is confidently two marbles off -> MAE = 1.0."""
    batch = blank_batch(2, seed=10)
    batch.score[:] = np.array([SCORE_OFFSET, SCORE_OFFSET], dtype=np.int64)  # both d = 0
    p, v, s, c = zeros_like_outputs(2)
    s[0, SCORE_OFFSET] = 40.0  # d = 0
    s[1, SCORE_OFFSET + 2] = 40.0  # d = +2
    out = validate(ScriptedNet(p, v, s, c), [batch], CPU)
    assert out["score_mae"] == pytest.approx(1.0, abs=1e-4)
    assert out["score_accuracy"] == pytest.approx(0.5)
    assert out["score_ce"] == pytest.approx(40.0 / 2.0, abs=0.5)


def test_score_ce_is_log_13_for_a_uniform_head() -> None:
    batch = blank_batch(3, seed=11)
    out = validate(ScriptedNet(*zeros_like_outputs(3)), [batch], CPU)
    assert out["score_ce"] == pytest.approx(math.log(SCORE_CLASSES), abs=1e-6)
    assert out["score_mae"] == pytest.approx(0.0, abs=1e-6)  # symmetric -> E[d] = 0


# ----- capture map -----------------------------------------------------------


def _capture_batch(size: int, target: np.ndarray) -> Batch:
    batch = blank_batch(size, seed=12)
    batch.capture_map[:] = target
    return batch


def test_capture_map_auc_is_one_for_perfect_separation() -> None:
    rng = np.random.default_rng(2)
    target = (rng.random((3, CAPTURE_MAP_CHANNELS, BOARD_H, BOARD_W)) < 0.3).astype(np.float32)
    batch = _capture_batch(3, target)
    p, v, s, _ = zeros_like_outputs(3)
    c = torch.from_numpy(np.where(target > 0.5, 4.0, -4.0).astype(np.float32))

    out = validate(ScriptedNet(p, v, s, c), [batch], CPU)
    assert out["capture_map_auc"] == pytest.approx(1.0)
    assert out["capture_map_separation"] > 0.9
    assert out["data_capture_map_positive_rate"] > 0.0


def test_capture_map_auc_is_one_half_for_constant_predictions() -> None:
    """Every score tied. The rank statistic must average tied ranks, or this
    silently reports 0.0 or 1.0 depending on sort order."""
    rng = np.random.default_rng(3)
    target = (rng.random((2, CAPTURE_MAP_CHANNELS, BOARD_H, BOARD_W)) < 0.5).astype(np.float32)
    batch = _capture_batch(2, target)
    out = validate(ScriptedNet(*zeros_like_outputs(2)), [batch], CPU)
    assert out["capture_map_auc"] == pytest.approx(0.5)
    assert out["capture_map_separation"] == pytest.approx(0.0, abs=1e-7)


def test_capture_map_auc_is_nan_when_one_class_is_absent() -> None:
    batch = _capture_batch(2, np.zeros((2, CAPTURE_MAP_CHANNELS, BOARD_H, BOARD_W), np.float32))
    out = validate(ScriptedNet(*zeros_like_outputs(2)), [batch], CPU)
    assert math.isnan(out["capture_map_auc"])


def test_auc_matches_a_brute_force_pair_count() -> None:
    """Mann-Whitney identity: AUC is the fraction of (positive, negative) pairs
    ranked correctly, ties counting a half."""
    rng = np.random.default_rng(4)
    scores = np.round(rng.random(60), 2)  # rounding forces ties
    labels = (rng.random(60) < 0.4).astype(np.float64)
    pos, neg = scores[labels > 0], scores[labels == 0]
    brute = float(np.mean((pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])))
    assert _auc(scores, labels) == pytest.approx(brute, abs=1e-9)


def test_capture_map_bce_matches_the_training_loss() -> None:
    """`validate` reduces per row for exactness across batches; that must agree
    with `train_step.capture_map_loss` over the same data."""
    batch = make_batch(size=5, seed=13)
    p, v, s, _ = zeros_like_outputs(5)
    torch.manual_seed(0)
    c = torch.randn(5, CAPTURE_MAP_CHANNELS, BOARD_H, BOARD_W)
    out = validate(ScriptedNet(p, v, s, c), [batch], CPU)
    reference = capture_map_loss(c, batch_to_tensors(batch, CPU).capture_map)
    assert out["capture_map_bce"] == pytest.approx(reference.item(), abs=1e-6)


def test_capture_map_metrics_ignore_off_board_cells() -> None:
    rng = np.random.default_rng(5)
    target = (rng.random((2, CAPTURE_MAP_CHANNELS, BOARD_H, BOARD_W)) < 0.3).astype(np.float32)
    logits = np.where(target > 0.5, 4.0, -4.0).astype(np.float32)
    off = VALID_CELL_MASK == 0.0
    target[:, :, off] = 1.0  # poison the dead slots in both directions
    logits[:, :, off] = -50.0

    batch = _capture_batch(2, target)
    p, v, s, _ = zeros_like_outputs(2)
    out = validate(ScriptedNet(p, v, s, torch.from_numpy(logits)), [batch], CPU)
    assert out["capture_map_auc"] == pytest.approx(1.0)
    assert out["capture_map_bce"] < 0.02


# ----- data health -----------------------------------------------------------


def test_data_health_stats() -> None:
    batch = blank_batch(4, seed=14)
    batch.value[:] = np.array([VALUE_WIN, VALUE_DRAW, VALUE_LOSS, VALUE_WIN])
    batch.score[:] = np.array([SCORE_OFFSET + 3, SCORE_OFFSET, SCORE_OFFSET - 1, SCORE_OFFSET + 4])
    batch.planes[:, PLY_PLANE, :, :] = np.array([0.5, 0.25, 0.75, 0.5], np.float32).reshape(
        4, 1, 1
    )

    out = validate(ScriptedNet(*zeros_like_outputs(4)), [batch], CPU, max_plies=200)
    assert out["positions"] == 4
    assert out["data_draw_rate"] == pytest.approx(0.25)
    assert out["data_decisive_rate"] == pytest.approx(0.75)
    assert out["data_mean_abs_score_diff"] == pytest.approx((3 + 0 + 1 + 4) / 4)
    assert out["data_mean_ply_fraction"] == pytest.approx(0.5)
    assert out["data_mean_ply"] == pytest.approx(100.0)


def test_mean_ply_is_nan_without_a_ply_cap() -> None:
    out = validate(ScriptedNet(*zeros_like_outputs(2)), [blank_batch(2)], CPU)
    assert math.isnan(out["data_mean_ply"])


# ----- namespacing, and the metric that is NOT a progress signal -------------


def test_prefix_namespaces_every_key_and_changes_nothing_else() -> None:
    batch = make_batch(size=3, seed=30)
    net = ScriptedNet(*zeros_like_outputs(3))
    bare = validate(net, [batch], CPU)
    net.cursor = 0
    rolling = validate(net, [batch], CPU, prefix=VAL_ROLLING_PREFIX)

    assert set(rolling) == {f"{VAL_ROLLING_PREFIX}{k}" for k in bare}
    assert all(k.startswith(VAL_ROLLING_PREFIX) for k in rolling)
    for key, value in bare.items():
        if key in NON_SCALAR_KEYS:
            continue
        got = rolling[f"{VAL_ROLLING_PREFIX}{key}"]
        if isinstance(value, float) and math.isnan(value):
            assert math.isnan(got), key
        else:
            assert got == value, key
    assert VAL_FROZEN_PREFIX != VAL_ROLLING_PREFIX


def test_frozen_holdout_entropy_is_a_dataset_property_not_a_progress_signal() -> None:
    """**The trap this naming exists to prevent.**

    On a frozen holdout the data never changes, so `data_policy_target_entropy`
    cannot change either — no matter what the network learns. It is a property
    of the held-out set, and a flat curve is it working correctly, not a
    plateau. Read `selfplay/policy_target_entropy` for the training signal.

    Enforced by scoring the *same* batch with two very different models: every
    `data_*` key must come out bit-identical, and the model-quality keys must
    not.
    """
    batch = make_batch(size=6, seed=31)
    torch.manual_seed(0)
    weak = ScriptedNet(*zeros_like_outputs(6))
    strong = ScriptedNet(
        torch.from_numpy(batch.policy).clone() * 12.0,
        torch.randn(6, VALUE_CLASSES) * 4.0,
        torch.randn(6, SCORE_CLASSES) * 4.0,
        torch.randn(6, CAPTURE_MAP_CHANNELS, BOARD_H, BOARD_W),
    )
    a = validate(weak, [batch], CPU, prefix=VAL_FROZEN_PREFIX, max_plies=200)
    b = validate(strong, [batch], CPU, prefix=VAL_FROZEN_PREFIX, max_plies=200)

    dataset_keys = {f"{VAL_FROZEN_PREFIX}{k}" for k in DATASET_METRIC_KEYS}
    assert dataset_keys, "the dataset-property namespace must not be empty"
    assert f"{VAL_FROZEN_PREFIX}data_policy_target_entropy" in dataset_keys
    for key in dataset_keys:
        assert a[key] == b[key] or (math.isnan(a[key]) and math.isnan(b[key])), key

    # ...and the model-quality metrics did move, so the test above is not
    # passing because both models are identical.
    assert a[f"{VAL_FROZEN_PREFIX}policy_top1_agreement"] != pytest.approx(
        b[f"{VAL_FROZEN_PREFIX}policy_top1_agreement"]
    )


def test_every_dataset_property_carries_the_data_prefix() -> None:
    """No metric that is a property of the held-out set may sit under a bare
    name next to the model-quality metrics — that is exactly how
    `val_policy_target_entropy` got read as progress twice."""
    out = validate(ScriptedNet(*zeros_like_outputs(4)), [make_batch(size=4, seed=32)], CPU)
    for key in ("policy_target_entropy", "policy_uniform_entropy", "policy_entropy_ratio",
                "mean_legal_moves", "decisive_rate", "draw_rate", "mean_ply",
                "mean_abs_score_diff", "policy_target_rate", "capture_map_positive_rate"):
        assert key not in out, f"{key} must be namespaced under {DATASET_KEY_PREFIX}"
        assert f"{DATASET_KEY_PREFIX}{key}" in out
    assert DATASET_METRIC_KEYS == frozenset(
        k for k in METRIC_KEYS if k.startswith(DATASET_KEY_PREFIX)
    )
    # The model-quality keys must NOT be under `data_`: the marker has to mean
    # something.
    for key in ("value_ce", "value_brier", "policy_kl_visits_from_prior", "loss_total"):
        assert key in out
        assert not key.startswith(DATASET_KEY_PREFIX)


def test_scalar_metrics_drops_the_calibration_curve() -> None:
    out = validate(ScriptedNet(*zeros_like_outputs(3)), [make_batch(size=3, seed=33)], CPU)
    scalars = scalar_metrics(out)
    assert set(out) - set(scalars) == NON_SCALAR_KEYS
    assert all(isinstance(v, float) for v in scalars.values())


def test_every_metric_is_json_serialisable() -> None:
    """A numpy scalar anywhere in here is a `TypeError` at `metrics.jsonl`
    write time, a generation after the mistake."""
    out = validate(
        ScriptedNet(*zeros_like_outputs(3)), [make_batch(size=3, seed=34)], CPU, max_plies=200
    )
    for key, value in out.items():
        # `NON_SCALAR_KEYS` rather than a literal: this test exists to catch a
        # metric that a generic logger cannot write, and hardcoding the current
        # curve's name would make it silently stop covering the next one.
        if key in NON_SCALAR_KEYS:
            continue
        assert type(value) in (int, float), (key, type(value))
    json.dumps(out)


# ----- plumbing --------------------------------------------------------------


def test_batching_does_not_change_the_result() -> None:
    """Split 6 positions as 6, then as 4+2. Every metric must match — a mean of
    per-batch means would not, and the weighted policy CE least of all."""
    whole = make_batch(size=6, seed=15, policy_weight=np.array([1, 1, 0, 1, 0, 0]))
    parts = []
    for lo, hi in ((0, 4), (4, 6)):
        parts.append(
            Batch(**{k: v[lo:hi] for k, v in whole.__dict__.items()})
        )

    torch.manual_seed(0)
    outputs = (
        torch.randn(6, MOVE_SPACE),
        torch.randn(6, VALUE_CLASSES),
        torch.randn(6, SCORE_CLASSES),
        torch.randn(6, CAPTURE_MAP_CHANNELS, BOARD_H, BOARD_W),
    )
    net = ScriptedNet(*outputs)
    one = validate(net, [whole], CPU)
    net.cursor = 0
    split = validate(net, parts, CPU)

    for key, value in one.items():
        if isinstance(value, float) and not math.isnan(value):
            assert split[key] == pytest.approx(value, abs=1e-6), key
        elif isinstance(value, int):
            assert split[key] == value, key


def test_empty_input_returns_nan_metrics_with_the_same_keys() -> None:
    net = ScriptedNet(*zeros_like_outputs(2))
    populated = validate(net, [blank_batch(2)], CPU)
    empty = validate(net, [], CPU)
    assert set(empty) == set(populated)
    assert empty["positions"] == 0
    assert math.isnan(empty["value_ce"])
    assert len(empty["value_calibration"]) == 10


def test_zero_length_batches_are_skipped() -> None:
    net = ScriptedNet(*zeros_like_outputs(2))
    out = validate(net, [blank_batch(0), blank_batch(2)], CPU)
    assert out["positions"] == 2


def test_model_mode_is_restored_and_no_gradients_are_kept() -> None:
    torch.manual_seed(0)
    model = build("small")
    batches = [make_batch(size=2, seed=16)]

    model.train()
    validate(model, batches, CPU)
    assert model.training is True

    model.eval()
    validate(model, batches, CPU)
    assert model.training is False
    assert all(p.grad is None for p in model.parameters())


def test_mode_is_restored_even_when_a_batch_raises() -> None:
    class Exploding(nn.Module):
        def forward(self, x):
            raise ValueError("boom")

    model = Exploding()
    model.train()
    with pytest.raises(ValueError, match="boom"):
        validate(model, [blank_batch(1)], CPU)
    assert model.training is True


def test_real_network_produces_the_full_metric_set() -> None:
    """End-to-end smoke: a real `small` net over two batches. Values are not
    pinned — only that every documented key is present and finite-or-nan."""
    torch.manual_seed(0)
    model = build("small")
    batches = [make_batch(size=4, seed=17), make_batch(size=3, seed=18)]
    out = validate(model, batches, CPU, max_plies=200)

    assert set(out) == set(METRIC_KEYS) | NON_SCALAR_KEYS
    assert out["positions"] == 7
    assert out["policy_rows"] == 7
    assert math.isfinite(out["loss_total"])
    assert out["loss_total"] == pytest.approx(
        out["policy_ce"] + out["value_ce"] + 0.15 * out["score_ce"] + 0.15 * out["capture_map_bce"],
        abs=1e-6,
    )
    assert 0.0 <= out["policy_top1_agreement"] <= 1.0
