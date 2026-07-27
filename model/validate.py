"""Held-out validation — the feedback loop that was missing.

Every metric the previous run logged was *training* loss on the replay buffer,
so a total collapse of the training signal went unnoticed for three generations
([2026-07-27 review §7.2](../docs/2026-07-27-architecture-review.md)). This
module is the cheap instrumentation that would have caught it in minutes:
run a frozen set of positions from a withheld generation through the network
every generation and look at the numbers.

The single most important number here is **`policy_target_entropy`**. The failed
run sat at a policy loss of 4.13 ≈ `ln(62)` ≈ `ln(branching factor)`, meaning the
MCTS visit distributions carried *zero* information — search was returning
roughly uniform over legal moves and there was nothing for the policy head to
learn. `policy_uniform_entropy` (= `ln(mean_legal_moves)`) is reported right next
to it so the comparison is direct, and `policy_entropy_ratio` is the two divided:
**at 1.0 the data is worthless and no downstream metric means anything.**

Everything is computed under `torch.no_grad()` with the model in eval mode
(BatchNorm running stats, not batch stats — validation batches are not
statistically like training batches). The caller's train/eval mode is restored on
the way out, including if a batch raises.

Metrics returned by `validate` — a flat `dict` apart from `value_calibration`:

    loss_total, loss_policy, loss_value, loss_score, loss_capture_map

    policy_top1_agreement   argmax over legal moves == MCTS argmax
    policy_ce               weighted policy cross-entropy
    policy_target_entropy   mean entropy of the MCTS visit distribution  ← the one
    policy_uniform_entropy  ln(mean_legal_moves) — the pathological value
    policy_entropy_ratio    target / uniform; 1.0 means search learned nothing
    mean_legal_moves        branching factor over the same rows
    policy_rows             rows with policy_weight > 0

    value_ce, value_accuracy, value_ece, value_calibration

    score_ce, score_mae (marbles), score_accuracy

    capture_map_bce, capture_map_auc, capture_map_separation,
    capture_map_positive_rate

    positions, decisive_rate, draw_rate, mean_abs_score_diff,
    mean_ply_fraction, mean_ply, policy_target_rate

Undefined metrics are `nan` (e.g. `policy_top1_agreement` when no row carries a
policy target), never a silently misleading 0.0.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.batch import (
    SCORE_OFFSET,
    VALUE_DRAW,
    VALUE_WIN,
    Batch,
)
from model.encoder import PLY_PLANE, VALID_CELL_MASK
from model.train_step import (
    DEFAULT_LOSS_WEIGHTS,
    NUM_VALID_CELLS,
    LossWeights,
    batch_to_tensors,
    masked_policy_logits,
    policy_cross_entropy_rows,
    valid_cell_mask,
)

#: Default number of equal-width bins for the value calibration curve.
DEFAULT_CALIBRATION_BINS = 10

#: A capture-map target at or above this counts as a positive for AUC. The
#: targets are time-discounted probabilities, so they are continuous in [0, 1].
DEFAULT_CAPTURE_THRESHOLD = 0.5

#: Flat indices (into the 81-cell grid) of the 61 on-board cells.
_VALID_FLAT = np.flatnonzero(VALID_CELL_MASK.reshape(-1) > 0)


def validate(
    model: nn.Module,
    batches: Iterable[Batch],
    device: torch.device,
    *,
    weights: LossWeights = DEFAULT_LOSS_WEIGHTS,
    calibration_bins: int = DEFAULT_CALIBRATION_BINS,
    capture_threshold: float = DEFAULT_CAPTURE_THRESHOLD,
    max_plies: int | None = None,
) -> dict[str, object]:
    """Run `model` over `batches` and return the held-out metrics.

    `batches` is any iterable of `Batch` — a list, or a generator draining a
    frozen validation shard. It is consumed exactly once.

    `max_plies` is the game's ply cap. Plane 12 stores `ply / max_plies`, so the
    cap is the only way back to a human-legible mean ply; without it
    `mean_ply` is `nan` and only `mean_ply_fraction` is reported.
    """
    was_training = model.training
    model.eval()
    acc = _Accumulator()
    try:
        with torch.no_grad():
            for batch in batches:
                if batch.size == 0:
                    continue
                acc.add(model, batch, device)
    finally:
        model.train(was_training)

    return acc.finalise(
        weights=weights,
        calibration_bins=calibration_bins,
        capture_threshold=capture_threshold,
        max_plies=max_plies,
    )


# ----- accumulation ----------------------------------------------------------


class _Accumulator:
    """Collects small per-row arrays across batches and reduces them at the end.

    Per-row rather than per-batch so that the reductions are exact: a mean of
    per-batch means is wrong the moment batch sizes differ, and the weighted
    policy CE has to be normalised by the *global* weight sum.
    """

    def __init__(self) -> None:
        self._rows: dict[str, list[np.ndarray]] = {}

    def _push(self, name: str, values: torch.Tensor | np.ndarray) -> None:
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        self._rows.setdefault(name, []).append(np.asarray(values))

    def _get(self, name: str) -> np.ndarray:
        parts = self._rows.get(name)
        if not parts:
            return np.zeros(0, dtype=np.float64)
        return np.concatenate([p.reshape(-1) for p in parts]).astype(np.float64)

    def add(self, model: nn.Module, batch: Batch, device: torch.device) -> None:
        t = batch_to_tensors(batch, device)
        policy_logits, value_logits, score_logits, capture_logits = model(t.planes)

        # -- policy ----------------------------------------------------------
        masked = masked_policy_logits(policy_logits, t.legal_mask)
        self._push("policy_ce_rows", policy_cross_entropy_rows(policy_logits, t.policy, t.legal_mask))
        self._push("policy_weight", t.policy_weight)
        self._push("agree", (masked.argmax(dim=1) == t.policy.argmax(dim=1)).float())
        self._push("legal_count", t.legal_mask.sum(dim=1))
        # Entropy of the *target*: clamp inside the log only, so zero-probability
        # moves contribute 0·log(tiny) = 0 rather than 0·(−inf) = NaN.
        p = t.policy
        self._push("target_entropy", -(p * torch.log(p.clamp_min(1e-12))).sum(dim=1))

        # -- value -----------------------------------------------------------
        self._push("value_ce_rows", F.cross_entropy(value_logits, t.value, reduction="none"))
        value_probs = F.softmax(value_logits, dim=1)
        self._push("p_win", value_probs[:, VALUE_WIN])
        self._push("value_correct", (value_logits.argmax(dim=1) == t.value).float())
        self._push("value_target", t.value)

        # -- score -----------------------------------------------------------
        self._push("score_ce_rows", F.cross_entropy(score_logits, t.score, reduction="none"))
        score_probs = F.softmax(score_logits, dim=1)
        buckets = torch.arange(
            score_probs.shape[1], device=score_probs.device, dtype=score_probs.dtype
        ) - float(SCORE_OFFSET)
        expected_diff = (score_probs * buckets).sum(dim=1)
        true_diff = t.score.to(expected_diff.dtype) - float(SCORE_OFFSET)
        self._push("score_abs_err", (expected_diff - true_diff).abs())
        self._push("score_correct", (score_logits.argmax(dim=1) == t.score).float())
        self._push("abs_score_diff", true_diff.abs())

        # -- capture map ------------------------------------------------------
        # Masked BCE kept per row, so the global mean is exact regardless of how
        # the validation set was split into batches. Same normalisation as
        # `train_step.capture_map_loss` (per masked element); a test pins that
        # the two agree.
        n_rows = capture_logits.shape[0]
        mask = valid_cell_mask(capture_logits.device, capture_logits.dtype)
        elementwise = F.binary_cross_entropy_with_logits(
            capture_logits, t.capture_map, reduction="none"
        )
        per_row = (elementwise * mask).sum(dim=(1, 2, 3)) / (
            capture_logits.shape[1] * NUM_VALID_CELLS
        )
        self._push("capture_bce_rows", per_row)
        valid = torch.from_numpy(_VALID_FLAT).to(capture_logits.device)
        probs = torch.sigmoid(capture_logits).reshape(n_rows, -1, 81).index_select(2, valid)
        target = t.capture_map.reshape(n_rows, -1, 81).index_select(2, valid)
        self._push("capture_prob", probs.reshape(-1))
        self._push("capture_target", target.reshape(-1))

        # -- data health ------------------------------------------------------
        self._push("ply_fraction", t.planes[:, PLY_PLANE, 0, 0])

    # -- reduction ------------------------------------------------------------

    def finalise(
        self,
        *,
        weights: LossWeights,
        calibration_bins: int,
        capture_threshold: float,
        max_plies: int | None,
    ) -> dict[str, object]:
        nan = float("nan")
        policy_weight = self._get("policy_weight")
        n = policy_weight.size
        if n == 0:
            return _empty_metrics(calibration_bins)

        # -- policy, over rows that actually carry a target -------------------
        sel = policy_weight > 0
        weight_sum = float(policy_weight.sum())
        policy_ce_rows = self._get("policy_ce_rows")
        policy_ce = (
            float((policy_weight * policy_ce_rows).sum() / weight_sum) if weight_sum > 0 else nan
        )
        if sel.any():
            top1 = float(self._get("agree")[sel].mean())
            target_entropy = float(self._get("target_entropy")[sel].mean())
            mean_legal = float(self._get("legal_count")[sel].mean())
        else:
            top1 = nan
            target_entropy = nan
            mean_legal = float(self._get("legal_count").mean())
        uniform_entropy = math.log(mean_legal) if mean_legal > 0 else nan
        entropy_ratio = (
            target_entropy / uniform_entropy
            if uniform_entropy and uniform_entropy > 0 and not math.isnan(target_entropy)
            else nan
        )

        # -- value ------------------------------------------------------------
        value_ce = float(self._get("value_ce_rows").mean())
        value_accuracy = float(self._get("value_correct").mean())
        p_win = self._get("p_win")
        won = (self._get("value_target") == VALUE_WIN).astype(np.float64)
        calibration, ece = _calibration(p_win, won, calibration_bins)

        # -- score ------------------------------------------------------------
        score_ce = float(self._get("score_ce_rows").mean())
        score_mae = float(self._get("score_abs_err").mean())
        score_accuracy = float(self._get("score_correct").mean())

        # -- capture map -------------------------------------------------------
        capture_bce = float(self._get("capture_bce_rows").mean())
        cap_prob = self._get("capture_prob")
        cap_label = (self._get("capture_target") >= capture_threshold).astype(np.float64)
        capture_auc = _auc(cap_prob, cap_label)
        capture_sep = _separation(cap_prob, cap_label)

        # -- data health -------------------------------------------------------
        value_target = self._get("value_target")
        draw_rate = float((value_target == VALUE_DRAW).mean())
        ply_fraction = float(self._get("ply_fraction").mean())

        # Same weighting as the training objective, so the two curves are
        # directly comparable and a train/validation gap is readable off them.
        # With no policy targets at all the policy term is dropped rather than
        # propagating a NaN through the headline number.
        loss_total = (
            (policy_ce if weight_sum > 0 else 0.0)
            + weights.value * value_ce
            + weights.score * score_ce
            + weights.capture_map * capture_bce
        )

        return {
            # headline
            "loss_total": float(loss_total),
            "loss_policy": policy_ce,
            "loss_value": value_ce,
            "loss_score": score_ce,
            "loss_capture_map": capture_bce,
            # policy
            "policy_top1_agreement": top1,
            "policy_ce": policy_ce,
            "policy_target_entropy": target_entropy,
            "policy_uniform_entropy": uniform_entropy,
            "policy_entropy_ratio": entropy_ratio,
            "mean_legal_moves": mean_legal,
            "policy_rows": int(sel.sum()),
            # value
            "value_ce": value_ce,
            "value_accuracy": value_accuracy,
            "value_ece": ece,
            "value_calibration": calibration,
            # score
            "score_ce": score_ce,
            "score_mae": score_mae,
            "score_accuracy": score_accuracy,
            # capture map
            "capture_map_bce": capture_bce,
            "capture_map_auc": capture_auc,
            "capture_map_separation": capture_sep,
            "capture_map_positive_rate": float(cap_label.mean()) if cap_label.size else nan,
            # data health
            "positions": int(n),
            "decisive_rate": 1.0 - draw_rate,
            "draw_rate": draw_rate,
            "mean_abs_score_diff": float(self._get("abs_score_diff").mean()),
            "mean_ply_fraction": ply_fraction,
            "mean_ply": ply_fraction * max_plies if max_plies else nan,
            "policy_target_rate": float(sel.mean()),
        }


def _empty_metrics(calibration_bins: int) -> dict[str, object]:
    """Every metric as `nan` for an empty validation set, with the same keys —
    a caller logging these should not have to branch on emptiness."""
    nan = float("nan")
    keys = (
        "loss_total loss_policy loss_value loss_score loss_capture_map "
        "policy_top1_agreement policy_ce policy_target_entropy policy_uniform_entropy "
        "policy_entropy_ratio mean_legal_moves value_ce value_accuracy value_ece "
        "score_ce score_mae score_accuracy capture_map_bce capture_map_auc "
        "capture_map_separation capture_map_positive_rate decisive_rate draw_rate "
        "mean_abs_score_diff mean_ply_fraction mean_ply policy_target_rate"
    ).split()
    out: dict[str, object] = dict.fromkeys(keys, nan)
    out["positions"] = 0
    out["policy_rows"] = 0
    out["value_calibration"] = _empty_calibration(calibration_bins)
    return out


# ----- statistics ------------------------------------------------------------


def _empty_calibration(bins: int) -> list[dict[str, float]]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    return [
        {
            "bin_lower": float(edges[i]),
            "bin_upper": float(edges[i + 1]),
            "count": 0,
            "mean_predicted": float("nan"),
            "observed_win_rate": float("nan"),
        }
        for i in range(bins)
    ]


def _calibration(
    predicted: np.ndarray, observed: np.ndarray, bins: int
) -> tuple[list[dict[str, float]], float]:
    """Reliability curve for P(win) plus the expected calibration error.

    `ECE = Σ_b (n_b / N) · |mean predicted_b − observed frequency_b|`, the usual
    bin-weighted L1 gap. Empty bins contribute nothing and report `nan` rather
    than a fabricated 0.
    """
    edges = np.linspace(0.0, 1.0, bins + 1)
    # `digitize` on the interior edges puts p == 1.0 in the last bin, and
    # p == 0.0 in the first — both endpoints are legitimate probabilities.
    idx = np.digitize(predicted, edges[1:-1], right=False)
    curve: list[dict[str, float]] = []
    ece = 0.0
    total = float(predicted.size)
    for b in range(bins):
        in_bin = idx == b
        count = int(in_bin.sum())
        if count:
            mean_pred = float(predicted[in_bin].mean())
            rate = float(observed[in_bin].mean())
            ece += (count / total) * abs(mean_pred - rate)
        else:
            mean_pred = float("nan")
            rate = float("nan")
        curve.append(
            {
                "bin_lower": float(edges[b]),
                "bin_upper": float(edges[b + 1]),
                "count": count,
                "mean_predicted": mean_pred,
                "observed_win_rate": rate,
            }
        )
    return curve, ece


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC AUC via the Mann–Whitney rank statistic.

    sklearn is deliberately not a dependency, and the rank form is a handful of
    vectorised numpy calls: `AUC = (Σ ranks of positives − P(P+1)/2) / (P·N)`,
    with tied scores taking their group's mean rank. Returns `nan` when one
    class is absent, where AUC is undefined.
    """
    n = labels.size
    n_pos = float(labels.sum())
    n_neg = n - n_pos
    if n == 0 or n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ordered = scores[order]
    _, inverse, counts = np.unique(ordered, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)  # 1-based rank of each tie group's last element
    mean_rank = (ends - counts + 1 + ends) / 2.0
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = mean_rank[inverse]
    return float((ranks[labels > 0].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _separation(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mean predicted probability on positive cells minus that on negative
    cells. Cruder than AUC but scale-free and trivially interpretable: 0 means
    the capture-map head has found nothing."""
    pos = labels > 0
    if not pos.any() or pos.all():
        return float("nan")
    return float(scores[pos].mean() - scores[~pos].mean())


__all__ = [
    "DEFAULT_CALIBRATION_BINS",
    "DEFAULT_CAPTURE_THRESHOLD",
    "NUM_VALID_CELLS",
    "validate",
]
