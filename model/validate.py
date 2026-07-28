"""Held-out validation — the feedback loop that was missing.

Every metric the previous run logged was *training* loss on the replay buffer,
so a total collapse of the training signal went unnoticed for three generations
([2026-07-27 review §7.2](../docs/2026-07-27-architecture-review.md)). This
module is the cheap instrumentation that would have caught it in minutes.

## Two holdouts, and only one of them is a gate

`validate` is prefix-parameterised because it serves both (see
`model/replay_buffer.py`):

    val_rolling/   positions withheld from the NEWEST generation, never trained
                   on. Same distribution the network is being trained on, so a
                   rising CE here really is a generalisation failure. Gate on
                   this one.
    val_frozen/    a whole early generation, withheld for the life of the run.
                   The self-play distribution moves by design, so CE here
                   measures DRIFT from an obsolete distribution as much as
                   anything about the model. Read it; never gate on it.

That distinction is not cosmetic. Across generations 1→6 of the validation run,
value CE on the frozen gen-1 holdout rose 1.557 → 2.047 while value *accuracy*
on the same data also rose 0.639 → 0.677. Overfitting and distribution shift
were indistinguishable, because there was only one holdout and it was the wrong
one. `value_brier` (below) exists to break the other half of that ambiguity.

## `data_*` is a property of the DATASET, not of the model

Keys under `data_` describe the held-out positions themselves. On the **frozen**
holdout every one of them is **constant across generations by construction** —
the data does not change, so neither can they. A `data_*` curve that is flat is
working correctly; it is not a plateau, and it is not progress either way. Only
`selfplay/policy_target_entropy` (`model/export_game.py`, measured on the
current generation's fresh games) says anything about whether search is
improving. Reading `val_frozen/data_policy_target_entropy` as a progress signal
is the specific mistake this naming exists to prevent.

## The numbers that carry the signal

    policy_kl_visits_from_prior   KL(MCTS visits ‖ network prior) over legal
                                  moves. How much search improves on the raw
                                  network — the quantity the policy head is
                                  literally trying to close. It should SHRINK.
                                  Far better behaved than top-1 agreement,
                                  which sits at 0.026–0.037 against a ~1/60
                                  chance level and is nearly all noise.
    policy_top5_agreement         top-1's less noisy sibling.
    value_brier                   bounded in [0, 2] and not tail-dominated, so
                                  it separates "generally miscalibrated" (Brier
                                  and CE both rise) from "a few confident
                                  errors" (CE rises, Brier barely moves).
    value_ce, value_accuracy, value_ece, value_calibration

Everything is computed under `torch.no_grad()` with the model in eval mode
(BatchNorm running stats, not batch stats — validation batches are not
statistically like training batches). The caller's train/eval mode is restored on
the way out, including if a batch raises.

Full key set, before the caller's prefix — a flat `dict` apart from
`value_calibration`:

    loss_total, loss_policy, loss_value, loss_score, loss_capture_map

    policy_top1_agreement   argmax over legal moves == MCTS argmax
    policy_top5_agreement   MCTS argmax within the model's top 5 legal moves
    policy_ce               weighted policy cross-entropy
    policy_kl_visits_from_prior   ← the policy signal worth watching
    policy_rows             rows with policy_weight > 0

    value_ce, value_brier, value_accuracy, value_ece, value_calibration

    score_ce, score_mae (marbles), score_accuracy

    capture_map_bce, capture_map_auc, capture_map_separation

    positions

    data_policy_target_entropy    mean entropy of the MCTS visit distribution
    data_policy_uniform_entropy   ln(mean_legal_moves) — the pathological value
    data_policy_entropy_ratio     target / uniform; 1.0 means search learned nothing
    data_mean_legal_moves         branching factor over the same rows
    data_policy_target_rate       fraction of rows carrying a policy target
    data_decisive_rate, data_draw_rate, data_mean_abs_score_diff,
    data_mean_ply_fraction, data_mean_ply, data_capture_map_positive_rate

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
    VALUE_LOSS,
    VALUE_WIN,
    Batch,
)
from model.encoder import PLY_PLANE, VALID_CELL_MASK
from model.train_step import (
    DEFAULT_LOSS_WEIGHTS,
    NUM_VALID_CELLS,
    LossWeights,
    batch_to_tensors,
    masked_log_softmax,
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

#: Namespace for the rolling holdout — positions withheld from the newest
#: generation. Generalisation within the current distribution; gate on this.
VAL_ROLLING_PREFIX = "val_rolling/"

#: Namespace for the frozen holdout — a whole early generation. A drift
#: indicator only. **Never gate on anything under this prefix.**
VAL_FROZEN_PREFIX = "val_frozen/"

#: Marks keys describing the held-out DATA rather than the model. Constant
#: across generations on a frozen holdout, by construction.
DATASET_KEY_PREFIX = "data_"

#: Top-k used by `policy_top5_agreement`.
TOP_K = 5

#: Absolute-ply boundaries for the value calibration-by-phase curve.
#:
#: The question this answers is "how early in a game does the network know who
#: is winning", and a stronger network answers it sooner. Aggregate accuracy
#: cannot: it is dominated by late positions where a marble count settles the
#: matter and every network looks good. Opening accuracy is where the headroom
#: is, and it is invisible in the mean.
#:
#: Absolute ply, not fraction-of-game. Fraction would put the realised game
#: length in the denominator, which is itself an outcome — "70% of the way
#: through" means something different in a 60-ply game than a 190-ply one, and
#: the network cannot know which it is in. `ply / max_plies` (plane 12) divides
#: by the *cap*, a constant, so it is absolute ply rescaled and converts back
#: exactly.
#:
#: **Read these buckets as conditional.** Only games that lasted at least `k`
#: plies contribute to the bucket at `k`, and games that run long are
#: systematically the closer ones. Accuracy in the late buckets is measured on
#: a harder subsample and is not comparable to the early buckets in absolute
#: terms — compare a bucket against itself across generations.
PHASE_EDGES_PLIES: tuple[int, ...] = (0, 20, 40, 60, 100, 200)

#: The returned values that are not scalars. A generic logger (TensorBoard, a
#: CSV writer) should skip them; they belong in `metrics.jsonl`.
NON_SCALAR_KEYS: frozenset[str] = frozenset({"value_calibration", "value_phase"})


def validate(
    model: nn.Module,
    batches: Iterable[Batch],
    device: torch.device,
    *,
    weights: LossWeights = DEFAULT_LOSS_WEIGHTS,
    calibration_bins: int = DEFAULT_CALIBRATION_BINS,
    capture_threshold: float = DEFAULT_CAPTURE_THRESHOLD,
    max_plies: int | None = None,
    prefix: str = "",
) -> dict[str, object]:
    """Run `model` over `batches` and return the held-out metrics.

    `batches` is any iterable of `Batch` — a list, or a generator draining a
    holdout. It is consumed exactly once.

    `max_plies` is the game's ply cap. Plane 12 stores `ply / max_plies`, so the
    cap is the only way back to a human-legible mean ply; without it
    `data_mean_ply` is `nan` and only `data_mean_ply_fraction` is reported.

    `prefix` namespaces every returned key, so one function serves both
    holdouts: pass `VAL_ROLLING_PREFIX` or `VAL_FROZEN_PREFIX`. The default of
    `""` returns bare keys.
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

    metrics = acc.finalise(
        weights=weights,
        calibration_bins=calibration_bins,
        capture_threshold=capture_threshold,
        max_plies=max_plies,
    )
    return {f"{prefix}{k}": v for k, v in metrics.items()} if prefix else metrics


def scalar_metrics(metrics: dict[str, object]) -> dict[str, float]:
    """`metrics` reduced to the plain-number entries, for a generic logger.

    Drops `value_calibration` (a curve) and anything else non-numeric,
    whether or not a prefix has been applied.
    """
    return {
        k: float(v)
        for k, v in metrics.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }


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
        best = t.policy.argmax(dim=1)
        self._push("agree", (masked.argmax(dim=1) == best).float())
        # Top-k by rank rather than by `topk`: the number of legal moves the
        # model scores strictly above the MCTS choice. `topk` over the full
        # 2562-wide head would pad with illegal moves on a row with fewer than
        # k legal ones and score it as agreement for free.
        best_logit = masked.gather(1, best.unsqueeze(1))
        rank = (masked > best_logit).sum(dim=1)
        self._push("agree_topk", (rank < TOP_K).float())
        self._push("legal_count", t.legal_mask.sum(dim=1))
        # Entropy of the *target*: clamp inside the log only, so zero-probability
        # moves contribute 0·log(tiny) = 0 rather than 0·(−inf) = NaN.
        p = t.policy
        log_p = torch.log(p.clamp_min(1e-12))
        self._push("target_entropy", -(p * log_p).sum(dim=1))
        # KL(visits ‖ prior) over legal moves = Σ π log(π / p). Computed
        # directly rather than as `policy_ce − target_entropy` so a bug in the
        # cross-entropy cannot cancel itself out here. Rows with no policy
        # target come out as 0 (π is identically zero) and are dropped at
        # reduction time, not counted as a perfect match.
        log_q = masked_log_softmax(policy_logits, t.legal_mask)
        self._push("policy_kl_rows", (p * (log_p - log_q)).sum(dim=1))

        # -- value -----------------------------------------------------------
        self._push("value_ce_rows", F.cross_entropy(value_logits, t.value, reduction="none"))
        value_probs = F.softmax(value_logits, dim=1)
        self._push("p_win", value_probs[:, VALUE_WIN])
        self._push("value_correct", (value_logits.argmax(dim=1) == t.value).float())
        self._push("value_target", t.value)
        # Multiclass Brier: Σ_c (p_c − y_c)², in [0, 2]. Bounded and quadratic,
        # where CE is unbounded and dominated by its tail — the pair is what
        # tells "uniformly miscalibrated" apart from "a handful of confident
        # errors".
        onehot = F.one_hot(t.value, num_classes=value_probs.shape[1]).to(value_probs.dtype)
        self._push("value_brier_rows", ((value_probs - onehot) ** 2).sum(dim=1))
        # Probability assigned to the class the network actually picked. Paired
        # with accuracy per phase it separates the two ways a value head fails:
        # confident and wrong is a different problem from unsure and right.
        self._push("value_confidence", value_probs.max(dim=1).values)
        self._push("value_predicted", value_logits.argmax(dim=1))
        # Scalar expectation of the value head, on the same [-1, +1] scale as
        # the search's `q`, so the two are directly comparable.
        self._push("value_expected", value_probs[:, VALUE_WIN] - value_probs[:, VALUE_LOSS])
        self._push("q_target", t.q)

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
            topk = float(self._get("agree_topk")[sel].mean())
            target_entropy = float(self._get("target_entropy")[sel].mean())
            mean_legal = float(self._get("legal_count")[sel].mean())
            # Only over rows that carry a full-search target: a fast-searched
            # row's visit distribution is not the thing the prior is chasing.
            policy_kl = float(self._get("policy_kl_rows")[sel].mean())
        else:
            top1 = nan
            topk = nan
            target_entropy = nan
            policy_kl = nan
            mean_legal = float(self._get("legal_count").mean())
        uniform_entropy = math.log(mean_legal) if mean_legal > 0 else nan
        entropy_ratio = (
            target_entropy / uniform_entropy
            if uniform_entropy and uniform_entropy > 0 and not math.isnan(target_entropy)
            else nan
        )

        # -- value ------------------------------------------------------------
        value_ce = float(self._get("value_ce_rows").mean())
        value_brier = float(self._get("value_brier_rows").mean())
        value_correct = self._get("value_correct")
        value_accuracy = float(value_correct.mean())
        p_win = self._get("p_win")
        value_target = self._get("value_target")
        won = (value_target == VALUE_WIN).astype(np.float64)
        calibration, ece = _calibration(p_win, won, calibration_bins)

        # Accuracy as a function of how far into the game the position sits.
        # The aggregate is dominated by late positions, where a marble count
        # settles the outcome and every network scores well; the opening is
        # where the headroom is and where improvement actually shows.
        ply_fraction_rows = self._get("ply_fraction").astype(np.float64)
        confidence = self._get("value_confidence")
        is_draw_rows = (value_target == VALUE_DRAW).astype(np.float64)
        if max_plies:
            phase = _phase_curve(
                ply_fraction_rows * float(max_plies),
                value_correct,
                confidence,
                self._get("value_brier_rows"),
                is_draw_rows,
                PHASE_EDGES_PLIES,
            )
        else:
            # Plane 12 holds ply/max_plies; without the cap there is no way back
            # to an absolute ply, and bucketing the fraction would silently mean
            # something different for a run with a different cap.
            phase = _empty_phase(PHASE_EDGES_PLIES)
        early = phase[0]

        # The collapse this project actually suffered: the value head learned to
        # emit a constant when no game ever ended decisively, and three
        # generations passed before anybody noticed. Predicted draw rate against
        # the observed draw rate names it directly — a head predicting 90% draws
        # into 3% draws has stopped modelling anything, and `value_accuracy`
        # alone will not say so.
        predicted = self._get("value_predicted")
        predicted_draw_rate = float((predicted == VALUE_DRAW).mean())

        # Per-class recall. A head can look healthy in aggregate while being
        # blind to one outcome — losses are the expensive one to miss.
        def _recall(cls: int) -> float:
            mask = value_target == cls
            return float(value_correct[mask].mean()) if mask.any() else nan

        # How much the search moves the value head. The value analogue of
        # `policy_kl_visits_from_prior`: `q` is the search's root value for this
        # position, so the gap is what search is correcting. On the rolling
        # holdout `q` comes from the network being measured; on the frozen one
        # it comes from whichever network played that generation, which is why
        # this is reported and not alarmed on.
        v_net = self._get("value_expected")
        q_rows = self._get("q_target")
        value_q_gap = float(np.abs(v_net - q_rows).mean())
        value_q_sign_agreement = float((np.sign(v_net) == np.sign(q_rows)).mean())

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
        draw_rate = float(is_draw_rows.mean())
        ply_fraction = float(ply_fraction_rows.mean())

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
            # -- model quality: these move when the network moves -------------
            # headline
            "loss_total": float(loss_total),
            "loss_policy": policy_ce,
            "loss_value": value_ce,
            "loss_score": score_ce,
            "loss_capture_map": capture_bce,
            # policy
            "policy_top1_agreement": top1,
            "policy_top5_agreement": topk,
            "policy_ce": policy_ce,
            "policy_kl_visits_from_prior": policy_kl,
            "policy_rows": int(sel.sum()),
            # value
            "value_ce": value_ce,
            "value_brier": value_brier,
            "value_accuracy": value_accuracy,
            "value_ece": ece,
            "value_calibration": calibration,
            # The headline of the phase curve: how well the network knows the
            # outcome in the first 20 plies, where it cannot yet be read off a
            # marble count. This is the number that should climb.
            "value_accuracy_early": early["accuracy"],
            "value_brier_early": early["brier"],
            "value_confidence_early": early["mean_confidence"],
            "value_confidence": float(confidence.mean()),
            "value_predicted_draw_rate": predicted_draw_rate,
            "value_recall_win": _recall(VALUE_WIN),
            "value_recall_draw": _recall(VALUE_DRAW),
            "value_recall_loss": _recall(VALUE_LOSS),
            "value_q_gap": value_q_gap,
            "value_q_sign_agreement": value_q_sign_agreement,
            "value_phase": phase,
            # score
            "score_ce": score_ce,
            "score_mae": score_mae,
            "score_accuracy": score_accuracy,
            # capture map
            "capture_map_bce": capture_bce,
            "capture_map_auc": capture_auc,
            "capture_map_separation": capture_sep,
            "positions": int(n),
            # -- properties of the held-out DATA ------------------------------
            # Constant across generations on a frozen holdout, by construction.
            # Flat is correct here; flat is not a plateau and not progress.
            "data_policy_target_entropy": target_entropy,
            "data_policy_uniform_entropy": uniform_entropy,
            "data_policy_entropy_ratio": entropy_ratio,
            "data_policy_entropy_gap": (
                uniform_entropy - target_entropy
                if not (math.isnan(uniform_entropy) or math.isnan(target_entropy))
                else nan
            ),
            "data_mean_legal_moves": mean_legal,
            "data_policy_target_rate": float(sel.mean()),
            "data_capture_map_positive_rate": (
                float(cap_label.mean()) if cap_label.size else nan
            ),
            "data_decisive_rate": 1.0 - draw_rate,
            "data_draw_rate": draw_rate,
            "data_mean_abs_score_diff": float(self._get("abs_score_diff").mean()),
            "data_mean_ply_fraction": ply_fraction,
            "data_mean_ply": ply_fraction * max_plies if max_plies else nan,
        }


#: Every scalar key `validate` returns, in the order `finalise` emits them.
#: `_empty_metrics` is built from this so the two cannot drift apart.
METRIC_KEYS: tuple[str, ...] = (
    "loss_total",
    "loss_policy",
    "loss_value",
    "loss_score",
    "loss_capture_map",
    "policy_top1_agreement",
    "policy_top5_agreement",
    "policy_ce",
    "policy_kl_visits_from_prior",
    "policy_rows",
    "value_ce",
    "value_brier",
    "value_accuracy",
    "value_ece",
    "value_accuracy_early",
    "value_brier_early",
    "value_confidence_early",
    "value_confidence",
    "value_predicted_draw_rate",
    "value_recall_win",
    "value_recall_draw",
    "value_recall_loss",
    "value_q_gap",
    "value_q_sign_agreement",
    "score_ce",
    "score_mae",
    "score_accuracy",
    "capture_map_bce",
    "capture_map_auc",
    "capture_map_separation",
    "positions",
    "data_policy_target_entropy",
    "data_policy_uniform_entropy",
    "data_policy_entropy_ratio",
    "data_policy_entropy_gap",
    "data_mean_legal_moves",
    "data_policy_target_rate",
    "data_capture_map_positive_rate",
    "data_decisive_rate",
    "data_draw_rate",
    "data_mean_abs_score_diff",
    "data_mean_ply_fraction",
    "data_mean_ply",
)

#: Keys that describe the held-out positions rather than the model. On a frozen
#: holdout every one of these is constant for the life of the run.
DATASET_METRIC_KEYS: frozenset[str] = frozenset(
    k for k in METRIC_KEYS if k.startswith(DATASET_KEY_PREFIX)
)

#: Integer-valued keys — counts, not rates. `nan` would be a lie for these.
_COUNT_KEYS: frozenset[str] = frozenset({"positions", "policy_rows"})


def _empty_metrics(calibration_bins: int) -> dict[str, object]:
    """Every metric as `nan` for an empty validation set, with the same keys —
    a caller logging these should not have to branch on emptiness."""
    nan = float("nan")
    out: dict[str, object] = {k: 0 if k in _COUNT_KEYS else nan for k in METRIC_KEYS}
    out["value_calibration"] = _empty_calibration(calibration_bins)
    out["value_phase"] = _empty_phase(PHASE_EDGES_PLIES)
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


def _empty_phase(edges: tuple[int, ...]) -> list[dict[str, float]]:
    nan = float("nan")
    return [
        {
            "ply_lower": int(edges[i]),
            "ply_upper": int(edges[i + 1]),
            "count": 0,
            "mean_ply": nan,
            "accuracy": nan,
            "mean_confidence": nan,
            "brier": nan,
            "draw_rate": nan,
        }
        for i in range(len(edges) - 1)
    ]


def _phase_curve(
    ply: np.ndarray,
    correct: np.ndarray,
    confidence: np.ndarray,
    brier: np.ndarray,
    is_draw: np.ndarray,
    edges: tuple[int, ...],
) -> list[dict[str, float]]:
    """Value-head quality as a function of how far into the game the position is.

    The curve is the artifact worth reading; the scalars derived from it are a
    convenience for alarming and for TensorBoard. A network that is improving
    shows the *early* buckets climbing while the late ones sit near their
    ceiling — late positions are easy for everyone, so aggregate accuracy moves
    only slightly while the thing you care about moves a lot.

    `is_draw` is carried because a bucket's accuracy is uninterpretable without
    it: at the ply cap almost everything adjudicates, and a bucket that is 60%
    draws rewards a network that has learned to shrug.
    """
    curve: list[dict[str, float]] = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        # Half-open, except the last bucket, which must catch the ply cap itself.
        in_bin = (ply >= lo) & (ply < hi) if i < len(edges) - 2 else (ply >= lo) & (ply <= hi)
        count = int(in_bin.sum())
        nan = float("nan")
        curve.append(
            {
                "ply_lower": int(lo),
                "ply_upper": int(hi),
                "count": count,
                "mean_ply": float(ply[in_bin].mean()) if count else nan,
                "accuracy": float(correct[in_bin].mean()) if count else nan,
                "mean_confidence": float(confidence[in_bin].mean()) if count else nan,
                "brier": float(brier[in_bin].mean()) if count else nan,
                "draw_rate": float(is_draw[in_bin].mean()) if count else nan,
            }
        )
    return curve


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
    "DATASET_KEY_PREFIX",
    "DATASET_METRIC_KEYS",
    "DEFAULT_CALIBRATION_BINS",
    "DEFAULT_CAPTURE_THRESHOLD",
    "METRIC_KEYS",
    "NON_SCALAR_KEYS",
    "PHASE_EDGES_PLIES",
    "NUM_VALID_CELLS",
    "TOP_K",
    "VAL_FROZEN_PREFIX",
    "VAL_ROLLING_PREFIX",
    "scalar_metrics",
    "validate",
]
