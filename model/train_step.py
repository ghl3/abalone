"""One SGD step: forward, four-headed loss, backward, clip, optimizer step.

The loss (MODEL.md §6.6)::

    L = L_policy
      + w_v · L_value          3-way cross-entropy   (win, draw, loss)
      + w_s · L_score          13-way cross-entropy  (capture diff d ∈ [−6, +6])
      + w_c · L_capture_map    masked BCE-with-logits over the 61 valid cells

Defaults `w_v = 1.0`, `w_s = 0.15`, `w_c = 0.15` — the auxiliary heads exist to
shape the representation, not to dominate it.

Things here that are load-bearing and should not be "simplified":

**Illegal moves are masked before the softmax**, by substituting a large finite
negative logit rather than `-inf`. `-inf` makes an all-illegal row produce
`NaN` through `log_softmax`; `-1e9` degrades to a uniform row instead, and the
target is all-zero there anyway so it contributes nothing.

**The policy loss is normalised by the sum of `policy_weight`, not by the batch
size.** Playout cap randomisation gives only ~25% of rows a usable visit
distribution ([MODEL.md §7.2](../docs/MODEL.md)); dividing by `B` would make the
effective policy learning rate track the cap-randomisation rate, so retuning that
rate would silently retune the policy LR. An all-zero-weight batch yields a
policy loss of exactly 0 with zero gradient, not a NaN.

**`batch.q` is not a loss term.** The value head trains on the game outcome
alone — there is deliberately no z/q blend any more, and the ramp that used to
schedule one is gone. See
[ARCHITECTURE.md §5.5](../docs/ARCHITECTURE.md#55-training-batch): the blend
worked around an all-draws data distribution that the capture-handicap
curriculum removed. `q` stays in the batch for diagnostics and the review UI.

**No weight decay term is added to the loss.** Regularisation is the optimizer's
job and the optimizer is expected to be `AdamW`, whose weight decay is decoupled.
Adding an L2 penalty here would double-count it. This module never constructs an
optimizer, and assumes nothing about which one it is handed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.batch import Batch
from model.encoder import VALID_CELL_MASK

#: Substituted for illegal-move logits before the softmax. Large and negative,
#: but *finite*: `-inf` would give an all-illegal row a NaN gradient.
ILLEGAL_LOGIT = -1e9

#: Guards the policy-loss denominator when every row has weight 0.
_WEIGHT_EPS = 1e-8

#: Number of on-board cells in the 9×9 axial grid.
NUM_VALID_CELLS = int(VALID_CELL_MASK.sum())


@dataclass(frozen=True)
class LossWeights:
    """Relative weight of each head in the total loss. `policy` is fixed at 1.0
    by construction — it is the scale everything else is expressed against."""

    value: float = 1.0
    score: float = 0.15
    capture_map: float = 0.15


DEFAULT_LOSS_WEIGHTS = LossWeights()


@dataclass
class StepMetrics:
    """Per-step scalars, all plain floats and safe to log or average."""

    loss_total: float
    loss_policy: float
    loss_value: float
    loss_score: float
    loss_capture_map: float
    grad_norm: float
    #: Sum of `policy_weight` over the batch — the policy loss's denominator.
    #: Worth logging: if it drifts, the policy target supply has changed.
    policy_weight_sum: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class BatchTensors:
    """`Batch` moved onto a device as tensors. Field-for-field identical to
    `model.batch.Batch`; `q` is carried through but never read by the loss."""

    planes: torch.Tensor  # (B, 14, 9, 9) float32
    policy: torch.Tensor  # (B, 2562)     float32
    legal_mask: torch.Tensor  # (B, 2562) float32
    policy_weight: torch.Tensor  # (B,)   float32
    value: torch.Tensor  # (B,)           int64
    score: torch.Tensor  # (B,)           int64
    capture_map: torch.Tensor  # (B, 2, 9, 9) float32
    q: torch.Tensor  # (B,)               float32 — diagnostics only

    @property
    def size(self) -> int:
        return int(self.planes.shape[0])


@dataclass
class LossTerms:
    """The loss as tensors, before `.item()`. `total` is the one to backward."""

    total: torch.Tensor
    policy: torch.Tensor
    value: torch.Tensor
    score: torch.Tensor
    capture_map: torch.Tensor
    policy_weight_sum: torch.Tensor


# ----- tensor plumbing -------------------------------------------------------

_MASK_CACHE: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}


def valid_cell_mask(device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """`(1, 1, 9, 9)` on-board mask, cached per (device, dtype).

    Imported from the encoder so there is exactly one definition of which cells
    exist; the network holds its own copy of the same array as a buffer.
    """
    key = (torch.device(device), dtype)
    cached = _MASK_CACHE.get(key)
    if cached is None:
        cached = torch.from_numpy(np.ascontiguousarray(VALID_CELL_MASK)).to(
            device=key[0], dtype=dtype
        )
        cached = cached.reshape(1, 1, *VALID_CELL_MASK.shape)
        _MASK_CACHE[key] = cached
    return cached


def batch_to_tensors(batch: Batch, device: torch.device) -> BatchTensors:
    """Copy a `Batch` onto `device`. No dtype coercion — the batch contract in
    `model.batch` already pins them, and a silent cast here would hide a bug."""

    def to(arr: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(arr)).to(device)

    return BatchTensors(
        planes=to(batch.planes),
        policy=to(batch.policy),
        legal_mask=to(batch.legal_mask),
        policy_weight=to(batch.policy_weight),
        value=to(batch.value),
        score=to(batch.score),
        capture_map=to(batch.capture_map),
        q=to(batch.q),
    )


# ----- individual loss terms -------------------------------------------------


def masked_policy_logits(policy_logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    """Illegal entries replaced by `ILLEGAL_LOGIT` (finite, not `-inf`)."""
    return torch.where(
        legal_mask > 0, policy_logits, torch.full_like(policy_logits, ILLEGAL_LOGIT)
    )


def masked_log_softmax(policy_logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    """`log_softmax` over the legal moves only. Shared with `model.validate` so
    training and evaluation cannot drift apart on the masking convention."""
    return F.log_softmax(masked_policy_logits(policy_logits, legal_mask), dim=1)


def policy_cross_entropy_rows(
    policy_logits: torch.Tensor, policy_target: torch.Tensor, legal_mask: torch.Tensor
) -> torch.Tensor:
    """Per-row `-Σ π_i log p_i`, shape `(B,)`. Unweighted, unreduced."""
    log_probs = masked_log_softmax(policy_logits, legal_mask)
    return -(policy_target * log_probs).sum(dim=1)


def policy_loss(
    policy_logits: torch.Tensor,
    policy_target: torch.Tensor,
    legal_mask: torch.Tensor,
    policy_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Weighted policy cross-entropy, normalised by Σ weights.

    Returns `(loss, weight_sum)`. When `weight_sum` is 0 the numerator is 0 too,
    so the loss is exactly 0 and every gradient through it vanishes — no NaN.
    """
    rows = policy_cross_entropy_rows(policy_logits, policy_target, legal_mask)
    weight_sum = policy_weight.sum()
    numer = (policy_weight * rows).sum()
    return numer / weight_sum.clamp_min(_WEIGHT_EPS), weight_sum


def value_loss(value_logits: torch.Tensor, value_target: torch.Tensor) -> torch.Tensor:
    """3-way cross-entropy against the outcome class. No q blend — see module doc."""
    return F.cross_entropy(value_logits, value_target)


def score_loss(score_logits: torch.Tensor, score_target: torch.Tensor) -> torch.Tensor:
    """13-way cross-entropy against the final capture-differential class."""
    return F.cross_entropy(score_logits, score_target)


def capture_map_loss(
    capture_map_logits: torch.Tensor, capture_map_target: torch.Tensor
) -> torch.Tensor:
    """Masked BCE-with-logits over the 61 on-board cells.

    Normalised by the number of *masked* elements (`B · 2 · 61`), so the term's
    scale is a per-cell mean and does not depend on how many dead slots the 9×9
    grid happens to have. Off-board predictions and targets cannot influence it.
    """
    mask = valid_cell_mask(capture_map_logits.device, capture_map_logits.dtype)
    elementwise = F.binary_cross_entropy_with_logits(
        capture_map_logits, capture_map_target, reduction="none"
    )
    n_elements = capture_map_logits.shape[0] * capture_map_logits.shape[1] * NUM_VALID_CELLS
    if n_elements == 0:
        return capture_map_logits.sum() * 0.0
    return (elementwise * mask).sum() / n_elements


# ----- the whole loss --------------------------------------------------------


def compute_losses(
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    targets: BatchTensors,
    weights: LossWeights = DEFAULT_LOSS_WEIGHTS,
) -> LossTerms:
    """The full four-headed loss. `outputs` is `AbaloneNet.forward`'s tuple, in
    its documented order: (policy, value, score, capture_map) logits."""
    policy_logits, value_logits, score_logits, capture_map_logits = outputs

    l_policy, weight_sum = policy_loss(
        policy_logits, targets.policy, targets.legal_mask, targets.policy_weight
    )
    l_value = value_loss(value_logits, targets.value)
    l_score = score_loss(score_logits, targets.score)
    l_capture = capture_map_loss(capture_map_logits, targets.capture_map)

    total = (
        l_policy
        + weights.value * l_value
        + weights.score * l_score
        + weights.capture_map * l_capture
    )
    return LossTerms(
        total=total,
        policy=l_policy,
        value=l_value,
        score=l_score,
        capture_map=l_capture,
        policy_weight_sum=weight_sum,
    )


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Batch,
    *,
    device: torch.device,
    weights: LossWeights = DEFAULT_LOSS_WEIGHTS,
    grad_clip: float | None = 1.0,
) -> StepMetrics:
    """Forward, backward, clip, step. Returns the per-step scalars.

    `optimizer` is constructed by the caller — `AdamW` in this project, whose
    decoupled weight decay is why no L2 term appears in the loss.
    """
    targets = batch_to_tensors(batch, device)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    outputs = model(targets.planes)
    terms = compute_losses(outputs, targets, weights)

    if not torch.isfinite(terms.total):
        raise RuntimeError(
            "non-finite training loss: "
            f"total={terms.total.item()}, policy={terms.policy.item()}, "
            f"value={terms.value.item()}, score={terms.score.item()}, "
            f"capture_map={terms.capture_map.item()}. "
            "Training aborted to avoid corrupting the model."
        )

    terms.total.backward()

    if grad_clip is not None:
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip))
    else:
        grad_norm = float(
            sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
        )

    optimizer.step()

    return StepMetrics(
        loss_total=float(terms.total.detach()),
        loss_policy=float(terms.policy.detach()),
        loss_value=float(terms.value.detach()),
        loss_score=float(terms.score.detach()),
        loss_capture_map=float(terms.capture_map.detach()),
        grad_norm=grad_norm,
        policy_weight_sum=float(terms.policy_weight_sum.detach()),
    )
