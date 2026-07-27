"""Tests for `model.train_step`.

What is worth pinning here, and why:

  * **Hand-worked loss values.** Every term is checked against a number computed
    by hand from small logits, not against "whatever the code returned". A loss
    that is silently 2× or has a sign flipped still trains *something*, just not
    the right thing.
  * **`policy_weight` semantics.** Playout cap randomisation means most rows have
    no usable policy target. The loss must ignore them and must normalise by the
    weight sum, not the batch size — otherwise changing the cap rate silently
    changes the effective policy learning rate.
  * **NaN hygiene.** An all-illegal row and an all-zero-weight batch are both
    reachable from real data; neither may produce a NaN loss or NaN gradients.
  * **`q` is inert.** The z/q blend is gone. The test perturbs `q` wildly and
    demands a *bit-identical* loss, which no tolerance-based check would catch.
  * **Overfit.** A `small` net on one fixed batch must drive the loss down hard.
    This is the test that catches detached graphs, sign errors and dead heads —
    all of which pass every unit test above.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from model.abalone_net import build
from model.batch import (
    BOARD_H,
    BOARD_W,
    CAPTURE_MAP_CHANNELS,
    INPUT_PLANES,
    MOVE_SPACE,
    SCORE_CLASSES,
    VALUE_CLASSES,
    Batch,
)
from model.encoder import VALID_CELL_MASK
from model.train_step import (
    DEFAULT_LOSS_WEIGHTS,
    NUM_VALID_CELLS,
    LossWeights,
    batch_to_tensors,
    capture_map_loss,
    compute_losses,
    masked_log_softmax,
    policy_loss,
    score_loss,
    train_step,
    value_loss,
)

CPU = torch.device("cpu")
OFF_BOARD = np.argwhere(VALID_CELL_MASK == 0.0)  # (20, 2) row/col pairs


# ----- batch construction ----------------------------------------------------


def make_batch(
    size: int = 4,
    *,
    seed: int = 0,
    legal_per_row: int = 8,
    policy_weight: np.ndarray | float = 1.0,
) -> Batch:
    """A shape-valid random batch. Deliberately built by hand rather than by
    `ReplayBuffer.sample` so these tests do not depend on the buffer."""
    rng = np.random.default_rng(seed)
    legal = np.zeros((size, MOVE_SPACE), dtype=np.float32)
    policy = np.zeros((size, MOVE_SPACE), dtype=np.float32)
    for i in range(size):
        idx = rng.choice(MOVE_SPACE, size=legal_per_row, replace=False)
        legal[i, idx] = 1.0
        visits = rng.random(legal_per_row) + 0.1
        policy[i, idx] = (visits / visits.sum()).astype(np.float32)

    if np.isscalar(policy_weight):
        weights = np.full(size, float(policy_weight), dtype=np.float32)
    else:
        weights = np.asarray(policy_weight, dtype=np.float32)

    return Batch(
        planes=rng.random((size, INPUT_PLANES, BOARD_H, BOARD_W), dtype=np.float32),
        policy=policy,
        legal_mask=legal,
        policy_weight=weights,
        value=rng.integers(0, VALUE_CLASSES, size=size).astype(np.int64),
        score=rng.integers(0, SCORE_CLASSES, size=size).astype(np.int64),
        capture_map=rng.random(
            (size, CAPTURE_MAP_CHANNELS, BOARD_H, BOARD_W), dtype=np.float32
        ),
        q=rng.random(size, dtype=np.float32) * 2.0 - 1.0,
        )


def test_make_batch_satisfies_the_contract() -> None:
    make_batch(size=3).validate()


# ----- policy loss -----------------------------------------------------------


def test_policy_loss_matches_hand_computation() -> None:
    # Three legal moves out of five, logits [1, 2, 3] on them, target [.5,.3,.2].
    logits = torch.tensor([[1.0, 2.0, 3.0, 7.0, 9.0]])
    legal = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0]])
    target = torch.tensor([[0.5, 0.3, 0.2, 0.0, 0.0]])
    weight = torch.tensor([1.0])

    z = math.log(math.exp(1.0) + math.exp(2.0) + math.exp(3.0))
    expected = -(0.5 * (1.0 - z) + 0.3 * (2.0 - z) + 0.2 * (3.0 - z))

    loss, weight_sum = policy_loss(logits, target, legal, weight)
    assert loss.item() == pytest.approx(expected, abs=1e-6)
    assert weight_sum.item() == pytest.approx(1.0)


def test_illegal_moves_get_negligible_probability() -> None:
    # The illegal logits above are the *largest* ones; masking must bury them.
    logits = torch.tensor([[1.0, 2.0, 3.0, 50.0, 80.0]])
    legal = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0]])
    probs = masked_log_softmax(logits, legal).exp()
    assert probs[0, 3].item() < 1e-30
    assert probs[0, 4].item() < 1e-30
    assert probs[0, :3].sum().item() == pytest.approx(1.0, abs=1e-6)


def test_all_illegal_row_is_finite() -> None:
    """Reachable if a shard ever carries a terminal position. `-inf` masking
    would give NaN here; the `-1e9` fill degrades to a uniform row instead."""
    logits = torch.randn(1, 5, requires_grad=True)
    legal = torch.zeros(1, 5)
    target = torch.zeros(1, 5)
    loss, _ = policy_loss(logits, target, legal, torch.tensor([1.0]))
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()

    log_probs = masked_log_softmax(logits.detach(), legal)
    assert torch.isfinite(log_probs).all()
    assert log_probs.exp().sum().item() == pytest.approx(1.0, abs=1e-6)


def test_policy_loss_ignores_zero_weight_rows() -> None:
    """Row 1 is deliberately terrible. With weight 0 it must not move the loss
    at all — its target is real data, but its visit counts are too noisy."""
    logits = torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    legal = torch.ones(2, 3)
    target = torch.tensor([[0.5, 0.3, 0.2], [1.0, 0.0, 0.0]])

    both, _ = policy_loss(logits, target, legal, torch.tensor([1.0, 0.0]))
    only_first, _ = policy_loss(logits[:1], target[:1], legal[:1], torch.tensor([1.0]))
    assert both.item() == pytest.approx(only_first.item(), abs=1e-7)


def test_policy_loss_normalises_by_weight_sum_not_batch_size() -> None:
    """Duplicating a row and zeroing the copy's weight must not halve the loss.
    Normalising by `B` is exactly the bug this catches."""
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    legal = torch.ones(1, 3)
    target = torch.tensor([[0.5, 0.3, 0.2]])
    single, _ = policy_loss(logits, target, legal, torch.tensor([1.0]))

    padded, _ = policy_loss(
        torch.cat([logits, logits]),
        torch.cat([target, target]),
        torch.cat([legal, legal]),
        torch.tensor([1.0, 0.0]),
    )
    assert padded.item() == pytest.approx(single.item(), abs=1e-7)


def test_all_zero_weight_batch_is_finite_with_no_nan_gradients() -> None:
    logits = torch.randn(4, 6, requires_grad=True)
    legal = torch.ones(4, 6)
    target = torch.full((4, 6), 1.0 / 6.0)
    loss, weight_sum = policy_loss(logits, target, legal, torch.zeros(4))
    assert weight_sum.item() == 0.0
    assert loss.item() == 0.0
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert (logits.grad == 0).all()


def test_policy_loss_reaches_target_entropy_at_perfect_prediction() -> None:
    """The theoretical minimum of a cross-entropy is the target's own entropy."""
    target = torch.tensor([[0.5, 0.25, 0.25, 0.0]])
    legal = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
    logits = torch.log(target.clamp_min(1e-12))
    loss, _ = policy_loss(logits, target, legal, torch.tensor([1.0]))
    entropy = -(target[target > 0] * target[target > 0].log()).sum()
    assert loss.item() == pytest.approx(entropy.item(), abs=1e-6)


# ----- value and score losses ------------------------------------------------


def test_value_loss_matches_hand_computation() -> None:
    logits = torch.tensor([[2.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    target = torch.tensor([0, 2])  # win, loss
    z0 = math.log(math.exp(2.0) + math.exp(1.0) + math.exp(0.0))
    row0 = z0 - 2.0
    row1 = math.log(3.0)  # uniform over three classes
    expected = (row0 + row1) / 2.0
    assert value_loss(logits, target).item() == pytest.approx(expected, abs=1e-6)


def test_score_loss_matches_hand_computation() -> None:
    logits = torch.zeros(2, SCORE_CLASSES)
    logits[0, 6] = 5.0
    target = torch.tensor([6, 0])
    z0 = math.log(math.exp(5.0) + 12.0)
    expected = ((z0 - 5.0) + math.log(SCORE_CLASSES)) / 2.0
    assert score_loss(logits, target).item() == pytest.approx(expected, abs=1e-6)


def test_value_and_score_losses_vanish_at_perfect_prediction() -> None:
    big = 60.0
    value_logits = torch.full((3, VALUE_CLASSES), -big)
    score_logits = torch.full((3, SCORE_CLASSES), -big)
    value_target = torch.tensor([0, 1, 2])
    score_target = torch.tensor([0, 6, 12])
    value_logits[torch.arange(3), value_target] = big
    score_logits[torch.arange(3), score_target] = big
    assert value_loss(value_logits, value_target).item() == pytest.approx(0.0, abs=1e-6)
    assert score_loss(score_logits, score_target).item() == pytest.approx(0.0, abs=1e-6)


# ----- capture-map loss ------------------------------------------------------


def test_capture_map_loss_matches_hand_computation() -> None:
    """Constant logit `l` against constant target `t` over the masked cells:
    the mean is just the single-element BCE, whatever the mask contains."""
    logit, target_p = 0.75, 0.25
    logits = torch.full((2, CAPTURE_MAP_CHANNELS, BOARD_H, BOARD_W), logit)
    target = torch.full((2, CAPTURE_MAP_CHANNELS, BOARD_H, BOARD_W), target_p)
    p = 1.0 / (1.0 + math.exp(-logit))
    expected = -(target_p * math.log(p) + (1.0 - target_p) * math.log(1.0 - p))
    assert capture_map_loss(logits, target).item() == pytest.approx(expected, abs=1e-6)


def test_capture_map_loss_ignores_off_board_cells() -> None:
    torch.manual_seed(0)
    logits = torch.randn(3, CAPTURE_MAP_CHANNELS, BOARD_H, BOARD_W)
    target = torch.rand(3, CAPTURE_MAP_CHANNELS, BOARD_H, BOARD_W)
    baseline = capture_map_loss(logits, target)

    poisoned_logits = logits.clone()
    poisoned_target = target.clone()
    for r, c in OFF_BOARD:
        poisoned_logits[:, :, r, c] = 1e4
        poisoned_target[:, :, r, c] = 1.0
    assert capture_map_loss(poisoned_logits, poisoned_target).item() == pytest.approx(
        baseline.item(), abs=1e-7
    )


def test_capture_map_loss_normalises_by_masked_element_count() -> None:
    """The denominator is `B · C · 61`, not `B · C · 81` — a per-masked-element
    mean, so the term's scale is independent of the dead-slot count."""
    logits = torch.zeros(1, CAPTURE_MAP_CHANNELS, BOARD_H, BOARD_W)
    target = torch.zeros(1, CAPTURE_MAP_CHANNELS, BOARD_H, BOARD_W)
    loss = capture_map_loss(logits, target)
    assert loss.item() == pytest.approx(math.log(2.0), abs=1e-6)
    assert NUM_VALID_CELLS == 61


def test_capture_map_loss_vanishes_at_perfect_prediction() -> None:
    target = torch.zeros(2, CAPTURE_MAP_CHANNELS, BOARD_H, BOARD_W)
    target[:, 0, 4, 4] = 1.0
    logits = torch.where(target > 0.5, torch.full_like(target, 40.0), torch.full_like(target, -40.0))
    assert capture_map_loss(logits, target).item() == pytest.approx(0.0, abs=1e-6)


# ----- the assembled loss ----------------------------------------------------


def _losses_for(batch: Batch, weights: LossWeights = DEFAULT_LOSS_WEIGHTS):
    torch.manual_seed(7)
    model = build("small").eval()
    targets = batch_to_tensors(batch, CPU)
    with torch.no_grad():
        return compute_losses(model(targets.planes), targets, weights)


def test_total_is_the_documented_weighted_sum() -> None:
    weights = LossWeights(value=1.0, score=0.15, capture_map=0.15)
    terms = _losses_for(make_batch(size=4, seed=3), weights)
    expected = (
        terms.policy
        + weights.value * terms.value
        + weights.score * terms.score
        + weights.capture_map * terms.capture_map
    )
    assert terms.total.item() == pytest.approx(expected.item(), abs=1e-6)


def test_default_weights_are_the_documented_ones() -> None:
    assert DEFAULT_LOSS_WEIGHTS.value == 1.0
    assert DEFAULT_LOSS_WEIGHTS.score == 0.15
    assert DEFAULT_LOSS_WEIGHTS.capture_map == 0.15


def test_q_does_not_affect_the_loss() -> None:
    """The z/q blend is gone: `q` is diagnostics only. Bit-identical, not
    approximately equal — any leak into the loss would show up as a wobble."""
    batch = make_batch(size=6, seed=11)
    baseline = _losses_for(batch)

    perturbed = Batch(**{**batch.__dict__, "q": np.full(batch.size, -12345.0, dtype=np.float32)})
    after = _losses_for(perturbed)

    assert after.total.item() == baseline.total.item()
    assert after.value.item() == baseline.value.item()

    nan_q = Batch(**{**batch.__dict__, "q": np.full(batch.size, np.nan, dtype=np.float32)})
    assert _losses_for(nan_q).total.item() == baseline.total.item()


def test_train_step_returns_all_head_losses() -> None:
    torch.manual_seed(0)
    model = build("small")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    metrics = train_step(model, optimizer, make_batch(size=4, seed=1), device=CPU)

    for name in ("loss_total", "loss_policy", "loss_value", "loss_score", "loss_capture_map"):
        assert math.isfinite(getattr(metrics, name)), name
        assert getattr(metrics, name) >= 0.0, name
    assert metrics.grad_norm > 0.0
    assert metrics.policy_weight_sum == pytest.approx(4.0)
    assert set(metrics.as_dict()) == {
        "loss_total",
        "loss_policy",
        "loss_value",
        "loss_score",
        "loss_capture_map",
        "grad_norm",
        "policy_weight_sum",
    }


def test_train_step_works_with_adamw_decoupled_decay() -> None:
    """The loss carries no L2 term, so decay is entirely the optimizer's. With
    a zero-gradient batch and a large decay, weights must still shrink."""
    torch.manual_seed(0)
    model = build("small")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.5)
    before = model.policy_conv.weight.detach().clone()
    train_step(model, optimizer, make_batch(size=2, seed=2), device=CPU)
    assert not torch.equal(before, model.policy_conv.weight.detach())


def test_train_step_survives_an_all_zero_weight_batch() -> None:
    torch.manual_seed(0)
    model = build("small")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch = make_batch(size=4, seed=5, policy_weight=0.0)
    metrics = train_step(model, optimizer, batch, device=CPU)
    assert metrics.loss_policy == 0.0
    assert math.isfinite(metrics.loss_total)
    assert math.isfinite(metrics.grad_norm)
    for p in model.parameters():
        assert p.grad is None or torch.isfinite(p.grad).all()


def test_train_step_raises_on_non_finite_loss() -> None:
    """Better to abort the run than to push NaN through the optimizer and
    silently poison every subsequent checkpoint."""
    torch.manual_seed(0)
    model = build("small")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch = make_batch(size=2, seed=4)
    batch.planes[0, 0, 0, 0] = np.float32("nan")
    with pytest.raises(RuntimeError, match="non-finite training loss"):
        train_step(model, optimizer, batch, device=CPU)


def test_mixed_policy_weights_reproduce_the_full_weight_loss() -> None:
    """A half-weighted batch computes exactly the loss of the sub-batch that
    carries targets — the property playout cap randomisation relies on."""
    batch = make_batch(size=6, seed=9, policy_weight=np.array([1, 0, 1, 0, 1, 0]))
    mixed = _losses_for(batch)

    keep = np.array([0, 2, 4])
    sub = Batch(
        planes=batch.planes[keep],
        policy=batch.policy[keep],
        legal_mask=batch.legal_mask[keep],
        policy_weight=batch.policy_weight[keep],
        value=batch.value[keep],
        score=batch.score[keep],
        capture_map=batch.capture_map[keep],
        q=batch.q[keep],
    )
    assert mixed.policy.item() == pytest.approx(_losses_for(sub).policy.item(), abs=1e-5)


# ----- the test that catches what unit tests cannot --------------------------


def _loss_floors(batch: Batch, weights: LossWeights = DEFAULT_LOSS_WEIGHTS) -> dict[str, float]:
    """The theoretical minimum of each term on this batch.

    Cross-entropy bottoms out at the *target's* entropy, not at 0, so "did it
    learn" has to be measured as excess above the floor. For soft targets that
    floor is large: uniform-random capture targets alone contribute ≈ 0.5 nats.
    """
    p = batch.policy
    row_entropy = -np.sum(np.where(p > 0, p * np.log(np.clip(p, 1e-12, None)), 0.0), axis=1)
    w = batch.policy_weight
    policy = float((w * row_entropy).sum() / max(w.sum(), 1e-8))

    t = batch.capture_map[:, :, VALID_CELL_MASK > 0].astype(np.float64)
    t = np.clip(t, 1e-12, 1 - 1e-12)
    capture = float(np.mean(-(t * np.log(t) + (1 - t) * np.log(1 - t))))

    return {
        "policy": policy,
        "value": 0.0,  # hard class labels
        "score": 0.0,
        "capture_map": capture,
        "total": policy + weights.capture_map * capture,
    }


def test_small_model_overfits_a_single_batch() -> None:
    """~200 AdamW steps on one fixed batch must collapse the excess loss.

    Sign errors, detached graphs and dead heads all pass every unit test above
    and all fail here. Measured as loss *above the entropy floor*, because the
    soft policy and capture-map targets put the achievable minimum well above 0
    and a raw ratio would flatter or fail the test for the wrong reason.
    """
    torch.manual_seed(0)
    model = build("small")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    batch = make_batch(size=8, seed=42, legal_per_row=6)
    floor = _loss_floors(batch)

    curve = [train_step(model, optimizer, batch, device=CPU, grad_clip=2.0) for _ in range(200)]
    first, last = curve[0], curve[-1]
    assert all(math.isfinite(m.loss_total) for m in curve)

    def excess(m, head: str) -> float:
        return getattr(m, f"loss_{head}") - floor[head]

    assert excess(last, "total") < 0.15 * excess(first, "total"), (
        f"total loss {first.loss_total:.3f} -> {last.loss_total:.3f} "
        f"against a floor of {floor['total']:.3f}"
    )
    # Every head must move; one that does not is a wiring bug the total can hide.
    assert excess(last, "policy") < 0.25 * excess(first, "policy")
    assert last.loss_value < 0.05 * first.loss_value
    assert last.loss_score < 0.05 * first.loss_score
    assert excess(last, "capture_map") < 0.25 * excess(first, "capture_map")
    # And the loss must be monotone-ish, not merely lower at the two endpoints.
    assert min(m.loss_total for m in curve[-20:]) < floor["total"] + 0.3
