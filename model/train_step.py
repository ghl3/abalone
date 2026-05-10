"""Single SGD step: forward, masked policy cross-entropy, blended value
loss, backward, optimizer step. Returns per-step metrics.

Loss formulation (AlphaZero-style with our IQ-blended value target):

    policy_loss = - sum_i pi_i * log_softmax(logits)_i  (over legal i)
    value_target = z_weight * z + (1 - z_weight) * q
    value_loss = MSE(v, value_target)
    total = policy_loss + value_loss_weight * value_loss

The legal-move mask is applied to the logits *before* softmax by
setting illegal entries to a large negative number (`-1e9`); softmax
then puts ~zero probability on them.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.replay_buffer import Batch


@dataclass
class StepMetrics:
    loss_total: float
    loss_policy: float
    loss_value: float
    grad_norm: float


def value_target_blend_weight(gen: int, blend_start: float, blend_end: float, done_by_gen: int) -> float:
    """Linear ramp on the weight applied to terminal-z. At `gen=0` the
    target is `blend_start * z + (1 - blend_start) * q`; by `done_by_gen`
    it's `blend_end * z + (1 - blend_end) * q`."""
    if done_by_gen <= 0:
        return blend_end
    t = max(0.0, min(1.0, gen / done_by_gen))
    return blend_start + t * (blend_end - blend_start)


def batch_to_tensors(
    batch: Batch, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    planes = torch.from_numpy(batch.planes).to(device)
    policy = torch.from_numpy(batch.policy).to(device)
    legal_mask = torch.from_numpy(batch.legal_mask).to(device)
    z = torch.from_numpy(batch.z).to(device)
    q = torch.from_numpy(batch.q).to(device)
    return planes, policy, legal_mask, z, q


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Batch,
    *,
    device: torch.device,
    value_loss_weight: float,
    z_weight: float,
    grad_clip: float | None = 1.0,
) -> StepMetrics:
    planes, policy_target, legal_mask, z, q = batch_to_tensors(batch, device)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    policy_logits, value = model(planes)

    # Mask illegal moves before softmax. Use -1e9 instead of -inf to
    # avoid NaN gradients when an entire row's mask is zero (shouldn't
    # happen for valid training data, but defensive).
    masked_logits = torch.where(
        legal_mask > 0, policy_logits, torch.full_like(policy_logits, -1e9)
    )
    log_probs = F.log_softmax(masked_logits, dim=1)
    # Cross-entropy: -sum(pi * log p). pi is already zero on illegal
    # moves (from the buffer), so the multiplication suffices.
    policy_loss = -(policy_target * log_probs).sum(dim=1).mean()

    value_target = z_weight * z + (1.0 - z_weight) * q
    value_pred = value.squeeze(-1)
    value_loss = F.mse_loss(value_pred, value_target)

    total = policy_loss + value_loss_weight * value_loss
    if not torch.isfinite(total):
        raise RuntimeError(
            f"non-finite training loss: total={total.item()}, "
            f"policy={policy_loss.item()}, value={value_loss.item()}. "
            f"Training aborted to avoid corrupting the model."
        )
    total.backward()

    if grad_clip is not None:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip).item()
    else:
        grad_norm = float(
            sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None)
            ** 0.5
        )

    optimizer.step()

    return StepMetrics(
        loss_total=float(total.detach().cpu()),
        loss_policy=float(policy_loss.detach().cpu()),
        loss_value=float(value_loss.detach().cpu()),
        grad_norm=float(grad_norm),
    )
