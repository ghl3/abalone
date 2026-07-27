"""Abalone policy / value / score / capture-map network (MODEL.md §5–§6).

Input: `(B, 14, 9, 9)` float32 planes — see `encoder.py` and MODEL.md §5.
The 9×9 axial grid holds 61 valid cells; the other 20 slots are off-board.

`forward` returns a plain tuple in this **fixed order**::

    (policy_logits, value_logits, score_logits, capture_map_logits)
     (B, 2562)      (B, 3)        (B, 13)       (B, 2, 9, 9)

* `policy_logits`   — flat over the move-index space. Unmasked, unnormalised;
  consumers apply the legal mask and the softmax.
* `value_logits`    — **who wins**: logits over `(win, draw, loss)` from the
  side-to-move's POV. This is the RL value function MCTS backs up, via
  `E[value] = P(win) − P(loss)`.
* `score_logits`    — **by how much**: logits over the final capture
  differential `d ∈ [−6, +6]` (13 buckets, index `d + 6`). Auxiliary.
* `capture_map_logits` — raw (pre-sigmoid) `(2, 9, 9)` map. Channel 0 = own
  losses, channel 1 = opponent losses. The sigmoid lives in the loss, not
  here, so training can use `binary_cross_entropy_with_logits`.

Design notes worth not "fixing":

**Plain `Conv2d` is correct on the hex board.** In axial coordinates all six
hex neighbours land inside a standard 3×3 kernel: `(0, ±1)` = E/W,
`(±1, ±1)` = NE/SW, `(±1, 0)` = NW/SE. The two remaining kernel corners are
distance-2 non-neighbours that the network learns to ignore. No custom hex
convolution is needed.

**Off-board cells are masked everywhere.** The input is masked on entry, and
activations are multiplied by the valid-cell mask after the stem and after
every residual block, so signal cannot accumulate in the 20 dead slots or
leak into a valid cell through a 3×3 kernel. Masking the *input* matters as
much as masking the blocks: a 3×3 kernel at a valid cell reads its off-board
neighbours, so without it, junk in a dead slot would change on-board outputs.

**The policy head is convolutional, not dense.** The move space *is* a
`(42, 9, 9)` tensor read out through a fixed index table — see
`model/policy_map.py`. The previous dense `16·81 → 2562` head held 92% of all
parameters (3.32 M); the conv head costs ~48 k, which frees the whole budget
for the trunk where representation quality actually comes from.

**The value/score head pools globally on purpose.** A position's value is
invariant under the D6 board symmetry, so a masked mean+max pool gets that
invariance structurally instead of spending capacity learning it. The max
pool masks off-board cells to `-inf`, not 0, so a dead slot can never win the
max.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.encoder import VALID_CELL_MASK
from model.policy_map import POLICY_GATHER, POLICY_PLANES

# ----- shape constants -------------------------------------------------------

INPUT_CHANNELS = 14
BOARD_H = 9
BOARD_W = 9

MOVE_SPACE = 2562  # matches abalone_game::move_index::MOVE_SPACE
VALUE_BUCKETS = 3  # (win, draw, loss)
SCORE_BUCKETS = 13  # capture differential d ∈ [−6, +6]
CAPTURE_MAP_CHANNELS = 2  # 0 = own losses, 1 = opponent losses

VS_HEAD_CHANNELS = 32  # width of the shared value/score 1×1 conv
VS_HIDDEN = 128  # width of the shared value/score MLP
SE_REDUCTION = 8


# ----- configuration ---------------------------------------------------------


@dataclass(frozen=True)
class NetConfig:
    """Trunk shape. Heads are fixed by the game, only the trunk scales."""

    blocks: int
    channels: int
    se: bool = False

    def __str__(self) -> str:  # e.g. "10×128" or "10×128+se"
        return f"{self.blocks}×{self.channels}" + ("+se" if self.se else "")


PRESETS: dict[str, NetConfig] = {
    # ~1.1 M params — fast iteration, CI, smoke runs.
    "small": NetConfig(blocks=6, channels=96),
    # ~3.0 M params — default training runs.
    "base": NetConfig(blocks=10, channels=128),
    # ~9.4 M params — final run, if throughput allows.
    "large": NetConfig(blocks=14, channels=192),
}

DEFAULT_PRESET = "base"
DEFAULT_CONFIG = PRESETS[DEFAULT_PRESET]

# Convenience aliases for callers that just want the default shape.
NUM_RES_BLOCKS = DEFAULT_CONFIG.blocks
CHANNELS = DEFAULT_CONFIG.channels


# ----- building blocks -------------------------------------------------------


class SqueezeExcite(nn.Module):
    """Masked global context → per-channel gate.

    Abalone's most important state is partly *global* (capture counters,
    material, "am I one push from losing"). SE lets pooled context modulate
    channels directly instead of forcing it to propagate spatially through
    the trunk. Off by default; see MODEL.md Q3.
    """

    def __init__(self, channels: int, reduction: int = SE_REDUCTION) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Mean over the 61 valid cells only — dead slots must not dilute it.
        pooled = (x * mask).sum(dim=(2, 3)) / mask.sum()
        gate = torch.sigmoid(self.fc2(F.relu(self.fc1(pooled))))
        return x * gate.unsqueeze(-1).unsqueeze(-1)


class ResidualBlock(nn.Module):
    """conv-bn-relu-conv-bn, (optional SE), residual add, relu.

    Stride 1 throughout, so spatial dims stay 9×9.
    """

    def __init__(self, channels: int, se: bool = False) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SqueezeExcite(channels) if se else None

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.se is not None:
            out = self.se(out, mask)
        return F.relu(out + x)


# ----- the model ------------------------------------------------------------


class AbaloneNet(nn.Module):
    """Residual trunk + four heads. See the module docstring for the contract."""

    def __init__(self, config: NetConfig = DEFAULT_CONFIG) -> None:
        super().__init__()
        self.config = config
        c = config.channels

        # The valid-cell mask is derived from board geometry, not learned, so
        # it is a non-persistent buffer: it moves with `.to(device)` but never
        # enters a checkpoint. Same for the policy gather table.
        mask = torch.from_numpy(np.array(VALID_CELL_MASK, dtype=np.float32)).reshape(
            1, 1, BOARD_H, BOARD_W
        )
        self.register_buffer("valid_mask", mask, persistent=False)
        self.register_buffer("off_board_mask", mask < 0.5, persistent=False)
        self.register_buffer(
            "policy_gather",
            torch.from_numpy(np.array(POLICY_GATHER, dtype=np.int64)),
            persistent=False,
        )

        # Stem: lift 14 input planes to the working width.
        self.stem_conv = nn.Conv2d(INPUT_CHANNELS, c, kernel_size=3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(c)

        # Trunk.
        self.blocks = nn.ModuleList(ResidualBlock(c, se=config.se) for _ in range(config.blocks))

        # Policy head: 3×3 so each anchor sees local context, then a fixed
        # gather onto the flat move space. ~48 k params at C = 128.
        self.policy_conv = nn.Conv2d(c, POLICY_PLANES, kernel_size=3, padding=1)

        # Value / score head: shared 1×1 → masked mean+max pool → MLP → two
        # softmax outputs. Pooling makes the head D6-invariant by construction.
        self.vs_conv = nn.Conv2d(c, VS_HEAD_CHANNELS, kernel_size=1, bias=False)
        self.vs_bn = nn.BatchNorm2d(VS_HEAD_CHANNELS)
        self.vs_fc = nn.Linear(2 * VS_HEAD_CHANNELS, VS_HIDDEN)
        self.value_fc = nn.Linear(VS_HIDDEN, VALUE_BUCKETS)
        self.score_fc = nn.Linear(VS_HIDDEN, SCORE_BUCKETS)

        # Capture-map head: raw per-cell logits, sigmoid applied in the loss.
        self.capture_conv = nn.Conv2d(c, CAPTURE_MAP_CHANNELS, kernel_size=1)

    # -- forward -------------------------------------------------------------

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """`(B, 14, 9, 9)` → `(policy_logits, value_logits, score_logits, capture_map_logits)`.

        Shapes: `(B, 2562)`, `(B, 3)`, `(B, 13)`, `(B, 2, 9, 9)`. No softmax,
        no sigmoid, no legal-masking — all of that belongs to the consumer.
        """
        mask = self.valid_mask

        # Mask on entry: a 3×3 kernel at a valid cell reads its off-board
        # neighbours, so unmasked junk in a dead slot would reach the board.
        h = x * mask
        h = F.relu(self.stem_bn(self.stem_conv(h))) * mask
        for block in self.blocks:
            h = block(h, mask) * mask

        return (
            self._policy(h),
            *self._value_and_score(h),
            self.capture_conv(h),
        )

    def _policy(self, h: torch.Tensor) -> torch.Tensor:
        p = self.policy_conv(h)  # (B, 42, 9, 9)
        p = p.flatten(1)  # (B, 3402), index = plane * 81 + cell
        return p.index_select(1, self.policy_gather)  # (B, 2562)

    def _value_and_score(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        v = F.relu(self.vs_bn(self.vs_conv(h)))
        # BN carries a bias, so the head's own activations need re-masking
        # before pooling even though `h` is already masked.
        v = v * self.valid_mask
        mean = v.sum(dim=(2, 3)) / self.valid_mask.sum()
        # −inf, not 0: a dead slot must never win the max. (61 cells are always
        # valid, so no row is all −inf and the amax is always finite.)
        peak = v.masked_fill(self.off_board_mask, float("-inf")).amax(dim=(2, 3))
        z = F.relu(self.vs_fc(torch.cat([mean, peak], dim=1)))  # (B, 128)
        return self.value_fc(z), self.score_fc(z)

    # -- introspection -------------------------------------------------------

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameter_breakdown(self) -> dict[str, int]:
        """Params per component. Used by the smoke test and by tests that
        assert the trunk — not the policy head — holds the budget."""
        groups = {
            "stem": ("stem_conv", "stem_bn"),
            "trunk": ("blocks",),
            "policy_head": ("policy_conv",),
            "value_score_head": ("vs_conv", "vs_bn", "vs_fc", "value_fc", "score_fc"),
            "capture_head": ("capture_conv",),
        }
        out = dict.fromkeys(groups, 0)
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            for group, prefixes in groups.items():
                if name.startswith(prefixes):
                    out[group] += param.numel()
                    break
            else:  # pragma: no cover - guards against an unclassified param
                raise AssertionError(f"parameter {name!r} not covered by parameter_breakdown")
        return out


# ----- constructors ---------------------------------------------------------


def build(config: NetConfig | str = DEFAULT_CONFIG) -> AbaloneNet:
    """Build a net from a `NetConfig` or a preset name (`small`/`base`/`large`)."""
    if isinstance(config, str):
        try:
            config = PRESETS[config]
        except KeyError:
            raise ValueError(
                f"unknown preset {config!r}; expected one of {sorted(PRESETS)}"
            ) from None
    return AbaloneNet(config)


def build_default() -> AbaloneNet:
    """The `base` preset: 10 × 128, ~3.0 M params."""
    return build(DEFAULT_CONFIG)


if __name__ == "__main__":
    # Smoke test: build every preset, print the parameter breakdown, and run
    # a dummy forward through the default one.
    for name in ("small", "base", "large"):
        net = build(name)
        total = net.num_parameters()
        parts = net.parameter_breakdown()
        print(f"{name:>5} {net.config!s:>9}  {total:>10,} params")
        for group, n in parts.items():
            print(f"        {group:<18} {n:>10,}  ({100.0 * n / total:5.1f}%)")

    net = build_default()
    net.eval()
    x = torch.zeros(2, INPUT_CHANNELS, BOARD_H, BOARD_W)
    policy, value, score, capture = net(x)
    print(
        "forward: "
        f"policy {tuple(policy.shape)}, value {tuple(value.shape)}, "
        f"score {tuple(score.shape)}, capture_map {tuple(capture.shape)}"
    )
    assert policy.shape == (2, MOVE_SPACE)
    assert value.shape == (2, VALUE_BUCKETS)
    assert score.shape == (2, SCORE_BUCKETS)
    assert capture.shape == (2, CAPTURE_MAP_CHANNELS, BOARD_H, BOARD_W)
    assert torch.isfinite(policy).all()
    assert torch.isfinite(value).all()
    assert torch.isfinite(score).all()
    assert torch.isfinite(capture).all()
