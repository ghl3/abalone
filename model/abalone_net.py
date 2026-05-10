"""Abalone policy+value network: 4-block × 64-channel ResNet.

Input: `(B, 6, 9, 9)` float32 planes (see `encoder.py` for the layout).
Outputs:
    policy_logits: `(B, 2562)`  — flat over the move-index space.
    value:         `(B, 1)`     — `tanh`-bounded eval from side-to-move POV.

The 6 channels are arranged so that "side to move" is *always* in the
first channel; this lets the same network play both colors without a
side-to-move embedding. Captures and ply are scalar features broadcast
to full 9×9 planes for spatial conditioning.

Architecture sized for fast training on M1 MPS and small ONNX export
(target <2MB compressed for the browser).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----- shape constants -------------------------------------------------------

INPUT_CHANNELS = 6
BOARD_H = 9
BOARD_W = 9

NUM_RES_BLOCKS = 4
CHANNELS = 64

# Width of the 1×1 conv at the start of the policy head. The
# subsequent FC layer is `POLICY_HEAD_CHANNELS * 9 * 9 → MOVE_SPACE`,
# i.e. (~81C, 2562), which dominates the model's parameter count.
# 16 channels → ~3.6M total params (vs ~7M for 32 channels), with
# roughly 20% faster inference and minimal capacity loss on this domain.
POLICY_HEAD_CHANNELS = 16
VALUE_HEAD_CHANNELS = 1
VALUE_HIDDEN = 256

# Must match `abalone_game::move_index::MOVE_SPACE`. Hardcoded here
# because there's no compelling reason to plumb it through config.
MOVE_SPACE = 2562


# ----- building blocks -------------------------------------------------------


class ResidualBlock(nn.Module):
    """Standard pre-AlphaZero residual block: conv-bn-relu-conv-bn,
    residual add, relu. Stride 1 throughout, so spatial dims are
    preserved at 9×9."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        out = F.relu(out)
        return out


# ----- the model ------------------------------------------------------------


class AbaloneNet(nn.Module):
    def __init__(
        self,
        input_channels: int = INPUT_CHANNELS,
        blocks: int = NUM_RES_BLOCKS,
        channels: int = CHANNELS,
        policy_head_channels: int = POLICY_HEAD_CHANNELS,
        value_head_channels: int = VALUE_HEAD_CHANNELS,
        value_hidden: int = VALUE_HIDDEN,
        move_space: int = MOVE_SPACE,
    ) -> None:
        super().__init__()

        # Stem: lift 6 input channels to working width.
        self.stem_conv = nn.Conv2d(input_channels, channels, kernel_size=3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(channels)

        # Tower.
        self.blocks = nn.ModuleList(ResidualBlock(channels) for _ in range(blocks))

        # Policy head: 1×1 to small width, flatten, dense to MOVE_SPACE.
        self.policy_conv = nn.Conv2d(channels, policy_head_channels, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(policy_head_channels)
        self.policy_fc = nn.Linear(policy_head_channels * BOARD_H * BOARD_W, move_space)

        # Value head: 1×1 to 1 channel, flatten, MLP to scalar.
        self.value_conv = nn.Conv2d(channels, value_head_channels, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(value_head_channels)
        self.value_fc1 = nn.Linear(value_head_channels * BOARD_H * BOARD_W, value_hidden)
        self.value_fc2 = nn.Linear(value_hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Stem.
        h = F.relu(self.stem_bn(self.stem_conv(x)))
        # Tower.
        for block in self.blocks:
            h = block(h)
        # Policy.
        p = F.relu(self.policy_bn(self.policy_conv(h)))
        p = p.flatten(1)
        policy_logits = self.policy_fc(p)
        # Value.
        v = F.relu(self.value_bn(self.value_conv(h)))
        v = v.flatten(1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))
        return policy_logits, v

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_default() -> AbaloneNet:
    return AbaloneNet()


if __name__ == "__main__":
    # Smoke test: build the model and run a dummy forward.
    net = build_default()
    print(f"AbaloneNet: {net.num_parameters():,} params")
    x = torch.zeros(2, INPUT_CHANNELS, BOARD_H, BOARD_W)
    logits, v = net(x)
    print(f"forward: logits {tuple(logits.shape)}, value {tuple(v.shape)}")
    assert logits.shape == (2, MOVE_SPACE)
    assert v.shape == (2, 1)
