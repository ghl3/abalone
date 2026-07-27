"""Export a trained `AbaloneNet` to ONNX.

Signature (ARCHITECTURE.md §5.3, MODEL.md §6):

```
input   planes         (B, 14, 9, 9)  float32
output  policy_logits  (B, 2562)      float32   unmasked, unnormalised
        value          (B, 3)         float32   logits over (win, draw, loss)
        score          (B, 13)        float32   logits over d ∈ [−6, +6]
        capture_map    (B, 2, 9, 9)   float32   raw, pre-sigmoid
```

Batch is dynamic on every tensor. The export:

  - Sets the network to eval mode (BN runs in inference mode), then restores
    the caller's `training` flag and device.
  - Deliberately does **not** fuse softmax, sigmoid or legal-masking into the
    graph — keeping logits lets the Rust evaluator and `onnxruntime-web` apply
    the legal mask before normalising.

`value` and `score` are distinct heads and the names are load-bearing:
`value` is *who wins* (the RL value function MCTS backs up), `score` is *by
how much* (the final capture differential). See MODEL.md §6.0.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from model.abalone_net import (
    BOARD_H,
    BOARD_W,
    INPUT_CHANNELS,
    PRESETS,
    AbaloneNet,
    build,
)

INPUT_NAME = "planes"
OUTPUT_POLICY_NAME = "policy_logits"
OUTPUT_VALUE_NAME = "value"
OUTPUT_SCORE_NAME = "score"
OUTPUT_CAPTURE_MAP_NAME = "capture_map"

OUTPUT_NAMES = [
    OUTPUT_POLICY_NAME,
    OUTPUT_VALUE_NAME,
    OUTPUT_SCORE_NAME,
    OUTPUT_CAPTURE_MAP_NAME,
]


def export(model: AbaloneNet, out_path: Path | str, opset: int = 17) -> None:
    """Export `model` to `out_path` (atomic via temp + rename). The
    model's `training` flag and device are restored before returning."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Remember and restore so the caller doesn't have to.
    was_training = model.training
    orig_device = next(model.parameters()).device
    model.eval().to("cpu")
    dummy = torch.zeros(1, INPUT_CHANNELS, BOARD_H, BOARD_W, dtype=torch.float32)

    # Write to a temp file in the same directory, then atomic rename.
    fd, tmp_name = tempfile.mkstemp(
        dir=out_path.parent, prefix=out_path.name + ".", suffix=".tmp"
    )
    import os

    os.close(fd)
    try:
        # `dynamo=False` selects the legacy exporter, which emits a
        # single self-contained .onnx (weights inlined). The newer
        # dynamo-based exporter writes weights to a sidecar `.data`
        # file, which doesn't survive our temp + rename pattern. The
        # 10×128 model is ~12 MB — far under the 2 GB protobuf limit
        # where external-data becomes mandatory.
        torch.onnx.export(
            model,
            dummy,
            tmp_name,
            input_names=[INPUT_NAME],
            output_names=OUTPUT_NAMES,
            dynamic_axes={name: {0: "batch"} for name in [INPUT_NAME, *OUTPUT_NAMES]},
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )
        os.replace(tmp_name, out_path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    finally:
        model.to(orig_device)
        if was_training:
            model.train()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, help="path to a .pt file")
    parser.add_argument("--out", required=True, help="output .onnx path")
    parser.add_argument(
        "--preset",
        default=None,
        choices=sorted(PRESETS),
        help="trunk preset; defaults to the one recorded in the checkpoint, else 'base'",
    )
    args = parser.parse_args()

    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    # Checkpoints written by the training loop may record the trunk shape;
    # the CLI flag wins if given, and `base` is the fallback.
    preset = args.preset or state.get("preset", "base")
    net = build(preset)
    net.load_state_dict(state["model"])
    export(net, args.out)
    print(f"exported {args.out} ({preset}, {net.num_parameters():,} params)")
