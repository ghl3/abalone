"""Export a trained `AbaloneNet` to ONNX. The exported model:

  - Sets the network to eval mode (BN runs in inference mode).
  - Has a fixed batch dimension expanded to dynamic at export time so
    inference can use any batch size.
  - Uses input/output names that the `ort`-based Rust evaluator
    expects: `planes` in, `(policy_logits, value)` out.

We deliberately do NOT fuse softmax into the policy head — keeping
logits gives the Rust side flexibility (it can apply legal-mask
clipping then softmax itself).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from model.abalone_net import (
    BOARD_H,
    BOARD_W,
    INPUT_CHANNELS,
    AbaloneNet,
)

INPUT_NAME = "planes"
OUTPUT_POLICY_NAME = "policy_logits"
OUTPUT_VALUE_NAME = "value"


def export(model: AbaloneNet, out_path: Path | str, opset: int = 17) -> None:
    """Export `model` to `out_path` (atomic via temp + rename)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = model.eval().to("cpu")
    dummy = torch.zeros(1, INPUT_CHANNELS, BOARD_H, BOARD_W, dtype=torch.float32)

    # Write to a temp file in the same directory, then atomic rename.
    fd, tmp_name = tempfile.mkstemp(
        dir=out_path.parent, prefix=out_path.name + ".", suffix=".tmp"
    )
    import os

    os.close(fd)
    try:
        torch.onnx.export(
            model,
            dummy,
            tmp_name,
            input_names=[INPUT_NAME],
            output_names=[OUTPUT_POLICY_NAME, OUTPUT_VALUE_NAME],
            dynamic_axes={
                INPUT_NAME: {0: "batch"},
                OUTPUT_POLICY_NAME: {0: "batch"},
                OUTPUT_VALUE_NAME: {0: "batch"},
            },
            opset_version=opset,
            do_constant_folding=True,
        )
        os.replace(tmp_name, out_path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="path to a .pt file")
    parser.add_argument("--out", required=True, help="output .onnx path")
    args = parser.parse_args()

    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    net = AbaloneNet()
    net.load_state_dict(state["model"])
    export(net, args.out)
    print(f"exported {args.out}")
