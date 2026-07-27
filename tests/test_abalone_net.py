"""Tests for `model.abalone_net`: shapes, masking, parameter budget, export.

The properties worth pinning down:

  * **Output contract.** Four heads, fixed tuple order, exact shapes at more
    than one batch size — the Rust evaluator and the browser both index these
    positionally.
  * **Off-board masking.** Two inputs that differ *only* in the 20 dead slots
    must produce bit-identical outputs. This is the whole point of masking the
    input and every block: a 3×3 kernel at a valid cell reads its off-board
    neighbours, so nothing else would guarantee it.
  * **Parameter budget.** The old dense head held 92% of all parameters. The
    regression test is not "total params ≈ 3 M" but "the *trunk* holds the
    budget and the policy head is small".
  * **Numerics.** All logits finite (the −inf used for the masked max pool
    must not escape into an output), softmax normalised.
  * **ONNX export.** Writes a real file with the documented signature.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from model.abalone_net import (
    BOARD_H,
    BOARD_W,
    CAPTURE_MAP_CHANNELS,
    DEFAULT_CONFIG,
    INPUT_CHANNELS,
    MOVE_SPACE,
    PRESETS,
    SCORE_BUCKETS,
    VALUE_BUCKETS,
    AbaloneNet,
    NetConfig,
    build,
    build_default,
)
from model.encoder import VALID_CELL_MASK

OFF_BOARD = np.argwhere(VALID_CELL_MASK == 0.0)  # (20, 2) row/col pairs


@pytest.fixture(scope="module")
def net() -> AbaloneNet:
    """A `small` net in eval mode — enough for every structural property and
    ~3× faster than `base`."""
    torch.manual_seed(0)
    m = build("small")
    m.eval()
    return m


def _random_input(batch: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand(batch, INPUT_CHANNELS, BOARD_H, BOARD_W, generator=g)


# ----- construction ----------------------------------------------------------


def test_presets_exist_and_default_is_base():
    assert set(PRESETS) == {"small", "base", "large"}
    assert PRESETS["small"] == NetConfig(blocks=6, channels=96)
    assert PRESETS["base"] == NetConfig(blocks=10, channels=128)
    assert PRESETS["large"] == NetConfig(blocks=14, channels=192)
    assert DEFAULT_CONFIG == PRESETS["base"]
    assert build_default().config == PRESETS["base"]
    assert AbaloneNet().config == PRESETS["base"]


def test_build_accepts_name_or_config():
    assert build("small").config == PRESETS["small"]
    assert build(NetConfig(blocks=2, channels=16)).config.channels == 16
    with pytest.raises(ValueError):
        build("enormous")


def test_geometry_buffers_are_not_persistent():
    """The mask and gather table are derived from board geometry, so they must
    not bloat (or version-lock) checkpoints."""
    m = build("small")
    geometry = {"valid_mask", "off_board_mask", "policy_gather"}
    assert geometry <= dict(m.named_buffers()).keys()
    assert geometry.isdisjoint(m.state_dict().keys())
    # BN running stats, by contrast, are learned and must persist.
    assert "stem_bn.running_mean" in m.state_dict()


# ----- forward shapes --------------------------------------------------------


@pytest.mark.parametrize("batch", [1, 8])
def test_forward_shapes(net, batch):
    policy, value, score, capture = net(_random_input(batch))
    assert policy.shape == (batch, MOVE_SPACE) == (batch, 2562)
    assert value.shape == (batch, VALUE_BUCKETS) == (batch, 3)
    assert score.shape == (batch, SCORE_BUCKETS) == (batch, 13)
    assert capture.shape == (batch, CAPTURE_MAP_CHANNELS, BOARD_H, BOARD_W)


def test_forward_returns_plain_tuple_of_four(net):
    out = net(_random_input(2))
    assert type(out) is tuple
    assert len(out) == 4


def test_se_variant_runs():
    torch.manual_seed(0)
    m = build(NetConfig(blocks=2, channels=32, se=True)).eval()
    policy, value, score, capture = m(_random_input(3))
    assert policy.shape == (3, MOVE_SPACE)
    assert value.shape == (3, VALUE_BUCKETS)
    assert score.shape == (3, SCORE_BUCKETS)
    assert capture.shape == (3, CAPTURE_MAP_CHANNELS, BOARD_H, BOARD_W)
    assert m.num_parameters() > build(NetConfig(blocks=2, channels=32)).num_parameters()


def test_gradients_reach_every_head(net):
    m = build(NetConfig(blocks=2, channels=16))
    policy, value, score, capture = m(_random_input(2))
    (policy.sum() + value.sum() + score.sum() + capture.sum()).backward()
    missing = [n for n, p in m.named_parameters() if p.grad is None]
    assert not missing, f"no gradient reached {missing}"


# ----- masking ---------------------------------------------------------------


def test_off_board_cells_cannot_change_outputs(net):
    """Inputs differing ONLY on the 20 off-board slots are indistinguishable."""
    a = _random_input(4, seed=1)
    b = a.clone()
    for r, q in OFF_BOARD:
        b[:, :, r, q] = torch.randn(4, INPUT_CHANNELS) * 100.0
    assert not torch.equal(a, b)

    with torch.no_grad():
        out_a = net(a)
        out_b = net(b)
    for name, x, y in zip(
        ("policy", "value", "score", "capture_map"), out_a, out_b, strict=True
    ):
        assert torch.allclose(x, y, atol=0, rtol=0), f"{name} leaked off-board signal"


def test_off_board_cells_cannot_win_the_max_pool(net):
    """A huge value in a dead slot must not move the value/score head.

    Covered by the test above in aggregate; asserted separately because a
    max-pool masked with 0 instead of −inf fails *only* this way.
    """
    a = _random_input(2, seed=2)
    b = a.clone()
    for r, q in OFF_BOARD:
        b[:, :, r, q] = 1e6
    with torch.no_grad():
        _, va, sa, _ = net(a)
        _, vb, sb, _ = net(b)
    assert torch.allclose(va, vb, atol=0, rtol=0)
    assert torch.allclose(sa, sb, atol=0, rtol=0)


def test_capture_map_is_spatially_constant_off_board(net):
    """Off-board capture-map logits are just the conv bias — no signal."""
    with torch.no_grad():
        _, _, _, capture = net(_random_input(2, seed=3))
    r0, q0 = OFF_BOARD[0]
    for r, q in OFF_BOARD:
        for c in range(CAPTURE_MAP_CHANNELS):
            assert torch.allclose(capture[:, c, r, q], capture[0, c, r0, q0])


# ----- numerics --------------------------------------------------------------


def test_all_outputs_finite(net):
    """−inf is used internally for the masked max pool; it must not escape."""
    with torch.no_grad():
        outs = net(_random_input(4, seed=4))
    for name, t in zip(("policy", "value", "score", "capture_map"), outs, strict=True):
        assert torch.isfinite(t).all(), f"{name} contains inf/nan"


def test_zero_input_is_finite(net):
    with torch.no_grad():
        outs = net(torch.zeros(2, INPUT_CHANNELS, BOARD_H, BOARD_W))
    assert all(torch.isfinite(t).all() for t in outs)


def test_value_and_score_softmax_normalised(net):
    with torch.no_grad():
        _, value, score, _ = net(_random_input(5, seed=5))
    v = torch.softmax(value, dim=1)
    s = torch.softmax(score, dim=1)
    assert torch.allclose(v.sum(1), torch.ones(5), atol=1e-6)
    assert torch.allclose(s.sum(1), torch.ones(5), atol=1e-6)
    assert (v >= 0).all() and (s >= 0).all()


# ----- parameter budget ------------------------------------------------------


def test_base_parameter_count_in_band():
    total = build("base").num_parameters()
    assert 2.6e6 < total < 3.4e6, f"base is {total:,} params, expected ~3.0M"


def test_breakdown_sums_to_total():
    m = build("base")
    assert sum(m.parameter_breakdown().values()) == m.num_parameters()


def test_trunk_holds_the_budget_and_policy_head_is_small():
    """The regression this whole architecture exists to prevent: the old dense
    head was 3,321,282 params — 92% of the model."""
    m = build("base")
    parts = m.parameter_breakdown()
    total = m.num_parameters()

    assert parts["policy_head"] < 100_000, parts
    assert parts["policy_head"] / total < 0.05, parts
    assert parts["trunk"] / total > 0.90, parts
    # Auxiliary heads are rounding errors by construction.
    assert parts["value_score_head"] < 50_000
    assert parts["capture_head"] < 5_000


@pytest.mark.parametrize(
    ("preset", "lo", "hi"),
    [("small", 0.9e6, 1.3e6), ("base", 2.6e6, 3.4e6), ("large", 8.5e6, 10.0e6)],
)
def test_preset_sizes_match_the_design_doc(preset, lo, hi):
    total = build(preset).num_parameters()
    assert lo < total < hi, f"{preset} is {total:,} params"


# ----- ONNX export -----------------------------------------------------------


def test_export_onnx_writes_a_real_model(tmp_path):
    from model.export_onnx import (
        INPUT_NAME,
        OUTPUT_NAMES,
        export,
    )

    m = build(NetConfig(blocks=2, channels=32))
    m.train()  # export must restore this
    out = tmp_path / "net.onnx"
    export(m, out)

    assert out.exists()
    assert out.stat().st_size > 10_000, "exported file is implausibly small"
    assert m.training, "export did not restore the training flag"
    assert not list(tmp_path.glob("*.tmp")), "temp file left behind"

    onnx = pytest.importorskip("onnx", reason="onnx not installed")
    model = onnx.load(str(out))
    onnx.checker.check_model(model)

    inputs = {i.name: i for i in model.graph.input}
    outputs = {o.name: o for o in model.graph.output}
    assert list(inputs) == [INPUT_NAME]
    assert list(outputs) == OUTPUT_NAMES

    def dims(vi):
        return [
            d.dim_param if d.HasField("dim_param") else d.dim_value
            for d in vi.type.tensor_type.shape.dim
        ]

    assert dims(inputs[INPUT_NAME]) == ["batch", INPUT_CHANNELS, BOARD_H, BOARD_W]
    assert dims(outputs["policy_logits"]) == ["batch", MOVE_SPACE]
    assert dims(outputs["value"]) == ["batch", VALUE_BUCKETS]
    assert dims(outputs["score"]) == ["batch", SCORE_BUCKETS]
    assert dims(outputs["capture_map"]) == [
        "batch",
        CAPTURE_MAP_CHANNELS,
        BOARD_H,
        BOARD_W,
    ]


def test_export_is_atomic_on_failure(tmp_path, monkeypatch):
    """A failed export must leave no partial file and no stray temp."""
    import model.export_onnx as export_onnx

    def boom(*args, **kwargs):
        raise RuntimeError("simulated exporter failure")

    monkeypatch.setattr(export_onnx.torch.onnx, "export", boom)

    out = tmp_path / "net.onnx"
    with pytest.raises(RuntimeError, match="simulated exporter failure"):
        export_onnx.export(build(NetConfig(blocks=1, channels=8)), out)
    assert not out.exists()
    assert not list(tmp_path.glob("*")), "export left files behind"
