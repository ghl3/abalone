"""Cross-language conformance: the Rust encoder vs. `model.encoder`.

The `(14, 9, 9)` plane encoding is implemented twice — once in
`crates/selfplay/src/encoder.rs` for the self-play hot path, once in
`model/encoder.py` for training. Nothing in either language's own test
suite can catch the two drifting apart, and they *did* drift: a swapped
capture-plane bug was live for three generations and survived a 43-test
Python suite that exercised the encoder heavily (ARCHITECTURE.md §5).

This module is the guard. It runs the Rust `dump-golden` binary, which
emits several hundred positions together with the planes and legal-move
indices Rust computed for them, then rebuilds every one of those planes
in Python and asserts **exact** float equality — `np.array_equal`, not
`np.allclose`. An `allclose` comparison would pass a plane swap whenever
both counters happened to be equal, which is most positions; only exact
equality over the asymmetric handicap fixtures closes that hole.

If the binary is missing and cannot be built the module skips with the
exact command to run, so a clean checkout still gets a green suite.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from model.encoder import (
    BROADSIDE_OFFSET,
    MOVE_SPACE,
    NUM_INPUT_CHANNELS,
    PLANE_SIZE,
    PositionRecord,
    decode_broadside,
    decode_inline,
    encode_broadside,
    encode_inline,
    encode_position,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
BIN_NAME = "dump-golden"
BUILD_CMD = f"cargo build --release -p abalone-selfplay --bin {BIN_NAME}"
# A cold build pulls the ONNX Runtime binaries, which is slow but bounded.
BUILD_TIMEOUT_S = 900
DUMP_TIMEOUT_S = 300

SKIP_HINT = (
    f"the Rust `{BIN_NAME}` binary is required for the encoder conformance "
    f"test and could not be {{reason}}. Build it with:\n\n    {BUILD_CMD}\n\n"
    "then re-run pytest. Without it, the Rust and Python encoders are "
    "untested against each other."
)


def _find_bin() -> Path | None:
    """Locate `dump-golden`, mirroring `model/eval.py::_bin`."""
    for candidate in (
        REPO_ROOT / "target" / "release" / BIN_NAME,
        REPO_ROOT / "target" / "debug" / BIN_NAME,
    ):
        if candidate.exists():
            return candidate
    found = shutil.which(BIN_NAME)
    return Path(found) if found is not None else None


def _build_bin() -> Path | None:
    """Try a release build of `dump-golden`. Returns None if that is not
    possible here (no cargo, no network, compile error)."""
    if shutil.which("cargo") is None:
        return None
    try:
        subprocess.run(
            BUILD_CMD.split(),
            cwd=REPO_ROOT,
            check=True,
            timeout=BUILD_TIMEOUT_S,
            capture_output=True,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return None
    return _find_bin()


@pytest.fixture(scope="module")
def fixtures() -> dict:
    """Run `dump-golden` into a temp file and parse the JSON."""
    binary = _find_bin()
    if binary is None:
        binary = _build_bin()
    if binary is None:
        pytest.skip(SKIP_HINT.format(reason="found or built"))

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "golden.json"
        try:
            subprocess.run(
                [str(binary), str(out)],
                cwd=REPO_ROOT,
                check=True,
                timeout=DUMP_TIMEOUT_S,
                capture_output=True,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as e:
            stderr = getattr(e, "stderr", b"") or b""
            pytest.skip(
                SKIP_HINT.format(reason="run")
                + f"\n({binary} failed: {stderr.decode(errors='replace')[-2000:]})"
            )
        if not out.exists():
            pytest.skip(SKIP_HINT.format(reason="run"))
        return json.loads(out.read_text())


@pytest.fixture(scope="module")
def records(fixtures: dict) -> list[dict]:
    return fixtures["positions"]


def _to_record(rec: dict) -> PositionRecord:
    """Rebuild the Python-side record from the Rust JSON.

    The fixture reports losses per *colour* (`black_losses` = marbles Black
    has had pushed off). The encoder wants them side-to-move relative, so
    the mapping is chosen by `turn` — and getting this flip wrong in either
    direction is exactly the historical bug, which is why the fixtures cover
    asymmetric handicaps for both sides to move.
    """
    black_to_move = rec["turn"] == 0
    own_losses = rec["black_losses"] if black_to_move else rec["white_losses"]
    opp_losses = rec["white_losses"] if black_to_move else rec["black_losses"]
    return PositionRecord(
        own_bb=(rec["own_bb_hi"] << 64) | rec["own_bb_lo"],
        opp_bb=(rec["opp_bb_hi"] << 64) | rec["opp_bb_lo"],
        own_losses=own_losses,
        opp_losses=opp_losses,
        ply=rec["ply"],
        max_plies=rec["max_plies"],
        turn=rec["turn"],
    )


# ----- header / coverage -----------------------------------------------------


class TestFixtureHeader:
    def test_shape_constants_agree(self, fixtures: dict):
        assert fixtures["num_input_channels"] == NUM_INPUT_CHANNELS
        assert fixtures["plane_size"] == PLANE_SIZE
        assert fixtures["move_space"] == MOVE_SPACE
        assert fixtures["board_h"] == 9
        assert fixtures["board_w"] == 9

    def test_enough_positions(self, records: list[dict]):
        assert len(records) >= 200, (
            f"only {len(records)} fixtures; the guard is only as good as its "
            "coverage (see dump_golden.rs::positions)"
        )

    def test_both_sides_to_move_are_covered(self, records: list[dict]):
        turns = {r["turn"] for r in records}
        assert turns == {0, 1}

    def test_full_handicap_grid_is_covered(self, records: list[dict]):
        """Both loss counters over 0..=5 independently, including the
        asymmetric pairs a plane swap would otherwise hide."""
        seen = {(r["black_losses"], r["white_losses"]) for r in records}
        missing = {(b, w) for b in range(6) for w in range(6)} - seen
        assert not missing, f"handicap pairs absent from the fixtures: {sorted(missing)}"
        for pair in [(0, 4), (4, 0), (5, 1), (1, 5)]:
            assert pair in seen, f"asymmetric pair {pair} must be covered"

    def test_asymmetric_records_exist_for_each_side_to_move(self, records: list[dict]):
        """The plane swap is only observable when the two counters differ.
        Assert we have such positions with Black to move AND with White to
        move, so neither branch of the POV flip goes untested."""
        for turn in (0, 1):
            n = sum(
                1
                for r in records
                if r["turn"] == turn and r["black_losses"] != r["white_losses"]
            )
            assert n >= 10, f"only {n} asymmetric-loss fixtures with turn={turn}"

    def test_several_ply_caps_are_covered(self, records: list[dict]):
        caps = {r["max_plies"] for r in records}
        assert len(caps) >= 3, f"ply normaliser under-covered: caps seen = {caps}"

    def test_varied_plies_are_covered(self, records: list[dict]):
        plies = {r["ply"] for r in records}
        assert len(plies) >= 20
        assert max(plies) > 20, "fixtures must include deep positions, not just openings"


# ----- the actual conformance assertion --------------------------------------


class TestPlaneConformance:
    def test_every_record_matches_exactly(self, records: list[dict]):
        """The whole point. Exact equality, every record, every float."""
        mismatches: list[str] = []
        for rec in records:
            expected = np.asarray(rec["planes"], dtype=np.float32).reshape(
                NUM_INPUT_CHANNELS, 9, 9
            )
            actual = encode_position(_to_record(rec))
            assert actual.dtype == np.float32
            assert actual.shape == expected.shape
            if not np.array_equal(actual, expected):
                bad = sorted(
                    {int(c) for c in np.argwhere(actual != expected)[:, 0].tolist()}
                )
                mismatches.append(
                    f"  {rec['label']}: turn={rec['turn']} "
                    f"black_losses={rec['black_losses']} "
                    f"white_losses={rec['white_losses']} ply={rec['ply']}/"
                    f"{rec['max_plies']} -> planes {bad} differ"
                )
        assert not mismatches, (
            f"{len(mismatches)}/{len(records)} positions encode differently in "
            "Rust and Python:\n" + "\n".join(mismatches[:25])
        )

    def test_no_record_is_trivially_equal(self, records: list[dict]):
        """Sanity check on the check: the fixtures must actually contain
        non-zero, varied planes, or exact equality would be vacuous."""
        sums = {float(np.asarray(r["planes"], dtype=np.float32).sum()) for r in records}
        assert len(sums) > 5, "fixture planes are suspiciously uniform"

    def test_dtype_and_shape_are_the_contract(self, records: list[dict]):
        planes = encode_position(_to_record(records[0]))
        assert planes.shape == (NUM_INPUT_CHANNELS, 9, 9)
        assert planes.dtype == np.float32
        assert planes.size == PLANE_SIZE


# ----- legal-move indices ----------------------------------------------------


class TestLegalMoveConformance:
    def test_indices_are_in_range_and_sorted(self, records: list[dict]):
        for rec in records:
            idxs = rec["legal_move_indices"]
            assert idxs == sorted(idxs), f"{rec['label']}: indices not sorted"
            assert len(set(idxs)) == len(idxs), f"{rec['label']}: duplicate indices"
            for i in idxs:
                assert 0 <= i < MOVE_SPACE, f"{rec['label']}: move index {i} out of range"

    def test_indices_round_trip_through_the_python_decoders(self, records: list[dict]):
        """Every index Rust calls legal must decode and re-encode to itself
        under `model.encoder`'s move maths — the other half of the §5.1
        cross-language contract."""
        for rec in records:
            for idx in rec["legal_move_indices"]:
                if idx < BROADSIDE_OFFSET:
                    anchor, d, size = decode_inline(idx)
                    assert 0 <= d < 6
                    assert 1 <= size <= 3
                    assert encode_inline(anchor, d, size) == idx
                else:
                    anchor, gd, md, size = decode_broadside(idx)
                    assert 2 <= size <= 3
                    assert gd != md
                    assert encode_broadside(anchor, gd, md, size) == idx

    def test_openings_have_a_plausible_branching_factor(self, records: list[dict]):
        """Guards against the fixtures degenerating to empty move lists,
        which would make the round-trip test vacuous."""
        counts = [len(r["legal_move_indices"]) for r in records]
        assert max(counts) > 40, f"max branching {max(counts)} looks wrong"
        non_empty = sum(1 for c in counts if c > 0)
        assert non_empty > 0.9 * len(counts)


# ----- the guard's own guard -------------------------------------------------


class TestGuardDetectsDivergence:
    """A conformance test that cannot fail is worse than none. These
    mutate the *expected* side (never the encoder) in the two ways the
    implementations have historically drifted, and assert the comparison
    notices."""

    def test_swapping_the_loss_plane_groups_is_detected(self, records: list[dict]):
        # The exact historical bug: planes 2-6 hold the opponent's losses
        # and 7-11 the side-to-move's.
        caught = 0
        for rec in records:
            if rec["black_losses"] == rec["white_losses"]:
                continue  # a swap is invisible when the counters agree
            expected = np.asarray(rec["planes"], dtype=np.float32).reshape(
                NUM_INPUT_CHANNELS, 9, 9
            )
            swapped = expected.copy()
            swapped[2:7], swapped[7:12] = expected[7:12].copy(), expected[2:7].copy()
            if not np.array_equal(encode_position(_to_record(rec)), swapped):
                caught += 1
        assert caught > 0, "no fixture would notice a swap of the loss plane groups"

    def test_a_wrong_ply_normaliser_is_detected(self, records: list[dict]):
        # The old encoder divided by a hardcoded 400 instead of max_plies.
        caught = 0
        for rec in records:
            if rec["ply"] == 0:
                continue
            expected = np.asarray(rec["planes"], dtype=np.float32).reshape(
                NUM_INPUT_CHANNELS, 9, 9
            )
            wrong = expected.copy()
            wrong[12].fill(min(rec["ply"], 400) / 400.0)
            if not np.array_equal(encode_position(_to_record(rec)), wrong):
                caught += 1
        assert caught > 0, "no fixture would notice a hardcoded ply normaliser"

    def test_masking_the_constant_planes_is_detected(self, records: list[dict]):
        # Constant planes must cover all 81 cells, off-board slots included.
        from model.encoder import VALID_CELL_MASK

        caught = 0
        for rec in records:
            expected = np.asarray(rec["planes"], dtype=np.float32).reshape(
                NUM_INPUT_CHANNELS, 9, 9
            )
            masked = expected.copy()
            masked[2:13] *= VALID_CELL_MASK
            if not np.array_equal(encode_position(_to_record(rec)), masked):
                caught += 1
        assert caught > 0, "no fixture would notice the constant planes being masked"


if __name__ == "__main__":
    raise SystemExit(pytest.main([os.path.abspath(__file__), "-v"]))
