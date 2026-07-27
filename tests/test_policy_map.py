"""Tests for `model.policy_map`: the fixed `(42, 9, 9)` → 2562 gather table.

This table is the whole reason the policy head can be convolutional
(MODEL.md §6.2), and it is a *silent* failure mode: a wrong entry aliases one
move onto another's logit and training still runs, just wrong. So we test it
three ways:

  * Structural: length, dtype, range, and — the critical one — **injectivity**.
  * Against the encoder's `decode_inline` / `decode_broadside`, which mirror
    `crates/game/src/move_index.rs`, for every index (2562 is cheap).
  * Counting: each of the 42 planes is used by exactly 61 moves and each of
    the 61 valid cells by exactly 42, i.e. the table is a bijection onto the
    valid sub-grid.
"""

from __future__ import annotations

import numpy as np
import pytest

from model.encoder import (
    BROADSIDE_DIRS,
    BROADSIDE_OFFSET,
    CELL_TO_COMPACT,
    COMPACT_TO_CELL,
    MOVE_SPACE,
    NUM_VALID,
    POSITIVE_DIRS,
    decode_broadside,
    decode_inline,
)
from model.policy_map import (
    BOARD_CELLS,
    BROADSIDE_PLANES,
    INLINE_PLANES,
    POLICY_GATHER,
    POLICY_PLANES,
)

FLAT_SIZE = POLICY_PLANES * BOARD_CELLS  # 42 * 81 = 3402


# ----- structural ------------------------------------------------------------


def test_plane_count_matches_move_space():
    assert INLINE_PLANES == 18  # 6 dirs × 3 sizes
    assert BROADSIDE_PLANES == 24  # 3 group dirs × 4 move dirs × 2 sizes
    assert POLICY_PLANES == 42
    assert POLICY_PLANES * NUM_VALID == MOVE_SPACE == 2562


def test_gather_shape_and_dtype():
    assert POLICY_GATHER.shape == (MOVE_SPACE,)
    assert POLICY_GATHER.dtype == np.int64


def test_gather_in_range():
    assert POLICY_GATHER.min() >= 0
    assert POLICY_GATHER.max() < FLAT_SIZE


def test_gather_is_injective():
    """The critical invariant: no two moves may share a logit."""
    assert len(np.unique(POLICY_GATHER)) == MOVE_SPACE


def test_gather_targets_only_valid_cells():
    cells = POLICY_GATHER % BOARD_CELLS
    assert all(CELL_TO_COMPACT[c] != 255 for c in np.unique(cells))


def test_plane_and_cell_usage_counts():
    """Bijection onto (42 planes × 61 valid cells)."""
    planes, cells = np.divmod(POLICY_GATHER, BOARD_CELLS)
    plane_counts = np.bincount(planes, minlength=POLICY_PLANES)
    assert plane_counts.tolist() == [NUM_VALID] * POLICY_PLANES

    used_cells, cell_counts = np.unique(cells, return_counts=True)
    assert len(used_cells) == NUM_VALID
    assert cell_counts.tolist() == [POLICY_PLANES] * NUM_VALID


def test_inline_and_broadside_planes_do_not_overlap():
    planes = POLICY_GATHER // BOARD_CELLS
    assert planes[:BROADSIDE_OFFSET].max() < INLINE_PLANES
    assert planes[BROADSIDE_OFFSET:].min() >= INLINE_PLANES


# ----- cross-check against the move-index spec -------------------------------


def _expected_entry(idx: int) -> int:
    """Independently recompute the gather target from `model.encoder`'s
    decode, which mirrors `abalone_game::move_index::decode`."""
    if idx < BROADSIDE_OFFSET:
        anchor, d, size = decode_inline(idx)
        plane = d * 3 + (size - 1)
    else:
        anchor, group_dir, move_dir, size = decode_broadside(idx)
        gi = POSITIVE_DIRS.index(group_dir)
        mi = BROADSIDE_DIRS[gi].index(move_dir)
        plane = INLINE_PLANES + gi * 8 + mi * 2 + (size - 2)
    return plane * BOARD_CELLS + anchor


@pytest.mark.parametrize(
    "idx",
    [
        0,  # first inline: anchor 0, E, size 1
        1,
        17,  # last move kind of anchor 0
        18,  # anchor 1, E, size 1
        BROADSIDE_OFFSET - 1,  # last inline
        BROADSIDE_OFFSET,  # first broadside
        BROADSIDE_OFFSET + 1,
        BROADSIDE_OFFSET + 23,
        BROADSIDE_OFFSET + 24,
        MOVE_SPACE - 1,  # last broadside
    ],
)
def test_sample_indices_match_decode(idx):
    assert int(POLICY_GATHER[idx]) == _expected_entry(idx)


def test_all_indices_match_decode():
    """Exhaustive — 2562 entries is cheap and this is the spec check."""
    expected = np.array([_expected_entry(i) for i in range(MOVE_SPACE)], dtype=np.int64)
    assert np.array_equal(POLICY_GATHER, expected)


def test_anchor_major_layout_holds():
    """Every anchor uses the same 42 planes, differing only in the cell.

    This is the property that makes the head convolutional: the plane index
    is anchor-independent.
    """
    planes, cells = np.divmod(POLICY_GATHER, BOARD_CELLS)
    for a in range(NUM_VALID):
        cell = int(COMPACT_TO_CELL[a])
        inline_slice = planes[a * 18 : (a + 1) * 18]
        assert inline_slice.tolist() == list(range(INLINE_PLANES))
        assert (cells[a * 18 : (a + 1) * 18] == cell).all()

        lo = BROADSIDE_OFFSET + a * 24
        broadside_slice = planes[lo : lo + 24]
        assert broadside_slice.tolist() == list(range(INLINE_PLANES, POLICY_PLANES))
        assert (cells[lo : lo + 24] == cell).all()
