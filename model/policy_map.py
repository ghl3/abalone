"""The fixed gather table that turns a `(42, 9, 9)` policy tensor into the
flat 2562-entry move space.

This is the piece that lets the policy head be *convolutional* instead of
dense (MODEL.md §6.2). Both halves of the move index are **anchor-major**:

```
inline     idx =        anchor_compact · 18 + dir · 3          + (size − 1)
broadside  idx = 1098 + anchor_compact · 24 + gi · 8 + mi · 2  + (size − 2)
```

Because the anchor is the most significant factor, everything *below* it —
`rem = idx % 18` for inline, `rem = (idx − 1098) % 24` for broadside — is a
per-anchor "move kind" that is identical at every anchor. There are
`18 + 24 = 42` such kinds, and `42 × 61 = 2562` exactly, so the move space
*is* a `(42, 9, 9)` tensor sampled at the 61 valid cells.

The 42 planes:

| Planes | Kind | Decomposition |
| --- | --- | --- |
| `0..18` | inline | 6 directions × 3 sizes |
| `18..42` | broadside | 3 group dirs × 4 move dirs × 2 sizes |

Read out through `POLICY_GATHER[idx] = plane · 81 + COMPACT_TO_CELL[anchor]`,
which indexes the head's output after `flatten(1)`.

The table is a bijection onto its image: 2562 distinct flat offsets. That is
the invariant worth guarding — a collision would silently alias two different
moves onto one logit. `tests/test_policy_map.py` asserts it, and this module
re-derives the decode from `model.encoder` at import time as a cheap
self-check that the table still agrees with `crates/game/src/move_index.rs`.
"""

from __future__ import annotations

import numpy as np

from model.encoder import BROADSIDE_OFFSET, COMPACT_TO_CELL, MOVE_SPACE, NUM_VALID

# ----- shape constants -------------------------------------------------------

BOARD_CELLS = 81  # 9 × 9, of which NUM_VALID = 61 are on-board

INLINE_PLANES = 18  # 6 directions × 3 sizes
BROADSIDE_PLANES = 24  # 3 group directions × 4 move directions × 2 sizes
POLICY_PLANES = INLINE_PLANES + BROADSIDE_PLANES  # 42

assert POLICY_PLANES * NUM_VALID == MOVE_SPACE, (
    f"{POLICY_PLANES} planes × {NUM_VALID} valid cells != {MOVE_SPACE} moves"
)


# ----- the table -------------------------------------------------------------


def _build_policy_gather() -> np.ndarray:
    """`POLICY_GATHER[idx]` = flat offset into a flattened `(42, 9, 9)` tensor.

    Mirrors `abalone_game::move_index::decode` exactly; see the module
    docstring for the algebra.
    """
    out = np.zeros(MOVE_SPACE, dtype=np.int64)
    for idx in range(MOVE_SPACE):
        if idx < BROADSIDE_OFFSET:
            a, rem = divmod(idx, INLINE_PLANES)
            plane = rem
        else:
            a, rem = divmod(idx - BROADSIDE_OFFSET, BROADSIDE_PLANES)
            plane = INLINE_PLANES + rem
        out[idx] = plane * BOARD_CELLS + int(COMPACT_TO_CELL[a])
    return out


POLICY_GATHER = _build_policy_gather()
POLICY_GATHER.flags.writeable = False

# Injectivity is the load-bearing property: two moves must never share a
# logit. Cheap enough (2562 entries) to assert on every import.
assert len(np.unique(POLICY_GATHER)) == MOVE_SPACE, "POLICY_GATHER is not injective"


# ----- self-check runnable as `python -m model.policy_map` -------------------


def _sanity_checks() -> None:
    from model.encoder import (
        BROADSIDE_DIRS,
        CELL_TO_COMPACT,
        POSITIVE_DIRS,
        decode_broadside,
        decode_inline,
    )

    assert POLICY_GATHER.shape == (MOVE_SPACE,)
    assert POLICY_GATHER.dtype == np.int64
    assert POLICY_GATHER.min() >= 0
    assert POLICY_GATHER.max() < POLICY_PLANES * BOARD_CELLS

    # Every entry must agree with the encoder's decode of the same index.
    for idx in range(MOVE_SPACE):
        plane, cell = divmod(int(POLICY_GATHER[idx]), BOARD_CELLS)
        if idx < BROADSIDE_OFFSET:
            anchor, d, size = decode_inline(idx)
            assert plane == d * 3 + (size - 1)
        else:
            anchor, gd, md, size = decode_broadside(idx)
            gi = POSITIVE_DIRS.index(gd)
            mi = BROADSIDE_DIRS[gi].index(md)
            assert plane == INLINE_PLANES + gi * 8 + mi * 2 + (size - 2)
        assert cell == anchor
        assert CELL_TO_COMPACT[cell] != 255, "gather points at an off-board cell"

    print("policy_map sanity checks passed")
    print(f"  POLICY_PLANES  = {POLICY_PLANES} ({INLINE_PLANES} inline + {BROADSIDE_PLANES} broadside)")
    print(f"  POLICY_GATHER  shape={POLICY_GATHER.shape} dtype={POLICY_GATHER.dtype}")
    print(f"  distinct targets = {len(np.unique(POLICY_GATHER))} / {MOVE_SPACE}")


if __name__ == "__main__":
    _sanity_checks()
