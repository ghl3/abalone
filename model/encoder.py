"""Position encoding and D6 symmetry augmentation.

Plane layout (matches `abalone_net.py`'s `INPUT_CHANNELS = 6`):
    0: own marbles    9×9 binary  (current side-to-move's marbles)
    1: opp marbles    9×9 binary
    2: own captures   9×9 constant — `pushed_off[opp_side] / 6`
    3: opp captures   9×9 constant — `pushed_off[own_side] / 6`
    4: ply            9×9 constant — `ply / 400`
    5: valid mask     9×9 binary   — 1 for the 61 on-board cells

Channels 0/1 are oriented to side-to-move (NOT to a fixed colour), so
the same network drives both sides without a side-to-move embedding.

Symmetry: 12 elements of D6. We pre-compute two permutation tables at
module import:
    `CELL_PERM[s, c]`         — cell `c` maps to cell `CELL_PERM[s, c]`
    `MOVE_PERM[s, idx]`       — move index `idx` maps to `MOVE_PERM[s, idx]`

Symmetry application is then pure indexing: pixel permutation for the
binary planes, integer remap for the policy vector.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ----- coordinate / cell helpers ---------------------------------------------

NUM_CELLS = 81
NUM_VALID = 61
NUM_DIRS = 6
MOVE_SPACE = 2562  # matches abalone_game::move_index::MOVE_SPACE
NUM_SYMS = 12

# Direction shifts in cell-index space, in the engine's canonical order.
# E=0, NE=1, NW=2, W=3, SW=4, SE=5
DIR_SHIFTS = np.array([1, 10, 9, -1, -10, -9], dtype=np.int32)
POSITIVE_DIRS = (0, 1, 2)  # E, NE, NW
# For each positive group_dir gi, the four sideways move_dirs, in engine order.
BROADSIDE_DIRS = (
    (1, 2, 5, 4),  # E:  NE NW SE SW
    (0, 3, 2, 5),  # NE: E  W  NW SE
    (0, 3, 1, 4),  # NW: E  W  NE SW
)
# Opposite direction lookup.
OPP_DIR = (3, 4, 5, 0, 1, 2)  # E↔W, NE↔SW, NW↔SE

# Inline ↔ broadside split in the move-index space.
INLINE_OFFSET = 0
BROADSIDE_OFFSET = 1098
N_INLINE = 1098  # 61 * 6 * 3
N_BROADSIDE = 1464  # 61 * 3 * 4 * 2


def _is_valid(r: int, q: int) -> bool:
    return 0 <= r < 9 and 0 <= q < 9 and abs(q - r) <= 4


def _cell_to_compact() -> np.ndarray:
    """`CELL_TO_COMPACT[c]` = compact 0..61 index, or 255 for off-board."""
    out = np.full(NUM_CELLS, 255, dtype=np.uint8)
    idx = 0
    for c in range(NUM_CELLS):
        r, q = divmod(c, 9)
        if _is_valid(r, q):
            out[c] = idx
            idx += 1
    return out


def _compact_to_cell() -> np.ndarray:
    """`COMPACT_TO_CELL[i]` = the actual cell index for compact `i`."""
    out = np.zeros(NUM_VALID, dtype=np.uint8)
    idx = 0
    for c in range(NUM_CELLS):
        r, q = divmod(c, 9)
        if _is_valid(r, q):
            out[idx] = c
            idx += 1
    return out


CELL_TO_COMPACT = _cell_to_compact()
COMPACT_TO_CELL = _compact_to_cell()
VALID_CELL_MASK = (CELL_TO_COMPACT != 255).astype(np.float32).reshape(9, 9)


# ----- D6 symmetry: matrices, cell perm, dir perm ----------------------------

# Centered-coord matrices for the 12 elements. Same as Rust's
# `SYM_MATS`: 0..6 are R^k (60° rotations), 6..12 are R^k * F (reflection
# F is the swap u<->v, fixing the NE-SW diagonal).
def _build_sym_mats() -> np.ndarray:
    R = np.array([[1, -1], [1, 0]], dtype=np.int32)
    F = np.array([[0, 1], [1, 0]], dtype=np.int32)
    mats = []
    cur = np.eye(2, dtype=np.int32)
    for _ in range(6):
        mats.append(cur.copy())
        cur = R @ cur
    cur = F.copy()
    for _ in range(6):
        mats.append(cur.copy())
        cur = R @ cur
    return np.stack(mats)


SYM_MATS = _build_sym_mats()  # (12, 2, 2)


def _build_cell_perm() -> np.ndarray:
    """`CELL_PERM[s, c]` for s in 0..12, c in 0..81. Off-board cells
    map to 255 (sentinel; they should never be queried)."""
    out = np.full((NUM_SYMS, NUM_CELLS), 255, dtype=np.int16)
    for s in range(NUM_SYMS):
        m = SYM_MATS[s]
        for c in range(NUM_CELLS):
            r, q = divmod(c, 9)
            if not _is_valid(r, q):
                continue
            u, v = q - 4, r - 4
            u2 = m[0, 0] * u + m[0, 1] * v
            v2 = m[1, 0] * u + m[1, 1] * v
            r2, q2 = v2 + 4, u2 + 4
            assert _is_valid(r2, q2), f"sym {s} sent {(r, q)} off-board"
            out[s, c] = r2 * 9 + q2
    return out


CELL_PERM = _build_cell_perm()


def _build_dir_perm() -> np.ndarray:
    """`DIR_PERM[s, d]` = where direction d is sent under sym s."""
    # Direction (u, v) deltas in engine order: E NE NW W SW SE.
    deltas = np.array(
        [(1, 0), (1, 1), (0, 1), (-1, 0), (-1, -1), (0, -1)], dtype=np.int32
    )
    inv = {tuple(v): i for i, v in enumerate(deltas.tolist())}
    out = np.zeros((NUM_SYMS, NUM_DIRS), dtype=np.int8)
    for s in range(NUM_SYMS):
        m = SYM_MATS[s]
        for d in range(NUM_DIRS):
            du, dv = deltas[d]
            du2 = m[0, 0] * du + m[0, 1] * dv
            dv2 = m[1, 0] * du + m[1, 1] * dv
            out[s, d] = inv[(int(du2), int(dv2))]
    return out


DIR_PERM = _build_dir_perm()


# ----- move-index encoding (mirror of abalone_game::move_index) -------------


def encode_inline(anchor: int, dir_idx: int, size: int) -> int:
    a = int(CELL_TO_COMPACT[anchor])
    return a * 18 + dir_idx * 3 + (size - 1)


def decode_inline(idx: int) -> tuple[int, int, int]:
    a = idx // 18
    rem = idx % 18
    d = rem // 3
    size = (rem % 3) + 1
    return int(COMPACT_TO_CELL[a]), d, size


def encode_broadside(anchor: int, group_dir: int, move_dir: int, size: int) -> int:
    a = int(CELL_TO_COMPACT[anchor])
    gi = POSITIVE_DIRS.index(group_dir)
    mi = BROADSIDE_DIRS[gi].index(move_dir)
    return BROADSIDE_OFFSET + a * 24 + gi * 8 + mi * 2 + (size - 2)


def decode_broadside(idx: int) -> tuple[int, int, int, int]:
    j = idx - BROADSIDE_OFFSET
    a = j // 24
    rem = j % 24
    gi = rem // 8
    rem2 = rem % 8
    mi = rem2 // 2
    size = (rem2 % 2) + 2
    anchor = int(COMPACT_TO_CELL[a])
    group_dir = POSITIVE_DIRS[gi]
    move_dir = BROADSIDE_DIRS[gi][mi]
    return anchor, group_dir, move_dir, size


def _line_on_board(anchor: int, dir_idx: int, size: int) -> bool:
    """All cells `anchor + i*shift` for `i in 0..size` are on-board."""
    shift = int(DIR_SHIFTS[dir_idx])
    for i in range(size):
        c = anchor + i * shift
        if c < 0 or c >= NUM_CELLS or CELL_TO_COMPACT[c] == 255:
            return False
    return True


def _build_move_perm() -> np.ndarray:
    """`MOVE_PERM[s, idx]` = where move `idx` is sent under sym `s`.
    Pre-computed once; thereafter sym application to a policy vector
    is a single fancy-indexing op.

    The 2562-index space includes structurally invalid moves (e.g. a
    size-3 inline whose line extends off the board). Those never come
    out of `legal_moves()` and never appear in policy vectors we'd
    actually rotate, so we map them to themselves (identity) and the
    permutation stays well-defined as a bijection over the valid subset
    plus a fixed-point on the invalid subset."""
    out = np.zeros((NUM_SYMS, MOVE_SPACE), dtype=np.int32)
    for s in range(NUM_SYMS):
        for idx in range(MOVE_SPACE):
            if idx < BROADSIDE_OFFSET:
                anchor, d, size = decode_inline(idx)
                if not _line_on_board(anchor, d, size):
                    out[s, idx] = idx
                    continue
                anchor2 = int(CELL_PERM[s, anchor])
                d2 = int(DIR_PERM[s, d])
                out[s, idx] = encode_inline(anchor2, d2, size)
            else:
                anchor, gd, md, size = decode_broadside(idx)
                if not _line_on_board(anchor, gd, size):
                    out[s, idx] = idx
                    continue
                anchor2 = int(CELL_PERM[s, anchor])
                gd2 = int(DIR_PERM[s, gd])
                md2 = int(DIR_PERM[s, md])
                if gd2 in POSITIVE_DIRS:
                    out[s, idx] = encode_broadside(anchor2, gd2, md2, size)
                else:
                    dg_shift = int(DIR_SHIFTS[gd2])
                    other_end = anchor2 + (size - 1) * dg_shift
                    out[s, idx] = encode_broadside(
                        other_end, OPP_DIR[gd2], md2, size
                    )
    return out


MOVE_PERM = _build_move_perm()


# ----- plane construction ---------------------------------------------------


@dataclass
class PositionRecord:
    """Mirror of one parquet row, parsed into native types."""

    own_bb: int  # u128, the side-to-move's bitboard
    opp_bb: int
    own_pushed_off: int  # number of OWN marbles already pushed off
    opp_pushed_off: int
    ply: int
    turn: int  # 0 = Black, 1 = White; informational only


def bb_to_plane(bb: int) -> np.ndarray:
    """Decode a 128-bit bitboard into a 9×9 binary float plane."""
    out = np.zeros((9, 9), dtype=np.float32)
    for c in range(NUM_CELLS):
        if (bb >> c) & 1:
            r, q = divmod(c, 9)
            out[r, q] = 1.0
    return out


def encode_position(rec: PositionRecord) -> np.ndarray:
    """Build the 6-plane (6, 9, 9) tensor for one position."""
    planes = np.zeros((6, 9, 9), dtype=np.float32)
    planes[0] = bb_to_plane(rec.own_bb)
    planes[1] = bb_to_plane(rec.opp_bb)
    planes[2].fill(rec.own_pushed_off / 6.0)
    planes[3].fill(rec.opp_pushed_off / 6.0)
    planes[4].fill(min(rec.ply, 400) / 400.0)
    planes[5] = VALID_CELL_MASK
    return planes


# ----- symmetry application -------------------------------------------------


def apply_sym_to_planes(planes: np.ndarray, sym: int) -> np.ndarray:
    """Permute the spatial dimensions of `planes` (..., 9, 9) by sym
    `s`. Channels are unchanged. Constant-broadcast channels are
    invariant under permutation; we still apply the index gather to
    keep the code uniform.

    Convention: a marble at cell `c` ends up at cell `CELL_PERM[s, c]`.
    Implementation: for each output cell c', pull from input cell
    such that `c -> c'`, i.e., from `CELL_PERM_INV[s, c']`. We compute
    the inverse on the fly via `argsort` for small fixed `s`."""
    perm = CELL_PERM[sym]  # (81,)
    # Inverse permutation (over valid cells; off-board entries are 255).
    inv = np.full(NUM_CELLS, 255, dtype=np.int16)
    for c in range(NUM_CELLS):
        p = perm[c]
        if p != 255:
            inv[p] = c
    # Apply: out[r', q'] = in[inv[r'*9+q']]
    flat_in = planes.reshape(*planes.shape[:-2], 81)
    out_flat = np.zeros_like(flat_in)
    valid = inv != 255
    out_flat[..., valid] = flat_in[..., inv[valid]]
    return out_flat.reshape(planes.shape)


def apply_sym_to_policy(policy: np.ndarray, sym: int) -> np.ndarray:
    """Permute a length-`MOVE_SPACE` policy vector by sym. The marble
    that *was* the best move at index `i` is now at index
    `MOVE_PERM[s, i]` after rotation.

    `policy.shape[-1]` must be `MOVE_SPACE`."""
    assert policy.shape[-1] == MOVE_SPACE
    out = np.zeros_like(policy)
    perm = MOVE_PERM[sym]  # (MOVE_SPACE,)
    out[..., perm] = policy
    return out


# ----- self-checks runnable as `python -m model.encoder` --------------------


def _sanity_checks() -> None:
    # Symmetries form a group: applying s then s should return to identity.
    for s in range(NUM_SYMS):
        for c in range(NUM_CELLS):
            if CELL_PERM[s, c] == 255:
                continue
            # Find the inverse element by checking which sym round-trips.
        # Just smoke-test that 60° applied 6 times is identity.
    for c in range(NUM_CELLS):
        if CELL_PERM[1, c] == 255:
            continue
        cur = c
        for _ in range(6):
            cur = int(CELL_PERM[1, cur])
        assert cur == c, f"R^6 not identity for cell {c}"

    # MOVE_PERM round-trips for all moves under R^6.
    for idx in range(MOVE_SPACE):
        cur = idx
        for _ in range(6):
            cur = int(MOVE_PERM[1, cur])
        assert cur == idx, f"R^6 didn't round-trip move {idx} (got {cur})"

    # F is involution.
    for c in range(NUM_CELLS):
        if CELL_PERM[6, c] == 255:
            continue
        assert CELL_PERM[6, CELL_PERM[6, c]] == c
    for idx in range(MOVE_SPACE):
        assert MOVE_PERM[6, MOVE_PERM[6, idx]] == idx

    print("encoder sanity checks passed")
    print(f"  CELL_PERM       shape={CELL_PERM.shape}")
    print(f"  MOVE_PERM       shape={MOVE_PERM.shape}")
    print(f"  VALID_CELL_MASK on-board cells = {int(VALID_CELL_MASK.sum())}")


if __name__ == "__main__":
    _sanity_checks()
