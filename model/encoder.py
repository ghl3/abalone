"""Position encoding and D6 symmetry augmentation.

Plane layout — `(14, 9, 9)`, matching `abalone_net.py`'s
`INPUT_CHANNELS = 14`, `selfplay/src/encoder.rs`, and MODEL.md §5:

    0     own marbles     9×9 binary   — the side-to-move's marbles
    1     opp marbles     9×9 binary
    2–6   own losses      9×9 constant — thermometer, `own_losses >= k+1`
    7–11  opp losses      9×9 constant — thermometer, `opp_losses >= k+1`
    12    ply             9×9 constant — `ply / max_plies`, clamped to [0, 1]
    13    valid mask      9×9 binary   — 1 for the 61 on-board cells

Channels 0/1 are oriented to side-to-move (NOT to a fixed colour), so
the same network drives both sides without a side-to-move embedding.

**Thermometer, not a `count / 6` scalar.** Losses are an ordinal quantity
with a hard threshold at 6; `>=1, >=2, ...` lets a single linear layer
read off both the magnitude and "one away from losing".

**Constant planes are filled across all 81 cells**, off-board slots
included. The network masks off-board activations internally, so the
encoder must not pre-mask — and the Rust encoder does exactly the same.

**This layout is a cross-language contract.** `tests/test_conformance.py`
replays golden fixtures emitted by the Rust `dump-golden` binary through
`encode_position` and asserts bitwise float equality. A swapped
capture-plane divergence between the two encoders was live for three
generations before that guard existed; do not change this file without
regenerating and re-running the conformance fixtures.

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

# ----- plane layout (mirror of selfplay/src/encoder.rs) ----------------------

NUM_INPUT_CHANNELS = 14
BOARD_H = 9
BOARD_W = 9
PLANE_SIZE = NUM_INPUT_CHANNELS * BOARD_H * BOARD_W  # 1134

OWN_MARBLES_PLANE = 0
OPP_MARBLES_PLANE = 1
OWN_LOSSES_PLANE = 2
OPP_LOSSES_PLANE = 7
# Thresholds per thermometer: `>=1 .. >=5`. Six losses is terminal, so a
# `>=6` plane would only ever be set in a position that is already over.
LOSS_THERMOMETER_PLANES = 5
PLY_PLANE = 12
VALID_MASK_PLANE = 13

assert OPP_LOSSES_PLANE == OWN_LOSSES_PLANE + LOSS_THERMOMETER_PLANES
assert PLY_PLANE == OPP_LOSSES_PLANE + LOSS_THERMOMETER_PLANES
assert NUM_INPUT_CHANNELS == VALID_MASK_PLANE + 1

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
    """Mirror of one parquet row, parsed into native types.

    The loss counters are named for *whose marbles were lost*, not for who
    did the pushing. The Rust `Board::pushed_off[s]` field counts the
    OPPONENT marbles side `s` has pushed off — it reads naturally as the
    opposite of what it holds, and reading it as "s's losses" is precisely
    the bug that swapped the capture planes for three generations. Here
    `own_losses` unambiguously means *marbles of the side to move that
    have been pushed off the board* (Rust: `board.lost(game.turn)`).
    """

    own_bb: int  # u128, the side-to-move's bitboard
    opp_bb: int
    own_losses: int  # marbles the SIDE TO MOVE has lost (had pushed off)
    opp_losses: int  # marbles the opponent has lost
    ply: int
    max_plies: int  # the game's own ply cap; normalises plane 12
    turn: int  # 0 = Black, 1 = White; informational only


def bb_to_plane(bb: int) -> np.ndarray:
    """Decode a 128-bit bitboard into a 9×9 binary float plane."""
    out = np.zeros((9, 9), dtype=np.float32)
    for c in range(NUM_CELLS):
        if (bb >> c) & 1:
            r, q = divmod(c, 9)
            out[r, q] = 1.0
    return out


def ply_fraction(ply: int, max_plies: int) -> float:
    """`ply / max_plies` clamped to [0, 1]. A zero cap is degenerate
    configuration; it yields 0 rather than a ZeroDivisionError, and
    `selfplay/src/encoder.rs::ply_fraction` agrees."""
    if max_plies == 0:
        return 0.0
    return min(max(ply / max_plies, 0.0), 1.0)


def encode_position(rec: PositionRecord) -> np.ndarray:
    """Build the `(14, 9, 9)` input tensor for one position.

    Byte-for-byte identical to `selfplay/src/encoder.rs::encode_planes`;
    `tests/test_conformance.py` enforces that against golden fixtures.
    """
    planes = np.zeros((NUM_INPUT_CHANNELS, BOARD_H, BOARD_W), dtype=np.float32)

    # 0/1 — marbles, side-to-move relative.
    planes[OWN_MARBLES_PLANE] = bb_to_plane(rec.own_bb)
    planes[OPP_MARBLES_PLANE] = bb_to_plane(rec.opp_bb)

    # 2–11 — loss thermometers. Plane `base + k` is 1.0 iff `losses >= k+1`.
    # NOTE the direction: 2–6 are the losses of the side TO MOVE.
    for k in range(LOSS_THERMOMETER_PLANES):
        threshold = k + 1
        if rec.own_losses >= threshold:
            planes[OWN_LOSSES_PLANE + k].fill(1.0)
        if rec.opp_losses >= threshold:
            planes[OPP_LOSSES_PLANE + k].fill(1.0)

    # 12 — normalised ply, against the game's own cap (not a hardcoded
    # constant), so changing the cap moves both implementations together.
    planes[PLY_PLANE].fill(ply_fraction(rec.ply, rec.max_plies))

    # 13 — valid-cell mask. The one plane that is NOT filled across all 81
    # cells; everything above deliberately covers the off-board slots too.
    planes[VALID_MASK_PLANE] = VALID_CELL_MASK
    return planes


# ----- symmetry application -------------------------------------------------


def _build_cell_perm_inv() -> np.ndarray:
    """`CELL_PERM_INV[s, c']` = the cell that maps *to* `c'` under sym `s`.

    Precomputed once at import. This used to be rebuilt with an 81-iteration
    Python loop on every call — which the replay buffer made a per-*example*
    cost inside the sampling loop."""
    out = np.full((NUM_SYMS, NUM_CELLS), 255, dtype=np.int16)
    for s in range(NUM_SYMS):
        for c in range(NUM_CELLS):
            p = CELL_PERM[s, c]
            if p != 255:
                out[s, p] = c
    return out


CELL_PERM_INV = _build_cell_perm_inv()

# Off-board slots, as a flat 81-length boolean mask. These are fixed points of
# every board symmetry — see `apply_sym_to_planes`.
_OFF_BOARD_FLAT = CELL_PERM_INV[0] == 255


def apply_sym_to_planes(planes: np.ndarray, sym: int) -> np.ndarray:
    """Permute the spatial dimensions of `planes` (..., 9, 9) by sym `s`.
    Channels are unchanged.

    Convention: a value at cell `c` ends up at cell `CELL_PERM[s, c]`.
    Implementation: for each output cell `c'`, pull from `CELL_PERM_INV[s, c']`.

    Off-board cells are carried through UNCHANGED rather than zeroed. The 20
    off-board slots have no image under the board symmetry, so they are fixed
    points of it; zeroing them would make this function do two things at once
    (permute *and* mask) and, at identity, not be a no-op.

    That distinction is not academic. `encode_position` fills the constant
    planes (losses, ply) across all 81 cells, so a zeroing implementation made
    augmented training examples differ from inference-time encodings in exactly
    those 20 slots — a silent train/inference divergence. Masking off-board
    cells is the *network's* job (it does so on input and after every block),
    not the augmentation's."""
    inv = CELL_PERM_INV[sym]
    flat_in = planes.reshape(*planes.shape[:-2], 81)
    out_flat = flat_in.copy()
    valid = ~_OFF_BOARD_FLAT
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

    # Plane layout: shape, thermometer direction, and the deliberate lack of
    # masking on the constant planes.
    probe = encode_position(
        PositionRecord(
            own_bb=0, opp_bb=0, own_losses=3, opp_losses=1,
            ply=50, max_plies=200, turn=0,
        )
    )
    assert probe.shape == (NUM_INPUT_CHANNELS, BOARD_H, BOARD_W)
    assert probe.dtype == np.float32
    assert [probe[OWN_LOSSES_PLANE + k][0, 0] for k in range(5)] == [1, 1, 1, 0, 0]
    assert [probe[OPP_LOSSES_PLANE + k][0, 0] for k in range(5)] == [1, 0, 0, 0, 0]
    assert probe[PLY_PLANE].sum() == 81 * 0.25, "constant planes must not be masked"
    assert probe[VALID_MASK_PLANE].sum() == NUM_VALID

    print("encoder sanity checks passed")
    print(f"  input planes    {NUM_INPUT_CHANNELS} x {BOARD_H} x {BOARD_W} = {PLANE_SIZE}")
    print(f"  CELL_PERM       shape={CELL_PERM.shape}")
    print(f"  MOVE_PERM       shape={MOVE_PERM.shape}")
    print(f"  VALID_CELL_MASK on-board cells = {int(VALID_CELL_MASK.sum())}")


if __name__ == "__main__":
    _sanity_checks()
