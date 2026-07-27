"""Replay buffer: streaming parquet ingest, compact storage, vectorised sampling.

Shard schema v2 is normative in `docs/ARCHITECTURE.md` §5.4; the batch this
module produces is normative in §5.5 and pinned by `model/batch.py`.

Three properties drive the design.

**Store bitboards, not planes.** A position costs ~48 bytes of fixed-size
columns here (four `u64` bitboard halves plus a handful of small scalars)
against `14 × 81 × 4 = 4536` bytes for a dense `float32` plane stack. Planes
are reconstructed at sample time by `np.unpackbits` over the whole batch at
once, which is cheaper than the memory traffic of reading them back from a
resident array would have been. The ragged search result (`child_move_idxs` /
`child_visits`) is the dominant term in practice — see `nbytes()`.

**No Python loop over the batch.** `sample()` is a fixed number of numpy
operations regardless of `batch_size`: one gather over the fixed-size columns,
one ragged gather (CSR-style) over the search results and capture map, one
`unpackbits`, and a handful of scatters. The D6 symmetry is applied by
indexing precomputed permutation tables — `CELL_PERM_INV` for the spatial
planes, `MOVE_PERM` for policy and legality, `CAP_PERM` for the capture map —
so augmentation costs no Python-level work per example either.

**Amortised ingest.** Shards land in a per-generation pending list (O(rows
added)) and are merged into one contiguous columnar store the first time
something samples. Self-play writes one file per game, so a generation ingests
~200 shards and then serves thousands of batches: exactly one merge per
generation cycle, not one re-copy per file.

Capture-map channels are POV-relative but cells are absolute, so the capture
map is *spatial* and must be permuted by the same symmetry as `planes` and
`policy`. Getting that wrong trains the auxiliary head against rotated labels
without any visible error; `tests/test_replay_buffer.py` pins it.

Eviction is by absolute generation: `evict_below(K)` drops every generation
below `K`, except rows in the frozen holdout, which outlive the window by
design. The training loop calls it with `current_gen - replay_buffer_gens`.

**Two holdouts, and they measure different things.**

*Frozen* — `mark_holdout(gen, count=N)` withholds **approximately N positions**
of `gen` as validation-only. Those rows are never returned by `sample()`, are
not counted by `total_size()` (so they cannot satisfy the training loop's
minimum-buffer check), and survive `evict_below` so the set stays fixed for the
life of the run. Because the self-play distribution moves by design,
cross-entropy against a generation-1 frozen set increasingly measures **drift
away from an obsolete distribution**, not generalisation. It is a drift
indicator. Do not gate on it.

**It is bounded on purpose.** Freezing the whole generation — what this did
before — threw away the largest generation in the run: games are longest under
random play, so generation 1 of a live run was 57,699 positions against
generation 2's 33,717, and every one of them was withheld permanently to feed a
metric nothing is allowed to gate on. Worse, it emptied the pool outside the
current generation, which silently disabled the *rolling* holdout (see the
minimum-buffer guard in `train_loop`) — the one measurement that could have
caught overfitting. A drift indicator needs enough rows to evaluate on and not
one more; `validation.holdout_positions` sets that, defaulting to
`validation.positions` because scoring more rows than a validation pass draws is
pure waste. The un-held remainder of the generation is ordinary trainable data
and ages out of the window normally.

*Rolling* — `mark_rolling_holdout(gen, fraction=...)` withholds a slice of the
*newest* generation. Those rows are excluded from `sample()` permanently, so
they are never trained on, and they are drawn from the distribution the network
is actually being trained on right now. That is the set worth gating on.

**Both slices are taken by whole game**, never by row — `_select_whole_games` is
shared so they cannot drift apart. Positions from one game share the same `z`
and `score_diff` labels and are near-duplicates of each other; a row-level split
would leave sibling positions on both sides and report memorisation as
generalisation. `whole_games=False` exists for tests that need an exact row
count.

Both selections are deterministic in `(seed, gen)` and idempotent, so a resumed
run reproduces exactly the same rows rather than quietly re-rolling a split
across data the network has already trained on.

`model/validate.py` scores either set; `iter_holdout_batches(...)` is the
fixed-seed, augmentation-off iterator that makes the curve comparable across
generations.

`buffer_metrics()` is the `buffer/*` block — size, generation span, seeded
fraction, and `epochs_this_gen = steps * batch_size / size`, the number that
says whether a generation's training is a light touch or twenty passes over the
same data.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from model.batch import (
    BOARD_H,
    BOARD_W,
    CAPTURE_MAP_CHANNELS,
    INPUT_PLANES,
    MOVE_SPACE,
    Batch,
    score_class,
    value_class,
)
from model.encoder import (
    CELL_PERM,
    LOSS_THERMOMETER_PLANES,
    MOVE_PERM,
    NUM_CELLS,
    NUM_SYMS,
    OPP_LOSSES_PLANE,
    OPP_MARBLES_PLANE,
    OWN_LOSSES_PLANE,
    OWN_MARBLES_PLANE,
    PLY_PLANE,
    VALID_CELL_MASK,
    VALID_MASK_PLANE,
)
from model.encoder import MOVE_SPACE as _ENCODER_MOVE_SPACE
from model.encoder import NUM_INPUT_CHANNELS as _ENCODER_INPUT_PLANES

# The batch contract and the encoder are separately authored mirrors of the same
# cross-language spec. If they ever disagree, fail at import rather than at the
# next silent broadcast.
assert MOVE_SPACE == _ENCODER_MOVE_SPACE, "MOVE_SPACE disagrees between batch and encoder"
assert INPUT_PLANES == _ENCODER_INPUT_PLANES, "plane count disagrees between batch and encoder"

#: Flat size of the sparse capture-map index space: `channel * 81 + cell`.
CAP_MAP_SIZE = CAPTURE_MAP_CHANNELS * NUM_CELLS  # 162

#: Namespace for `buffer_metrics`. Properties of the replay buffer, not of the
#: model and not of this generation's self-play.
BUFFER_PREFIX = "buffer/"

#: Columns required by shard schema v2. A v1 shard (`pushed_off_black`, no
#: `is_full_search`, no capture map) fails loudly here instead of training on
#: silently wrong labels.
REQUIRED_COLUMNS: tuple[str, ...] = (
    "game_id",
    "seed",
    "opening",
    "handicap_black",
    "handicap_white",
    "own_bb_lo",
    "own_bb_hi",
    "opp_bb_lo",
    "opp_bb_hi",
    "black_losses",
    "white_losses",
    "turn",
    "ply",
    "max_plies",
    "is_full_search",
    "z",
    "score_diff",
    "q",
    "child_move_idxs",
    "child_visits",
    "cap_map_idx",
    "cap_map_val",
)

#: Per-row provenance from §5.4. Not part of the training batch — carried so a
#: buffered position can still be traced back to a reproducible game.
META_DTYPE = np.dtype(
    [
        ("game_id", "<u4"),
        ("seed", "<u8"),
        ("opening", "u1"),
        ("handicap_black", "u1"),
        ("handicap_white", "u1"),
    ]
)


# ----- precomputed permutation / lookup tables -------------------------------


def _build_cell_perm_inv() -> np.ndarray:
    """`CELL_PERM_INV[s, c'] = c` such that `CELL_PERM[s, c] == c'`.

    `apply_sym_to_planes` rebuilt this per call (per *example*, in the old
    sampler). It is a fixed 12×81 table; build it once.

    Off-board cells have no valid preimage and are left as fixed points. That
    is safe for anything derived from a bitboard, where off-board slots are
    always zero, and it keeps the table usable as a plain gather index with no
    sentinel handling in the hot path.
    """
    inv = np.tile(np.arange(NUM_CELLS, dtype=np.int64), (NUM_SYMS, 1))
    for s in range(NUM_SYMS):
        for c in range(NUM_CELLS):
            p = int(CELL_PERM[s, c])
            if p != 255:
                inv[s, p] = c
    return inv


def _build_cap_perm() -> np.ndarray:
    """`CAP_PERM[s, i]` for the flat capture-map index `i = channel*81 + cell`.

    Channels are POV-relative and therefore sym-invariant; only the cell moves.
    """
    out = np.empty((NUM_SYMS, CAP_MAP_SIZE), dtype=np.int64)
    for s in range(NUM_SYMS):
        for ch in range(CAPTURE_MAP_CHANNELS):
            for c in range(NUM_CELLS):
                p = int(CELL_PERM[s, c])
                if p == 255:
                    p = c
                out[s, ch * NUM_CELLS + c] = ch * NUM_CELLS + p
    return out


CELL_PERM_INV = _build_cell_perm_inv()
CAP_PERM = _build_cap_perm()

#: `_LOSS_THERMOMETER[n, k] = 1.0 iff n >= k + 1`, indexed by a `u8` loss count.
_LOSS_THERMOMETER = (
    np.arange(256, dtype=np.int64)[:, None] >= np.arange(1, LOSS_THERMOMETER_PLANES + 1)[None, :]
).astype(np.float32)

# Class-index tables built *from* the pinned mappings, so the vectorised path
# cannot drift from `value_class` / `score_class`. Indexed by `int8 + 128`.
_VALUE_CLASS_LUT = np.array([value_class(v) for v in range(-128, 128)], dtype=np.int64)
_SCORE_CLASS_LUT = np.array([score_class(v) for v in range(-128, 128)], dtype=np.int64)


# ----- columnar storage ------------------------------------------------------


def _ragged_take(
    offsets: np.ndarray, arrays: Sequence[np.ndarray], rows: np.ndarray
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    """Gather whole rows out of a CSR-style ragged column.

    Returns new offsets plus the compacted value arrays, in `rows` order.
    Vectorised: the per-row runs are materialised with one `repeat` and one
    `arange`, never a Python loop.
    """
    starts = offsets[rows]
    counts = offsets[rows + 1] - starts
    new_off = np.zeros(len(rows) + 1, dtype=np.int64)
    np.cumsum(counts, out=new_off[1:])
    total = int(new_off[-1])
    if total == 0:
        return new_off, tuple(a[:0].copy() for a in arrays)
    src = (
        np.arange(total, dtype=np.int64)
        - np.repeat(new_off[:-1], counts)
        + np.repeat(starts, counts)
    )
    return new_off, tuple(a[src] for a in arrays)


@dataclass
class _Columns:
    """Row-parallel columns for a set of positions.

    Everything is stored at the width the schema uses, POV-resolved once at
    ingest. Grouping the four bitboard halves into a single `(N, 4)` `<u8`
    array is not cosmetic: `bb.view(np.uint8)` then feeds `unpackbits`
    directly, decoding both players' boards in one call.
    """

    gen: np.ndarray  # (N,)   int32
    bb: np.ndarray  # (N, 4) '<u8'   own_lo, own_hi, opp_lo, opp_hi
    losses: np.ndarray  # (N, 2) uint8   own, opp — POV-resolved at ingest
    plies: np.ndarray  # (N, 2) uint16  ply, max_plies
    turn: np.ndarray  # (N,)   uint8   0 = Black, 1 = White
    is_full: np.ndarray  # (N,)   bool
    z: np.ndarray  # (N,)   int8
    score_diff: np.ndarray  # (N,)   int8
    q: np.ndarray  # (N,)   float32
    meta: np.ndarray  # (N,)   META_DTYPE
    #: Rolling-holdout flag, per row. True rows are never returned by
    #: `sample()`. Carried as a column rather than as a side table because
    #: `take()` and `concat()` reorder and rebuild the store, and a holdout
    #: that silently re-enters the training pool at the next eviction would be
    #: worse than no holdout at all.
    hold: np.ndarray  # (N,)   bool
    #: Frozen-holdout flag, per row. Same argument as `hold`, plus one more:
    #: `evict_below` keeps these rows while dropping the rest of their
    #: generation, which is only expressible per row.
    frozen: np.ndarray  # (N,)   bool
    child_off: np.ndarray  # (N+1,) int64
    child_idx: np.ndarray  # (E,)   uint16
    child_visits: np.ndarray  # (E,)   uint32
    cap_off: np.ndarray  # (N+1,) int64
    cap_idx: np.ndarray  # (F,)   uint16
    cap_val: np.ndarray  # (F,)   float32

    _ROW_FIELDS = (
        "gen",
        "bb",
        "losses",
        "plies",
        "turn",
        "is_full",
        "z",
        "score_diff",
        "q",
        "meta",
        "hold",
        "frozen",
    )
    _RAGGED_FIELDS = (
        ("child_off", ("child_idx", "child_visits")),
        ("cap_off", ("cap_idx", "cap_val")),
    )

    @property
    def size(self) -> int:
        return int(self.gen.shape[0])

    @property
    def nbytes(self) -> int:
        return sum(
            int(getattr(self, f).nbytes)
            for f in (
                *self._ROW_FIELDS,
                "child_off",
                "child_idx",
                "child_visits",
                "cap_off",
                "cap_idx",
                "cap_val",
            )
        )

    def take(self, rows: np.ndarray) -> _Columns:
        """A new `_Columns` holding `rows`, in the given order. Fully copied,
        so a sampled batch never aliases the store."""
        rows = np.asarray(rows, dtype=np.int64)
        kwargs = {f: getattr(self, f)[rows] for f in self._ROW_FIELDS}
        for off_name, val_names in self._RAGGED_FIELDS:
            new_off, vals = _ragged_take(
                getattr(self, off_name), [getattr(self, v) for v in val_names], rows
            )
            kwargs[off_name] = new_off
            kwargs.update(dict(zip(val_names, vals, strict=True)))
        return _Columns(**kwargs)

    @staticmethod
    def empty() -> _Columns:
        return _Columns(
            gen=np.zeros(0, np.int32),
            bb=np.zeros((0, 4), "<u8"),
            losses=np.zeros((0, 2), np.uint8),
            plies=np.zeros((0, 2), np.uint16),
            turn=np.zeros(0, np.uint8),
            is_full=np.zeros(0, np.bool_),
            z=np.zeros(0, np.int8),
            score_diff=np.zeros(0, np.int8),
            q=np.zeros(0, np.float32),
            meta=np.zeros(0, META_DTYPE),
            hold=np.zeros(0, np.bool_),
            frozen=np.zeros(0, np.bool_),
            child_off=np.zeros(1, np.int64),
            child_idx=np.zeros(0, np.uint16),
            child_visits=np.zeros(0, np.uint32),
            cap_off=np.zeros(1, np.int64),
            cap_idx=np.zeros(0, np.uint16),
            cap_val=np.zeros(0, np.float32),
        )

    @staticmethod
    def concat(parts: Sequence[_Columns]) -> _Columns:
        parts = [p for p in parts if p.size > 0]
        if not parts:
            return _Columns.empty()
        if len(parts) == 1:
            return parts[0]
        kwargs: dict[str, np.ndarray] = {
            f: np.concatenate([getattr(p, f) for p in parts]) for f in _Columns._ROW_FIELDS
        }
        for off_name, val_names in _Columns._RAGGED_FIELDS:
            chunks = [np.zeros(1, np.int64)]
            base = 0
            for p in parts:
                off = getattr(p, off_name)
                chunks.append(off[1:] + base)
                base += int(off[-1])
            kwargs[off_name] = np.concatenate(chunks)
            for v in val_names:
                kwargs[v] = np.concatenate([getattr(p, v) for p in parts])
        return _Columns(**kwargs)


# ----- parquet ingest --------------------------------------------------------


def _checked_column(table: pa.Table, name: str) -> pa.ChunkedArray:
    """Nulls would come back from `to_numpy` as NaN in a float array and then be
    cast to garbage integers. No shard column is nullable; say so out loud."""
    col = table.column(name)
    if col.null_count:
        raise ValueError(
            f"column {name!r} contains {col.null_count} nulls; shard columns are not nullable"
        )
    return col


def _scalar_column(table: pa.Table, name: str, dtype: np.dtype | str) -> np.ndarray:
    values = _checked_column(table, name).to_numpy(zero_copy_only=False)
    return np.ascontiguousarray(values, dtype=dtype)


def _list_column(
    table: pa.Table, name: str, dtype: np.dtype | str
) -> tuple[np.ndarray, np.ndarray]:
    """Read a `list<...>` column as (offsets int64, flat values)."""
    arr = _checked_column(table, name)
    if isinstance(arr, pa.ChunkedArray):
        arr = arr.combine_chunks()
        if isinstance(arr, pa.ChunkedArray):  # older pyarrow keeps the wrapper
            arr = (
                arr.chunk(0)
                if arr.num_chunks == 1
                else pa.concat_arrays([c for c in arr.iterchunks()])
            )
    offsets = np.asarray(arr.offsets.to_numpy(zero_copy_only=False), dtype=np.int64)
    values = np.asarray(arr.values.to_numpy(zero_copy_only=False), dtype=dtype)
    if offsets.size == 0:
        return np.zeros(1, np.int64), values[:0]
    # A sliced ListArray keeps the parent's value buffer; rebase onto it.
    values = np.ascontiguousarray(values[offsets[0] : offsets[-1]])
    return offsets - offsets[0], values


def _read_shard(path: Path, gen: int) -> _Columns:
    """Decode one shard file into `_Columns`. Raises on a non-v2 schema."""
    table = pq.read_table(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in table.column_names]
    if missing:
        raise ValueError(
            f"{path}: not a v2 shard — missing columns {missing}. "
            "See docs/ARCHITECTURE.md section 5.4."
        )
    n = table.num_rows
    if n == 0:
        return _Columns.empty()

    turn = _scalar_column(table, "turn", np.uint8)
    black_losses = _scalar_column(table, "black_losses", np.uint8)
    white_losses = _scalar_column(table, "white_losses", np.uint8)
    # THE inversion. `black_losses` is marbles BLACK HAS LOST, so for Black to
    # move (turn == 0) it is *own* losses. The v1 columns were named for who
    # did the pushing and were read as if named for who was pushed; that swap
    # trained three generations against mirrored capture planes.
    black_to_move = turn == 0
    own_losses = np.where(black_to_move, black_losses, white_losses)
    opp_losses = np.where(black_to_move, white_losses, black_losses)

    child_off, child_idx = _list_column(table, "child_move_idxs", np.uint16)
    _, child_visits = _list_column(table, "child_visits", np.uint32)
    cap_off, cap_idx = _list_column(table, "cap_map_idx", np.uint16)
    _, cap_val = _list_column(table, "cap_map_val", np.float32)

    if child_idx.shape != child_visits.shape:
        raise ValueError(f"{path}: child_move_idxs and child_visits have different lengths")
    if cap_idx.shape != cap_val.shape:
        raise ValueError(f"{path}: cap_map_idx and cap_map_val have different lengths")
    if child_idx.size and int(child_idx.max()) >= MOVE_SPACE:
        raise ValueError(f"{path}: child move index >= MOVE_SPACE ({MOVE_SPACE})")
    if cap_idx.size and int(cap_idx.max()) >= CAP_MAP_SIZE:
        raise ValueError(f"{path}: cap_map_idx >= {CAP_MAP_SIZE} (channel*81 + cell)")

    meta = np.zeros(n, META_DTYPE)
    meta["game_id"] = _scalar_column(table, "game_id", np.uint32)
    meta["seed"] = _scalar_column(table, "seed", np.uint64)
    meta["opening"] = _scalar_column(table, "opening", np.uint8)
    meta["handicap_black"] = _scalar_column(table, "handicap_black", np.uint8)
    meta["handicap_white"] = _scalar_column(table, "handicap_white", np.uint8)

    return _Columns(
        gen=np.full(n, gen, np.int32),
        bb=np.stack(
            [
                _scalar_column(table, "own_bb_lo", "<u8"),
                _scalar_column(table, "own_bb_hi", "<u8"),
                _scalar_column(table, "opp_bb_lo", "<u8"),
                _scalar_column(table, "opp_bb_hi", "<u8"),
            ],
            axis=1,
        ),
        losses=np.stack([own_losses, opp_losses], axis=1).astype(np.uint8),
        plies=np.stack(
            [
                _scalar_column(table, "ply", np.uint16),
                _scalar_column(table, "max_plies", np.uint16),
            ],
            axis=1,
        ),
        turn=turn,
        is_full=_scalar_column(table, "is_full_search", np.bool_),
        z=_scalar_column(table, "z", np.int8),
        score_diff=_scalar_column(table, "score_diff", np.int8),
        q=_scalar_column(table, "q", np.float32),
        meta=meta,
        hold=np.zeros(n, np.bool_),
        frozen=np.zeros(n, np.bool_),
        child_off=child_off,
        child_idx=child_idx,
        child_visits=child_visits,
        cap_off=cap_off,
        cap_idx=cap_idx,
        cap_val=cap_val,
    )


# ----- batch construction ----------------------------------------------------


def _decode_bitboards(bb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`(N, 4)` little-endian u64 halves -> two `(N, 81)` uint8 bit arrays.

    The four halves are contiguous, so one `unpackbits` over the raw bytes
    yields bit `c` of the own board at column `c` and of the opponent board at
    column `128 + c` — cell index and bit index coincide by construction.
    """
    if bb.shape[0] == 0:
        empty = np.zeros((0, NUM_CELLS), dtype=np.uint8)
        return empty, empty.copy()
    raw = np.ascontiguousarray(bb).view(np.uint8)  # (N, 32)
    bits = np.unpackbits(raw, axis=1, bitorder="little")  # (N, 256)
    return bits[:, :NUM_CELLS], bits[:, 128 : 128 + NUM_CELLS]


def _build_batch(cols: _Columns, syms: np.ndarray | None) -> Batch:
    """Densify `cols` into a `Batch`, optionally applying one D6 symmetry per
    row. `syms is None` means identity for every row and skips the gathers."""
    b = cols.size
    rows = np.arange(b)

    own_bits, opp_bits = _decode_bitboards(cols.bb)
    if syms is not None:
        inv = CELL_PERM_INV[syms]  # (B, 81)
        own_bits = np.take_along_axis(own_bits, inv, axis=1)
        opp_bits = np.take_along_axis(opp_bits, inv, axis=1)

    planes = np.zeros((b, INPUT_PLANES, BOARD_H, BOARD_W), dtype=np.float32)
    planes[:, OWN_MARBLES_PLANE] = own_bits.reshape(b, BOARD_H, BOARD_W)
    planes[:, OPP_MARBLES_PLANE] = opp_bits.reshape(b, BOARD_H, BOARD_W)
    therm_lo = OWN_LOSSES_PLANE
    therm_hi = OWN_LOSSES_PLANE + LOSS_THERMOMETER_PLANES
    planes[:, therm_lo:therm_hi] = _LOSS_THERMOMETER[cols.losses[:, 0]][:, :, None, None]
    planes[:, OPP_LOSSES_PLANE : OPP_LOSSES_PLANE + LOSS_THERMOMETER_PLANES] = _LOSS_THERMOMETER[
        cols.losses[:, 1]
    ][:, :, None, None]
    # Matches `encoder.ply_fraction`: float64 divide, clamp, then narrow.
    ply = cols.plies[:, 0].astype(np.float64)
    cap = cols.plies[:, 1].astype(np.float64)
    frac = np.clip(np.divide(ply, cap, out=np.zeros(b), where=cap > 0), 0.0, 1.0)
    planes[:, PLY_PLANE] = frac[:, None, None]
    planes[:, VALID_MASK_PLANE] = VALID_CELL_MASK

    # Policy / legality: one scatter each over the flattened ragged entries.
    counts = np.diff(cols.child_off)
    row_of = np.repeat(rows, counts)
    moves = cols.child_idx.astype(np.int64)
    if syms is not None:
        moves = MOVE_PERM[syms[row_of], moves]
    visits = cols.child_visits.astype(np.float32)
    totals = np.bincount(row_of, weights=visits, minlength=b).astype(np.float32)

    # Playout cap randomisation: fast-searched rows keep their value, score and
    # capture-map targets but must contribute nothing to the policy loss. Folding
    # the weight into the per-row normaliser zeroes them exactly, and costs a
    # length-B multiply instead of a (B, 2562) one.
    policy_weight = cols.is_full.astype(np.float32)
    scale = np.where(totals > 0.0, policy_weight / np.where(totals > 0.0, totals, 1.0), 0.0)

    policy = np.zeros((b, MOVE_SPACE), dtype=np.float32)
    legal_mask = np.zeros((b, MOVE_SPACE), dtype=np.float32)
    legal_mask[row_of, moves] = 1.0
    policy[row_of, moves] = visits * scale[row_of]

    cap_counts = np.diff(cols.cap_off)
    cap_row = np.repeat(rows, cap_counts)
    cap_idx = cols.cap_idx.astype(np.int64)
    if syms is not None:
        cap_idx = CAP_PERM[syms[cap_row], cap_idx]
    capture_map = np.zeros((b, CAP_MAP_SIZE), dtype=np.float32)
    capture_map[cap_row, cap_idx] = np.clip(cols.cap_val, 0.0, 1.0)

    return Batch(
        planes=planes,
        policy=policy,
        legal_mask=legal_mask,
        policy_weight=policy_weight,
        value=_VALUE_CLASS_LUT[cols.z.astype(np.int64) + 128],
        score=_SCORE_CLASS_LUT[cols.score_diff.astype(np.int64) + 128],
        capture_map=capture_map.reshape(b, CAPTURE_MAP_CHANNELS, BOARD_H, BOARD_W),
        q=np.ascontiguousarray(cols.q, dtype=np.float32),
    )


# ----- holdout selection -----------------------------------------------------

#: Extra entropy mixed into the frozen holdout's RNG so that, at the same
#: `(seed, gen)`, it does not walk the identical permutation the rolling holdout
#: walks. Both stay deterministic; they are merely independent.
_FROZEN_STREAM = 0x46524F5A  # b"FROZ"


def _select_whole_games(
    rows: np.ndarray, game_ids: np.ndarray, target: int, rng: np.random.Generator
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Add whole games in `rng` order until at least `target` positions are held.

    Returns `(picked row indices, the chosen game ids sorted)`. Shared by both
    holdouts so they cannot drift apart: positions inside one game share `z` and
    `score_diff` exactly, so a row-level split would score the model on
    near-duplicates of its own training data.

    The count therefore *overshoots* by up to one game's worth of positions —
    "approximately `target`" is the contract, and it is the right one: a holdout
    is measured in games, not rows.
    """
    uniq, counts = np.unique(game_ids, return_counts=True)
    chosen: list[int] = []
    held = 0
    for i in rng.permutation(uniq.size):
        chosen.append(int(uniq[i]))
        held += int(counts[i])
        if held >= target:
            break
    picked = rows[np.isin(game_ids, np.asarray(chosen, dtype=game_ids.dtype))]
    return picked, tuple(sorted(chosen))


# ----- the buffer ------------------------------------------------------------


class ReplayBuffer:
    """Rolling multi-generation buffer over shard files."""

    def __init__(self, augment: bool = True) -> None:
        self.augment = augment
        self._store: _Columns = _Columns.empty()
        self._pending: dict[int, list[_Columns]] = {}
        self._counts: dict[int, int] = {}
        #: gen -> rows currently rolling-held. Mirrors the `hold` column so
        #: `total_size()` stays O(generations) rather than O(buffer).
        self._rolling: dict[int, int] = {}
        #: gen -> the game ids that make up its rolling slice. Also the marker
        #: that makes `mark_rolling_holdout` idempotent.
        self._rolling_games: dict[int, tuple[int, ...]] = {}
        #: gen -> rows currently frozen-held. Mirrors the `frozen` column, for
        #: the same reason `_rolling` mirrors `hold`.
        self._frozen: dict[int, int] = {}
        #: gen -> the game ids making up its frozen slice; the idempotence
        #: marker for `mark_holdout`.
        self._frozen_games: dict[int, tuple[int, ...]] = {}
        self._pool_cache: dict[tuple, np.ndarray] = {}

    # -- ingest ---------------------------------------------------------------

    def ingest_shard(self, path: Path | str, gen: int) -> int:
        """Decode `path` and add its rows under `gen`. Returns rows added.

        O(rows added): the shard is parked in a per-generation pending list and
        merged into the contiguous store lazily, at most once per sampling
        burst. Repeated ingest into the same generation is therefore linear,
        not quadratic.
        """
        cols = _read_shard(Path(path), gen)
        self._pending.setdefault(gen, []).append(cols)
        self._counts[gen] = self._counts.get(gen, 0) + cols.size
        self._pool_cache.clear()
        return cols.size

    def _flush(self) -> None:
        if not self._pending:
            return
        parts = [self._store]
        for g in sorted(self._pending):
            parts.extend(self._pending[g])
        self._store = _Columns.concat(parts)
        self._pending.clear()
        self._pool_cache.clear()

    # -- accounting -----------------------------------------------------------

    def trainable_size(self, gen: int) -> int:
        """Rows of `gen` the optimiser may actually sample: everything buffered
        for it minus both holdouts' slices."""
        gen = int(gen)
        return (
            self._counts.get(gen, 0)
            - self._frozen.get(gen, 0)
            - self._rolling.get(gen, 0)
        )

    def total_size(self, *, include_holdout: bool = False) -> int:
        """Sampleable rows. Both holdouts are excluded by default, so the
        training loop's minimum-buffer check cannot be satisfied by data it is
        not allowed to train on — and so `epochs_this_gen` is computed against
        the data actually being trained on.

        The holdout generation contributes its *un-held remainder* here: only
        the frozen slice is withheld, not the generation that carries it.
        """
        if include_holdout:
            return sum(self._counts.values())
        return sum(self.trainable_size(g) for g in self._counts)

    def holdout_size(self, gen: int | None = None) -> int:
        """Rows in the *frozen* holdout — for `gen`, or across every generation."""
        if gen is None:
            return sum(self._frozen.values())
        return self._frozen.get(int(gen), 0)

    def rolling_holdout_size(self, gen: int | None = None) -> int:
        """Rows in the rolling holdout — for `gen`, or across every generation."""
        if gen is None:
            return sum(self._rolling.values())
        return self._rolling.get(int(gen), 0)

    def chunk_size(self, gen: int) -> int:
        """Rows buffered for `gen`, holdout or not; 0 if never ingested."""
        return self._counts.get(gen, 0)

    def generations(self) -> list[int]:
        return sorted(self._counts)

    def nbytes(self) -> int:
        """Resident bytes of position data. Divide by `total_size()` for the
        per-position cost; compare against `14 * 81 * 4 = 4536` for the dense
        plane representation this replaced."""
        self._flush()
        return self._store.nbytes

    # -- holdout --------------------------------------------------------------

    def mark_holdout(
        self,
        gen: int,
        *,
        count: int,
        seed: int = 0,
        whole_games: bool = True,
    ) -> int:
        """Freeze ~`count` positions of `gen` as validation-only. Returns rows held.

        The held rows are excluded from `sample()` and from `total_size()`, and
        they survive `evict_below` so the ruler stays fixed for the life of the
        run. **Everything else in `gen` stays trainable** and ages out of the
        rolling window like any other generation.

        This is the **frozen** holdout: a drift indicator, not a generalisation
        measure — see the module docstring. Gate on the rolling holdout instead.
        Which is exactly why it is bounded: withholding a whole generation to
        feed a metric nothing may gate on costs tens of thousands of real
        training positions, and (by emptying the pool outside the current
        generation) disables the rolling holdout that *can* be gated on.

        `whole_games=True` (the default) rounds up to whole games — positions in
        a game share `z` and `score_diff`, so a row-level split would score the
        model on near-duplicates of its training data.

        **Idempotent and deterministic in `(seed, gen)`.** Re-marking a
        generation that already has a slice returns the existing size and
        changes nothing, and a run resumed from shards reproduces the same rows.
        """
        gen = int(gen)
        if count < 0:
            raise ValueError(f"count must be non-negative, got {count}")
        if gen in self._frozen_games:
            return self._frozen.get(gen, 0)

        self._flush()
        # Rows already withheld by the rolling holdout are not candidates: they
        # are validation-only either way, and letting them count twice would
        # make `holdout_frozen_size + holdout_rolling_size` exceed the
        # generation.
        rows = np.flatnonzero((self._store.gen == gen) & ~self._store.hold)
        if rows.size == 0:
            return 0
        target = max(0, min(int(count), int(rows.size)))
        if target == 0:
            self._frozen_games[gen] = ()
            self._frozen[gen] = 0
            return 0

        rng = np.random.default_rng([seed, gen, _FROZEN_STREAM])
        if whole_games:
            picked, chosen = _select_whole_games(
                rows, self._store.meta["game_id"][rows], target, rng
            )
            self._frozen_games[gen] = chosen
        else:
            picked = rows[rng.permutation(rows.size)[:target]]
            self._frozen_games[gen] = ()

        self._store.frozen[picked] = True
        self._frozen[gen] = int(picked.size)
        self._pool_cache.clear()
        return int(picked.size)

    @property
    def holdout_gens(self) -> frozenset[int]:
        """Generations carrying a frozen slice, including empty ones."""
        return frozenset(self._frozen_games)

    def holdout_games(self, gen: int) -> tuple[int, ...]:
        """The `game_id`s making up `gen`'s frozen slice. Empty when the slice
        was taken row-wise (`whole_games=False`) or does not exist."""
        return self._frozen_games.get(int(gen), ())

    def mark_rolling_holdout(
        self,
        gen: int,
        *,
        fraction: float | None = None,
        count: int | None = None,
        seed: int = 0,
        whole_games: bool = True,
    ) -> int:
        """Withhold a slice of `gen` from training. Returns rows held out.

        Exactly one of `fraction` (of `gen`'s rows) or `count` (rows) gives the
        target size. The rows are excluded from `sample()` and from
        `total_size()` from here on — permanently, so "never trained on" is a
        property of the rows and not of when you happen to ask.

        **Idempotent.** Calling this again for a generation that already has a
        rolling slice returns the existing size and changes nothing, so a
        resumed or retried generation cannot quietly re-roll the split and
        contaminate it with rows the model has already seen.

        `whole_games=True` (the default) rounds the selection up to whole games:
        positions within a game share `z` and `score_diff` exactly, so a
        row-level split scores the model on near-duplicates of its training
        data. With `whole_games=False` the count is exact — useful for tests,
        misleading for measurement.

        Selection is deterministic in `(seed, gen)`, so a run resumed from
        shards reproduces the same split.
        """
        gen = int(gen)
        if (fraction is None) == (count is None):
            raise ValueError("mark_rolling_holdout needs exactly one of fraction or count")
        if gen in self._rolling_games:
            return self._rolling.get(gen, 0)
        if fraction is not None and not 0.0 <= fraction <= 1.0:
            raise ValueError(f"fraction must be in [0, 1], got {fraction}")
        if count is not None and count < 0:
            raise ValueError(f"count must be non-negative, got {count}")

        self._flush()
        rows = np.flatnonzero(self._store.gen == gen)
        if rows.size == 0:
            return 0
        target = math.ceil(float(fraction) * rows.size) if fraction is not None else int(count)
        target = max(0, min(target, int(rows.size)))
        if target == 0:
            self._rolling_games[gen] = ()
            self._rolling[gen] = 0
            return 0

        rng = np.random.default_rng([seed, gen])
        if whole_games:
            picked, chosen = _select_whole_games(
                rows, self._store.meta["game_id"][rows], target, rng
            )
            self._rolling_games[gen] = chosen
        else:
            picked = rows[rng.permutation(rows.size)[:target]]
            self._rolling_games[gen] = ()

        self._store.hold[picked] = True
        self._rolling[gen] = int(picked.size)
        self._pool_cache.clear()
        return int(picked.size)

    @property
    def rolling_holdout_gens(self) -> frozenset[int]:
        """Generations with a rolling slice, including empty ones."""
        return frozenset(self._rolling_games)

    @property
    def rolling_holdout_gen(self) -> int | None:
        """The newest generation carrying a non-empty rolling slice."""
        live = [g for g, n in self._rolling.items() if n > 0]
        return max(live) if live else None

    def rolling_holdout_games(self, gen: int) -> tuple[int, ...]:
        """The `game_id`s making up `gen`'s rolling slice. Empty when the slice
        was taken row-wise (`whole_games=False`) or does not exist."""
        return self._rolling_games.get(int(gen), ())

    # -- eviction -------------------------------------------------------------

    def evict_below(self, threshold_gen: int, *, include_holdout: bool = False) -> int:
        """Drop generations `< threshold_gen`. Returns rows removed.

        **Frozen-holdout rows survive** unless `include_holdout` — a validation
        set that silently ages out is worse than no validation set. Their
        generation is evicted *around* them: the un-held remainder is ordinary
        training data and leaves on schedule, and the generation stays in the
        buffer carrying only its frozen slice.

        Returns 0 without touching the store when nothing is actually
        removable. That fast path is load-bearing: the frozen generation sits
        below the threshold for the rest of the run, and this is called on every
        poll of the training loop.
        """
        removable = {}
        for g, n in self._counts.items():
            if g >= threshold_gen:
                continue
            keep = 0 if include_holdout else self._frozen.get(g, 0)
            if n - keep > 0:
                removable[g] = n - keep
        if not removable:
            return 0

        self._flush()  # a partially-evicted generation must be one contiguous set
        removed = sum(removable.values())
        for g in removable:
            keep = 0 if include_holdout else self._frozen.get(g, 0)
            self._rolling.pop(g, None)
            self._rolling_games.pop(g, None)
            if keep:
                self._counts[g] = keep
                continue
            del self._counts[g]
            self._frozen.pop(g, None)
            self._frozen_games.pop(g, None)
        if self._store.size:
            doomed = np.fromiter(removable, np.int32, len(removable))
            in_doomed = np.isin(self._store.gen, doomed)
            keep_mask = ~in_doomed
            if not include_holdout:
                keep_mask |= in_doomed & self._store.frozen
            self._store = self._store.take(np.flatnonzero(keep_mask))
        self._pool_cache.clear()
        return removed

    # -- sampling -------------------------------------------------------------

    def _pool(
        self,
        include: frozenset[int] | None,
        exclude: frozenset[int],
        rolling: str = "exclude",
        frozen: str = "exclude",
    ) -> np.ndarray | None:
        """Row indices eligible for sampling, or `None` meaning "every row".

        `rolling` and `frozen` select how each per-row holdout flag is applied:
        `"exclude"` drops held rows (training), `"only"` keeps just them
        (validation on that holdout), `"any"` ignores the flag.

        Cached: the filter only changes when the buffer does, and rebuilding it
        per batch would put an O(buffer) pass in front of every SGD step.
        """
        if include is None and not exclude and rolling == "any" and frozen == "any":
            return None
        self._flush()  # the filter is over store rows; pending ones are invisible
        key = (include, exclude, rolling, frozen)
        pool = self._pool_cache.get(key)
        if pool is None:
            gen = self._store.gen
            mask = np.ones(gen.shape[0], dtype=np.bool_)
            if include is not None:
                mask &= np.isin(gen, np.fromiter(include, np.int32, len(include)))
            if exclude:
                mask &= ~np.isin(gen, np.fromiter(exclude, np.int32, len(exclude)))
            for mode, flag in ((rolling, self._store.hold), (frozen, self._store.frozen)):
                if mode == "exclude":
                    mask &= ~flag
                elif mode == "only":
                    mask &= flag
            pool = np.flatnonzero(mask)
            self._pool_cache[key] = pool
        return pool

    def _sample(
        self,
        batch_size: int,
        rng: np.random.Generator | None,
        include: frozenset[int] | None,
        exclude: frozenset[int],
        augment: bool | None,
        force_sym: int | None,
        rolling: str = "exclude",
        frozen: str = "exclude",
    ) -> Batch:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self._flush()
        if self._store.size == 0:
            raise ValueError("replay buffer is empty")
        if rng is None:
            rng = np.random.default_rng()

        pool = self._pool(include, exclude, rolling, frozen)
        if pool is None:
            rows = rng.integers(0, self._store.size, size=batch_size)
        else:
            if pool.size == 0:
                raise ValueError(
                    f"no sampleable rows (include={sorted(include) if include else None}, "
                    f"exclude={sorted(exclude)}, rolling={rolling}, frozen={frozen})"
                )
            rows = pool[rng.integers(0, pool.size, size=batch_size)]

        if force_sym is not None:
            if not 0 <= force_sym < NUM_SYMS:
                raise ValueError(f"force_sym must be in [0, {NUM_SYMS}), got {force_sym}")
            syms = np.full(batch_size, force_sym, dtype=np.int64)
        elif self.augment if augment is None else augment:
            syms = rng.integers(0, NUM_SYMS, size=batch_size)
        else:
            syms = None
        return _build_batch(self._store.take(rows), syms)

    def sample(
        self,
        batch_size: int,
        rng: np.random.Generator | None = None,
        *,
        exclude_gens: Iterable[int] | None = None,
        augment: bool | None = None,
        force_sym: int | None = None,
    ) -> Batch:
        """Uniformly sample `batch_size` positions from the trainable window.

        Rows held by *either* holdout are always excluded; `exclude_gens`
        removes whole generations on top of that. `augment` overrides the
        instance default. `force_sym` applies one fixed symmetry to every row —
        for tests and for test-time augmentation in validation, where averaging
        over syms wants a deterministic sweep.
        """
        exclude = frozenset(exclude_gens or ())
        return self._sample(
            batch_size, rng, None, exclude, augment, force_sym, "exclude", "exclude"
        )

    def sample_from_gens(
        self,
        gens: Iterable[int],
        batch_size: int,
        rng: np.random.Generator | None = None,
        *,
        augment: bool | None = None,
        force_sym: int | None = None,
    ) -> Batch:
        """Sample only from `gens` — **every** row of them, holdout or not.

        Both holdout flags are ignored here: an explicit generation request
        means exactly that. Use `sample_frozen_holdout` / `sample_rolling_holdout`
        for the withheld slices; scoring validation through this method would
        report training data as held out.
        """
        include = frozenset(gens)
        if not include:
            raise ValueError("sample_from_gens requires at least one generation")
        return self._sample(
            batch_size, rng, include, frozenset(), augment, force_sym, "any", "any"
        )

    def sample_frozen_holdout(
        self,
        batch_size: int,
        rng: np.random.Generator | None = None,
        *,
        gens: Iterable[int] | None = None,
        augment: bool | None = False,
        force_sym: int | None = None,
    ) -> Batch:
        """Sample only from the frozen holdout — rows never trained on.

        With a fixed `rng` seed this yields the same batch every generation,
        which is what makes the `val_frozen/` curve comparable end to end. It is
        a drift indicator; do not gate on it.
        """
        include = frozenset(gens) if gens is not None else frozenset(self._frozen_games)
        if not include:
            raise ValueError("no frozen holdout has been marked")
        return self._sample(
            batch_size, rng, include, frozenset(), augment, force_sym, "any", "only"
        )

    def sample_rolling_holdout(
        self,
        batch_size: int,
        rng: np.random.Generator | None = None,
        *,
        gen: int | None = None,
        augment: bool | None = False,
        force_sym: int | None = None,
    ) -> Batch:
        """Sample only from the rolling holdout — rows never trained on.

        `gen` defaults to the newest generation carrying a slice, which is the
        one that measures generalisation *within the current distribution*.
        Augmentation is off by default: the point is a stable set, not more of
        it.
        """
        target = self.rolling_holdout_gen if gen is None else int(gen)
        if target is None:
            raise ValueError("no rolling holdout has been marked")
        return self._sample(
            batch_size, rng, frozenset({target}), frozenset(), augment, force_sym, "only", "any"
        )

    def iter_holdout_batches(
        self,
        *,
        kind: str,
        positions: int,
        batch_size: int,
        seed: int,
        gens: Iterable[int] | None = None,
    ) -> Iterator[Batch]:
        """Fixed-seed, augmentation-off draws from one of the two holdouts.

        `kind` is `"frozen"` (the frozen slices, `gens` defaults to
        `holdout_gens`) or `"rolling"` (the newest rolling slice, or `gens`'
        single generation). Either way only the *withheld* rows are drawn —
        never the un-held remainder of a holdout generation, which is training
        data. Sampling is with replacement, so this is a bootstrap rather than
        an enumeration — but with a fixed seed it is the *same* bootstrap every
        generation, which is what makes the curve comparable. Yields nothing
        when the requested set is empty, so the caller can treat "no holdout
        yet" as "no metrics" instead of an error.
        """
        if kind not in ("frozen", "rolling"):
            raise ValueError(f"kind must be 'frozen' or 'rolling', got {kind!r}")
        chosen = frozenset(gens) if gens is not None else None
        if kind == "frozen":
            chosen = chosen if chosen is not None else frozenset(self._frozen_games)
            if not chosen or not any(self._frozen.get(g, 0) for g in chosen):
                return
        else:
            target = self.rolling_holdout_gen if chosen is None else max(chosen)
            if target is None or self._rolling.get(target, 0) == 0:
                return
            chosen = frozenset({target})

        rng = np.random.default_rng(seed)
        remaining = int(positions)
        while remaining > 0:
            n = min(int(batch_size), remaining)
            if kind == "frozen":
                yield self.sample_frozen_holdout(n, rng, gens=chosen, augment=False)
            else:
                yield self.sample_rolling_holdout(
                    n, rng, gen=next(iter(chosen)), augment=False
                )
            remaining -= n


    # -- metrics --------------------------------------------------------------

    def seeded_fraction(self) -> float:
        """Fraction of *trainable* rows that came from a handicap-seeded game.

        `nan` with nothing trainable in the buffer — a rate with no denominator
        is undefined, not zero.
        """
        pool = self._pool(None, frozenset(), "exclude", "exclude")
        if pool is None:
            meta = self._store.meta
        else:
            if pool.size == 0:
                return float("nan")
            meta = self._store.meta[pool]
        if meta.size == 0:
            return float("nan")
        seeded = (meta["handicap_black"].astype(np.int64) + meta["handicap_white"]) > 0
        return float(seeded.mean())

    def buffer_metrics(
        self,
        *,
        gen: int | None = None,
        steps: int | None = None,
        batch_size: int | None = None,
        prefix: str = BUFFER_PREFIX,
    ) -> dict[str, float]:
        """The `buffer/*` block: what the training window actually contains.

        `epochs_this_gen = steps * batch_size / size` is the reason this exists.
        A generation that takes 20 raw passes over its own replay buffer is
        overfitting by construction, and reading that off a training curve after
        the fact is exactly the ambiguity this measurement layer is meant to
        remove. It is `nan` unless both `steps` and `batch_size` are supplied.

        `size` is the *trainable* row count: rows held by either holdout are
        excluded, so the epoch count is against the data the optimiser actually
        saw. The holdout generation's un-held remainder **is** in there — it is
        trained on like any other data, and leaving it out inflated
        `epochs_this_gen` by the ratio of the whole generation to the part of it
        that was really withheld.
        """
        nan = float("nan")
        size = self.total_size()
        gens = [g for g in self._counts if self.trainable_size(g) > 0]
        epochs = (
            float(steps) * float(batch_size) / float(size)
            if steps is not None and batch_size is not None and size > 0
            else nan
        )
        out: dict[str, float] = {
            "size": int(size),
            "size_with_holdout": int(self.total_size(include_holdout=True)),
            "holdout_frozen_size": int(self.holdout_size()),
            "holdout_rolling_size": int(self.rolling_holdout_size()),
            "generations": len(gens),
            "gen_min": int(min(gens)) if gens else nan,
            "gen_max": int(max(gens)) if gens else nan,
            "seeded_fraction": self.seeded_fraction(),
            "positions_this_gen": int(self.chunk_size(gen)) if gen is not None else nan,
            "epochs_this_gen": epochs,
        }
        return {f"{prefix}{k}": v for k, v in out.items()}


def find_shards_for_gen(shards_root: Path, gen: int) -> list[Path]:
    """List `*.parquet` files under `shards_root/gen_NNN/`, sorted."""
    gen_dir = Path(shards_root) / f"gen_{gen:03d}"
    if not gen_dir.exists():
        return []
    return sorted(gen_dir.glob("*.parquet"))
