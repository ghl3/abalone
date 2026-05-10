"""Replay buffer: streaming parquet ingest + uniform sampling with sym
augmentation.

Memory plan: the buffer holds one chunk per ingested shard file. Each
chunk is decoded once on ingest (bitboards → dense planes, sparse
visit lists → padded numpy arrays). At sample time we sample uniformly
across all chunks, build dense `policy` and `legal_mask` vectors from
the sparse representation, and apply a random D6 sym to each example.

Generation eviction is by absolute generation number — `evict_below(K)`
drops every chunk whose `gen < K`. The training loop calls this with
`(current_gen - replay_buffer_gens)` to maintain a rolling window.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from model.encoder import (
    MOVE_SPACE,
    NUM_SYMS,
    PositionRecord,
    apply_sym_to_planes,
    apply_sym_to_policy,
    encode_position,
)


@dataclass
class _Chunk:
    """One shard's decoded contents."""

    gen: int
    # All arrays indexed by row.
    planes: np.ndarray  # (N, 6, 9, 9) float32
    z: np.ndarray  # (N,) float32
    q: np.ndarray  # (N,) float32
    n_children: np.ndarray  # (N,) int16
    # Sparse policy: padded to `child_idxs.shape[1]` with -1 / 0.
    child_idxs: np.ndarray  # (N, M) int16
    child_visits: np.ndarray  # (N, M) int32

    @property
    def size(self) -> int:
        return int(self.planes.shape[0])


def _decode_parquet(path: Path, gen: int) -> _Chunk:
    """Read a shard file and decode it into a `_Chunk`."""
    table = pq.read_table(path)
    cols = table.to_pydict()
    n = len(cols["z"])
    if n == 0:
        return _Chunk(
            gen=gen,
            planes=np.zeros((0, 6, 9, 9), dtype=np.float32),
            z=np.zeros((0,), dtype=np.float32),
            q=np.zeros((0,), dtype=np.float32),
            n_children=np.zeros((0,), dtype=np.int16),
            child_idxs=np.zeros((0, 0), dtype=np.int16),
            child_visits=np.zeros((0, 0), dtype=np.int32),
        )

    own_lo = np.asarray(cols["own_bb_lo"], dtype=np.uint64)
    own_hi = np.asarray(cols["own_bb_hi"], dtype=np.uint64)
    opp_lo = np.asarray(cols["opp_bb_lo"], dtype=np.uint64)
    opp_hi = np.asarray(cols["opp_bb_hi"], dtype=np.uint64)
    pushed_off_black = np.asarray(cols["pushed_off_black"], dtype=np.int32)
    pushed_off_white = np.asarray(cols["pushed_off_white"], dtype=np.int32)
    turn = np.asarray(cols["turn"], dtype=np.int32)
    ply = np.asarray(cols["ply"], dtype=np.int32)
    z = np.asarray(cols["z"], dtype=np.float32)
    q = np.asarray(cols["q"], dtype=np.float32)

    # Decode each row's planes. The bitboard reconstruction needs Python
    # int math for the 128-bit ops, so this is a Python loop — but it
    # runs once per shard ingest, not per training batch.
    planes = np.zeros((n, 6, 9, 9), dtype=np.float32)
    for i in range(n):
        own_bb = (int(own_hi[i]) << 64) | int(own_lo[i])
        opp_bb = (int(opp_hi[i]) << 64) | int(opp_lo[i])
        # `pushed_off_black` = number of black marbles pushed off the
        # board (= white's captures). For training we want the value
        # from the current side's POV: own_pushed_off = my own marbles
        # I've already lost.
        if int(turn[i]) == 0:  # Black to move
            own_po = int(pushed_off_black[i])
            opp_po = int(pushed_off_white[i])
        else:  # White to move
            own_po = int(pushed_off_white[i])
            opp_po = int(pushed_off_black[i])
        rec = PositionRecord(
            own_bb=own_bb,
            opp_bb=opp_bb,
            own_pushed_off=own_po,
            opp_pushed_off=opp_po,
            ply=int(ply[i]),
            turn=int(turn[i]),
        )
        planes[i] = encode_position(rec)

    # Sparse children: pad each row to the file's max width.
    raw_idxs = cols["child_move_idxs"]  # list of lists
    raw_visits = cols["child_visits"]
    n_children = np.array([len(c) for c in raw_idxs], dtype=np.int16)
    max_width = int(n_children.max()) if n > 0 else 0
    child_idxs = np.full((n, max(max_width, 1)), -1, dtype=np.int16)
    child_visits = np.zeros((n, max(max_width, 1)), dtype=np.int32)
    for i in range(n):
        m = int(n_children[i])
        if m > 0:
            child_idxs[i, :m] = np.asarray(raw_idxs[i], dtype=np.int16)
            child_visits[i, :m] = np.asarray(raw_visits[i], dtype=np.int32)

    return _Chunk(
        gen=gen,
        planes=planes,
        z=z,
        q=q,
        n_children=n_children,
        child_idxs=child_idxs,
        child_visits=child_visits,
    )


@dataclass
class Batch:
    """Dense training batch."""

    planes: np.ndarray  # (B, 6, 9, 9) float32
    policy: np.ndarray  # (B, MOVE_SPACE) float32, normalized over legal moves
    legal_mask: np.ndarray  # (B, MOVE_SPACE) float32, 1 for legal, 0 otherwise
    z: np.ndarray  # (B,) float32
    q: np.ndarray  # (B,) float32


class ReplayBuffer:
    """Per-generation chunked buffer with rolling eviction."""

    def __init__(self, augment: bool = True) -> None:
        self.augment = augment
        # gen -> _Chunk. `dict` rather than list so out-of-order ingest
        # (resume) doesn't depend on iteration order.
        self._chunks: dict[int, _Chunk] = {}

    def ingest_shard(self, path: Path | str, gen: int) -> int:
        """Decode `path` and add to the buffer under `gen`. If a chunk
        already exists for this gen (multiple shard files per gen), the
        new rows are concatenated. Returns the number of rows added."""
        chunk = _decode_parquet(Path(path), gen)
        if gen in self._chunks:
            self._chunks[gen] = _concat(self._chunks[gen], chunk)
        else:
            self._chunks[gen] = chunk
        return chunk.size

    def evict_below(self, threshold_gen: int) -> int:
        """Drop chunks with `gen < threshold_gen`. Returns rows removed."""
        removed = 0
        for g in list(self._chunks.keys()):
            if g < threshold_gen:
                removed += self._chunks[g].size
                del self._chunks[g]
        return removed

    def total_size(self) -> int:
        return sum(c.size for c in self._chunks.values())

    def chunk_size(self, gen: int) -> int:
        """Number of positions buffered for `gen`, or 0 if not ingested."""
        chunk = self._chunks.get(gen)
        return chunk.size if chunk else 0

    def generations(self) -> list[int]:
        return sorted(self._chunks.keys())

    def sample(self, batch_size: int, rng: np.random.Generator | None = None) -> Batch:
        """Uniformly sample `batch_size` examples across all chunks."""
        if rng is None:
            rng = np.random.default_rng()
        if not self._chunks:
            raise ValueError("replay buffer is empty")

        gens = sorted(self._chunks.keys())
        sizes = np.array([self._chunks[g].size for g in gens], dtype=np.int64)
        total = int(sizes.sum())
        if total == 0:
            raise ValueError("replay buffer has zero total rows")
        cumsum = np.cumsum(sizes)
        flat = rng.integers(0, total, size=batch_size)
        chunk_pick = np.searchsorted(cumsum, flat, side="right")
        within = flat - np.where(chunk_pick > 0, cumsum[chunk_pick - 1], 0)

        planes = np.empty((batch_size, 6, 9, 9), dtype=np.float32)
        z = np.empty((batch_size,), dtype=np.float32)
        q = np.empty((batch_size,), dtype=np.float32)
        policy = np.zeros((batch_size, MOVE_SPACE), dtype=np.float32)
        legal_mask = np.zeros((batch_size, MOVE_SPACE), dtype=np.float32)

        syms = (
            rng.integers(0, NUM_SYMS, size=batch_size)
            if self.augment
            else np.zeros(batch_size, dtype=np.int64)
        )

        for i in range(batch_size):
            cgen = gens[int(chunk_pick[i])]
            wi = int(within[i])
            chunk = self._chunks[cgen]
            n = int(chunk.n_children[wi])
            idxs_i = chunk.child_idxs[wi, :n].astype(np.int64)
            visits_i = chunk.child_visits[wi, :n].astype(np.float32)
            total_visits = float(visits_i.sum())

            row_planes = chunk.planes[wi]
            row_policy = np.zeros(MOVE_SPACE, dtype=np.float32)
            row_mask = np.zeros(MOVE_SPACE, dtype=np.float32)
            if total_visits > 0.0:
                row_policy[idxs_i] = visits_i / total_visits
            row_mask[idxs_i] = 1.0

            s = int(syms[i])
            if s != 0:
                row_planes = apply_sym_to_planes(row_planes, s)
                row_policy = apply_sym_to_policy(row_policy, s)
                row_mask = apply_sym_to_policy(row_mask, s)

            planes[i] = row_planes
            policy[i] = row_policy
            legal_mask[i] = row_mask
            z[i] = chunk.z[wi]
            q[i] = chunk.q[wi]

        return Batch(planes=planes, policy=policy, legal_mask=legal_mask, z=z, q=q)


def _concat(a: _Chunk, b: _Chunk) -> _Chunk:
    assert a.gen == b.gen
    # Pad both to common max-width before concatenating sparse arrays.
    aw = a.child_idxs.shape[1]
    bw = b.child_idxs.shape[1]
    target_w = max(aw, bw, 1)

    def _pad(arr: np.ndarray, fill_value: int) -> np.ndarray:
        if arr.shape[1] == target_w:
            return arr
        pad = np.full(
            (arr.shape[0], target_w - arr.shape[1]), fill_value, dtype=arr.dtype
        )
        return np.concatenate([arr, pad], axis=1)

    return _Chunk(
        gen=a.gen,
        planes=np.concatenate([a.planes, b.planes], axis=0),
        z=np.concatenate([a.z, b.z]),
        q=np.concatenate([a.q, b.q]),
        n_children=np.concatenate([a.n_children, b.n_children]),
        child_idxs=np.concatenate([_pad(a.child_idxs, -1), _pad(b.child_idxs, -1)], axis=0),
        child_visits=np.concatenate(
            [_pad(a.child_visits, 0), _pad(b.child_visits, 0)], axis=0
        ),
    )


# Discover shard files matching the layout the trainer writes.
def find_shards_for_gen(shards_root: Path, gen: int) -> list[Path]:
    """List `*.parquet` files under `shards_root/gen_NNN/`, sorted."""
    gen_dir = Path(shards_root) / f"gen_{gen:03d}"
    if not gen_dir.exists():
        return []
    return sorted(gen_dir.glob("*.parquet"))
