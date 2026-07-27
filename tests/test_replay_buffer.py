"""Tests for `model.replay_buffer`.

Fixtures are synthetic parquet shards written here with pyarrow against the
schema in `docs/ARCHITECTURE.md` §5.4, rather than shards produced by the Rust
writer. That is deliberate: it pins the reader against the *written spec*, so a
writer that drifts from the spec fails these tests instead of defining them.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from model.batch import (
    BOARD_H,
    BOARD_W,
    CAPTURE_MAP_CHANNELS,
    INPUT_PLANES,
    MOVE_SPACE,
    SCORE_OFFSET,
    VALUE_DRAW,
    VALUE_LOSS,
    VALUE_WIN,
)
from model.encoder import (
    CELL_PERM,
    COMPACT_TO_CELL,
    LOSS_THERMOMETER_PLANES,
    MOVE_PERM,
    NUM_SYMS,
    OPP_LOSSES_PLANE,
    OPP_MARBLES_PLANE,
    OWN_LOSSES_PLANE,
    OWN_MARBLES_PLANE,
    PLY_PLANE,
    VALID_MASK_PLANE,
    PositionRecord,
    apply_sym_to_planes,
    apply_sym_to_policy,
    encode_inline,
    encode_position,
)
from model.replay_buffer import CAP_MAP_SIZE, ReplayBuffer, find_shards_for_gen

# ----- shard fixtures --------------------------------------------------------

SHARD_SCHEMA = pa.schema(
    [
        ("game_id", pa.uint32()),
        ("seed", pa.uint64()),
        ("opening", pa.uint8()),
        ("handicap_black", pa.uint8()),
        ("handicap_white", pa.uint8()),
        ("own_bb_lo", pa.uint64()),
        ("own_bb_hi", pa.uint64()),
        ("opp_bb_lo", pa.uint64()),
        ("opp_bb_hi", pa.uint64()),
        ("black_losses", pa.uint8()),
        ("white_losses", pa.uint8()),
        ("turn", pa.uint8()),
        ("ply", pa.uint16()),
        ("max_plies", pa.uint16()),
        ("move_played", pa.uint16()),
        ("is_full_search", pa.bool_()),
        ("z", pa.int8()),
        ("score_diff", pa.int8()),
        ("q", pa.float32()),
        ("child_move_idxs", pa.list_(pa.uint16())),
        ("child_visits", pa.list_(pa.uint32())),
        ("cap_map_idx", pa.list_(pa.uint16())),
        ("cap_map_val", pa.list_(pa.float32())),
    ]
)

MAX_PLIES = 200


def row(**overrides) -> dict:
    """One shard row with defaults; override whatever the test cares about."""
    r = {
        "game_id": 0,
        "seed": 0,
        "opening": 0,
        "handicap_black": 0,
        "handicap_white": 0,
        "own_bb": 0,
        "opp_bb": 0,
        "black_losses": 0,
        "white_losses": 0,
        "turn": 0,
        "ply": 0,
        "max_plies": MAX_PLIES,
        "move_played": 0,
        "is_full_search": True,
        "z": 0,
        "score_diff": 0,
        "q": 0.0,
        "child_move_idxs": [],
        "child_visits": [],
        "cap_map_idx": [],
        "cap_map_val": [],
    }
    unknown = set(overrides) - set(r)
    assert not unknown, f"unknown row field(s): {unknown}"
    r.update(overrides)
    return r


def write_shard(path: Path, rows: list[dict]) -> Path:
    """Write `rows` as a v2 parquet shard."""
    cols: dict[str, list] = {name: [] for name in SHARD_SCHEMA.names}
    for r in rows:
        for name in SHARD_SCHEMA.names:
            if name == "own_bb_lo":
                cols[name].append(r["own_bb"] & ((1 << 64) - 1))
            elif name == "own_bb_hi":
                cols[name].append((r["own_bb"] >> 64) & ((1 << 64) - 1))
            elif name == "opp_bb_lo":
                cols[name].append(r["opp_bb"] & ((1 << 64) - 1))
            elif name == "opp_bb_hi":
                cols[name].append((r["opp_bb"] >> 64) & ((1 << 64) - 1))
            else:
                cols[name].append(r[name])
    pq.write_table(pa.Table.from_pydict(cols, schema=SHARD_SCHEMA), path)
    return path


def bb(cells) -> int:
    out = 0
    for c in cells:
        out |= 1 << int(c)
    return out


def a_move(anchor_cell: int, direction: int = 0) -> int:
    """A structurally valid (size-1 inline) move index anchored at a cell.

    Structural validity matters: `MOVE_PERM` maps impossible indices to
    themselves, so a symmetry test built on one would pass vacuously.
    """
    return encode_inline(int(anchor_cell), direction, 1)


def random_rows(
    n: int, rng: np.random.Generator, *, n_children: int = 20, q_offset: float = 0.0
) -> list[dict]:
    """A plausible shard: disjoint boards, ~25% full searches, sparse capture
    maps, a mix of outcomes."""
    rows = []
    for i in range(n):
        cells = rng.permutation(COMPACT_TO_CELL)[:28]
        own, opp = cells[:14], cells[14:]
        anchors = rng.permutation(COMPACT_TO_CELL)[:n_children]
        moves = sorted({a_move(c, int(rng.integers(0, 6))) for c in anchors})
        visits = rng.integers(0, 50, size=len(moves)).tolist()
        if not any(visits):
            visits[0] = 7
        n_cap = int(rng.integers(0, 5))
        cap_cells = rng.permutation(COMPACT_TO_CELL)[:n_cap]
        cap_ch = rng.integers(0, CAPTURE_MAP_CHANNELS, size=n_cap)
        rows.append(
            row(
                game_id=i // 8,
                seed=1000 + i,
                turn=int(i % 2),
                black_losses=int(rng.integers(0, 6)),
                white_losses=int(rng.integers(0, 6)),
                ply=int(rng.integers(0, MAX_PLIES)),
                own_bb=bb(own),
                opp_bb=bb(opp),
                is_full_search=bool(rng.random() < 0.25),
                z=int(rng.integers(-1, 2)),
                score_diff=int(rng.integers(-6, 7)),
                # Unique per row: the batch carries `q` untouched, so it is a
                # usable fingerprint for matching a sampled example back to its
                # source row (and, with `q_offset`, back to its shard).
                q=q_offset + float(i) / 1000.0,
                child_move_idxs=[int(m) for m in moves],
                child_visits=[int(v) for v in visits],
                cap_map_idx=[
                    int(ch) * 81 + int(c) for ch, c in zip(cap_ch, cap_cells, strict=True)
                ],
                cap_map_val=[round(0.98 ** (k + 1), 4) for k in range(n_cap)],
            )
        )
    return rows


def source_index(batch, rows: list[dict]) -> np.ndarray:
    """Map each sampled example back to its source row via the `q` fingerprint."""
    keys = np.array([r["q"] for r in rows], dtype=np.float32)
    order = np.argsort(keys)
    pos = np.searchsorted(keys[order], batch.q)
    idx = order[pos]
    assert np.allclose(keys[idx], batch.q)
    return idx


@pytest.fixture(scope="module")
def rows(tmp_path_factory) -> list[dict]:
    return random_rows(64, np.random.default_rng(20260727))


@pytest.fixture(scope="module")
def shard(tmp_path_factory, rows) -> Path:
    return write_shard(tmp_path_factory.mktemp("shards") / "shard_t00_0000.parquet", rows)


@pytest.fixture(scope="module")
def shard_b(tmp_path_factory) -> Path:
    """A second shard whose `q` values are offset by 10, so a sampled batch
    says which shard each example came from."""
    rows_b = random_rows(64, np.random.default_rng(31337), q_offset=10.0)
    return write_shard(tmp_path_factory.mktemp("shards_b") / "shard_t00_0000.parquet", rows_b)


# ----- round trip: the pinned batch contract ---------------------------------


def test_sample_satisfies_the_batch_contract(shard, rows):
    rb = ReplayBuffer(augment=True)
    n = rb.ingest_shard(shard, gen=0)
    assert n == len(rows)
    assert rb.total_size() == len(rows)

    batch = rb.sample(37, np.random.default_rng(0))
    batch.validate()  # every field's shape and dtype
    assert batch.size == 37

    # Policy mass only ever lands on legal moves.
    assert np.all(batch.policy * (1.0 - batch.legal_mask) == 0.0)
    assert np.all(batch.legal_mask.sum(axis=1) > 0)
    # Full-search rows carry a normalised distribution.
    full = batch.policy_weight > 0
    if full.any():
        np.testing.assert_allclose(batch.policy[full].sum(axis=1), 1.0, atol=1e-5)
    assert np.all((batch.capture_map >= 0.0) & (batch.capture_map <= 1.0))
    assert np.all(np.isin(batch.value, [VALUE_WIN, VALUE_DRAW, VALUE_LOSS]))


def test_planes_match_the_pinned_encoder(shard, rows):
    """The vectorised bitboard decode must reproduce `encode_position`, which
    `tests/test_conformance.py` pins against the Rust encoder."""
    rb = ReplayBuffer(augment=False)
    rb.ingest_shard(shard, gen=0)
    batch = rb.sample(128, np.random.default_rng(3))
    idx = source_index(batch, rows)
    for b, i in enumerate(idx):
        r = rows[i]
        own_losses = r["black_losses"] if r["turn"] == 0 else r["white_losses"]
        opp_losses = r["white_losses"] if r["turn"] == 0 else r["black_losses"]
        expected = encode_position(
            PositionRecord(
                own_bb=r["own_bb"],
                opp_bb=r["opp_bb"],
                own_losses=own_losses,
                opp_losses=opp_losses,
                ply=r["ply"],
                max_plies=r["max_plies"],
                turn=r["turn"],
            )
        )
        np.testing.assert_array_equal(batch.planes[b], expected)


def test_value_and_score_are_class_indices(tmp_path):
    diffs = list(range(-8, 9))
    rows = [
        row(z=z, score_diff=d, q=float(k) / 1000.0)
        for k, (z, d) in enumerate((z, d) for z in (-1, 0, 1) for d in diffs)
    ]
    rb = ReplayBuffer(augment=False)
    rb.ingest_shard(write_shard(tmp_path / "s.parquet", rows), gen=0)
    batch = rb.sample(600, np.random.default_rng(1))
    idx = source_index(batch, rows)
    src_z = np.array([rows[i]["z"] for i in idx])
    src_d = np.array([rows[i]["score_diff"] for i in idx])

    assert batch.value.dtype == np.int64 and batch.score.dtype == np.int64
    np.testing.assert_array_equal(batch.value[src_z > 0], VALUE_WIN)
    np.testing.assert_array_equal(batch.value[src_z == 0], VALUE_DRAW)
    np.testing.assert_array_equal(batch.value[src_z < 0], VALUE_LOSS)
    np.testing.assert_array_equal(batch.score, np.clip(src_d, -6, 6) + SCORE_OFFSET)
    assert batch.score.min() == 0 and batch.score.max() == 12


# ----- the regression that motivated the v2 rename ---------------------------


def test_black_losses_column_means_marbles_black_has_lost(tmp_path):
    """Regression: v1 read `pushed_off_black` (marbles pushed off *by* Black)
    as Black's losses, so both loss thermometers were swapped for three
    generations. `black_losses` is Black's OWN losses; for Black to move it is
    `own_losses`, for White to move it is `opp_losses`. An implementation that
    inverts this fails here, and only here — every shape check still passes.
    """
    rows = [
        row(turn=0, black_losses=3, white_losses=1, q=0.0),
        row(turn=1, black_losses=3, white_losses=1, q=0.001),
    ]
    rb = ReplayBuffer(augment=False)
    rb.ingest_shard(write_shard(tmp_path / "s.parquet", rows), gen=0)
    batch = rb.sample(64, np.random.default_rng(0))
    idx = source_index(batch, rows)

    def thermometer(b: int, base: int) -> list[float]:
        return [float(batch.planes[b, base + k, 0, 0]) for k in range(LOSS_THERMOMETER_PLANES)]

    black_to_move = int(np.flatnonzero(idx == 0)[0])
    white_to_move = int(np.flatnonzero(idx == 1)[0])

    # Black to move: own losses = black_losses = 3, opp losses = 1.
    assert thermometer(black_to_move, OWN_LOSSES_PLANE) == [1, 1, 1, 0, 0]
    assert thermometer(black_to_move, OPP_LOSSES_PLANE) == [1, 0, 0, 0, 0]
    # White to move: the same row read from the other side, mirrored.
    assert thermometer(white_to_move, OWN_LOSSES_PLANE) == [1, 0, 0, 0, 0]
    assert thermometer(white_to_move, OPP_LOSSES_PLANE) == [1, 1, 1, 0, 0]


# ----- playout cap randomisation ---------------------------------------------


def test_fast_searched_rows_have_zero_policy_weight_and_zero_policy(tmp_path):
    moves = [a_move(c) for c in COMPACT_TO_CELL[:6]]
    visits = [10, 20, 30, 5, 5, 30]
    rows = [
        row(is_full_search=True, child_move_idxs=moves, child_visits=visits, q=0.0),
        row(is_full_search=False, child_move_idxs=moves, child_visits=visits, q=0.001),
    ]
    rb = ReplayBuffer(augment=True)
    rb.ingest_shard(write_shard(tmp_path / "s.parquet", rows), gen=0)
    batch = rb.sample(128, np.random.default_rng(11))
    idx = source_index(batch, rows)

    fast = idx == 1
    full = idx == 0
    assert fast.any() and full.any()

    assert batch.policy_weight.dtype == np.float32
    np.testing.assert_array_equal(batch.policy_weight[fast], 0.0)
    np.testing.assert_array_equal(batch.policy_weight[full], 1.0)

    # Exactly zero, not merely small: the policy loss must see nothing.
    assert np.count_nonzero(batch.policy[fast]) == 0
    # Legality, value, score and the capture map are still usable for them.
    assert np.all(batch.legal_mask[fast].sum(axis=1) == len(moves))
    np.testing.assert_allclose(batch.policy[full].sum(axis=1), 1.0, atol=1e-6)


# ----- capture map -----------------------------------------------------------


def test_capture_map_densification(tmp_path):
    entries = {(0, 40): 1.0, (0, 22): 0.5, (1, 58): 0.98, (1, 4 * 9 + 6): 0.25}
    rows = [
        row(
            cap_map_idx=[ch * 81 + cell for (ch, cell) in entries],
            cap_map_val=list(entries.values()),
        )
    ]
    rb = ReplayBuffer(augment=False)
    rb.ingest_shard(write_shard(tmp_path / "s.parquet", rows), gen=0)
    batch = rb.sample(4, np.random.default_rng(0))

    expected = np.zeros((CAPTURE_MAP_CHANNELS, BOARD_H, BOARD_W), dtype=np.float32)
    for (ch, cell), val in entries.items():
        expected[ch, cell // 9, cell % 9] = val
    for b in range(batch.size):
        np.testing.assert_array_equal(batch.capture_map[b], expected)
    assert batch.capture_map.sum() == pytest.approx(sum(entries.values()) * batch.size)


def test_empty_capture_map_is_all_zeros(tmp_path):
    rb = ReplayBuffer(augment=True)
    rb.ingest_shard(write_shard(tmp_path / "s.parquet", [row()]), gen=0)
    batch = rb.sample(4, np.random.default_rng(0))
    assert not batch.capture_map.any()


# ----- D6 augmentation -------------------------------------------------------


def test_augmentation_permutes_planes_policy_mask_and_capture_map_consistently(tmp_path):
    """`capture_map` is spatial. If it is not permuted by the same symmetry as
    the planes, the auxiliary head trains against rotated labels and nothing
    ever raises."""
    own_cell, opp_cell = 4 * 9 + 5, 2 * 9 + 2
    cap_own_cell, cap_opp_cell = 6 * 9 + 4, 3 * 9 + 1
    move = a_move(own_cell, 1)
    rows = [
        row(
            own_bb=bb([own_cell]),
            opp_bb=bb([opp_cell]),
            child_move_idxs=[move],
            child_visits=[9],
            cap_map_idx=[0 * 81 + cap_own_cell, 1 * 81 + cap_opp_cell],
            cap_map_val=[0.75, 0.5],
        )
    ]
    rb = ReplayBuffer(augment=True)
    rb.ingest_shard(write_shard(tmp_path / "s.parquet", rows), gen=0)

    for sym in range(NUM_SYMS):
        batch = rb.sample(1, np.random.default_rng(sym), force_sym=sym)
        planes, policy, mask, cmap = (
            batch.planes[0],
            batch.policy[0],
            batch.legal_mask[0],
            batch.capture_map[0],
        )

        def only(plane: np.ndarray, cell: int, what: str, s: int = sym) -> None:
            hits = np.flatnonzero(plane.reshape(-1))
            assert hits.tolist() == [cell], f"sym {s}: {what} at {hits.tolist()}, want [{cell}]"

        only(planes[OWN_MARBLES_PLANE], int(CELL_PERM[sym, own_cell]), "own marble")
        only(planes[OPP_MARBLES_PLANE], int(CELL_PERM[sym, opp_cell]), "opp marble")
        only(cmap[0], int(CELL_PERM[sym, cap_own_cell]), "capture map ch0")
        only(cmap[1], int(CELL_PERM[sym, cap_opp_cell]), "capture map ch1")
        assert cmap[0].sum() == pytest.approx(0.75)
        assert cmap[1].sum() == pytest.approx(0.5)

        moved = int(MOVE_PERM[sym, move])
        assert np.flatnonzero(policy).tolist() == [moved]
        assert np.flatnonzero(mask).tolist() == [moved]
        # The valid-cell mask is D6-invariant; a bad permutation would show here.
        assert planes[VALID_MASK_PLANE].sum() == 61


def test_augmentation_agrees_with_the_scalar_encoder_path(shard, rows):
    """Cross-check the vectorised sym against `apply_sym_to_planes` /
    `apply_sym_to_policy`, the reference implementations.

    Only the two marble planes are compared: `apply_sym_to_planes` writes just
    the 61 on-board cells and leaves the rest zero, which silently masks the
    constant planes that `encode_position` deliberately fills across all 81
    slots. This buffer permutes only the bitboard planes and rebuilds the
    constant ones, so its output matches `encode_position` — the actual
    cross-language contract — rather than that helper.
    """
    rb = ReplayBuffer(augment=True)
    rb.ingest_shard(shard, gen=0)
    plain = rb.sample(24, np.random.default_rng(5), augment=False)
    spatial = (OWN_MARBLES_PLANE, OPP_MARBLES_PLANE)
    for sym in (1, 4, 6, 9):
        rotated = rb.sample(24, np.random.default_rng(5), force_sym=sym)
        for b in range(plain.size):
            reference = apply_sym_to_planes(plain.planes[b], sym)
            np.testing.assert_array_equal(
                rotated.planes[b, spatial, :, :], reference[spatial, :, :]
            )
            np.testing.assert_allclose(
                rotated.policy[b], apply_sym_to_policy(plain.policy[b], sym), atol=1e-7
            )
            np.testing.assert_array_equal(
                rotated.legal_mask[b], apply_sym_to_policy(plain.legal_mask[b], sym)
            )
        # The constant planes are sym-invariant, and keep their full 81-cell
        # fill: augmented and unaugmented examples must be encoded alike.
        constant = [p for p in range(INPUT_PLANES) if p not in spatial]
        np.testing.assert_array_equal(rotated.planes[:, constant], plain.planes[:, constant])
        ply_plane = rotated.planes[:, PLY_PLANE].reshape(plain.size, -1)
        assert np.all(ply_plane == ply_plane[:, :1]), "off-board slots were masked"
        # Non-spatial fields are untouched by augmentation.
        np.testing.assert_array_equal(rotated.value, plain.value)
        np.testing.assert_array_equal(rotated.score, plain.score)
        np.testing.assert_array_equal(rotated.q, plain.q)


def test_augment_false_is_a_no_op(shard):
    rb = ReplayBuffer(augment=True)
    rb.ingest_shard(shard, gen=0)
    off = rb.sample(16, np.random.default_rng(2), augment=False)
    identity = rb.sample(16, np.random.default_rng(2), force_sym=0)
    np.testing.assert_array_equal(off.planes, identity.planes)
    np.testing.assert_array_equal(off.policy, identity.policy)
    np.testing.assert_array_equal(off.legal_mask, identity.legal_mask)
    np.testing.assert_array_equal(off.capture_map, identity.capture_map)

    # And the instance-level flag has the same effect as the per-call override.
    rb_off = ReplayBuffer(augment=False)
    rb_off.ingest_shard(shard, gen=0)
    np.testing.assert_array_equal(rb_off.sample(16, np.random.default_rng(2)).planes, off.planes)


def test_augmentation_actually_varies(shard):
    rb = ReplayBuffer(augment=True)
    rb.ingest_shard(shard, gen=0)
    rng = np.random.default_rng(0)
    assert not np.array_equal(rb.sample(16, rng).planes, rb.sample(16, rng).planes)


# ----- ingest, eviction, accounting ------------------------------------------


def test_repeated_ingest_into_one_generation_accumulates(shard, rows):
    rb = ReplayBuffer()
    for _ in range(5):
        rb.ingest_shard(shard, gen=3)
    assert rb.total_size() == 5 * len(rows)
    assert rb.chunk_size(3) == 5 * len(rows)
    assert rb.generations() == [3]
    rb.sample(8, np.random.default_rng(0)).validate()


def test_eviction_and_total_size_accounting(shard, rows):
    n = len(rows)
    rb = ReplayBuffer()
    for gen in (0, 5, 10):
        rb.ingest_shard(shard, gen=gen)
    rb.ingest_shard(shard, gen=10)
    assert rb.total_size() == 4 * n
    assert rb.generations() == [0, 5, 10]
    assert rb.chunk_size(10) == 2 * n
    assert rb.chunk_size(7) == 0

    rb.sample(8, np.random.default_rng(0))  # force a flush, then evict
    assert rb.evict_below(5) == n
    assert rb.total_size() == 3 * n
    assert rb.generations() == [5, 10]
    assert rb.evict_below(5) == 0

    # Surviving rows are intact and still sampleable.
    batch = rb.sample(64, np.random.default_rng(0))
    batch.validate()
    assert rb.evict_below(999) == 3 * n
    assert rb.total_size() == 0
    with pytest.raises(ValueError, match="empty"):
        rb.sample(4, np.random.default_rng(0))


def test_out_of_order_generation_ingest(shard, rows):
    rb = ReplayBuffer()
    for gen in (7, 2, 7, 4):
        rb.ingest_shard(shard, gen=gen)
    assert rb.generations() == [2, 4, 7]
    assert rb.chunk_size(7) == 2 * len(rows)
    rb.sample(32, np.random.default_rng(0)).validate()
    assert rb.evict_below(4) == len(rows)
    assert rb.generations() == [4, 7]


def test_empty_shard_is_ingestable(tmp_path, shard, rows):
    rb = ReplayBuffer()
    assert rb.ingest_shard(write_shard(tmp_path / "empty.parquet", []), gen=0) == 0
    assert rb.total_size() == 0
    rb.ingest_shard(shard, gen=1)
    assert rb.total_size() == len(rows)
    rb.sample(8, np.random.default_rng(0)).validate()


def test_v1_shard_is_rejected(tmp_path):
    """A shard with the old `pushed_off_*` naming must fail loudly rather than
    be silently misread."""
    path = tmp_path / "v1.parquet"
    pq.write_table(
        pa.table(
            {
                "own_bb_lo": pa.array([0], pa.uint64()),
                "own_bb_hi": pa.array([0], pa.uint64()),
                "opp_bb_lo": pa.array([0], pa.uint64()),
                "opp_bb_hi": pa.array([0], pa.uint64()),
                "pushed_off_black": pa.array([0], pa.uint8()),
                "pushed_off_white": pa.array([0], pa.uint8()),
                "turn": pa.array([0], pa.uint8()),
                "ply": pa.array([0], pa.uint16()),
                "z": pa.array([0], pa.int8()),
                "q": pa.array([0.0], pa.float32()),
                "child_move_idxs": pa.array([[1]], pa.list_(pa.uint16())),
                "child_visits": pa.array([[1]], pa.list_(pa.uint32())),
            }
        ),
        path,
    )
    with pytest.raises(ValueError, match="black_losses"):
        ReplayBuffer().ingest_shard(path, gen=0)


def test_corrupt_indices_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="MOVE_SPACE"):
        ReplayBuffer().ingest_shard(
            write_shard(
                tmp_path / "a.parquet",
                [row(child_move_idxs=[MOVE_SPACE + 1], child_visits=[1])],
            ),
            gen=0,
        )
    with pytest.raises(ValueError, match="cap_map_idx"):
        ReplayBuffer().ingest_shard(
            write_shard(
                tmp_path / "b.parquet",
                [row(cap_map_idx=[CAP_MAP_SIZE], cap_map_val=[1.0])],
            ),
            gen=0,
        )


def test_null_column_is_rejected(tmp_path):
    """`to_numpy` turns a null into NaN in a float array, which then casts to a
    garbage integer. Nulls must fail, not decode."""
    path = tmp_path / "nulls.parquet"
    write_shard(path, [row(black_losses=1)])
    table = pq.read_table(path)
    patched = table.set_column(
        table.schema.get_field_index("black_losses"),
        "black_losses",
        pa.array([None], pa.uint8()),
    )
    pq.write_table(patched, path)
    with pytest.raises(ValueError, match="nulls"):
        ReplayBuffer().ingest_shard(path, gen=0)


def test_find_shards_for_gen(tmp_path, rows):
    root = tmp_path / "shards"
    (root / "gen_004").mkdir(parents=True)
    for name in ("shard_t01_0001.parquet", "shard_t00_0000.parquet", "notes.txt"):
        (root / "gen_004" / name).write_bytes(b"")
    assert [p.name for p in find_shards_for_gen(root, 4)] == [
        "shard_t00_0000.parquet",
        "shard_t01_0001.parquet",
    ]
    assert find_shards_for_gen(root, 5) == []


# ----- holdout ---------------------------------------------------------------


def test_holdout_generation_is_never_sampled_or_counted(shard, shard_b, rows):
    n = len(rows)
    rb = ReplayBuffer(augment=False)
    rb.ingest_shard(shard, gen=0)  # q < 1
    rb.ingest_shard(shard_b, gen=1)  # q > 10
    rb.mark_holdout(1)

    assert rb.holdout_gens == frozenset({1})
    assert rb.total_size() == n
    assert rb.total_size(include_holdout=True) == 2 * n
    assert rb.holdout_size() == n
    assert rb.chunk_size(1) == n  # still buffered

    # Training never sees the held-out generation: every sampled q comes from
    # the gen-0 shard.
    train = rb.sample(512, np.random.default_rng(0))
    assert np.all(train.q < 1.0)

    held = rb.sample_from_gens([1], 32, np.random.default_rng(0))
    held.validate()
    assert np.all(held.q > 9.0)

    # Frozen: same seed, same batch, generation after generation.
    again = rb.sample_from_gens([1], 32, np.random.default_rng(0))
    np.testing.assert_array_equal(held.planes, again.planes)
    np.testing.assert_array_equal(held.value, again.value)

    # And exempt from the rolling window.
    assert rb.evict_below(10) == n
    assert rb.generations() == [1]
    assert rb.evict_below(10, include_holdout=True) == n
    assert rb.generations() == []


def test_exclude_gens_argument(shard, shard_b):
    rb = ReplayBuffer()
    rb.ingest_shard(shard, gen=0)  # q < 1
    rb.ingest_shard(shard_b, gen=1)  # q > 10
    batch = rb.sample(256, np.random.default_rng(0), exclude_gens=[0])
    batch.validate()
    assert np.all(batch.q > 9.0)
    assert np.all(rb.sample(256, np.random.default_rng(0), exclude_gens=[1]).q < 1.0)
    with pytest.raises(ValueError, match="no sampleable rows"):
        rb.sample(8, np.random.default_rng(0), exclude_gens=[0, 1])


# ----- memory and throughput -------------------------------------------------


def test_storage_is_bitboards_not_dense_planes(tmp_path):
    """Dense `(N, 14, 9, 9)` float32 planes cost 4536 B/position. Bitboards
    plus the ragged search result must land an order of magnitude under that."""
    n = 2000
    rows = random_rows(n, np.random.default_rng(7), n_children=20)
    rb = ReplayBuffer()
    rb.ingest_shard(write_shard(tmp_path / "s.parquet", rows), gen=0)
    rb.sample(8, np.random.default_rng(0))  # flush

    dense = INPUT_PLANES * BOARD_H * BOARD_W * 4
    per_position = rb.nbytes() / rb.total_size()
    print(f"\nstorage: {per_position:.0f} B/position vs {dense} B dense planes")
    assert rb.nbytes() < n * dense / 8


def test_sampling_throughput(shard, rows):
    """Report batches/second at batch 256 against the old per-example path.

    The floor is deliberately loose — this guards against a regression back to
    a Python loop over the batch, not against a slow machine.
    """
    rb = ReplayBuffer(augment=True)
    for gen in range(8):
        rb.ingest_shard(shard, gen=gen)
    rng = np.random.default_rng(0)
    rb.sample(256, rng)  # flush + warm up

    reps = 20
    start = time.perf_counter()
    for _ in range(reps):
        rb.sample(256, rng)
    fast = reps / (time.perf_counter() - start)

    # The old design, reconstructed: a Python loop over the batch, dense plane
    # encode per example, and `apply_sym_to_planes` rebuilding an 81-element
    # inverse permutation every single time.
    def legacy_batch(batch_size: int, rng: np.random.Generator) -> None:
        planes = np.empty((batch_size, INPUT_PLANES, BOARD_H, BOARD_W), np.float32)
        policy = np.zeros((batch_size, MOVE_SPACE), np.float32)
        mask = np.zeros((batch_size, MOVE_SPACE), np.float32)
        picks = rng.integers(0, len(rows), size=batch_size)
        syms = rng.integers(0, NUM_SYMS, size=batch_size)
        for i, (p, s) in enumerate(zip(picks, syms, strict=True)):
            r = rows[int(p)]
            rec = PositionRecord(
                own_bb=r["own_bb"],
                opp_bb=r["opp_bb"],
                own_losses=r["black_losses"] if r["turn"] == 0 else r["white_losses"],
                opp_losses=r["white_losses"] if r["turn"] == 0 else r["black_losses"],
                ply=r["ply"],
                max_plies=r["max_plies"],
                turn=r["turn"],
            )
            row_policy = np.zeros(MOVE_SPACE, np.float32)
            row_mask = np.zeros(MOVE_SPACE, np.float32)
            idxs = np.asarray(r["child_move_idxs"], dtype=np.int64)
            visits = np.asarray(r["child_visits"], dtype=np.float32)
            row_policy[idxs] = visits / max(visits.sum(), 1.0)
            row_mask[idxs] = 1.0
            planes[i] = apply_sym_to_planes(encode_position(rec), int(s))
            policy[i] = apply_sym_to_policy(row_policy, int(s))
            mask[i] = apply_sym_to_policy(row_mask, int(s))

    legacy_reps = 3
    start = time.perf_counter()
    for _ in range(legacy_reps):
        legacy_batch(256, rng)
    slow = legacy_reps / (time.perf_counter() - start)

    print(
        f"\nsample(256): {fast:.1f} batches/s vectorised vs {slow:.1f} batches/s "
        f"per-example ({fast / slow:.1f}x)"
    )
    # The ratio is the real guard: both paths are timed on the same machine at
    # the same moment, so it survives a loaded CI box in a way that an absolute
    # rate does not. Measured ~12-18x; 5x still means "not a Python loop".
    assert fast > 5.0 * slow
    assert fast > 20.0
