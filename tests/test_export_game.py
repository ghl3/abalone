"""Tests for `model.export_game` — shards to reviewable game JSON.

The fixtures here are synthetic parquet written against the schema in
`docs/ARCHITECTURE.md` §5.4 column for column, *not* against whatever the Rust
writer currently emits. The reader is a consumer of a cross-language contract;
pinning it to the written spec is the only way a divergence shows up as a failing
test rather than as three generations of quiet garbage.

The tests that matter most:

  * **POV signs.** Shard `z`, `score_diff` and `q` are relative to the side to
    move *at that position*, so they alternate down the trajectory. A game where
    Black wins must show alternating per-ply labels and a `result` fixed to
    Black's POV regardless of who moved last.
  * **Uniform visit distributions must yield entropy exactly `ln(n_legal)`.**
    That is the precise pathology of the failed run (policy loss pinned at
    `ln(62)`); `summarise_generation` exists to make it visible, so its arithmetic
    is pinned here.
  * **A truncated shard costs one file, not the batch.** Self-play workers get
    killed mid-write; the export must route around that.
  * **Plain `json.dumps` must work.** pyarrow leaks numpy scalars into anything
    that touches its values carelessly, and a numpy scalar in the tree is a
    `TypeError` at write time, generations after the mistake.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from model.encoder import MOVE_SPACE
from model.export_game import (
    REQUIRED_COLUMNS,
    SELFPLAY_PREFIX,
    SHARD_COLUMNS,
    ExportResult,
    ShardError,
    collect_games,
    export_game,
    export_generation,
    find_generations,
    format_generation_report,
    is_unseeded,
    load_exported_games,
    main,
    read_shard,
    selfplay_metrics,
    split_games,
    summarise_generation,
    terminated_naturally,
    visit_entropy,
)
from model.validate import VAL_FROZEN_PREFIX, VAL_ROLLING_PREFIX

# --------------------------------------------------------------------------- #
# Fixtures — the v2 shard schema, verbatim from ARCHITECTURE §5.4              #
# --------------------------------------------------------------------------- #

V2_SCHEMA = pa.schema(
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


def make_row(**overrides):
    """One shard row with plausible defaults; override what a test cares about."""
    row = {
        "game_id": 7,
        "seed": 8891,
        "opening": 1,
        "handicap_black": 2,
        "handicap_white": 1,
        "own_bb_lo": 0x00FF00FF00FF00FF,
        "own_bb_hi": 0x1,
        "opp_bb_lo": 0xFF00FF00FF00FF00,
        "opp_bb_hi": 0x2,
        "black_losses": 0,
        "white_losses": 0,
        "turn": 0,
        "ply": 0,
        "max_plies": 200,
        "move_played": 1204,
        "is_full_search": True,
        "z": 1,
        "score_diff": 3,
        "q": 0.25,
        "child_move_idxs": [1204, 881, 42, 900],
        "child_visits": [10, 40, 20, 30],
        "cap_map_idx": [81 + 40],
        "cap_map_val": [0.98],
    }
    row.update(overrides)
    return row


def make_game(
    game_id=7,
    n_plies=6,
    z_black=1,
    score_black=3,
    q_black=0.25,
    seed=8891,
    opening=1,
    handicap=(2, 1),
    max_plies=200,
    full_search_period=1,
    child_idxs=(1204, 881, 42, 900),
    child_visits=(10, 40, 20, 30),
    losses=None,
):
    """A whole game's rows, POV-flipped per ply exactly as the writer must.

    `z_black`, `score_black` and `q_black` are stated once from Black's POV; each
    row stores them relative to whoever is to move at that ply. That flip is the
    thing the export has to undo, so the fixture applies it explicitly rather
    than letting a helper hide it.

    `losses` is the per-ply `(black_losses, white_losses)` *before* that ply's
    move, matching the writer. It defaults to the handicap held constant, which
    is a game in which nothing was ever pushed off; pass a ramp to give the game
    captures.
    """
    rows = []
    for ply in range(n_plies):
        turn = ply % 2
        sign = 1 if turn == 0 else -1
        lost = tuple(handicap) if losses is None else tuple(losses[ply])
        rows.append(
            make_row(
                game_id=game_id,
                seed=seed,
                opening=opening,
                handicap_black=handicap[0],
                handicap_white=handicap[1],
                black_losses=int(lost[0]),
                white_losses=int(lost[1]),
                turn=turn,
                ply=ply,
                max_plies=max_plies,
                move_played=(child_idxs[0] + ply) % MOVE_SPACE,
                is_full_search=(ply % full_search_period == 0),
                z=sign * z_black,
                score_diff=sign * score_black,
                q=sign * q_black,
                child_move_idxs=list(child_idxs),
                child_visits=list(child_visits),
            )
        )
    return rows


def losses_ramp(n_plies, handicap, captures_before_last):
    """Per-ply `(black_losses, white_losses)` for a game in which
    `captures_before_last` marbles come off Black over the first plies.

    The final move's capture is deliberately absent — the writer records losses
    *before* each move, so the capture that ends a game is in no row, and
    reconstructing it is `export_game`'s job.
    """
    out = []
    for ply in range(n_plies):
        taken = min(captures_before_last, max(0, ply))
        out.append((handicap[0] + taken, handicap[1]))
    return out


def write_shard(path, rows):
    """Write rows as a v2 parquet shard."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = {name: [row[name] for row in rows] for name in V2_SCHEMA.names}
    pq.write_table(pa.table(columns, schema=V2_SCHEMA), path)
    return path


def to_table(rows):
    columns = {name: [row[name] for row in rows] for name in V2_SCHEMA.names}
    return pa.table(columns, schema=V2_SCHEMA)


def assert_builtin_scalars(obj, path="game"):
    """No numpy scalars anywhere in the tree.

    `type(x) is int` rather than `isinstance`: `np.float64` subclasses `float`
    and would sail through an isinstance check, then blow up somewhere else.
    """
    if isinstance(obj, dict):
        for key, value in obj.items():
            assert type(key) is str, f"{path}: non-str key {key!r} ({type(key)})"
            assert_builtin_scalars(value, f"{path}.{key}")
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            assert_builtin_scalars(value, f"{path}[{i}]")
    else:
        assert type(obj) in (str, int, float, bool, type(None)), (
            f"{path}: {obj!r} is {type(obj)}, not a builtin scalar"
        )


# --------------------------------------------------------------------------- #
# Schema                                                                       #
# --------------------------------------------------------------------------- #


def test_required_columns_are_a_subset_of_the_schema():
    assert set(REQUIRED_COLUMNS) <= set(SHARD_COLUMNS)
    # The fixture is the spec; if these diverge one of them is wrong.
    assert set(SHARD_COLUMNS) == set(V2_SCHEMA.names)


# --------------------------------------------------------------------------- #
# split_games                                                                  #
# --------------------------------------------------------------------------- #


def test_split_games_groups_by_id_and_sorts_by_ply():
    rows = make_game(game_id=3, n_plies=4) + make_game(game_id=11, n_plies=5)
    # Interleave and scramble: contiguity and ply order must not be relied upon.
    scrambled = [rows[i] for i in (7, 0, 5, 2, 8, 1, 4, 6, 3)]
    assert len(scrambled) == len(rows)

    games = split_games(to_table(scrambled))

    assert list(games) == [3, 11]
    assert [r["ply"] for r in games[3]] == [0, 1, 2, 3]
    assert [r["ply"] for r in games[11]] == [0, 1, 2, 3, 4]


def test_split_games_handles_several_games_in_one_file(tmp_path):
    rows = make_game(game_id=1, n_plies=3) + make_game(game_id=2, n_plies=7)
    path = write_shard(tmp_path / "shard_t00_0000.parquet", rows)

    games = split_games(read_shard(path))

    assert sorted(games) == [1, 2]
    assert len(games[1]) == 3
    assert len(games[2]) == 7


def test_collect_games_merges_one_game_across_shard_files(tmp_path):
    rows = make_game(game_id=5, n_plies=8)
    # Split the game across two files, second half first, so neither file order
    # nor within-file order can carry the result.
    write_shard(tmp_path / "shard_t00_0000.parquet", rows[4:])
    write_shard(tmp_path / "shard_t01_0000.parquet", rows[:4])

    games, skipped = collect_games(sorted(tmp_path.glob("*.parquet")))

    assert skipped == []
    assert list(games) == [5]
    assert [r["ply"] for r in games[5]] == list(range(8))


# --------------------------------------------------------------------------- #
# POV sign conventions — the thing most likely to be silently wrong            #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n_plies", [5, 6])
def test_per_ply_pov_alternates_while_result_is_fixed_to_black(n_plies):
    """Black wins by 3. Per-ply labels alternate; `result` does not move."""
    game = export_game(make_game(n_plies=n_plies, z_black=1, score_black=3, q_black=0.25), "r", 4)

    # Fixed POV: Black's, regardless of who happened to move last.
    assert game["result"]["outcome"] == "black_wins"
    assert game["result"]["score_diff"] == 3
    assert game["result"]["plies"] == n_plies

    for move in game["moves"]:
        sign = 1 if move["ply"] % 2 == 0 else -1
        assert move["turn"] == ("black" if sign == 1 else "white")
        # `value_pov` is the whole point: the UI must never have to infer it.
        assert move["value_pov"] == move["turn"]
        assert move["z"] == sign * 1
        assert move["score_diff"] == sign * 3
        assert move["q"] == pytest.approx(sign * 0.25)

    # And the alternation is real, not a constant that happens to match.
    assert [m["z"] for m in game["moves"][:4]] == [1, -1, 1, -1]


def test_white_win_normalises_to_black_pov():
    """A White win is `score_diff` negative from Black's POV, both parities."""
    for n_plies in (4, 5):
        rows = make_game(n_plies=n_plies, z_black=-1, score_black=-2)
        game = export_game(rows, "r", 0)
        assert game["result"]["outcome"] == "white_wins"
        assert game["result"]["score_diff"] == -2
        # The White-to-move rows store +1/+2 — a win for them.
        white_moves = [m for m in game["moves"] if m["turn"] == "white"]
        assert all(m["z"] == 1 and m["score_diff"] == 2 for m in white_moves)


def test_draw_outcome():
    game = export_game(make_game(n_plies=4, z_black=0, score_black=0), "r", 0)
    assert game["result"]["outcome"] == "draw"
    assert game["result"]["score_diff"] == 0
    assert all(m["z"] == 0 for m in game["moves"])


def test_inconsistent_pov_labels_are_rejected():
    """If the writer's signs disagree, a fixed-POV result would be a coin flip."""
    rows = make_game(n_plies=4, z_black=1, score_black=3)
    rows[2]["z"] = -1  # Black to move at ply 2, but labelled a loss
    with pytest.raises(ShardError, match="z disagrees"):
        export_game(rows, "r", 0)

    rows = make_game(n_plies=4, z_black=1, score_black=3)
    rows[3]["score_diff"] = 5
    with pytest.raises(ShardError, match="score_diff disagrees"):
        export_game(rows, "r", 0)


# --------------------------------------------------------------------------- #
# Move records                                                                 #
# --------------------------------------------------------------------------- #


def test_visits_sorted_descending_and_total_consistent():
    game = export_game(
        make_game(n_plies=2, child_idxs=(1204, 881, 42, 900), child_visits=(10, 40, 20, 30)),
        "r",
        0,
    )
    for move in game["moves"]:
        assert move["visits"] == [[881, 40], [900, 30], [42, 20], [1204, 10]]
        counts = [v for _, v in move["visits"]]
        assert counts == sorted(counts, reverse=True)
        assert move["total_visits"] == sum(counts) == 100


def test_visit_ties_break_on_move_index_for_determinism():
    game = export_game(
        make_game(n_plies=1, child_idxs=(900, 42, 1204), child_visits=(7, 7, 7)), "r", 0
    )
    assert game["moves"][0]["visits"] == [[42, 7], [900, 7], [1204, 7]]


def test_move_carries_flat_index_only_no_notation():
    game = export_game(make_game(n_plies=2), "r", 0)
    move = game["moves"][0]
    assert move["idx"] == 1204
    # Notation is rendered client-side by the WASM engine, which owns `Move`.
    assert "notation" not in move


def test_both_full_search_values_round_trip():
    rows = make_game(n_plies=6, full_search_period=3)
    game = export_game(rows, "r", 0)
    flags = [m["is_full_search"] for m in game["moves"]]
    assert flags == [True, False, False, True, False, False]
    assert all(type(f) is bool for f in flags)


def test_q_is_rounded_not_left_as_float32_noise():
    rows = make_game(n_plies=1, q_black=0.18)
    game = export_game(rows, "r", 0)
    # float32(0.18) is 0.18000000715255737; six decimals is lossless for display.
    assert game["moves"][0]["q"] == 0.18


def test_out_of_range_move_index_is_rejected():
    rows = make_game(n_plies=2)
    rows[1]["move_played"] = MOVE_SPACE + 5  # u16 holds it; the spec does not
    with pytest.raises(ShardError, match="outside move space"):
        export_game(rows, "r", 0)


def test_mismatched_child_columns_are_rejected():
    rows = make_game(n_plies=1)
    rows[0]["child_visits"] = [1, 2]
    with pytest.raises(ShardError, match="length mismatch"):
        export_game(rows, "r", 0)


# --------------------------------------------------------------------------- #
# Metadata round-trip and serialisability                                      #
# --------------------------------------------------------------------------- #


def test_metadata_round_trips():
    rows = make_game(
        game_id=117, n_plies=3, seed=8891, opening=1, handicap=(2, 1), max_plies=200
    )
    game = export_game(rows, "hardy-feather-20260510-2220", 34)

    assert game["run_id"] == "hardy-feather-20260510-2220"
    assert game["gen"] == 34
    assert game["game_id"] == 117
    assert game["opening"] == "belgian_daisy"
    assert game["handicap"] == [2, 1]  # [black_conceded, white_conceded]
    assert game["seed"] == 8891
    assert game["max_plies"] == 200


def test_standard_opening_and_unknown_opening_code():
    assert export_game(make_game(n_plies=1, opening=0), "r", 0)["opening"] == "standard"
    assert export_game(make_game(n_plies=1, opening=9), "r", 0)["opening"] == "unknown_9"


def test_handicap_is_ordered_black_then_white():
    game = export_game(make_game(n_plies=1, handicap=(0, 4)), "r", 0)
    assert game["handicap"] == [0, 4]


def test_json_dumps_works_without_a_custom_encoder(tmp_path):
    """pyarrow values must not leak numpy scalars into the tree."""
    path = write_shard(tmp_path / "shard_t00_0000.parquet", make_game(n_plies=4))
    games = split_games(read_shard(path))
    game = export_game(games[7], "run", 3)

    blob = json.dumps(game)  # would raise TypeError on a numpy scalar
    assert json.loads(blob) == game
    assert_builtin_scalars(game)


def test_per_game_constants_must_actually_be_constant():
    rows = make_game(n_plies=4)
    rows[2]["seed"] = 12345
    with pytest.raises(ShardError, match="seed is not constant"):
        export_game(rows, "r", 0)


def test_empty_game_is_rejected():
    with pytest.raises(ShardError, match="no rows"):
        export_game([], "r", 0)


# --------------------------------------------------------------------------- #
# export_generation                                                            #
# --------------------------------------------------------------------------- #


def test_export_generation_writes_one_file_per_game(tmp_path):
    shards = tmp_path / "shards" / "gen_003"
    write_shard(shards / "shard_t00_0000.parquet", make_game(game_id=1, n_plies=4))
    write_shard(
        shards / "shard_t01_0000.parquet",
        make_game(game_id=12, n_plies=6) + make_game(game_id=7, n_plies=2),
    )
    out = tmp_path / "games" / "gen_003"

    result = export_generation(shards, out, "run-x", 3)

    assert isinstance(result, ExportResult)
    assert result.skipped == []
    assert result.shards_read == 2
    assert sorted(p.name for p in out.iterdir()) == [
        "game_0001.json",
        "game_0007.json",
        "game_0012.json",
    ]
    on_disk = load_exported_games(out)
    assert [g["game_id"] for g in on_disk] == [1, 7, 12]
    assert all(g["run_id"] == "run-x" and g["gen"] == 3 for g in on_disk)


def test_unreadable_and_partial_shards_are_skipped_not_fatal(tmp_path):
    shards = tmp_path / "shards" / "gen_001"
    good = write_shard(shards / "shard_t00_0000.parquet", make_game(game_id=1, n_plies=4))

    # Not parquet at all.
    (shards / "shard_t01_0000.parquet").write_bytes(b"this is not a parquet file")
    # Truncated mid-write — a self-play worker killed between page and footer.
    (shards / "shard_t02_0000.parquet").write_bytes(good.read_bytes()[: len(good.read_bytes()) // 2])
    # Valid parquet, wrong schema (an older writer, or a column renamed).
    pq.write_table(pa.table({"game_id": [1], "ply": [0]}), shards / "shard_t03_0000.parquet")

    result = export_generation(shards, tmp_path / "out", "run-x", 1)

    assert [g["game_id"] for g in result.games] == [1]
    assert len(result.skipped) == 3
    reasons = " ".join(s.reason for s in result.skipped)
    assert "unreadable parquet" in reasons
    assert "missing columns" in reasons
    assert (tmp_path / "out" / "game_0001.json").exists()


def test_a_corrupt_game_does_not_abort_the_others(tmp_path):
    shards = tmp_path / "shards" / "gen_001"
    bad = make_game(game_id=2, n_plies=4)
    bad[1]["z"] = 1  # White to move at ply 1 but labelled a win for Black
    write_shard(
        shards / "shard_t00_0000.parquet", make_game(game_id=1, n_plies=4) + bad + make_game(game_id=3, n_plies=2)
    )

    result = export_generation(shards, tmp_path / "out", "run-x", 1)

    assert [g["game_id"] for g in result.games] == [1, 3]
    assert len(result.skipped) == 1
    assert result.skipped[0].source == "game 2"


def test_export_generation_limit(tmp_path):
    shards = tmp_path / "shards" / "gen_002"
    rows = []
    for gid in range(5):
        rows += make_game(game_id=gid, n_plies=2)
    write_shard(shards / "shard_t00_0000.parquet", rows)

    result = export_generation(shards, tmp_path / "out", "run-x", 2, limit=2)

    assert [g["game_id"] for g in result.games] == [0, 1]
    assert len(list((tmp_path / "out").iterdir())) == 2


def test_missing_shard_directory_is_reported_not_raised(tmp_path):
    result = export_generation(tmp_path / "nope", tmp_path / "out", "run-x", 9)
    assert result.games == []
    assert len(result.skipped) == 1
    assert "no shard files" in result.skipped[0].reason
    assert not (tmp_path / "out").exists()  # no empty directories left behind


def test_find_generations(tmp_path):
    for name in ("gen_001", "gen_010", "gen_003"):
        (tmp_path / name).mkdir(parents=True)
    (tmp_path / "not_a_gen").mkdir()
    (tmp_path / "gen_099").write_text("a file, not a directory")

    assert find_generations(tmp_path) == [1, 3, 10]
    assert find_generations(tmp_path / "absent") == []


# --------------------------------------------------------------------------- #
# summarise_generation — the health check that was missing                     #
# --------------------------------------------------------------------------- #


def test_visit_entropy_of_uniform_is_ln_n():
    for n in (2, 7, 62):
        assert visit_entropy([13] * n) == pytest.approx(math.log(n))
    assert visit_entropy([1, 0, 0, 0]) == pytest.approx(0.0)
    assert math.isnan(visit_entropy([]))
    assert math.isnan(visit_entropy([0, 0]))


def test_uniform_visit_distribution_yields_entropy_equal_to_ln_legal():
    """The exact pathology of the failed run, pinned.

    Policy loss sat at 4.13 ≈ ln(62) for three generations: search returned a
    flat distribution over legal moves, so the policy target carried zero
    information. If this assertion ever weakens, the metric stops being able to
    report that.
    """
    n_legal = 62
    idxs = tuple(range(n_legal))
    visits = (8,) * n_legal
    games = [
        export_game(
            make_game(game_id=g, n_plies=4, child_idxs=idxs, child_visits=visits), "r", 0
        )
        for g in range(3)
    ]

    summary = summarise_generation(games)

    assert summary["mean_legal_moves"] == pytest.approx(n_legal)
    assert summary["policy_target_entropy"] == pytest.approx(math.log(n_legal))
    assert summary["policy_uniform_entropy"] == pytest.approx(math.log(n_legal))
    assert summary["policy_entropy_ratio"] == pytest.approx(1.0)


def test_peaked_visit_distribution_is_far_below_uniform():
    n_legal = 62
    idxs = tuple(range(n_legal))
    visits = (800 - (n_legal - 1),) + (1,) * (n_legal - 1)
    games = [export_game(make_game(n_plies=2, child_idxs=idxs, child_visits=visits), "r", 0)]

    summary = summarise_generation(games)

    assert summary["policy_uniform_entropy"] == pytest.approx(math.log(n_legal))
    assert summary["policy_target_entropy"] < 0.5 * math.log(n_legal)
    assert summary["policy_entropy_ratio"] < 0.5


def test_summary_only_counts_full_search_positions_for_entropy():
    """Fast-search rows have no usable policy target and must not dilute it."""
    n_legal = 8
    idxs = tuple(range(n_legal))
    rows = make_game(n_plies=4, full_search_period=2, child_idxs=idxs, child_visits=(4,) * n_legal)
    # Give the fast-search rows a degenerate distribution over fewer children.
    for row in rows:
        if not row["is_full_search"]:
            row["child_move_idxs"] = [0, 1]
            row["child_visits"] = [5, 0]
    summary = summarise_generation([export_game(rows, "r", 0)])

    assert summary["positions"] == 4
    assert summary["policy_rows"] == 2
    assert summary["full_search_rate"] == pytest.approx(0.5)
    assert summary["mean_legal_moves"] == pytest.approx(n_legal)
    assert summary["policy_target_entropy"] == pytest.approx(math.log(n_legal))


def test_summary_aggregates_outcomes_plies_and_handicaps():
    games = [
        export_game(make_game(game_id=1, n_plies=4, z_black=1, score_black=3, handicap=(2, 1)), "r", 0),
        export_game(make_game(game_id=2, n_plies=6, z_black=-1, score_black=-1, handicap=(2, 1)), "r", 0),
        export_game(make_game(game_id=3, n_plies=2, z_black=0, score_black=0, handicap=(0, 0)), "r", 0),
    ]

    summary = summarise_generation(games)

    assert summary["games"] == 3
    assert summary["positions"] == 12
    assert summary["decisive_rate"] == pytest.approx(2 / 3)
    assert summary["draw_rate"] == pytest.approx(1 / 3)
    assert summary["black_win_rate"] == pytest.approx(1 / 3)
    assert summary["white_win_rate"] == pytest.approx(1 / 3)
    assert summary["mean_plies"] == pytest.approx(4.0)
    assert summary["mean_abs_score_diff"] == pytest.approx(4 / 3)
    assert summary["handicap_distribution"] == {"0-0": 1, "2-1": 2}


# --------------------------------------------------------------------------- #
# The curriculum control signal (MODEL.md §4)                                  #
# --------------------------------------------------------------------------- #


def _game(game_id, *, handicap, plies, max_plies=200, z_black=1, score_black=3):
    """One game with the two properties the curriculum measures: whether it was
    seeded, and whether it ended before the ply cap."""
    return export_game(
        make_game(
            game_id=game_id,
            n_plies=plies,
            max_plies=max_plies,
            handicap=handicap,
            z_black=z_black,
            score_black=score_black,
        ),
        "r",
        0,
    )


def test_natural_termination_is_measured_over_unseeded_games_only():
    """Three unseeded games, one of which ran to the cap; the seeded games are
    not in the denominator however they ended."""
    games = [
        _game(1, handicap=(0, 0), plies=40),  # natural
        _game(2, handicap=(0, 0), plies=90),  # natural
        _game(3, handicap=(0, 0), plies=200),  # hit the cap
        _game(4, handicap=(5, 5), plies=6),  # seeded, natural, not counted
        _game(5, handicap=(0, 3), plies=200),  # seeded, capped, not counted
    ]

    summary = summarise_generation(games)

    assert summary["unseeded_games"] == 3
    assert summary["seeded_games"] == 2
    assert summary["natural_terminations"] == 2
    assert summary["natural_termination_rate"] == pytest.approx(2 / 3)
    assert summary["seeded_natural_termination_rate"] == pytest.approx(0.5)


def test_a_seeded_game_that_drew_zero_zero_is_an_unseeded_game():
    """Handicaps are drawn independently per side, so a game selected for
    seeding legitimately comes out at (0, 0). "Unseeded" is defined by the
    recorded handicap — which is what the position actually was — and not by
    whether self-play intended to seed it. There is nothing in the shard that
    records the intent, and nothing that should."""
    summary = summarise_generation([_game(1, handicap=(0, 0), plies=30)])
    assert summary["unseeded_games"] == 1
    assert summary["natural_termination_rate"] == pytest.approx(1.0)


def test_a_game_that_reaches_the_cap_did_not_terminate_naturally():
    """The whole detection rule: with the no-progress rule off, a game either
    reaches six captures or hits `max_plies`, so `plies < max_plies` *is*
    natural termination. The boundary is what matters — `plies == max_plies` is
    the adjudicated case."""
    assert terminated_naturally(_game(1, handicap=(0, 0), plies=199, max_plies=200))
    assert not terminated_naturally(_game(2, handicap=(0, 0), plies=200, max_plies=200))
    assert is_unseeded(_game(3, handicap=(0, 0), plies=10))
    assert not is_unseeded(_game(4, handicap=(0, 1), plies=10))


def test_the_seeded_split_of_the_diagnostics_is_reported_separately():
    """Seeded games start a capture from the end: short, decisive, wide margins.
    Averaged together with unseeded ones the headline decisive rate is
    uninterpretable, which is the reason the split exists."""
    games = [
        _game(1, handicap=(5, 5), plies=8, z_black=1, score_black=6),
        _game(2, handicap=(5, 4), plies=12, z_black=1, score_black=6),
        _game(3, handicap=(0, 0), plies=200, z_black=0, score_black=0),
        _game(4, handicap=(0, 0), plies=200, z_black=1, score_black=1),
    ]

    summary = summarise_generation(games)

    assert summary["seeded_decisive_rate"] == pytest.approx(1.0)
    assert summary["seeded_mean_abs_score_diff"] == pytest.approx(6.0)
    assert summary["seeded_mean_plies"] == pytest.approx(10.0)
    assert summary["unseeded_decisive_rate"] == pytest.approx(0.5)
    assert summary["unseeded_mean_abs_score_diff"] == pytest.approx(0.5)
    assert summary["unseeded_mean_plies"] == pytest.approx(200.0)
    # The headline numbers still cover the whole generation.
    assert summary["decisive_rate"] == pytest.approx(0.75)


def test_no_unseeded_games_gives_nan_not_zero():
    """A generation with every game seeded has no estimate at all. Zero would
    read as "the network never finishes a game" and hold the curriculum for the
    right-looking wrong reason."""
    summary = summarise_generation([_game(1, handicap=(1, 2), plies=10)])
    assert summary["unseeded_games"] == 0
    assert math.isnan(summary["natural_termination_rate"])


def test_the_measurement_covers_every_game_not_just_the_exported_ones(tmp_path):
    """`export.games_per_gen` caps what is *written*. A controller reading only
    the written subset would be measuring the export limit: at 20-of-200 with a
    0.7 seeding rate it would see ~6 unseeded games a generation and hold
    forever."""
    rows = []
    for gid in range(10):
        rows += make_game(
            game_id=gid,
            n_plies=4 if gid < 3 else 60,
            max_plies=60,
            handicap=(0, 0) if gid < 6 else (3, 2),
        )
    shards = tmp_path / "shards"
    write_shard(shards / "shard_t00_0000.parquet", rows)

    result = export_generation(shards, tmp_path / "out", "run-x", 5, limit=2)

    assert len(result.games) == 2, "the export limit still caps what is written"
    assert result.summary["games"] == 10
    assert result.summary["unseeded_games"] == 6
    assert result.summary["natural_terminations"] == 3
    assert result.summary["natural_termination_rate"] == pytest.approx(0.5)
    # And the report a human reads is the full-generation one.
    report = "\n".join(format_generation_report(result))
    assert "10 games" in report
    assert "natural termination 50.0%" in report


def test_summary_of_nothing_is_nan_not_a_misleading_zero():
    summary = summarise_generation([])
    assert summary["games"] == 0
    assert summary["positions"] == 0
    for key in (
        "decisive_rate",
        "mean_plies",
        "mean_abs_score_diff",
        "natural_termination_rate",
        "unseeded_decisive_rate",
        "seeded_decisive_rate",
        "policy_target_entropy",
        "policy_uniform_entropy",
        "policy_entropy_ratio",
        "policy_entropy_gap",
        "mean_legal_moves",
        "captures_per_100_plies",
        "mean_captures",
        "unseeded_captures_per_100_plies",
        "seeded_captures_per_100_plies",
    ):
        assert math.isnan(summary[key]), key


# --------------------------------------------------------------------------- #
# captures — the bloodbath indicator                                           #
# --------------------------------------------------------------------------- #


def _capture_game(game_id, *, plies, max_plies, handicap, captures_before_last):
    return export_game(
        make_game(
            game_id=game_id,
            n_plies=plies,
            max_plies=max_plies,
            handicap=handicap,
            losses=losses_ramp(plies, handicap, captures_before_last),
        ),
        "r",
        0,
    )


def test_a_natural_termination_counts_the_final_capture_that_no_row_records():
    """The writer records losses *before* each move, so the capture that ends
    the game is in no row. A game short of its cap ended *because* the sixth
    capture landed, and a move pushes off at most one marble — so the missing
    count is exactly one."""
    game = _capture_game(1, plies=30, max_plies=200, handicap=(0, 0), captures_before_last=5)
    assert game["result"]["captures"] == 6
    assert game["result"]["captures_in_play"] == 6


def test_a_capped_game_does_not_invent_a_final_capture():
    """At the cap the last move's capture is genuinely unknowable. Counting one
    anyway would inflate the rate on exactly the games — long, grinding ones —
    where the indicator is supposed to read low."""
    game = _capture_game(2, plies=200, max_plies=200, handicap=(0, 0), captures_before_last=3)
    assert game["result"]["captures"] == 3
    assert game["result"]["captures_in_play"] == 3


def test_the_seeded_head_start_is_not_counted_as_captures_in_play():
    """A game seeded at (3, 2) was *given* five captures. Counting them would
    make every seeded game read as a bloodbath and make the indicator track the
    handicap rate instead of the play."""
    game = _capture_game(3, plies=20, max_plies=200, handicap=(3, 2), captures_before_last=0)
    assert game["result"]["captures"] == 6  # 3 + 2 given, plus the one that ended it
    assert game["result"]["captures_in_play"] == 1


def test_captures_per_100_plies_on_games_with_known_capture_counts():
    """Two unseeded games: 6 captures in 30 plies and 6 in 90. Pooled that is
    12 captures over 120 plies = 10.0 per 100 plies."""
    games = [
        _capture_game(1, plies=30, max_plies=200, handicap=(0, 0), captures_before_last=5),
        _capture_game(2, plies=90, max_plies=200, handicap=(0, 0), captures_before_last=5),
    ]
    summary = summarise_generation(games)
    assert summary["captures_per_100_plies"] == pytest.approx(100.0 * 12 / 120)
    assert summary["mean_captures"] == pytest.approx(6.0)
    assert summary["unseeded_captures_per_100_plies"] == pytest.approx(10.0)
    assert math.isnan(summary["seeded_captures_per_100_plies"])


def test_captures_per_100_plies_falls_as_games_lengthen():
    """The direction the metric has to move in. Same six captures, three times
    the plies: a third of the rate. If this ever reads flat, the indicator
    cannot distinguish a bloodbath from positional play."""
    short = summarise_generation(
        [_capture_game(1, plies=30, max_plies=400, handicap=(0, 0), captures_before_last=5)]
    )
    long = summarise_generation(
        [_capture_game(1, plies=90, max_plies=400, handicap=(0, 0), captures_before_last=5)]
    )
    assert short["captures_per_100_plies"] == pytest.approx(20.0)
    assert long["captures_per_100_plies"] == pytest.approx(20.0 / 3.0)


def test_captures_split_by_seeding():
    """Seeded games need one capture to finish and unseeded ones need six, so a
    pooled rate moves with the curriculum even when neither population does."""
    games = [
        _capture_game(1, plies=50, max_plies=200, handicap=(0, 0), captures_before_last=5),
        _capture_game(2, plies=10, max_plies=200, handicap=(5, 5), captures_before_last=0),
    ]
    summary = summarise_generation(games)
    assert summary["unseeded_mean_captures"] == pytest.approx(6.0)
    assert summary["seeded_mean_captures"] == pytest.approx(1.0)
    assert summary["unseeded_captures_per_100_plies"] == pytest.approx(12.0)
    assert summary["seeded_captures_per_100_plies"] == pytest.approx(10.0)
    assert summary["captures_per_100_plies"] == pytest.approx(100.0 * 7 / 60)


def test_captures_are_nan_for_games_exported_before_the_metric_existed():
    """An old `runs/` directory must read back as "not measured", never as zero
    captures — which would look like the best positional play ever recorded."""
    game = _capture_game(1, plies=30, max_plies=200, handicap=(0, 0), captures_before_last=5)
    del game["result"]["captures_in_play"]
    summary = summarise_generation([game])
    assert math.isnan(summary["captures_per_100_plies"])
    assert math.isnan(summary["mean_captures"])
    # Everything else is still measurable from an old export.
    assert summary["mean_plies"] == pytest.approx(30.0)


# --------------------------------------------------------------------------- #
# entropy gap, the full split, and the `selfplay/` namespace                   #
# --------------------------------------------------------------------------- #


def test_policy_entropy_gap_is_uniform_minus_target():
    """MODEL.md §8.2 states the criterion as a gap. Reporting it directly means
    a reader checks the criterion instead of subtracting two logged numbers."""
    n_legal = 62
    idxs = tuple(range(n_legal))
    peaked = (800 - (n_legal - 1),) + (1,) * (n_legal - 1)
    summary = summarise_generation(
        [export_game(make_game(n_plies=2, child_idxs=idxs, child_visits=peaked), "r", 0)]
    )
    assert summary["policy_entropy_gap"] == pytest.approx(
        summary["policy_uniform_entropy"] - summary["policy_target_entropy"]
    )
    assert summary["policy_entropy_gap"] > 0.0

    flat = summarise_generation(
        [export_game(make_game(n_plies=2, child_idxs=idxs, child_visits=(8,) * n_legal), "r", 0)]
    )
    # The pathological case: search learned nothing, so the gap is zero.
    assert flat["policy_entropy_gap"] == pytest.approx(0.0, abs=1e-9)


def test_every_rate_is_reported_on_both_sides_of_the_split():
    """No rate may exist pooled-only or on one side only: a reader comparing
    generations needs the same key set every time."""
    summary = summarise_generation(
        [
            _capture_game(1, plies=30, max_plies=200, handicap=(0, 0), captures_before_last=5),
            _capture_game(2, plies=10, max_plies=200, handicap=(4, 1), captures_before_last=0),
        ]
    )
    split_keys = {
        "decisive_rate",
        "draw_rate",
        "black_win_rate",
        "white_win_rate",
        "natural_termination_rate",
        "mean_plies",
        "mean_abs_score_diff",
        "mean_captures",
        "captures_per_100_plies",
        "full_search_rate",
        "positions",
    }
    for key in split_keys:
        assert f"unseeded_{key}" in summary, key
        assert f"seeded_{key}" in summary, key
    # `natural_termination_rate` bare is the UNSEEDED rate — the curriculum
    # control signal — not a pooled one. `model/curriculum.py` reads it.
    assert summary["natural_termination_rate"] == summary["unseeded_natural_termination_rate"]
    assert summary["unseeded_positions"] + summary["seeded_positions"] == summary["positions"]


def test_selfplay_metrics_namespaces_every_number_and_drops_the_histogram():
    summary = summarise_generation(
        [_capture_game(1, plies=30, max_plies=200, handicap=(0, 0), captures_before_last=5)]
    )
    metrics = selfplay_metrics(summary)

    assert all(k.startswith(SELFPLAY_PREFIX) for k in metrics)
    assert all(isinstance(v, float) for v in metrics.values())
    assert f"{SELFPLAY_PREFIX}captures_per_100_plies" in metrics
    assert f"{SELFPLAY_PREFIX}policy_entropy_gap" in metrics
    assert f"{SELFPLAY_PREFIX}positions" in metrics
    assert f"{SELFPLAY_PREFIX}seeded_games" in metrics
    assert f"{SELFPLAY_PREFIX}unseeded_games" in metrics
    # The handicap histogram is the one non-scalar, and it is not renamed into
    # the scalar namespace under a misleading name.
    assert f"{SELFPLAY_PREFIX}handicap_distribution" not in metrics
    assert set(summary) - {k[len(SELFPLAY_PREFIX):] for k in metrics} == {
        "handicap_distribution"
    }
    json.dumps(metrics)


def test_selfplay_metrics_is_a_different_namespace_from_the_validation_ones():
    """`selfplay/policy_target_entropy` is the training signal and it moves.
    `val_frozen/data_policy_target_entropy` is a constant. Reading one for the
    other is the mistake the namespaces exist to prevent, so they must not
    collide."""
    assert SELFPLAY_PREFIX not in (VAL_FROZEN_PREFIX, VAL_ROLLING_PREFIX)
    summary = summarise_generation(
        [_capture_game(1, plies=30, max_plies=200, handicap=(0, 0), captures_before_last=5)]
    )
    keys = set(selfplay_metrics(summary))
    assert not any(k.startswith((VAL_FROZEN_PREFIX, VAL_ROLLING_PREFIX)) for k in keys)
    assert f"{SELFPLAY_PREFIX}policy_target_entropy" in keys


def test_summary_is_json_serialisable():
    games = [export_game(make_game(n_plies=2), "r", 0)]
    assert json.loads(json.dumps(summarise_generation(games)))["games"] == 1


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #


def _make_run(tmp_path, gens=(1, 2)):
    run_dir = tmp_path / "runs" / "hardy-feather-20260510-2220"
    for gen in gens:
        write_shard(
            run_dir / "shards" / f"gen_{gen:03d}" / "shard_t00_0000.parquet",
            make_game(game_id=1, n_plies=4) + make_game(game_id=2, n_plies=6, z_black=0, score_black=0),
        )
    return run_dir


def test_cli_exports_one_generation(tmp_path, capsys):
    run_dir = _make_run(tmp_path)

    assert main(["--run-dir", str(run_dir), "--gen", "2"]) == 0

    out = capsys.readouterr().out
    assert "gen 002" in out
    assert "policy target entropy" in out
    assert "exported 2 game(s)" in out
    assert (run_dir / "games" / "gen_002" / "game_0001.json").exists()
    assert not (run_dir / "games" / "gen_001").exists()
    # run_id defaults to the run directory name.
    game = json.loads((run_dir / "games" / "gen_002" / "game_0001.json").read_text())
    assert game["run_id"] == "hardy-feather-20260510-2220"
    assert game["gen"] == 2


def test_cli_all_gens_with_limit_and_custom_out_dir(tmp_path, capsys):
    run_dir = _make_run(tmp_path, gens=(1, 2, 3))
    out_root = tmp_path / "exported"

    code = main(
        [
            "--run-dir",
            str(run_dir),
            "--all-gens",
            "--out-dir",
            str(out_root),
            "--limit",
            "1",
            "--run-id",
            "override",
        ]
    )

    assert code == 0
    assert sorted(p.name for p in out_root.iterdir()) == ["gen_001", "gen_002", "gen_003"]
    for gen in (1, 2, 3):
        files = sorted((out_root / f"gen_{gen:03d}").iterdir())
        assert [p.name for p in files] == ["game_0001.json"]
        assert json.loads(files[0].read_text())["run_id"] == "override"
    assert "exported 3 game(s)" in capsys.readouterr().out


def test_cli_reports_missing_run(tmp_path, capsys):
    assert main(["--run-dir", str(tmp_path / "absent"), "--all-gens"]) == 1
    assert "no generations found" in capsys.readouterr().out


def test_report_warns_when_visit_distributions_are_uniform():
    n_legal = 62
    idxs = tuple(range(n_legal))
    uniform = export_game(
        make_game(n_plies=2, child_idxs=idxs, child_visits=(8,) * n_legal), "r", 0
    )
    peaked = export_game(
        make_game(n_plies=2, child_idxs=idxs, child_visits=(700,) + (1,) * (n_legal - 1)), "r", 0
    )

    flat = "\n".join(format_generation_report(ExportResult(gen=1, games=[uniform])))
    sharp = "\n".join(format_generation_report(ExportResult(gen=1, games=[peaked])))

    assert "WARNING" in flat
    assert "policy target entropy" in flat
    assert "WARNING" not in sharp
