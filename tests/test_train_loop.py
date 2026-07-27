"""Unit tests for the training loop's pure helpers.

The loop itself is deliberately not unit-tested: it spawns subprocesses, writes
parquet and runs SGD, and a mock of all that tests the mock. What *is* tested
here is every decision the loop makes that could silently be wrong — the
predicates and the arithmetic — plus the two parsers that sit on a contract
with another process.

The interesting cases are the ones that already caused an incident:

* `entropy_ratio_alarm` is the guard for the failure that ran unnoticed for
  three generations (policy loss pinned at `ln(branching)`).
* `retention_victims` must never collect the file a resume needs, which the
  old symlink-resolving implementation could (review §7.3).
* `git_sha_drift` is the detection that `config_hash` cannot provide, because
  the hash covers YAML and the plane layout lives in Rust.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from model.eval import (
    KIND_FLOOR,
    KIND_FROZEN,
    KIND_TRAILING,
    LadderOpponent,
    MatchResult,
    _format_player,
    _slug,
    clamped_fraction,
    ladder_summary,
    mean_elo,
    model_spec,
    opponent_label,
)
from model.state import GenRecord, RunState
from model.train_loop import (
    LadderRung,
    _count_completed_games,
    _fmt_secs,
    entropy_ratio_alarm,
    epoch_budget_steps,
    epochs_alarm,
    epochs_over_buffer,
    git_sha_drift,
    ladder_opponents,
    namespaced,
    overfit_alarm,
    plan_resume,
    retention_victims,
    should_run_ladder,
    should_validate,
    tb_step_offset,
)

# --------------------------------------------------------------------------- #
# Resume resolution                                                            #
# --------------------------------------------------------------------------- #


def test_resume_after_a_clean_generation_starts_the_next_one() -> None:
    plan = plan_resume(current_gen=7, current_phase="complete")
    assert plan.resume_ckpt_gen == 7
    assert plan.start_gen == 8
    assert plan.discard_gen is None


@pytest.mark.parametrize("phase", ["self_play", "training", "export", "validate", "ladder"])
def test_resume_mid_generation_redoes_it_from_self_play(phase: str) -> None:
    """`current_gen` counts *completed* generations, so a crash in any phase of
    gen 8 leaves `current_gen = 7` and a partial `shards/gen_008/`. The shards
    are partial by definition — the generation restarts from self-play."""
    plan = plan_resume(current_gen=7, current_phase=phase)
    assert plan.resume_ckpt_gen == 7
    assert plan.start_gen == 8
    assert plan.discard_gen == 8


def test_resume_from_bootstrap() -> None:
    plan = plan_resume(current_gen=0, current_phase="complete")
    assert plan == plan_resume(0, "complete")
    assert (plan.resume_ckpt_gen, plan.start_gen, plan.discard_gen) == (0, 1, None)


def test_resume_never_loads_a_checkpoint_that_does_not_exist_yet() -> None:
    """The checkpoint to load is always the last *completed* generation, never
    the one in progress — that .pt was never written."""
    for gen in range(0, 20):
        for phase in ("complete", "training", "ladder"):
            assert plan_resume(gen, phase).resume_ckpt_gen == gen


# --------------------------------------------------------------------------- #
# git SHA drift                                                                #
# --------------------------------------------------------------------------- #


def test_git_sha_drift_detected() -> None:
    assert git_sha_drift("abc123", "def456")


def test_git_sha_match_is_not_drift() -> None:
    assert not git_sha_drift("abc123", "abc123")


@pytest.mark.parametrize(("recorded", "current"), [("", "abc"), ("abc", ""), ("", "")])
def test_unknown_sha_is_not_evidence_of_drift(recorded: str, current: str) -> None:
    """A tarball checkout or a build outside a repo yields no SHA. Absence of
    evidence must not warn — the warning would be permanent and ignored."""
    assert not git_sha_drift(recorded, current)


def test_state_records_and_round_trips_the_git_sha(tmp_path: Path) -> None:
    state = RunState.fresh("run-x", "hash-y", git_sha="0123456789ab")
    state.append_history(GenRecord(gen=1, train_loss_total=1.5))
    state.save_atomic(tmp_path / "state.json", fsync=False)
    loaded = RunState.load(tmp_path / "state.json")
    assert loaded.git_sha == "0123456789ab"
    assert loaded.history[0].train_loss_total == 1.5


def test_state_load_tolerates_unknown_keys(tmp_path: Path) -> None:
    """A state file written by a newer build must not hard-fail an older one —
    and `promoted`, from the deleted gating era, is exactly such a key."""
    (tmp_path / "state.json").write_text(
        '{"run_id": "r", "current_gen": 3, "invented_later": 1, '
        '"history": [{"gen": 1, "promoted": true}]}'
    )
    loaded = RunState.load(tmp_path / "state.json")
    assert loaded.current_gen == 3
    assert loaded.history[0].gen == 1


# --------------------------------------------------------------------------- #
# Ladder scheduling                                                            #
# --------------------------------------------------------------------------- #


def test_ladder_runs_on_the_cadence() -> None:
    assert [
        g for g in range(1, 21) if should_run_ladder(g, total_gens=20, every_gens=5)
    ] == [5, 10, 15, 20]


def test_ladder_runs_on_the_final_generation_off_cadence() -> None:
    """The last generation's number is the one anybody quotes; it must not be
    missed because 12 is not a multiple of 5."""
    assert should_run_ladder(12, total_gens=12, every_gens=5, run_on_final=True)
    assert not should_run_ladder(12, total_gens=12, every_gens=5, run_on_final=False)
    assert not should_run_ladder(11, total_gens=12, every_gens=5, run_on_final=True)


@pytest.mark.parametrize("every", [0, -1])
def test_ladder_disabled_means_disabled(every: int) -> None:
    """Including on the final generation: "off" should mean off."""
    assert not any(
        should_run_ladder(g, total_gens=10, every_gens=every, run_on_final=True)
        for g in range(1, 11)
    )


def test_ladder_every_generation() -> None:
    assert all(should_run_ladder(g, 3, every_gens=1) for g in range(1, 4))


# --------------------------------------------------------------------------- #
# Ladder opponents                                                             #
# --------------------------------------------------------------------------- #


def _ckpts(tmp_path: Path, *gens: int) -> None:
    for g in gens:
        (tmp_path / f"gen_{g:03d}.onnx").write_bytes(b"")


def test_floor_anchors_get_fewer_games_than_checkpoint_rungs(tmp_path: Path) -> None:
    """A floor rung answers "is anything catastrophically broken", which is a
    binary question; a checkpoint rung answers "how much stronger am I", which
    needs enough games for the interval to exclude zero."""
    _ckpts(tmp_path, 1)
    out = ladder_opponents(
        ["random", "heuristic@100"], [1], [], gen=5, ckpt_dir=tmp_path,
        games=40, floor_games=16,
    )
    assert [(o.spec, o.games, o.kind) for o in out] == [
        ("random", 16, KIND_FLOOR),
        ("heuristic@100", 16, KIND_FLOOR),
        (model_spec(tmp_path / "gen_001.onnx"), 40, KIND_FROZEN),
    ]


def test_ladder_opponents_keeps_fixed_anchors(tmp_path: Path) -> None:
    fixed = ["random", "heuristic@100"]
    out = ladder_opponents(fixed, [], [], gen=5, ckpt_dir=tmp_path)
    assert [o.spec for o in out] == fixed
    assert all(o.kind == KIND_FLOOR for o in out)


def test_ladder_opponents_adds_existing_frozen_checkpoints(tmp_path: Path) -> None:
    _ckpts(tmp_path, 1, 10)
    out = ladder_opponents(["random"], [1, 10], [], gen=20, ckpt_dir=tmp_path)
    assert [o.spec for o in out] == [
        "random",
        model_spec(tmp_path / "gen_001.onnx"),
        model_spec(tmp_path / "gen_010.onnx"),
    ]


def test_trailing_rungs_are_relative_to_the_current_generation(tmp_path: Path) -> None:
    """The rung that does not saturate: it moves with the network, so it still
    has resolution once every fixed anchor is being swept."""
    _ckpts(tmp_path, 1, 15, 18)
    out = ladder_opponents([], [1], [5, 2], gen=20, ckpt_dir=tmp_path)
    assert [(Path(o.spec).name, o.kind) for o in out] == [
        ("gen_001.onnx", KIND_FROZEN),
        ("gen_015.onnx", KIND_TRAILING),
        ("gen_018.onnx", KIND_TRAILING),
    ]


def test_a_trailing_rung_that_lands_on_a_frozen_one_is_deduplicated(tmp_path: Path) -> None:
    """`frozen_gens: [25]` with `trailing_gens: [5]` collide at generation 30.
    Playing the same checkpoint twice costs a rung's worth of wall-clock and
    measures it twice under two names."""
    _ckpts(tmp_path, 25)
    out = ladder_opponents([], [25], [5], gen=30, ckpt_dir=tmp_path)
    assert [(Path(o.spec).name, o.kind) for o in out] == [("gen_025.onnx", KIND_FROZEN)]


def test_ladder_opponents_skips_a_collected_checkpoint(tmp_path: Path) -> None:
    """Retention may have taken it. Dropping the rung beats crashing a ladder
    five hours into a run."""
    out = ladder_opponents(["random"], [1], [5], gen=20, ckpt_dir=tmp_path)
    assert [o.spec for o in out] == ["random"]


def test_a_generation_is_never_its_own_ladder_opponent(tmp_path: Path) -> None:
    _ckpts(tmp_path, 5)
    assert ladder_opponents([], [5], [], gen=5, ckpt_dir=tmp_path) == []
    # `trailing_gens: [0]` is rejected by the config, but the resolver must not
    # rely on that to avoid playing itself.
    assert ladder_opponents([], [], [0], gen=5, ckpt_dir=tmp_path) == []
    out = ladder_opponents([], [5], [], gen=6, ckpt_dir=tmp_path)
    assert [o.spec for o in out] == [model_spec(tmp_path / "gen_005.onnx")]


def test_ladder_opponents_deduplicates_and_orders(tmp_path: Path) -> None:
    _ckpts(tmp_path, 1, 3)
    out = ladder_opponents([], [3, 1, 3], [], gen=9, ckpt_dir=tmp_path)
    assert [o.spec for o in out] == [
        model_spec(tmp_path / "gen_001.onnx"),
        model_spec(tmp_path / "gen_003.onnx"),
    ]


# --------------------------------------------------------------------------- #
# Validation scheduling                                                        #
# --------------------------------------------------------------------------- #


def test_validation_starts_after_the_holdout_generation_exists() -> None:
    """Generation 1 *is* the holdout; there is nothing frozen to score against
    until it has been produced."""
    run = [
        g
        for g in range(1, 8)
        if should_validate(g, enabled=True, holdout_gen=1, every_gens=1)
    ]
    assert run == [2, 3, 4, 5, 6, 7]


def test_validation_cadence_counts_from_the_holdout() -> None:
    run = [
        g
        for g in range(1, 12)
        if should_validate(g, enabled=True, holdout_gen=2, every_gens=3)
    ]
    assert run == [5, 8, 11]


def test_validation_can_be_disabled() -> None:
    assert not should_validate(5, enabled=False, holdout_gen=1, every_gens=1)
    assert not should_validate(5, enabled=True, holdout_gen=1, every_gens=0)


# --------------------------------------------------------------------------- #
# The entropy-ratio alarm — the most important predicate in the file           #
# --------------------------------------------------------------------------- #


def test_uniform_policy_targets_raise_the_alarm() -> None:
    """`policy_target_entropy == ln(mean legal moves)` means MCTS returned a
    flat visit distribution: the policy target carries zero information, and
    the previous run sat there for three generations without anyone noticing."""
    assert entropy_ratio_alarm(1.0, 0.95)


def test_a_ratio_just_over_the_threshold_raises_the_alarm() -> None:
    assert entropy_ratio_alarm(0.951, 0.95)


def test_a_healthy_ratio_is_silent() -> None:
    assert not entropy_ratio_alarm(0.62, 0.95)


def test_the_threshold_itself_is_not_an_alarm() -> None:
    assert not entropy_ratio_alarm(0.95, 0.95)


@pytest.mark.parametrize("value", [None, float("nan"), "n/a"])
def test_an_unknown_ratio_is_not_an_alarm(value) -> None:
    """No measurement is not the same as a bad measurement; crying wolf on
    `nan` trains people to ignore the warning that matters."""
    assert not entropy_ratio_alarm(value, 0.95)


def test_alarm_matches_the_arithmetic_it_stands_for() -> None:
    """Sanity-check against the quantity itself: a uniform distribution over
    62 legal moves has entropy exactly `ln(62)`, so the ratio is 1."""
    branching = 62
    uniform_entropy = math.log(branching)
    target_entropy = math.log(branching)  # search learned nothing
    assert entropy_ratio_alarm(target_entropy / uniform_entropy, 0.95)
    # A distribution concentrated on ~4 moves is healthy.
    assert not entropy_ratio_alarm(math.log(4) / uniform_entropy, 0.95)


# --------------------------------------------------------------------------- #
# TensorBoard step offsets                                                     #
# --------------------------------------------------------------------------- #


def _hist(*steps: int) -> list[GenRecord]:
    return [GenRecord(gen=i + 1, train_steps=n) for i, n in enumerate(steps)]


def test_step_offset_is_zero_before_any_generation_has_run() -> None:
    assert tb_step_offset([]) == 0


def test_step_offsets_do_not_collide_when_a_generation_exceeds_the_minimum() -> None:
    """The regression this replaced.

    `(gen - 1) * steps_per_gen_min` only holds while every generation runs
    exactly the minimum. The budget is derived from `target_epochs_per_gen`, so
    a generation exceeds it as soon as the buffer is big enough — and then the
    next generation starts *inside* the previous one's range, every chart
    overlays itself and the x-axis stops being a timeline.
    """
    min_steps = 400
    runs = [400, 900, 1500]  # gen 2 and 3 outgrew the minimum

    old = [(gen - 1) * min_steps for gen in (1, 2, 3, 4)]
    assert old[2] < old[1] + runs[1], "the old formula collides, as it must for this test"

    offsets, history = [], []
    for n in runs:
        offsets.append(tb_step_offset(history))
        history.append(GenRecord(gen=len(history) + 1, train_steps=n))
    offsets.append(tb_step_offset(history))

    # Each generation's range [offset+1, offset+steps] is strictly after the
    # previous one's, and there are no gaps.
    assert offsets == [0, 400, 1300, 2800]
    for i, n in enumerate(runs):
        assert offsets[i] + n == offsets[i + 1]


def test_step_offset_survives_a_resume_from_state_json(tmp_path: Path) -> None:
    """It is accumulated from `state.history`, which `state.json` persists — so
    a resumed run picks up where the event file left off instead of overwriting
    generation 1."""
    state = RunState(run_id="r", config_hash="h")
    for gen, n in ((1, 400), (2, 900), (3, 1500)):
        state.append_history(GenRecord(gen=gen, train_steps=n))
    path = tmp_path / "state.json"
    state.save_atomic(path, fsync=False)

    assert tb_step_offset(RunState.load(path).history) == 2800


def test_step_offset_ignores_generations_that_took_no_steps() -> None:
    """`train_steps` is `None` for a generation that never got a batch, and
    `None` is not zero-ish enough for `sum` on its own."""
    assert tb_step_offset([GenRecord(gen=1), GenRecord(gen=2, train_steps=7)]) == 7
    assert tb_step_offset(_hist(0, 0, 5)) == 5


# --------------------------------------------------------------------------- #
# Retention                                                                    #
# --------------------------------------------------------------------------- #


def _files(tmp_path: Path, gens: range, ext: str) -> list[Path]:
    out = []
    for g in gens:
        p = tmp_path / f"gen_{g:03d}.{ext}"
        p.write_bytes(b"")
        out.append(p)
    return out


def test_retention_keeps_the_last_k(tmp_path: Path) -> None:
    files = _files(tmp_path, range(1, 11), "pt")
    victims = retention_victims(files, keep=3, protected=set())
    assert victims == files[:7]


def test_retention_keeps_everything_when_under_the_limit(tmp_path: Path) -> None:
    files = _files(tmp_path, range(1, 4), "pt")
    assert retention_victims(files, keep=10, protected=set()) == []


def test_retention_never_collects_a_protected_file(tmp_path: Path) -> None:
    """The resume checkpoint and anything `state.json` points at. Protection is
    by explicit path: `best.onnx` degrades to a plain copy where symlinks are
    unavailable, and resolving a copy protects nothing."""
    files = _files(tmp_path, range(1, 11), "onnx")
    protected = {files[0], files[2]}
    victims = retention_victims(files, keep=3, protected=protected)
    assert not (protected & set(victims))
    assert files[1] in victims


def test_retention_is_ordered_by_generation_not_by_mtime(tmp_path: Path) -> None:
    """Files are created out of order on purpose: retention must key off the
    zero-padded generation in the name, which is what makes it deterministic."""
    paths = []
    for g in (7, 1, 4, 10, 2):
        p = tmp_path / f"gen_{g:03d}.pt"
        p.write_bytes(b"")
        paths.append(p)
    victims = retention_victims(paths, keep=2, protected=set())
    assert [p.name for p in victims] == ["gen_001.pt", "gen_002.pt", "gen_004.pt"]


def test_retention_keep_zero_collects_everything_unprotected(tmp_path: Path) -> None:
    files = _files(tmp_path, range(1, 4), "pt")
    assert retention_victims(files, keep=0, protected={files[-1]}) == files[:-1]


def test_retention_treats_negative_keep_as_zero(tmp_path: Path) -> None:
    files = _files(tmp_path, range(1, 4), "pt")
    assert retention_victims(files, keep=-5, protected=set()) == files


# --------------------------------------------------------------------------- #
# The self-play log contract                                                   #
# --------------------------------------------------------------------------- #


SELFPLAY_LINE = (
    "  [t{t}] game {g}: {p} plies, final=Wins(Black), handicap=2/1, score_diff=+3\n"
)


def test_completed_games_parsed_from_the_log(tmp_path: Path) -> None:
    log = tmp_path / "selfplay.log"
    log.write_text(
        "selfplay-batch: model=x out=y games=3\n"
        + SELFPLAY_LINE.format(t=0, g=0, p=100)
        + SELFPLAY_LINE.format(t=1, g=1, p=140)
        + SELFPLAY_LINE.format(t=0, g=2, p=60)
        + "selfplay-batch: 3 games done in 4s\n"
    )
    games, mean_plies = _count_completed_games(log)
    assert games == 3
    assert mean_plies == pytest.approx(100.0)


def test_unparseable_lines_do_not_break_the_counter(tmp_path: Path) -> None:
    """A panic backtrace lands in the same stream. Losing the progress display
    because of it would be a bad trade."""
    log = tmp_path / "selfplay.log"
    log.write_text(
        SELFPLAY_LINE.format(t=0, g=0, p=100)
        + "thread '<unnamed>' panicked at src/lib.rs:42:\n"
        + "  [t0] game garbage: ? plies, final=Draw\n"
        + SELFPLAY_LINE.format(t=0, g=1, p=200)
    )
    assert _count_completed_games(log) == (2, 150.0)


def test_missing_log_is_zero_not_an_error(tmp_path: Path) -> None:
    assert _count_completed_games(tmp_path / "nope.log") == (0, 0.0)


# --------------------------------------------------------------------------- #
# Match-result parsing — tolerant by construction                              #
# --------------------------------------------------------------------------- #


def test_match_result_ignores_unknown_keys_but_keeps_them() -> None:
    """`cls(**d)` on a fixed dataclass turns every field the Rust side adds
    into a TypeError raised *after* the match has already been played."""
    result = MatchResult.from_dict(
        {
            "player_a": "model:a.onnx@400",
            "player_b": "heuristic@800",
            "games": 10,
            "wins_a": 6,
            "wins_b": 2,
            "draws": 2,
            "winrate_a": 0.7,
            "score_a": 0.7,
            "elo_a": 147.2,
            "distinct_transcripts": 10,
            "a_field_invented_next_week": [1, 2, 3],
        }
    )
    assert result.wins_a == 6
    assert result.elo_a == pytest.approx(147.2)
    assert result["a_field_invented_next_week"] == [1, 2, 3]
    assert result.get("nope", "default") == "default"
    assert "a_field_invented_next_week" in result


def test_winrate_a_is_now_the_standard_score() -> None:
    """It changed meaning: `(wins + 0.5·draws) / games`. The old definition
    survives as `wins_only_rate_a`, and confusing the two is how a drawish
    match looks like a loss."""
    d = {"games": 10, "wins_a": 6, "wins_b": 2, "draws": 2}
    result = MatchResult.from_dict(
        {**d, "winrate_a": 0.7, "score_a": 0.7, "wins_only_rate_a": 0.6}
    )
    assert result.winrate_a == pytest.approx(0.7)
    assert result.score_a == result.winrate_a
    assert result.wins_only_rate_a == pytest.approx(0.6)


def test_missing_fields_become_nan_not_zero() -> None:
    result = MatchResult.from_dict({"games": 4})
    assert math.isnan(result.elo_a)
    assert math.isnan(result.score_a)


def test_suspicious_transcripts_flags_the_determinism_bug() -> None:
    """A 21-game gate that produced two distinct games is a 2-game experiment
    with a large denominator (review §3.2)."""
    assert MatchResult.from_dict(
        {"games": 21, "distinct_transcripts": 2}
    ).suspicious_transcripts
    assert not MatchResult.from_dict(
        {"games": 21, "distinct_transcripts": 21}
    ).suspicious_transcripts
    # Two games genuinely can only be two transcripts.
    assert not MatchResult.from_dict(
        {"games": 2, "distinct_transcripts": 2}
    ).suspicious_transcripts


def test_match_result_summary_survives_a_sparse_payload() -> None:
    """The summary line is logged inside an exception-free path; it must not be
    the thing that takes the run down."""
    assert MatchResult.from_dict({"games": 1}).summary()


# --------------------------------------------------------------------------- #
# Player specs                                                                 #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "spec",
    [
        "random",
        "heuristic",
        "heuristic@100",
        "heuristic@800",
        "model:checkpoints/gen_001.onnx",
        "model:checkpoints/gen_001.onnx@400",
    ],
)
def test_valid_player_specs_pass_through_unchanged(spec: str) -> None:
    """`@N` is the whole point of the ladder: `heuristic@100` and
    `heuristic@800` are different opponents."""
    assert _format_player(spec) == spec


def test_a_path_becomes_a_model_spec() -> None:
    assert _format_player(Path("runs/x/checkpoints/gen_003.onnx")) == (
        "model:runs/x/checkpoints/gen_003.onnx"
    )


@pytest.mark.parametrize("spec", ["heuristic@", "hueristic", "model:", "", "model@100"])
def test_malformed_player_specs_are_rejected(spec: str) -> None:
    with pytest.raises(ValueError):
        _format_player(spec)


def test_model_spec_builds_the_sim_override() -> None:
    assert model_spec(Path("a/b.onnx")) == "model:a/b.onnx"
    assert model_spec(Path("a/b.onnx"), 400) == "model:a/b.onnx@400"


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("random", "random"),
        ("heuristic@800", "heuristic_800"),
        ("model:runs/x/checkpoints/gen_003.onnx", "gen_003"),
        ("model:runs/x/checkpoints/gen_003.onnx@400", "gen_003_400"),
    ],
)
def test_slug_is_filesystem_safe_and_distinguishes_rungs(spec: str, expected: str) -> None:
    """Two rungs collapsing onto one filename would overwrite each other's JSON."""
    assert _slug(spec) == expected


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("random", "random"),
        ("heuristic@800", "heuristic@800"),
        ("model:/abs/runs/x/checkpoints/gen_003.onnx", "model:gen_003.onnx"),
        ("model:/abs/runs/x/checkpoints/gen_003.onnx@400", "model:gen_003.onnx@400"),
    ],
)
def test_opponent_label_shortens_absolute_paths(spec: str, expected: str) -> None:
    """Frozen rungs are passed to `eval-match` as absolute paths; an absolute
    path in a log line buries the numbers it is there to show."""
    assert opponent_label(spec) == expected


def test_only_trailing_rungs_are_a_moving_reference() -> None:
    """What `mean_elo` selects on. A floor anchor and a frozen checkpoint are
    the same strength at generation 5 and at generation 45; `gen - k` is not."""
    assert LadderOpponent("random", 16, KIND_FLOOR).is_fixed_reference
    assert LadderOpponent("model:a/gen_001.onnx", 40, KIND_FROZEN).is_fixed_reference
    assert not LadderOpponent("model:a/gen_015.onnx", 40, KIND_TRAILING).is_fixed_reference
    assert LadderOpponent("model:/abs/x/gen_003.onnx", 40).label == "model:gen_003.onnx"


# --------------------------------------------------------------------------- #
# Elo aggregation                                                              #
# --------------------------------------------------------------------------- #


def _rung(opponent: str, elo: float, kind: str = KIND_FLOOR, **extra) -> LadderRung:
    return LadderRung(
        opponent=opponent,
        result=MatchResult.from_dict({"elo_a": elo, **extra}),
        kind=kind,
    )


def test_mean_elo_averages_the_fixed_reference_rungs() -> None:
    """Floor anchors and absolute frozen checkpoints: everything whose strength
    is the same number at generation 5 and generation 45."""
    rungs = [
        _rung("random", 400.0),
        _rung("heuristic@100", 100.0),
        _rung("model:checkpoints/gen_001.onnx", -100.0, KIND_FROZEN),
    ]
    assert mean_elo(rungs) == pytest.approx(400 / 3)


def test_mean_elo_excludes_trailing_rungs() -> None:
    """`gen - k` gets stronger every time the network does. Averaging it in
    would flatten, by construction, the curve the ladder exists to draw."""
    fixed = [_rung("random", 400.0), _rung("heuristic@100", 200.0)]
    with_trailing = [*fixed, _rung("model:checkpoints/gen_015.onnx", 0.0, KIND_TRAILING)]
    assert mean_elo(with_trailing) == pytest.approx(mean_elo(fixed))


def test_mean_elo_is_nan_with_nothing_to_average() -> None:
    assert math.isnan(mean_elo([]))
    assert math.isnan(mean_elo([_rung("model:a.onnx", 10.0, KIND_TRAILING)]))


def test_mean_elo_skips_unmeasured_rungs() -> None:
    rungs = [_rung("random", 300.0), _rung("heuristic@100", float("nan"))]
    assert mean_elo(rungs) == pytest.approx(300.0)


# --------------------------------------------------------------------------- #
# Clamping — the failure that made "+545" look like a measurement              #
# --------------------------------------------------------------------------- #


def test_a_swept_ladder_reports_that_it_measured_nothing() -> None:
    """Generation 6 of the validation run, exactly: four rungs, four 12-0-0
    sweeps, four identical sample-size bounds, and a mean reported as if it
    were a strength."""
    rungs = [
        _rung("random", 544.7, KIND_FLOOR, elo_a_clamped=True),
        _rung("heuristic@100", 544.7, KIND_FLOOR, elo_a_clamped=True),
        _rung("model:gen_001.onnx", 544.7, KIND_FROZEN, elo_a_clamped=True),
        _rung("model:gen_004.onnx", 544.7, KIND_TRAILING, elo_a_clamped=True),
    ]
    assert clamped_fraction(rungs) == pytest.approx(1.0)
    summary = ladder_summary(rungs)
    assert summary["ladder/all_clamped"] == 1.0
    assert summary["ladder/clamped_fraction"] == pytest.approx(1.0)


def test_a_ladder_with_resolution_is_not_flagged() -> None:
    rungs = [
        _rung("random", 544.7, KIND_FLOOR, elo_a_clamped=True),
        _rung("model:gen_015.onnx", 120.0, KIND_TRAILING, elo_a_clamped=False),
    ]
    assert clamped_fraction(rungs) == pytest.approx(0.5)
    assert ladder_summary(rungs)["ladder/all_clamped"] == 0.0


def test_clamped_fraction_is_nan_with_nothing_measured() -> None:
    """Not 0.0: "no rungs were clamped" and "no rungs were played" are
    different facts and only one of them is reassuring."""
    assert math.isnan(clamped_fraction([]))


def test_ladder_summary_emits_every_rung_not_just_the_mean() -> None:
    rungs = [
        _rung("random", 400.0, KIND_FLOOR, elo_a_ci95_lo=300.0, elo_a_ci95_hi=500.0),
        _rung("model:gen_010.onnx", 120.0, KIND_TRAILING),
    ]
    summary = ladder_summary(rungs)
    for key in (
        "ladder/elo/random",
        "ladder/elo_ci95_lo/random",
        "ladder/elo_ci95_hi/random",
        "ladder/score/random",
        "ladder/clamped/random",
        "ladder/elo/gen_010",
        "ladder/rungs",
        "ladder/clamped_fraction",
        "ladder/elo_mean",
    ):
        assert key in summary, key
    assert summary["ladder/elo_ci95_lo/random"] == pytest.approx(300.0)


def test_the_mean_elo_carries_its_own_interval() -> None:
    """An Elo without an interval is an opinion. Independent matches, so the
    standard errors add in quadrature."""
    rungs = [
        _rung("random", 400.0, KIND_FLOOR, elo_a_stderr=30.0),
        _rung("heuristic@100", 200.0, KIND_FLOOR, elo_a_stderr=40.0),
    ]
    summary = ladder_summary(rungs)
    se = math.sqrt(30.0**2 + 40.0**2) / 2
    assert summary["ladder/elo_mean_stderr"] == pytest.approx(se)
    assert summary["ladder/elo_mean_ci95_lo"] == pytest.approx(300.0 - 1.96 * se)
    assert summary["ladder/elo_mean_ci95_hi"] == pytest.approx(300.0 + 1.96 * se)


# --------------------------------------------------------------------------- #
# Step budget in epochs — the "20 raw epochs in one generation" bug            #
# --------------------------------------------------------------------------- #


def test_the_step_budget_tracks_the_buffer() -> None:
    """The validation run's generation 3: a 1500-step cap over a 19k-position
    buffer at batch 256 is 20.2 raw epochs, and nobody chose 20 — games had
    shortened from 148 plies to 67 and the buffer shrank underneath a
    constant."""
    assert epochs_over_buffer(1500, 256, 19_000) == pytest.approx(20.2, abs=0.05)
    # Expressed in epochs, the same buffer asks for ~110 steps.
    assert epoch_budget_steps(19_000, 256, 1.5, 100, 1500) == 111


def test_the_budget_is_clamped_at_both_ends() -> None:
    assert epoch_budget_steps(1_000_000, 256, 1.5, 400, 4000) == 4000
    assert epoch_budget_steps(1_000, 256, 1.5, 400, 4000) == 400


def test_no_epoch_target_is_the_old_fixed_cap() -> None:
    assert epoch_budget_steps(19_000, 256, None, 400, 1500) == 1500


def test_an_empty_buffer_asks_for_the_minimum_not_a_division_by_zero() -> None:
    assert epoch_budget_steps(0, 256, 1.5, 400, 4000) == 400
    assert math.isnan(epochs_over_buffer(400, 256, 0))


@pytest.mark.parametrize(
    ("epochs", "expected"), [(20.2, True), (8.01, True), (8.0, False), (1.5, False)]
)
def test_epochs_alarm(epochs: float, expected: bool) -> None:
    assert epochs_alarm(epochs, 8.0) is expected


@pytest.mark.parametrize("value", [None, float("nan")])
def test_an_unmeasured_epoch_count_is_not_an_alarm(value) -> None:
    assert not epochs_alarm(value, 8.0)


# --------------------------------------------------------------------------- #
# Overfitting — the signal the frozen holdout could not give cleanly           #
# --------------------------------------------------------------------------- #


def test_rolling_loss_pulling_away_from_train_loss_is_an_alarm() -> None:
    """Both are the same weighted total over the same distribution; one is on
    data the network trained on and one is not. The gap is memorisation."""
    assert overfit_alarm(4.9, 4.4, 0.35)


def test_a_small_generalisation_gap_is_normal() -> None:
    assert not overfit_alarm(4.5, 4.4, 0.35)


def test_a_rolling_loss_below_the_train_loss_is_never_an_alarm() -> None:
    """It happens: the training mean is over a whole generation of steps, the
    holdout is scored once at the end with better weights."""
    assert not overfit_alarm(4.0, 4.4, 0.35)


@pytest.mark.parametrize(
    ("rolling", "train"),
    [(None, 4.4), (4.9, None), (float("nan"), 4.4), (4.9, float("nan"))],
)
def test_an_unmeasured_holdout_is_not_an_alarm(rolling, train) -> None:
    assert not overfit_alarm(rolling, train, 0.35)


# --------------------------------------------------------------------------- #
# Generic metric namespacing                                                   #
# --------------------------------------------------------------------------- #


def test_namespaced_prefixes_bare_keys() -> None:
    assert namespaced("train", {"loss_total": 1.0}) == {"train/loss_total": 1.0}


def test_namespaced_passes_through_keys_that_already_carry_one() -> None:
    """The measurement layer emits its own prefixes. A consumer that blindly
    re-prefixed would produce `val_frozen/val_frozen/loss_total` the day that
    landed."""
    assert namespaced("train", {"buffer/size": 3}) == {"buffer/size": 3}


def test_namespacing_does_not_enumerate_metric_names() -> None:
    """The property that matters: a key nobody has written code for still gets
    logged. A whitelist is how a new diagnostic ends up existing only in the
    source."""
    out = namespaced("selfplay", {"a_metric_added_next_week": 7.0})
    assert out == {"selfplay/a_metric_added_next_week": 7.0}


# --------------------------------------------------------------------------- #
# Formatting                                                                   #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("secs", "expected"), [(0.4, "0s"), (45.0, "45s"), (90.0, "1.5m"), (7200.0, "2.00h")]
)
def test_fmt_secs(secs: float, expected: str) -> None:
    assert _fmt_secs(secs) == expected


def test_entropy_alarm_never_reads_the_frozen_holdout():
    """The frozen holdout's `data_*` metrics describe generation 1's dataset —
    made by a random network — and are constant for the life of the run.

    A fallback to them fired a "search is producing no information" alarm at
    generation 2 of a run whose current data had just improved from 0.960 to
    0.804. Alarming on a constant is a false positive every generation forever,
    which is how people learn to ignore warnings.
    """
    import inspect

    from model import train_loop

    src = inspect.getsource(train_loop)
    # Locate the holdout alarm and assert the frozen prefix is not one of its inputs.
    marker = "search is producing (near) no information"
    assert marker in src, "the holdout entropy alarm has moved; update this test"
    start = src.index("ratio = out.get(")
    end = src.index(marker)
    selector = src[start:end]
    assert "VAL_ROLLING_PREFIX" in selector
    assert "VAL_FROZEN_PREFIX" not in selector, (
        "the entropy alarm must not read the frozen holdout — it is constant by "
        "construction and produces a false alarm every generation"
    )
