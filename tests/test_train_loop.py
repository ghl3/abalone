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

from model.eval import MatchResult, _format_player, _slug, mean_elo, model_spec
from model.state import GenRecord, RunState
from model.train_loop import (
    LadderRung,
    _count_completed_games,
    _fmt_secs,
    entropy_ratio_alarm,
    git_sha_drift,
    ladder_opponents,
    opponent_label,
    plan_resume,
    retention_victims,
    should_run_ladder,
    should_validate,
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


def test_ladder_opponents_keeps_fixed_anchors(tmp_path: Path) -> None:
    fixed = ["random", "heuristic@100", "heuristic@800"]
    assert ladder_opponents(fixed, [], gen=5, ckpt_dir=tmp_path) == fixed


def test_ladder_opponents_adds_existing_frozen_checkpoints(tmp_path: Path) -> None:
    (tmp_path / "gen_001.onnx").write_bytes(b"")
    (tmp_path / "gen_010.onnx").write_bytes(b"")
    out = ladder_opponents(["random"], [1, 10], gen=20, ckpt_dir=tmp_path)
    assert out == ["random", model_spec(tmp_path / "gen_001.onnx"), model_spec(tmp_path / "gen_010.onnx")]


def test_ladder_opponents_skips_a_collected_checkpoint(tmp_path: Path) -> None:
    """Retention may have taken it. Dropping the rung beats crashing a ladder
    five hours into a run."""
    assert ladder_opponents(["random"], [1], gen=20, ckpt_dir=tmp_path) == ["random"]


def test_a_generation_is_never_its_own_ladder_opponent(tmp_path: Path) -> None:
    (tmp_path / "gen_005.onnx").write_bytes(b"")
    assert ladder_opponents([], [5], gen=5, ckpt_dir=tmp_path) == []
    assert ladder_opponents([], [5], gen=6, ckpt_dir=tmp_path) == [
        model_spec(tmp_path / "gen_005.onnx")
    ]


def test_ladder_opponents_deduplicates_and_orders(tmp_path: Path) -> None:
    for g in (1, 3):
        (tmp_path / f"gen_{g:03d}.onnx").write_bytes(b"")
    out = ladder_opponents([], [3, 1, 3], gen=9, ckpt_dir=tmp_path)
    assert out == [model_spec(tmp_path / "gen_001.onnx"), model_spec(tmp_path / "gen_003.onnx")]


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


# --------------------------------------------------------------------------- #
# Elo aggregation                                                              #
# --------------------------------------------------------------------------- #


def _rung(opponent: str, elo: float) -> LadderRung:
    return LadderRung(opponent=opponent, result=MatchResult.from_dict({"elo_a": elo}))


def test_mean_elo_averages_the_fixed_anchors() -> None:
    rungs = [_rung("random", 400.0), _rung("heuristic@100", 100.0), _rung("heuristic@800", -100.0)]
    assert mean_elo(rungs) == pytest.approx(400 / 3)


def test_mean_elo_excludes_frozen_checkpoints() -> None:
    """A frozen earlier checkpoint is a moving reference across generations;
    averaging against it would make the tracked best-checkpoint number
    meaningless."""
    fixed = [_rung("random", 400.0), _rung("heuristic@100", 200.0)]
    with_frozen = [*fixed, _rung("model:checkpoints/gen_001.onnx", 0.0)]
    assert mean_elo(with_frozen) == pytest.approx(mean_elo(fixed))


def test_mean_elo_is_nan_with_nothing_to_average() -> None:
    assert math.isnan(mean_elo([]))
    assert math.isnan(mean_elo([_rung("model:a.onnx", 10.0)]))


def test_mean_elo_skips_unmeasured_rungs() -> None:
    rungs = [_rung("random", 300.0), _rung("heuristic@100", float("nan"))]
    assert mean_elo(rungs) == pytest.approx(300.0)


# --------------------------------------------------------------------------- #
# Formatting                                                                   #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("secs", "expected"), [(0.4, "0s"), (45.0, "45s"), (90.0, "1.5m"), (7200.0, "2.00h")]
)
def test_fmt_secs(secs: float, expected: str) -> None:
    assert _fmt_secs(secs) == expected
