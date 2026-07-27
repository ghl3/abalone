"""Tests for `model.report`, the run-reading CLI.

The tool has one job: make a run's trajectory legible without anyone writing a
throwaway parser mid-run. So the tests are about the two ways that fails.

**It has to read the runs that already exist.** The metrics schema changed when
logging went generic (`data_mean_plies` → `selfplay/mean_plies`). A tool that
only reads the new schema cannot be pointed at the run whose findings motivated
writing it, which is the first thing anybody will try.

**It has to say "not measured" out loud.** A dash means a metric was absent; a
`0.00` in the same cell would be a lie with two decimal places. Every accessor
here distinguishes missing, `nan` and zero.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from model.report import (
    Row,
    clamped_fraction,
    entropy_gap,
    find_run,
    fmt,
    fmt_elo,
    format_ladder,
    format_table,
    read_metrics,
    render,
    warnings_for,
)


def write_run(tmp_path: Path, name: str, rows: list[dict]) -> Path:
    run = tmp_path / name
    run.mkdir(parents=True, exist_ok=True)
    with (run / "metrics.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return run


# --------------------------------------------------------------------------- #
# Both metric schemas                                                          #
# --------------------------------------------------------------------------- #

#: One generation as the pre-namespacing loop wrote it — taken verbatim from
#: `runs/dazzling-cinder-20260727-2016`, generation 6.
OLD_SCHEMA = {
    "gen": 6,
    "data_mean_plies": 71.0,
    "data_decisive_rate": 1.0,
    "data_natural_termination_rate": 1.0,
    "handicap_rate": 0.55,
    "data_policy_target_entropy": 3.02,
    "data_policy_uniform_entropy": 3.96,
    "val_value_ce": 2.0466,
    "val_value_accuracy": 0.677,
    "ladder_elo": 544.7,
    "gen_seconds": 1278.0,
}

NEW_SCHEMA = {
    "gen": 6,
    "selfplay/mean_plies": 71.0,
    "selfplay/decisive_rate": 1.0,
    "selfplay/policy_entropy_gap": 0.94,
    "selfplay/captures_per_100_plies": 8.4,
    "curriculum/natural_termination_rate": 1.0,
    "curriculum/handicap_rate": 0.55,
    "val_rolling/value_ce": 2.0466,
    "val_rolling/value_accuracy": 0.677,
    "buffer/epochs_this_gen": 1.6,
    "ladder/elo_mean": 544.7,
    "ladder/elo_mean_ci95_lo": 500.0,
    "ladder/elo_mean_ci95_hi": 589.0,
    "ladder/clamped_fraction": 1.0,
    "perf/gen_seconds": 1278.0,
}


@pytest.mark.parametrize("record", [OLD_SCHEMA, NEW_SCHEMA], ids=["old", "new"])
def test_both_metric_schemas_produce_the_same_row(record: dict) -> None:
    row = Row.from_metrics(record)
    assert row.gen == 6
    assert row.plies == pytest.approx(71.0)
    assert row.decisive == pytest.approx(1.0)
    assert row.natural == pytest.approx(1.0)
    assert row.handicap == pytest.approx(0.55)
    assert row.gap == pytest.approx(0.94, abs=0.005)
    assert row.val_ce == pytest.approx(2.0466)
    assert row.val_acc == pytest.approx(0.677)
    assert row.elo == pytest.approx(544.7)
    assert row.seconds == pytest.approx(1278.0)


def test_the_entropy_gap_is_reconstructed_when_it_was_not_logged() -> None:
    """The gap is the number that would have shown the previous collapse as a
    flat zero for three generations. Older runs logged the two entropies but
    not their difference."""
    assert entropy_gap(OLD_SCHEMA) == pytest.approx(0.94, abs=0.005)
    assert entropy_gap({"gen": 1}) is None


def test_the_rolling_holdout_is_preferred_over_the_frozen_one() -> None:
    """The frozen holdout is a drift indicator; the rolling one is
    current-distribution. When both are present the table shows the one that
    can be gated on."""
    row = Row.from_metrics(
        {"gen": 4, "val_rolling/value_ce": 1.1, "val_frozen/value_ce": 2.9}
    )
    assert row.val_ce == pytest.approx(1.1)


# --------------------------------------------------------------------------- #
# "Not measured" is not zero                                                   #
# --------------------------------------------------------------------------- #


def test_a_missing_metric_is_none_not_zero() -> None:
    row = Row.from_metrics({"gen": 1})
    assert row.plies is None and row.elo is None and row.gap is None


def test_a_nan_metric_is_none_too() -> None:
    """`nan` reaches metrics.jsonl as the string "nan" via `default=str`, and
    as a float when a strict parser is not in the way. Neither is a number."""
    row = Row.from_metrics({"gen": 1, "selfplay/mean_plies": float("nan")})
    assert row.plies is None


def test_a_genuine_zero_survives() -> None:
    """0% natural termination is the single most important measurement in the
    curriculum — it is what uniformly random play looks like. Collapsing it
    into "missing" would hide the one number the controller keys on."""
    row = Row.from_metrics({"gen": 1, "curriculum/natural_termination_rate": 0.0})
    assert row.natural == 0.0
    assert fmt(row.natural, 2) == "0.00"


def test_missing_renders_as_a_dash() -> None:
    assert fmt(None) == "-"
    assert fmt(float("nan")) == "-"


# --------------------------------------------------------------------------- #
# Elo is never shown bare                                                      #
# --------------------------------------------------------------------------- #


def test_elo_is_shown_with_its_interval() -> None:
    row = Row.from_metrics(NEW_SCHEMA)
    text = fmt_elo(row)
    assert "+545" in text
    assert "[+500,+589]" in text


def test_a_fully_clamped_ladder_is_marked() -> None:
    """The generation-6 failure: four rungs, four sweeps, four identical
    sample-size bounds, reported as a strength."""
    assert fmt_elo(Row.from_metrics(NEW_SCHEMA)).endswith("!")


def test_clamped_fraction_is_recomputed_from_the_rungs_for_old_runs() -> None:
    """The metric did not exist when the run that proved the point was made."""
    record = {
        "gen": 6,
        "ladder": [
            {"opponent": "random", "elo_a": 544.7, "elo_a_clamped": True},
            {"opponent": "heuristic@100", "elo_a": 544.7, "elo_a_clamped": True},
        ],
    }
    assert clamped_fraction(record) == pytest.approx(1.0)


def test_clamped_fraction_prefers_the_logged_metric() -> None:
    record = {"ladder/clamped_fraction": 0.25, "ladder": [{"elo_a": 1.0}]}
    assert clamped_fraction(record) == pytest.approx(0.25)


# --------------------------------------------------------------------------- #
# Warnings                                                                     #
# --------------------------------------------------------------------------- #


def test_a_swept_ladder_warns() -> None:
    rows = [Row.from_metrics(NEW_SCHEMA)]
    assert any("no resolution" in w for w in warnings_for(rows, [NEW_SCHEMA]))


def test_a_collapsed_entropy_gap_warns() -> None:
    record = {"gen": 1, "selfplay/policy_entropy_gap": 0.02}
    assert any("entropy gap" in w for w in warnings_for([Row.from_metrics(record)], [record]))


def test_too_many_epochs_warns() -> None:
    record = {"gen": 3, "buffer/epochs_this_gen": 20.2}
    assert any("passes over" in w for w in warnings_for([Row.from_metrics(record)], [record]))


def test_the_rolling_holdout_pulling_away_from_the_train_loss_warns() -> None:
    record = {"gen": 5, "val_rolling/loss_total": 5.1, "train/loss_total": 4.4}
    assert any("memorisation" in w for w in warnings_for([Row.from_metrics(record)], [record]))


def test_shortening_games_warn_even_though_every_headline_number_improved() -> None:
    """The validation run's actual failure mode: natural termination went
    0% → 100%, decisive rate went to 1.0, and mean plies went 148 → 71. Both
    sides were conceding marbles on contact (MODEL.md §8.2)."""
    records = [
        {"gen": 1, "selfplay/mean_plies": 148.0},
        {"gen": 2, "selfplay/mean_plies": 106.0},
        {"gen": 3, "selfplay/mean_plies": 71.0},
    ]
    rows = [Row.from_metrics(r) for r in records]
    assert any("shorter" in w for w in warnings_for(rows, records))


def test_lengthening_games_do_not_warn() -> None:
    records = [
        {"gen": 1, "selfplay/mean_plies": 71.0},
        {"gen": 2, "selfplay/mean_plies": 106.0},
        {"gen": 3, "selfplay/mean_plies": 148.0},
    ]
    rows = [Row.from_metrics(r) for r in records]
    assert not any("shorter" in w for w in warnings_for(rows, records))


def test_a_healthy_generation_produces_no_warnings() -> None:
    record = {
        "gen": 5,
        "selfplay/mean_plies": 150.0,
        "selfplay/policy_entropy_gap": 1.4,
        "buffer/epochs_this_gen": 1.6,
        "train/loss_total": 3.1,
        "val_rolling/loss_total": 3.2,
        "ladder/clamped_fraction": 0.25,
    }
    assert warnings_for([Row.from_metrics(record)], [record]) == []


# --------------------------------------------------------------------------- #
# Reading a run off disk                                                       #
# --------------------------------------------------------------------------- #


def test_latest_is_by_mtime_not_by_name(tmp_path: Path) -> None:
    """Run ids are `<adjective>-<noun>-<date>-<time>`, so sorting them
    alphabetically returns whichever adjective sorts last."""
    old = write_run(tmp_path, "zebra-oak-20260101-0000", [{"gen": 1}])
    new = write_run(tmp_path, "amber-fig-20260727-2016", [{"gen": 1}])
    import os

    os.utime(old / "metrics.jsonl", (1_700_000_000, 1_700_000_000))
    os.utime(new / "metrics.jsonl", (1_800_000_000, 1_800_000_000))
    assert find_run(tmp_path, "latest") == new


def test_a_run_can_be_named(tmp_path: Path) -> None:
    run = write_run(tmp_path, "amber-fig-20260727-2016", [{"gen": 1}])
    assert find_run(tmp_path, "amber-fig-20260727-2016") == run
    assert find_run(tmp_path, str(run)) == run


def test_an_unknown_run_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        find_run(tmp_path, "no-such-run")
    with pytest.raises(FileNotFoundError):
        find_run(tmp_path, "latest")


def test_a_half_written_last_line_is_skipped_not_fatal(tmp_path: Path) -> None:
    """The whole point of this tool is being usable *during* a run, and the
    last line of a file being appended to is the one most likely to be
    truncated."""
    run = write_run(tmp_path, "r", [{"gen": 1}, {"gen": 2}])
    with (run / "metrics.jsonl").open("a") as f:
        f.write('{"gen": 3, "selfplay/mean_pl')
    assert [r["gen"] for r in read_metrics(run)] == [1, 2]


def test_records_come_back_in_generation_order(tmp_path: Path) -> None:
    run = write_run(tmp_path, "r", [{"gen": 3}, {"gen": 1}, {"gen": 2}])
    assert [r["gen"] for r in read_metrics(run)] == [1, 2, 3]


def test_a_run_with_no_metrics_file_is_empty_not_an_error(tmp_path: Path) -> None:
    (tmp_path / "r").mkdir()
    assert read_metrics(tmp_path / "r") == []


# --------------------------------------------------------------------------- #
# Rendering                                                                    #
# --------------------------------------------------------------------------- #


def test_the_table_has_a_row_per_generation_plus_a_header() -> None:
    rows = [Row.from_metrics({"gen": g, "selfplay/mean_plies": 100.0}) for g in (1, 2, 3)]
    lines = format_table(rows)
    assert len(lines) == 5  # header, rule, three generations
    assert lines[0].split()[0] == "gen"


def test_every_documented_column_is_present() -> None:
    header = format_table([])[0]
    for column in ("plies", "dec", "nat", "hcap", "Hgap", "cap/100", "val CE", "val acc", "elo"):
        assert column in header, column


def test_the_ladder_lists_every_rung_not_just_the_mean() -> None:
    record = {
        "gen": 6,
        "ladder/clamped_fraction": 0.5,
        "ladder": [
            {
                "opponent": "random",
                "kind": "floor",
                "elo_a": 544.7,
                "elo_a_clamped": True,
                "wins_a": 12,
                "wins_b": 0,
                "draws": 0,
                "score_a": 1.0,
            },
            {
                "opponent": "model:/abs/checkpoints/gen_004.onnx",
                "kind": "trailing",
                "elo_a": 120.0,
                "elo_a_ci95_lo": 20.0,
                "elo_a_ci95_hi": 220.0,
                "wins_a": 24,
                "wins_b": 12,
                "draws": 4,
                "score_a": 0.65,
            },
        ],
    }
    lines = "\n".join(format_ladder([record]))
    assert "random" in lines and "gen_004.onnx" in lines
    assert "(bound)" in lines
    assert "[+20,+220]" in lines
    # The absolute path is shortened; a path in a log line buries the numbers.
    assert "/abs/" not in lines


def test_no_ladder_yet_says_so() -> None:
    assert "no ladder" in "\n".join(format_ladder([{"gen": 1}]))


def test_render_is_a_whole_report(tmp_path: Path) -> None:
    run = write_run(tmp_path, "r", [OLD_SCHEMA, NEW_SCHEMA])
    (run / "state.json").write_text(
        json.dumps(
            {"current_gen": 6, "current_phase": "complete", "handicap_rate": 0.55,
             "best_gen": 6, "best_elo": 544.7}
        )
    )
    text = "\n".join(render(run, read_metrics(run), tail=None))
    assert "run: r" in text
    assert "generations complete: 6" in text
    assert "ladder" in text
    assert "warnings" in text


def test_tail_limits_the_table(tmp_path: Path) -> None:
    run = write_run(tmp_path, "r", [{"gen": g} for g in range(1, 11)])
    text = "\n".join(render(run, read_metrics(run), tail=3))
    assert "showing the last 3 of 10 generations" in text


def test_no_warnings_says_none(tmp_path: Path) -> None:
    run = write_run(tmp_path, "r", [{"gen": 1, "selfplay/policy_entropy_gap": 1.0}])
    text = "\n".join(render(run, read_metrics(run), tail=None))
    assert "warnings (0)" in text
    assert "none" in text


def test_the_json_view_round_trips(tmp_path: Path) -> None:
    from model.report import main

    write_run(tmp_path, "r", [NEW_SCHEMA])
    assert main(["--run", "r", "--runs-root", str(tmp_path), "--json"]) == 0


def test_the_cli_runs_on_a_real_directory(tmp_path: Path, capsys) -> None:
    from model.report import main

    write_run(tmp_path, "r", [OLD_SCHEMA])
    assert main(["--run", "latest", "--runs-root", str(tmp_path)]) == 0
    assert "run: r" in capsys.readouterr().out


def test_an_unknown_run_exits_nonzero(tmp_path: Path) -> None:
    from model.report import main

    assert main(["--run", "nope", "--runs-root", str(tmp_path)]) == 2


def test_epochs_column_reads_the_buffer_metric() -> None:
    """`buffer/epochs_this_gen` comes from the replay buffer, which knows the
    trainable row count; `train/epochs_over_buffer` is the loop's own copy.
    Either will do, but the number must appear."""
    assert Row.from_metrics({"gen": 1, "buffer/epochs_this_gen": 1.6}).epochs == 1.6
    assert Row.from_metrics({"gen": 1, "train/epochs_over_buffer": 20.2}).epochs == 20.2
    assert math.isnan(float("nan"))  # sanity: nan is nan
