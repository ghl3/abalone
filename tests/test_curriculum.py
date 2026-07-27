"""Tests for the capture-handicap curriculum ratchet (MODEL.md §4).

The controller is a pure function of `(current_rate, stats, config, gen)`, which
is the whole reason it is testable at all: everything that decides whether the
curriculum moves is here, and nothing that decides it needs a subprocess, a
shard or a network.

The test that matters most is
`test_uniformly_random_play_does_not_move_the_rate`. The obvious control signal
— "the decisive rate of unseeded games" — is wrong, and wrong in a way that
looks right until a run has been wasted. Read the comment there before changing
the signal.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from model.config import HandicapAnnealConfig, RunConfig
from model.curriculum import (
    REASON_AT_FLOOR,
    REASON_BELOW_TARGET,
    REASON_INSUFFICIENT_SAMPLE,
    REASON_NOT_MEASURABLE,
    REASON_OFF,
    REASON_SCHEDULE,
    REASON_STEPPED,
    AnnealStats,
    decide_handicap_rate,
    resolve_initial_rate,
    scheduled_rate,
    step_multiple,
)
from model.state import GenRecord, RunState


def controller(**overrides) -> HandicapAnnealConfig:
    cfg = HandicapAnnealConfig(
        mode="controller",
        target_natural_termination=0.25,
        step=0.05,
        floor=0.10,
        min_unseeded_games=20,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def stats(rate: float | None, games: int = 50) -> AnnealStats:
    return AnnealStats(unseeded_games=games, natural_termination_rate=rate)


# --------------------------------------------------------------------------- #
# The control signal — the part that is easy to get wrong                      #
# --------------------------------------------------------------------------- #


def test_uniformly_random_play_does_not_move_the_rate() -> None:
    """The premature-annealing trap, pinned.

    These are the measured statistics of Belgian Daisy under *uniformly random*
    play with adjudication and no handicap, over 200 playouts: 63% of games are
    decisive (mean |margin| 0.89 — adjudication at the ply cap resolves almost
    anything on one lucky push) and **0%** terminate naturally (mean plies came
    out at exactly the cap: 400.0 / 200.0 / 200.0, so not one game in 200 ended
    on its own).

    A controller keyed on decisive rate fires here — against a network that has
    learned nothing — and pulls the curriculum out from under generation one.
    A controller keyed on natural termination does not move. If somebody
    "simplifies" the signal back to decisive rate, this test is what fails.
    """
    random_play = AnnealStats(unseeded_games=200, natural_termination_rate=0.0)
    # 63% decisive would clear any sane threshold; it is not an input at all.
    d = decide_handicap_rate(0.7, random_play, controller(), gen=1)
    assert d.rate == 0.7
    assert not d.stepped
    assert d.reason == REASON_BELOW_TARGET


def test_a_network_that_finishes_games_moves_the_rate() -> None:
    """The other side of the same coin: natural termination is what rises when
    the network actually learns to close a game out.

    0.30 against a 0.25 target is a multiple of 1.2, so the step is
    0.05 x 1.2 = 0.06.
    """
    d = decide_handicap_rate(0.7, stats(0.30), controller(), gen=8)
    assert d.stepped
    assert d.rate == pytest.approx(0.64)
    assert d.reason == REASON_STEPPED


# --------------------------------------------------------------------------- #
# Firing conditions                                                            #
# --------------------------------------------------------------------------- #


def test_the_target_itself_fires() -> None:
    """`>=`, not `>`: a target you can sit exactly on and never trigger is a
    target that silently means something else. Exactly on target is a multiple
    of 1, i.e. the configured step and no more."""
    d = decide_handicap_rate(0.7, stats(0.25), controller(), gen=3)
    assert d.stepped and d.rate == pytest.approx(0.65)


def test_just_below_the_target_holds() -> None:
    d = decide_handicap_rate(0.7, stats(0.2499), controller(), gen=3)
    assert not d.stepped
    assert d.rate == 0.7
    assert d.reason == REASON_BELOW_TARGET


def test_a_small_sample_holds_even_far_above_the_target() -> None:
    """19 unseeded games with 100% natural termination is one plausible run of
    luck, not evidence. The sample gate is checked before the target."""
    d = decide_handicap_rate(0.7, stats(1.0, games=19), controller(), gen=3)
    assert not d.stepped
    assert d.rate == 0.7
    assert d.reason == REASON_INSUFFICIENT_SAMPLE
    assert "min_unseeded_games" in d.message


def test_the_sample_threshold_itself_is_enough() -> None:
    d = decide_handicap_rate(0.7, stats(0.5, games=20), controller(), gen=3)
    assert d.stepped


def test_no_unseeded_games_at_all_is_a_hold_not_a_crash() -> None:
    d = decide_handicap_rate(
        0.7, AnnealStats(unseeded_games=0, natural_termination_rate=math.nan),
        controller(min_unseeded_games=0), gen=3,
    )
    assert not d.stepped
    assert d.reason == REASON_INSUFFICIENT_SAMPLE


def test_the_no_progress_rule_makes_the_signal_unmeasurable() -> None:
    """With the no-progress rule on, a game can stop short of the cap without
    six captures, so `plies < max_plies` stops meaning "terminated naturally"
    and the shards cannot tell the two apart. The controller goes inert rather
    than acting on a number that means something else."""
    d = decide_handicap_rate(
        0.7, AnnealStats(unseeded_games=100, natural_termination_rate=None),
        controller(), gen=3,
    )
    assert not d.stepped
    assert d.reason == REASON_NOT_MEASURABLE
    assert "no-progress" in d.message


# --------------------------------------------------------------------------- #
# Monotonicity and the floor                                                   #
# --------------------------------------------------------------------------- #


def test_the_step_is_proportional_to_how_far_above_target_the_signal_is() -> None:
    """The fix for the timidity the validation run exposed.

    `natural_termination_rate` hit 1.00 at generation 4 — four times the 0.25
    target, the signal completely saturated — and a fixed 0.05 step still
    needed eight more generations to reach the floor. The step now scales with
    `signal / target`, so a 4x signal buys 4x the step and the crutch comes out
    in three generations instead of fourteen.
    """
    d = decide_handicap_rate(0.7, stats(1.0), controller(), gen=3)
    assert d.stepped
    assert d.rate == pytest.approx(0.50)  # 0.7 - 0.05 * 4


def test_the_multiplier_is_clamped_at_both_ends() -> None:
    """Below the target the controller holds, so the lower clamp never bites in
    practice; above `max_step_multiple` it does, and it is what keeps this a
    ratchet rather than a controller with gain. One lucky generation must not
    be able to dump the entire curriculum."""
    assert step_multiple(0.25, 0.25, 4.0) == pytest.approx(1.0)
    assert step_multiple(0.10, 0.25, 4.0) == pytest.approx(1.0)
    assert step_multiple(0.50, 0.25, 4.0) == pytest.approx(2.0)
    assert step_multiple(1.00, 0.25, 4.0) == pytest.approx(4.0)
    # A target so low that every signal saturates still cannot exceed K.
    assert step_multiple(1.00, 0.01, 4.0) == pytest.approx(4.0)


def test_a_saturated_signal_still_cannot_punch_through_the_floor() -> None:
    cfg = controller(max_step_multiple=4.0)
    rate = 0.7
    for gen in range(1, 20):
        d = decide_handicap_rate(rate, stats(1.0), cfg, gen=gen)
        assert d.rate >= cfg.floor
        rate = d.rate
    assert rate == pytest.approx(cfg.floor)


def test_max_step_multiple_of_one_is_the_old_fixed_step() -> None:
    """The escape hatch, pinned: whatever the signal, K=1 moves exactly
    `step`."""
    for signal in (0.25, 0.5, 1.0):
        d = decide_handicap_rate(0.7, stats(signal), controller(max_step_multiple=1.0), gen=3)
        assert d.rate == pytest.approx(0.65)


def test_the_rate_never_increases_when_the_signal_collapses() -> None:
    """Step once, then feed it a network that has regressed. Holding is the
    right response: it already knows the endgames, and re-teaching them costs
    the generations that were supposed to be learning openings."""
    rate = 0.7
    # 0.40 / 0.25 = 1.6, so the step is 0.05 * 1.6 = 0.08.
    rate = decide_handicap_rate(rate, stats(0.40), controller(), gen=3).rate
    assert rate == pytest.approx(0.62)
    stepped = 0
    for gen, signal in enumerate([0.0, 0.05, 0.9999, 0.0], start=4):
        after = decide_handicap_rate(rate, stats(signal), controller(), gen=gen)
        assert after.rate <= rate + 1e-12, "the curriculum must never go backwards"
        stepped += after.stepped
        rate = after.rate
    assert stepped == 1, "only the 0.9999 generation is at or above target"
    # 0.9999 / 0.25 = 3.9996, just under the clamp, so that generation moved
    # 0.05 * 3.9996 = 0.19998.
    assert rate == pytest.approx(0.42002)


def test_the_floor_is_reached_exactly_and_never_overshot() -> None:
    """0.7 down to 0.1 in steps of 0.05 lands on the floor; the point is that a
    step of 0.07 (which does not divide the gap) also lands exactly on it."""
    for step in (0.05, 0.07, 0.3):
        cfg = controller(step=step)
        rate = 0.7
        for gen in range(1, 60):
            rate = decide_handicap_rate(rate, stats(0.9), cfg, gen=gen).rate
            assert rate >= cfg.floor, f"step {step} overshot the floor: {rate}"
        assert rate == pytest.approx(cfg.floor)


def test_at_the_floor_it_holds_and_says_so() -> None:
    d = decide_handicap_rate(0.10, stats(0.9), controller(), gen=30)
    assert not d.stepped
    assert d.rate == pytest.approx(0.10)
    assert d.reason == REASON_AT_FLOOR
    assert "floor" in d.message


def test_repeated_stepping_does_not_accumulate_float_noise() -> None:
    """0.7 - 0.05 - 0.05 is 0.5999999999999999 in binary floating point, and
    that noise ends up in `state.json` and in every log line. A signal sitting
    exactly on the target gives a multiple of 1, so this is the plain
    three-times-0.05 arithmetic."""
    rate = 0.7
    for gen in range(1, 4):
        rate = decide_handicap_rate(rate, stats(0.25), controller(), gen=gen).rate
    assert rate == 0.55
    assert repr(rate) == "0.55"


def test_a_proportional_step_is_rounded_too() -> None:
    """0.7 - 0.05*1.6 is 0.6200000000000001 without the rounding."""
    d = decide_handicap_rate(0.7, stats(0.40), controller(), gen=3)
    assert repr(d.rate) == "0.62"


# --------------------------------------------------------------------------- #
# mode: off                                                                    #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("signal", [0.0, 0.25, 1.0])
def test_off_never_moves_whatever_the_signal(signal: float) -> None:
    d = decide_handicap_rate(0.7, stats(signal), controller(mode="off"), gen=40)
    assert d.rate == 0.7
    assert not d.stepped
    assert d.reason == REASON_OFF


# --------------------------------------------------------------------------- #
# mode: schedule                                                               #
# --------------------------------------------------------------------------- #


def test_schedule_picks_the_most_recent_entry_at_or_below_the_generation() -> None:
    sched = {5: 0.5, 10: 0.3, 20: 0.15}
    assert scheduled_rate(sched, 1) is None
    assert scheduled_rate(sched, 5) == pytest.approx(0.5)
    assert scheduled_rate(sched, 9) == pytest.approx(0.5)
    assert scheduled_rate(sched, 10) == pytest.approx(0.3)
    assert scheduled_rate(sched, 19) == pytest.approx(0.3)
    assert scheduled_rate(sched, 1000) == pytest.approx(0.15)


def test_schedule_is_order_independent() -> None:
    a = {20: 0.15, 5: 0.5, 10: 0.3}
    b = {5: 0.5, 10: 0.3, 20: 0.15}
    assert [scheduled_rate(a, g) for g in range(1, 30)] == [
        scheduled_rate(b, g) for g in range(1, 30)
    ]


def test_schedule_ignores_the_measurement_entirely() -> None:
    cfg = controller(mode="schedule", schedule={10: 0.3})
    # 0% natural termination, and it steps anyway: a schedule is an assertion
    # about the run, not about the data.
    d = decide_handicap_rate(0.7, stats(0.0), cfg, gen=10)
    assert d.stepped
    assert d.rate == pytest.approx(0.3)
    assert d.reason == REASON_SCHEDULE


def test_schedule_before_the_first_entry_holds() -> None:
    d = decide_handicap_rate(0.7, stats(0.9), controller(mode="schedule", schedule={10: 0.3}), gen=4)
    assert not d.stepped
    assert d.rate == 0.7


def test_schedule_is_monotone_and_respects_the_floor() -> None:
    """An entry above the current rate cannot raise it, and one below the floor
    cannot punch through it — both invariants are run-wide, not controller-only.
    A schedule that genuinely wants zero says `floor: 0.0`."""
    cfg = controller(mode="schedule", floor=0.1, schedule={5: 0.9, 10: 0.0})
    assert decide_handicap_rate(0.4, stats(0.0), cfg, gen=5).rate == pytest.approx(0.4)
    assert decide_handicap_rate(0.4, stats(0.0), cfg, gen=10).rate == pytest.approx(0.1)


def test_an_empty_schedule_is_a_no_op() -> None:
    d = decide_handicap_rate(0.7, stats(0.9), controller(mode="schedule"), gen=40)
    assert d.rate == 0.7 and not d.stepped


# --------------------------------------------------------------------------- #
# Reading the measurement                                                      #
# --------------------------------------------------------------------------- #


def test_stats_from_a_generation_summary() -> None:
    summary = {"unseeded_games": 42, "natural_termination_rate": 0.3}
    s = AnnealStats.from_summary(summary)
    assert (s.unseeded_games, s.natural_termination_rate) == (42, 0.3)


def test_stats_report_nothing_measurable_when_no_progress_is_on() -> None:
    summary = {"unseeded_games": 42, "natural_termination_rate": 0.3}
    s = AnnealStats.from_summary(summary, no_progress_plies=40)
    assert s.natural_termination_rate is None
    assert s.unseeded_games == 42


def test_stats_from_an_empty_or_missing_summary() -> None:
    for summary in (None, {}, {"games": 0}):
        s = AnnealStats.from_summary(summary)
        assert s.unseeded_games == 0
        assert math.isnan(s.natural_termination_rate)


# --------------------------------------------------------------------------- #
# State: the rate must survive a resume                                        #
# --------------------------------------------------------------------------- #


def test_the_annealed_rate_round_trips_through_state_json(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    state = RunState.fresh("run-x", "hash", "sha", handicap_rate=0.7)
    state.handicap_rate = decide_handicap_rate(
        state.handicap_rate, stats(0.25), controller(), gen=3
    ).rate
    state.current_gen = 3
    state.append_history(GenRecord(gen=3, handicap_rate=0.7, handicap_rate_next=0.65))
    state.save_atomic(path, fsync=False)

    reloaded = RunState.load(path)
    assert reloaded.handicap_rate == pytest.approx(0.65)
    assert reloaded.history[0].handicap_rate == pytest.approx(0.7)
    assert reloaded.history[0].handicap_rate_next == pytest.approx(0.65)
    # And the value on disk is the plain number, not a float32-ish artefact.
    assert json.loads(path.read_text())["handicap_rate"] == 0.65


def test_a_resume_continues_the_curriculum_instead_of_restarting_it() -> None:
    """The whole point of persisting it: a run resumed at generation 30 must not
    quietly go back to the generation-one seeding rate."""
    assert resolve_initial_rate(0.35, cfg_rate=0.7) == pytest.approx(0.35)


def test_a_state_file_that_predates_the_annealer_falls_back_to_config() -> None:
    assert resolve_initial_rate(None, cfg_rate=0.7) == pytest.approx(0.7)


def test_state_written_by_an_older_build_loads_without_the_field(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"run_id": "old", "current_gen": 4, "history": []}))
    state = RunState.load(path)
    assert state.handicap_rate is None
    assert resolve_initial_rate(state.handicap_rate, 0.7) == pytest.approx(0.7)


# --------------------------------------------------------------------------- #
# The config hash — an annealed rate must not invalidate a resume              #
# --------------------------------------------------------------------------- #


def test_annealing_does_not_change_the_config_hash() -> None:
    """`handicap_rate` is in `config_hash` *and* changes during a run. That is
    only safe because the changing copy lives in `state.json` and the config's
    stays at its initial value. If anything ever writes the annealed rate back
    into the config, every resume after the first step is refused."""
    cfg = RunConfig()
    before = cfg.hash()
    state = RunState.fresh("run-x", before, handicap_rate=cfg.self_play.handicap_rate)
    for gen in range(1, 10):
        state.handicap_rate = decide_handicap_rate(
            state.handicap_rate, stats(0.9), cfg.self_play.handicap_anneal, gen=gen
        ).rate
    assert state.handicap_rate < cfg.self_play.handicap_rate
    assert cfg.hash() == before
    assert cfg.hash() == state.config_hash


def test_the_rate_reaches_the_self_play_command_line(monkeypatch, tmp_path: Path) -> None:
    """The last link in the chain. An annealed rate that never reaches
    `selfplay-batch` is a curriculum that moves only in the log file."""
    from model import eval as eval_mod

    captured: list[list[str]] = []
    monkeypatch.setattr(eval_mod, "_bin", lambda name: Path("/bin/echo"))
    monkeypatch.setattr(
        eval_mod.subprocess, "Popen", lambda cmd, **kw: captured.append(cmd) or object()
    )

    state = RunState.fresh("run-x", "hash", handicap_rate=0.7)
    for gen in range(1, 4):
        state.handicap_rate = decide_handicap_rate(
            state.handicap_rate, stats(0.25), controller(), gen=gen
        ).rate

    eval_mod.start_self_play(
        model_onnx=tmp_path / "net.onnx",
        out_dir=tmp_path,
        games=1, sims_fast=1, sims_full=1, full_search_rate=0.5, c_puct=1.4,
        batch_size=1, virtual_loss=1.0, fpu_reduction=0.25, temperature_plies=1,
        temperature=1.0, opening="belgian",
        handicap_rate=resolve_initial_rate(state.handicap_rate, 0.7),
        handicap_max=5, random_opening_plies=2, max_plies=60, no_progress_plies=0,
        capture_gamma=0.98, dirichlet_alpha=0.2, dirichlet_eps=0.25, shard_games=1,
        threads=1, seed=0,
    )

    cmd = captured[0]
    assert cmd[cmd.index("--handicap-rate") + 1] == "0.55"


def test_changing_the_anneal_settings_does_change_the_hash() -> None:
    """It changes which positions the run trains on from generation two onward,
    which is exactly what the hash is for."""
    base = RunConfig()
    changed = RunConfig()
    changed.self_play.handicap_anneal.target_natural_termination = 0.4
    assert changed.hash() != base.hash()
