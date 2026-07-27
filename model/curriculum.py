"""Capture-handicap curriculum annealing (MODEL.md §4).

Self-play seeds a `handicap_rate` fraction of games near the capture threshold
so that terminal states sit inside the search horizon in generation one. That
crutch has to come out: at generation 40, a static 0.7 would still spend 70% of
the compute on artificially-seeded positions and never shift attention to the
ones real play produces. This module owns *when* it comes out.

**The control signal is the natural termination rate of unseeded games** — the
fraction of games recorded with `handicap == (0, 0)` that reach six captures
**before** the ply cap.

**Why not the decisive rate.** It is the intuitive choice and it is wrong.
Measured over 200 uniformly-random playouts, Belgian Daisy with adjudication and
no handicap is already **63% decisive**, because adjudication at the ply cap
resolves almost any game on a ±1 capture margin (mean |margin| 0.89 — one lucky
push). A threshold on decisive rate therefore fires in generation one against a
*random* network and pulls the crutch before it has done anything. Over those
same 200 playouts the natural termination rate was **0.0%**: mean plies came out
at exactly the cap, 400.0 / 200.0 / 200.0, so not one game in 200 ended on its
own. It rises only when the network genuinely learns to finish games, which is
precisely when the endgame curriculum is no longer earning its keep.

Do not "fix" the signal back to decisive rate.
`tests/test_curriculum.py::test_uniformly_random_play_does_not_move_the_rate`
is the regression test for exactly that change.

**Detecting natural termination from shard data.** With the no-progress rule
disabled (`no_progress_plies: 0`, the default everywhere), `Game::state()` in
`crates/game/src/game.rs` terminates on exactly two conditions: a side reaching
`WIN_THRESHOLD = 6` captures, or `ply >= max_plies` (adjudicated on capture
differential). So a game whose recorded ply count is short of its own
`max_plies` ended on captures, and that is the whole test. With the no-progress
rule *enabled* a third exit appears — `plies_since_capture >= no_progress_plies`
— which also stops short of the cap without six captures. The shard schema
cannot distinguish the two after the fact: `black_losses`/`white_losses` are
recorded *before* each move, so the capture that ends the game is not in any
row, and `score_diff` is a differential rather than a count. Rather than guess,
`AnnealStats.from_summary` reports the signal as unmeasurable when the rule is
on and the controller holds the rate for the whole run.

**The rule is a ratchet.** Monotone non-increasing, one step per generation, and
it never goes below `floor`. A curriculum does not go backwards: if the network
regresses, holding is the right response, and re-teaching it endgames it already
knows is not. Monotone also means the loop cannot oscillate, which is what makes
a controller this crude safe to leave unattended for 50 generations.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from model.config import HandicapAnnealConfig

# Reasons a decision came out the way it did. These are logged and land in
# `metrics.jsonl`, so they are part of what a run is read back with.
REASON_OFF = "off"
REASON_SCHEDULE = "schedule"
REASON_STEPPED = "stepped"
REASON_BELOW_TARGET = "below_target"
REASON_INSUFFICIENT_SAMPLE = "insufficient_sample"
REASON_NOT_MEASURABLE = "not_measurable"
REASON_AT_FLOOR = "at_floor"

#: Rounding for stored/compared rates. Repeated subtraction of 0.05 from 0.7 in
#: binary floating point drifts (0.7 - 0.05 - 0.05 = 0.5999999999999999) and
#: that noise ends up in log lines and `state.json`.
_RATE_DIGITS = 6


@dataclass(frozen=True)
class AnnealStats:
    """The measurement the controller acts on, and nothing else.

    `natural_termination_rate` is `None` when the statistic cannot be measured
    from the data at all (the no-progress rule is on — see the module docstring)
    and `nan` when there simply were no unseeded games to measure. Both hold.
    """

    #: Games this generation recorded with `handicap_black == handicap_white == 0`.
    unseeded_games: int
    #: Fraction of *those* that reached six captures before the ply cap.
    natural_termination_rate: float | None

    @classmethod
    def from_summary(
        cls, summary: Mapping[str, Any] | None, *, no_progress_plies: int = 0
    ) -> AnnealStats:
        """Read the two numbers out of `export_game.summarise_generation`.

        `no_progress_plies != 0` makes natural termination unidentifiable from
        the shards, so the statistic is reported as `None` rather than as a
        number that would silently mean something else.
        """
        summary = summary or {}
        unseeded = summary.get("unseeded_games", 0)
        try:
            unseeded = int(unseeded)
        except (TypeError, ValueError):
            unseeded = 0
        if no_progress_plies:
            return cls(unseeded_games=unseeded, natural_termination_rate=None)
        rate = summary.get("natural_termination_rate")
        try:
            rate = float(rate)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            rate = float("nan")
        return cls(unseeded_games=unseeded, natural_termination_rate=rate)


@dataclass(frozen=True)
class AnnealDecision:
    """One generation's verdict. `message` is what gets logged verbatim."""

    rate: float
    previous: float
    stepped: bool
    reason: str
    message: str


def _round(rate: float) -> float:
    return round(float(rate), _RATE_DIGITS)


def _pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    v = float(value)
    return "n/a" if math.isnan(v) else f"{100.0 * v:.1f}%"


def scheduled_rate(schedule: Mapping[Any, Any], gen: int) -> float | None:
    """The most recent schedule entry with key `<= gen`, or `None` if there is
    none yet. Same "from this generation onward" semantics as
    `train.lr_schedule` — deliberately, so there is one rule to remember."""
    best: float | None = None
    for key in sorted(int(k) for k in schedule):
        if gen >= key:
            best = float(schedule[key])
    return best


def resolve_initial_rate(state_rate: float | None, cfg_rate: float) -> float:
    """The rate a generation starts from.

    `state.json` is authoritative once a run has one — otherwise resuming a run
    at generation 30 would silently restart the curriculum at 0.7. A state file
    written before this feature existed has no rate at all; it falls back to the
    config's initial value.
    """
    return _round(cfg_rate if state_rate is None else state_rate)


def decide_handicap_rate(
    current_rate: float,
    stats: AnnealStats,
    cfg: HandicapAnnealConfig,
    gen: int,
) -> AnnealDecision:
    """Ratchet `current_rate` down by at most one step. Pure.

    Called once per generation, after that generation's self-play has been
    measured; the result applies to the *next* generation. Two invariants hold
    in every mode: the rate never increases, and it never lands below
    `cfg.floor`.
    """
    previous = _round(current_rate)
    floor = _round(cfg.floor)

    def hold(reason: str, message: str) -> AnnealDecision:
        return AnnealDecision(
            rate=previous, previous=previous, stepped=False, reason=reason, message=message
        )

    if cfg.mode == "off":
        return hold(
            REASON_OFF,
            f"handicap {previous:.2f} (fixed — handicap_anneal.mode is off)",
        )

    if cfg.mode == "schedule":
        entry = scheduled_rate(cfg.schedule, gen)
        if entry is None:
            return hold(
                REASON_SCHEDULE,
                f"handicap {previous:.2f} (schedule — no entry at or before gen {gen})",
            )
        # Monotone, and never below the floor: the invariants are run-wide, not
        # controller-only. A schedule that wants zero must say `floor: 0.0`.
        target = max(floor, min(previous, _round(entry)))
        if target == previous:
            return hold(
                REASON_SCHEDULE,
                f"handicap {previous:.2f} (schedule — already at or below the entry "
                f"for gen {gen}, {entry:.2f})",
            )
        return AnnealDecision(
            rate=target,
            previous=previous,
            stepped=True,
            reason=REASON_SCHEDULE,
            message=(
                f"handicap {previous:.2f} → {target:.2f} (schedule entry for gen {gen})"
            ),
        )

    # ---- mode == "controller" ------------------------------------------------
    ntr = stats.natural_termination_rate
    target_str = f"target {_pct(cfg.target_natural_termination)}"

    if ntr is None:
        return hold(
            REASON_NOT_MEASURABLE,
            f"handicap {previous:.2f} (hold — the no-progress rule is on, so a game can "
            f"stop early without six captures and natural termination cannot be read off "
            f"the shards; the controller is inert for this run)",
        )
    if math.isnan(ntr):
        return hold(
            REASON_INSUFFICIENT_SAMPLE,
            f"handicap {previous:.2f} (hold — no unseeded games to measure)",
        )

    sample = f"{_pct(ntr)} natural termination over {stats.unseeded_games} unseeded game(s)"
    if stats.unseeded_games < cfg.min_unseeded_games:
        return hold(
            REASON_INSUFFICIENT_SAMPLE,
            f"handicap {previous:.2f} (hold — {stats.unseeded_games} unseeded game(s) "
            f"< min_unseeded_games {cfg.min_unseeded_games}; the estimate is too noisy to "
            f"act on, {_pct(ntr)} notwithstanding)",
        )
    if ntr < cfg.target_natural_termination:
        return hold(
            REASON_BELOW_TARGET,
            f"handicap {previous:.2f} (hold — {sample}, below {target_str})",
        )
    if previous <= floor:
        return hold(
            REASON_AT_FLOOR,
            f"handicap {previous:.2f} (hold at floor {floor:.2f} — {sample}, at or above "
            f"{target_str}; the floor is permanent endgame coverage, not a bug)",
        )

    new_rate = max(floor, _round(previous - cfg.step))
    return AnnealDecision(
        rate=new_rate,
        previous=previous,
        stepped=True,
        reason=REASON_STEPPED,
        message=(
            f"handicap {previous:.2f} → {new_rate:.2f} ({sample}, at or above "
            f"{target_str}; step {cfg.step:.2f}, floor {floor:.2f})"
        ),
    )


__all__ = [
    "REASON_AT_FLOOR",
    "REASON_BELOW_TARGET",
    "REASON_INSUFFICIENT_SAMPLE",
    "REASON_NOT_MEASURABLE",
    "REASON_OFF",
    "REASON_SCHEDULE",
    "REASON_STEPPED",
    "AnnealDecision",
    "AnnealStats",
    "decide_handicap_rate",
    "resolve_initial_rate",
    "scheduled_rate",
]
