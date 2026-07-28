"""Read a run back: the per-generation trajectory, the ladder, the warnings.

A training run emits `runs/<id>/metrics.jsonl`, one JSON object per generation,
and `runs/<id>/state.json`. Both are complete and neither is readable — the
generation-6 validation run wrote 60-key objects, and answering "are the games
getting longer" meant writing a throwaway parser. Twice, mid-run. This is that
parser, promoted.

```
    uv run python -m model.report --run latest
    uv run python -m model.report --run dazzling-cinder-20260727-2016 --tail 10
    uv run python -m model.report --run latest --json
```

The columns are the ones that decide whether a run is working:

| column      | what it tells you |
| --- | --- |
| `plies`     | mean game length. Should **rise** toward 120–200 as defence appears |
| `dec`       | decisive rate — fraction of games with a winner |
| `nat`       | natural termination of *unseeded* games: the curriculum's control signal |
| `hcap`      | live capture-handicap seeding rate, annealed down as `nat` rises |
| `Hgap`      | `ln(legal moves) − policy target entropy`. **Zero means search learned nothing** |
| `cap/100`   | captures per 100 plies. Should **fall** as `plies` rises |
| `val CE`    | value cross-entropy on the holdout |
| `val acc`   | value accuracy on the holdout |
| `epochs`    | raw passes over the replay buffer this generation |
| `elo`       | ladder Elo over fixed-reference rungs, with its 95% interval |

Two metric schemas are read. Runs from before the namespacing change wrote flat
keys (`data_mean_plies`, `val_value_ce`); runs after it write namespaced ones
(`selfplay/mean_plies`, `val_frozen/value_ce`). Both are accepted, because the
first thing anyone does with a tool like this is point it at the run that
motivated writing it.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Ladder rungs whose Elo is a sample-size bound rather than a measurement. At
#: 1.0 the ladder has no resolution left and the headline number is fiction.
ALL_CLAMPED = 1.0

#: `policy_entropy_gap` below this means the MCTS visit distribution is within
#: a whisker of uniform: the policy target carries no information.
ENTROPY_GAP_FLOOR = 0.15

#: Raw passes over the replay buffer in one generation above which a run is
#: overfitting by construction. Mirrors `train.epochs_per_gen_warn`'s default.
EPOCHS_WARN = 8.0

#: `val_rolling` total loss minus mean training loss, above which the network
#: is memorising the buffer. Mirrors `validation.overfit_warn_delta`.
OVERFIT_WARN = 0.35

#: The four heads, their default weight in the total loss (mirrors
#: `LossWeightsConfig`), and — the reason this table exists — how many
#: independent labels each one actually gets.
#:
#: `value` and `score` are labelled *per game*: every position in a game shares
#: the same `z` and the same final capture differential. So their effective
#: sample size is the number of games in the buffer (hundreds), not the number
#: of positions (tens of thousands), and each label is seen once per position
#: per epoch — of the order of a hundred times a generation. `policy` and
#: `capture_map` are labelled per position and have no such ceiling.
#:
#: That asymmetry is not a hypothesis. At generation 5 of ruby-panther the
#: train→rolling gap decomposed as value +0.232, score +0.244×0.15, capture_map
#: +0.011×0.15 and policy **−0.074** — the two per-game heads accounted for more
#: than the whole gap, and the per-position heads generalised at or better than
#: training. Reading that off the total loss alone is impossible, which is why
#: the decomposition is printed rather than left to be recomputed by hand.
HEADS: tuple[tuple[str, float, str], ...] = (
    ("value", 1.00, "per-game"),
    ("score", 0.15, "per-game"),
    ("policy", 1.00, "per-position"),
    ("capture_map", 0.15, "per-position"),
)


# --------------------------------------------------------------------------- #
# Reading a run                                                                #
# --------------------------------------------------------------------------- #


def find_run(runs_root: Path, run: str) -> Path:
    """Resolve `--run`: a run id, a path, or `latest`.

    `latest` is by `state.json` mtime, not by name — run ids are
    `<adjective>-<noun>-<date>-<time>` and sorting them alphabetically returns
    whichever adjective happens to sort last.

    It is deliberately `state.json` and not `metrics.jsonl`: `metrics.jsonl` is
    only appended when a generation *completes*, so a run still inside its first
    generation has none — which made `latest` skip past the live run and report
    on a finished one, exactly when this tool is most wanted. `state.json` is
    written at run start and at every phase transition, so it exists throughout.
    """
    if run not in ("latest", ""):
        for candidate in (Path(run), runs_root / run):
            if candidate.is_dir():
                return candidate
        raise FileNotFoundError(f"no run directory for {run!r} (looked in {runs_root})")
    if not runs_root.is_dir():
        raise FileNotFoundError(f"no runs directory at {runs_root}")
    def freshness(p: Path) -> float:
        # Newest of whichever markers exist. state.json alone is a run that has
        # not finished a generation yet; metrics.jsonl alone is a partial or
        # copied directory. Both should be reachable.
        times = [
            (p / name).stat().st_mtime
            for name in ("state.json", "metrics.jsonl")
            if (p / name).exists()
        ]
        return max(times) if times else float("-inf")

    candidates = [
        p
        for p in runs_root.glob("*")
        if p.is_dir() and ((p / "state.json").exists() or (p / "metrics.jsonl").exists())
    ]
    if not candidates:
        raise FileNotFoundError(
            f"no run in {runs_root} has a state.json or metrics.jsonl"
        )
    return max(candidates, key=freshness)


def read_metrics(run_dir: Path) -> list[dict[str, Any]]:
    """Every generation record, oldest first.

    Truncated or half-written lines are skipped rather than fatal: this tool's
    whole point is being usable *during* a run, and the last line of a file
    being appended to is exactly the line most likely to be incomplete.
    """
    path = run_dir / "metrics.jsonl"
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            out.append(row)
    return sorted(out, key=lambda r: _num(r, "gen") or 0.0)


# --------------------------------------------------------------------------- #
# Schema tolerance                                                             #
# --------------------------------------------------------------------------- #


def _num(row: dict[str, Any], *keys: str) -> float | None:
    """First key present with a real numeric value, else `None`.

    `None` and `nan` are both "not measured" here — a table cell reading `-` is
    honest, and a cell reading `0.00` because a key was missing is not.
    """
    for key in keys:
        if key not in row:
            continue
        v = row[key]
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        f = float(v)
        if not math.isnan(f):
            return f
    return None


def get(row: dict[str, Any], *keys: str) -> float | None:
    """A metric under any of its spellings, new schema first."""
    return _num(row, *keys)


def clamped_fraction(row: dict[str, Any]) -> float | None:
    """Fraction of this generation's ladder rungs whose Elo is a bound.

    Preferred from the measurement layer; recomputed from the rung list for
    runs that predate it — which is the whole reason the metric exists, so the
    tool had better be able to show it for the run that proved the point.
    """
    direct = get(row, "ladder/clamped_fraction", "ladder_clamped_fraction")
    if direct is not None:
        return direct
    rungs = row.get("ladder")
    if not isinstance(rungs, list) or not rungs:
        return None
    measured = [r for r in rungs if isinstance(r, dict) and _num(r, "elo_a") is not None]
    if not measured:
        return None
    return sum(1 for r in measured if r.get("elo_a_clamped")) / len(measured)


def entropy_gap(row: dict[str, Any]) -> float | None:
    """`ln(mean legal moves) − policy target entropy`.

    Preferred straight from the measurement layer; reconstructed from the two
    entropies for runs that predate it. This is the number the previous
    collapse would have shown as a flat zero for three generations.
    """
    gap = get(row, "selfplay/policy_entropy_gap", "data_policy_entropy_gap")
    if gap is not None:
        return gap
    uniform = get(row, "selfplay/policy_uniform_entropy", "data_policy_uniform_entropy")
    target = get(row, "selfplay/policy_target_entropy", "data_policy_target_entropy")
    return None if uniform is None or target is None else uniform - target


@dataclass(frozen=True)
class Row:
    """One generation, reduced to the columns that decide anything."""

    gen: int
    plies: float | None
    decisive: float | None
    natural: float | None
    handicap: float | None
    gap: float | None
    captures: float | None
    val_ce: float | None
    val_acc: float | None
    epochs: float | None
    elo: float | None
    elo_lo: float | None
    elo_hi: float | None
    clamped: float | None
    train_loss: float | None
    rolling_loss: float | None
    seconds: float | None
    positions: float | None

    @classmethod
    def from_metrics(cls, row: dict[str, Any]) -> Row:
        return cls(
            gen=int(_num(row, "gen") or 0),
            plies=get(row, "selfplay/mean_plies", "data_mean_plies", "mean_plies"),
            decisive=get(
                row, "selfplay/decisive_rate", "data_decisive_rate", "decisive_rate"
            ),
            natural=get(
                row,
                "curriculum/natural_termination_rate",
                "selfplay/natural_termination_rate",
                "data_natural_termination_rate",
                "natural_termination_rate",
            ),
            handicap=get(row, "curriculum/handicap_rate", "handicap_rate"),
            gap=entropy_gap(row),
            captures=get(
                row,
                "selfplay/captures_per_100_plies",
                "data_captures_per_100_plies",
            ),
            # The rolling holdout first: it is current-distribution and can be
            # gated on. The frozen one is a drift indicator (MODEL.md §8.1).
            val_ce=get(
                row, "val_rolling/value_ce", "val_frozen/value_ce", "val_value_ce"
            ),
            val_acc=get(
                row,
                "val_rolling/value_accuracy",
                "val_frozen/value_accuracy",
                "val_value_accuracy",
            ),
            epochs=get(
                row, "buffer/epochs_this_gen", "train/epochs_over_buffer"
            ),
            elo=get(row, "ladder/elo_mean", "ladder_elo"),
            elo_lo=get(row, "ladder/elo_mean_ci95_lo", "ladder_elo_ci95_lo"),
            elo_hi=get(row, "ladder/elo_mean_ci95_hi", "ladder_elo_ci95_hi"),
            clamped=clamped_fraction(row),
            train_loss=get(row, "train/loss_total", "train_loss_total"),
            rolling_loss=get(row, "val_rolling/loss_total"),
            seconds=get(row, "perf/gen_seconds", "gen_seconds"),
            positions=get(row, "buffer/positions_this_gen", "positions"),
        )


# --------------------------------------------------------------------------- #
# Formatting                                                                   #
# --------------------------------------------------------------------------- #


def fmt(value: float | None, digits: int = 2, *, sign: bool = False) -> str:
    if value is None:
        return "-"
    f = float(value)
    if math.isnan(f):
        return "-"
    return f"{f:{'+' if sign else ''}.{digits}f}"


def fmt_secs(s: float | None) -> str:
    if s is None or math.isnan(float(s)):
        return "-"
    s = float(s)
    if s < 60:
        return f"{s:.0f}s"
    if s < 3600:
        return f"{s / 60:.1f}m"
    return f"{s / 3600:.2f}h"


def fmt_elo(row: Row) -> str:
    """Elo with its interval, or nothing. Never a bare number: a ladder Elo
    with no interval is what let "+545" pass for a measurement."""
    if row.elo is None or math.isnan(row.elo):
        return "-"
    out = f"{row.elo:+.0f}"
    if row.elo_lo is not None and row.elo_hi is not None:
        out += f" [{row.elo_lo:+.0f},{row.elo_hi:+.0f}]"
    if row.clamped is not None and row.clamped >= ALL_CLAMPED:
        out += " !"
    return out


#: `(header, width, accessor)`. The width includes the header, so a column
#: cannot silently overflow into its neighbour.
COLUMNS: tuple[tuple[str, int, Any], ...] = (
    ("gen", 4, lambda r: str(r.gen)),
    ("plies", 6, lambda r: fmt(r.plies, 0)),
    ("dec", 5, lambda r: fmt(r.decisive, 2)),
    ("nat", 5, lambda r: fmt(r.natural, 2)),
    ("hcap", 5, lambda r: fmt(r.handicap, 2)),
    ("Hgap", 5, lambda r: fmt(r.gap, 2)),
    ("cap/100", 7, lambda r: fmt(r.captures, 2)),
    ("val CE", 7, lambda r: fmt(r.val_ce, 4)),
    ("val acc", 7, lambda r: fmt(r.val_acc, 3)),
    ("epochs", 6, lambda r: fmt(r.epochs, 1)),
    ("time", 6, lambda r: fmt_secs(r.seconds)),
    ("elo [95% CI]", 22, fmt_elo),
)


def format_table(rows: list[Row]) -> list[str]:
    header = "  ".join(h.rjust(w) if h != "elo [95% CI]" else h.ljust(w) for h, w, _ in COLUMNS)
    out = [header, "─" * len(header)]
    for row in rows:
        cells = []
        for head, width, accessor in COLUMNS:
            text = accessor(row)
            cells.append(text.ljust(width) if head == "elo [95% CI]" else text.rjust(width))
        out.append("  ".join(cells).rstrip())
    return out


def format_ladder(records: list[dict[str, Any]]) -> list[str]:
    """Every rung of every ladder, with its interval. Never the mean alone."""
    out: list[str] = []
    for row in records:
        rungs = row.get("ladder")
        if not isinstance(rungs, list) or not rungs:
            continue
        gen = int(_num(row, "gen") or 0)
        clamped = clamped_fraction(row)
        head = f"gen {gen:>3}"
        if clamped is not None:
            head += f"   ({clamped:.0%} of rungs clamped)"
        out.append(head)
        for rung in rungs:
            if not isinstance(rung, dict):
                continue
            label = str(rung.get("opponent", "?"))
            if label.startswith("model:"):
                stem, sep, sims = label[len("model:") :].partition("@")
                label = Path(stem).name + (sep + sims if sep else "")
            elo = _num(rung, "elo_a")
            lo, hi = _num(rung, "elo_a_ci95_lo"), _num(rung, "elo_a_ci95_hi")
            ci = f" [{lo:+.0f},{hi:+.0f}]" if lo is not None and hi is not None else ""
            bound = "  (bound)" if rung.get("elo_a_clamped") else ""
            wld = (
                f"{int(rung.get('wins_a', 0))}-{int(rung.get('wins_b', 0))}-"
                f"{int(rung.get('draws', 0))}"
            )
            score = _num(rung, "score_a")
            kind = str(rung.get("kind", "") or "")
            out.append(
                f"    {label:<24} {kind:<9} {wld:>9}  "
                f"score {fmt(score, 3):>6}  elo {fmt(elo, 0, sign=True):>6}{ci}{bound}"
            )
    return out or ["    (no ladder has run yet)"]


# --------------------------------------------------------------------------- #
# Warnings — the same conditions the training loop warns on, after the fact    #
# --------------------------------------------------------------------------- #


def warnings_for(rows: list[Row], records: list[dict[str, Any]]) -> list[str]:
    """Everything worth stopping a run over, read off the trajectory.

    Deliberately the same set the loop emits live, because the loop's warnings
    scroll past and this does not.
    """
    out: list[str] = []
    for row in rows:
        g = f"gen {row.gen}"
        if row.gap is not None and row.gap < ENTROPY_GAP_FLOOR:
            out.append(
                f"{g}: policy entropy gap {row.gap:.3f} — MCTS visit distributions are "
                f"within a whisker of uniform, so the policy target carries no "
                f"information and nothing downstream means anything"
            )
        if row.clamped is not None and row.clamped >= ALL_CLAMPED:
            out.append(
                f"{g}: every ladder rung clamped — the ladder has no resolution left. "
                f"The Elo shown is the sample-size bound for the game count, not a "
                f"measurement. Add a stronger anchor (a later frozen_gens entry, or a "
                f"nearer trailing_gens offset)"
            )
        if row.epochs is not None and row.epochs > EPOCHS_WARN:
            out.append(
                f"{g}: {row.epochs:.1f} raw passes over the replay buffer in one "
                f"generation — overfitting by construction. Self-play is the cheap "
                f"half; make more games or take fewer steps"
            )
        if (
            row.rolling_loss is not None
            and row.train_loss is not None
            and row.rolling_loss - row.train_loss > OVERFIT_WARN
        ):
            record = next((r for r in records if r.get("gen") == row.gen), None)
            out.append(
                f"{g}: val_rolling loss {row.rolling_loss:.4f} exceeds training loss "
                f"{row.train_loss:.4f} by {row.rolling_loss - row.train_loss:.3f} — the "
                f"rolling holdout is the same distribution as the training data, so "
                f"this gap is memorisation of the buffer. "
                + (overfit_advice(record) if record else "")
            )
    measured = [r for r in rows if r.plies is not None]
    if len(measured) >= 3:
        first, last = measured[0], measured[-1]
        if last.plies is not None and first.plies is not None and last.plies < first.plies:
            out.append(
                f"trend: mean plies fell {first.plies:.0f} → {last.plies:.0f}. Games "
                f"getting shorter while natural termination is high means both sides "
                f"are defending badly, not that either is playing well (MODEL.md §8.2)"
            )
    if not records:
        out.append("this run has written no generation records yet")
    return out


def overfit_advice(record: dict[str, Any]) -> str:
    """Which head is doing the memorising, and therefore what to change.

    "More games" and "fewer steps" are opposite actions and the total loss
    cannot tell you which one you need. Per-game heads (`value`, `score`) are
    capped by the number of *games* in the buffer, so they want more self-play.
    Per-position heads (`policy`, `capture_map`) are capped by steps, so they
    want a smaller step budget. See `HEADS`.
    """
    weighted: dict[str, float] = {}
    for name, weight, labelling in HEADS:
        tr = _num(record, f"train/loss_{name}")
        va = _num(record, f"val_rolling/loss_{name}")
        if tr is not None and va is not None:
            weighted[labelling] = weighted.get(labelling, 0.0) + (va - tr) * weight
    total = sum(weighted.values())
    if not weighted or abs(total) < 1e-9:
        return ""
    per_game = weighted.get("per-game", 0.0) / total
    if per_game >= 0.8:
        return (
            f"{per_game * 100:.0f}% of it is the per-game heads (value, score), whose "
            f"labels are shared by every position in a game — the effective sample "
            f"size is the game count, not the position count. Raise "
            f"self_play.games_per_gen or widen train.replay_buffer_gens; cutting "
            f"steps will not help"
        )
    if per_game <= 0.2:
        return (
            f"{(1 - per_game) * 100:.0f}% of it is the per-position heads (policy, "
            f"capture_map), which are capped by steps rather than games — lower "
            f"train.target_epochs_per_gen"
        )
    return (
        f"per-game heads account for {per_game * 100:.0f}% of it; see the "
        f"generalisation table"
    )


def format_generalisation(records: list[dict[str, Any]]) -> list[str]:
    """Decompose the newest generation's train→val_rolling gap by head.

    The total loss says *whether* the network is memorising the buffer. This
    says *which head* is, and the answer decides the fix: a per-game head that
    overfits wants more games, while a per-position head that overfits wants
    fewer steps. Those are opposite actions, and the total cannot tell them
    apart. See `HEADS`.
    """
    row = next(
        (r for r in reversed(records) if _num(r, "val_rolling/loss_total") is not None),
        None,
    )
    if row is None:
        return ["    (no generation has a rolling holdout yet)"]

    measured: list[tuple[str, float, str, float, float, float]] = []
    for name, weight, labelling in HEADS:
        tr = _num(row, f"train/loss_{name}")
        va = _num(row, f"val_rolling/loss_{name}")
        if tr is not None and va is not None:
            measured.append((name, weight, labelling, tr, va, (va - tr) * weight))
    if not measured:
        return ["    (no per-head losses recorded)"]

    total = sum(m[5] for m in measured)
    out = [
        f"    gen {row.get('gen', '?')}: train → val_rolling, by head",
        f"      {'head':12}{'train':>9}{'rolling':>9}{'gap':>9}{'w':>6}"
        f"{'share':>8}  labels",
    ]
    for name, weight, labelling, tr, va, contribution in measured:
        # Shares are of the *weighted* gap, so they can exceed 100% or go
        # negative: a head that generalises better than it trains offsets the
        # others rather than adding to them.
        share = f"{contribution / total * 100:7.0f}%" if abs(total) > 1e-9 else f"{'-':>8}"
        out.append(
            f"      {name:12}{tr:9.4f}{va:9.4f}{va - tr:+9.3f}{weight:6.2f}"
            f"{share}  {labelling}"
        )
    out.append(f"      {'total':12}{'':9}{'':9}{total:+9.3f}")
    return out


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #


def render(run_dir: Path, records: list[dict[str, Any]], tail: int | None) -> list[str]:
    rows = [Row.from_metrics(r) for r in records]
    shown = rows[-tail:] if tail else rows
    shown_records = records[-tail:] if tail else records

    lines = [f"run: {run_dir.name}", f"path: {run_dir}"]
    state_path = run_dir / "state.json"
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text())
        except json.JSONDecodeError:
            state = {}
        best_elo = state.get("best_elo")
        lines.append(
            f"generations complete: {state.get('current_gen', '?')}"
            f"   phase: {state.get('current_phase', '?')}"
            f"   handicap: {fmt(state.get('handicap_rate'), 2)}"
            + (
                f"   best: gen {state.get('best_gen')} ({best_elo:+.0f} Elo)"
                if isinstance(best_elo, (int, float))
                else "   best: (no ladder has run)"
            )
        )
    if tail and len(rows) > tail:
        lines.append(f"showing the last {tail} of {len(rows)} generations")
    lines.append("")
    lines.extend(format_table(shown))
    lines.append("")
    lines.append("ladder")
    lines.extend(format_ladder(shown_records))
    lines.append("")
    lines.append("generalisation")
    lines.extend(format_generalisation(shown_records))
    lines.append("")

    problems = warnings_for(shown, records)
    lines.append(f"warnings ({len(problems)})")
    lines.extend(f"    ** {w}" for w in problems)
    if not problems:
        lines.append("    none")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m model.report", description=__doc__.split("\n\n")[0]
    )
    parser.add_argument(
        "--run", default="latest", help="run id, run directory, or 'latest' (default)"
    )
    parser.add_argument(
        "--runs-root", type=Path, default=REPO_ROOT / "runs", help="where runs live"
    )
    parser.add_argument("--tail", type=int, help="only the last N generations")
    parser.add_argument(
        "--json", action="store_true", help="emit the reduced rows as JSON instead"
    )
    args = parser.parse_args(argv)

    try:
        run_dir = find_run(args.runs_root, args.run)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    records = read_metrics(run_dir)
    if args.json:
        rows = [Row.from_metrics(r) for r in records]
        shown = rows[-args.tail :] if args.tail else rows
        print(
            json.dumps(
                {
                    "run": run_dir.name,
                    "generations": [vars(r) for r in shown],
                    "warnings": warnings_for(shown, records),
                },
                indent=2,
            )
        )
        return 0

    print("\n".join(render(run_dir, records, args.tail)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
