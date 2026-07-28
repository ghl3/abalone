"""Post-hoc anchor ladder with anchors that can actually resolve.

Run `ruby-panther` froze its config in memory at startup, so its own ladders
kept the broken `trailing_gens: [4]` for the whole run and only measured
anything at generation 8 — and then only by accident, because `frozen_gens:
[1, 6]` finally had generation 6 available to play.

Every checkpoint is retained and `eval-match` is a standalone binary, so the
measurement is recoverable without re-running training. This plays two probe
generations against a spread of earlier ones. Two probes rather than one
because a single Elo column is a snapshot; two is a slope, and the slope is
what "is it still learning" actually asks.

The spread matters as much as the probes. A ladder of near anchors saturates
at the bottom and a ladder of distant ones saturates at the top; measuring
against 11, 10, 9, 8, 6, 4, 2 and 1 means whatever the true strength gap is,
some rung sits inside the resolvable band and reports an interval instead of a
sample-size bound.

Usage:  ABALONE_USE_COREML=1 uv run python -m model.posthoc_ladder [--games 32]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from model.config import RunConfig
from model.eval import (
    KIND_FROZEN,
    LadderOpponent,
    clamped_fraction,
    mean_elo,
    model_spec,
    run_ladder,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Opponents for every probe, newest first. Anything not strictly earlier than
#: the probe is dropped, so one list serves both probes.
ANCHOR_GENS = (11, 10, 9, 8, 6, 4, 2, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="ruby-panther-20260727-2159")
    ap.add_argument("--probes", default="12,8", help="comma-separated generations to measure")
    ap.add_argument("--games", type=int, default=32)
    ap.add_argument("--simulations", type=int, default=200)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if os.environ.get("ABALONE_USE_COREML") != "1":
        # The CPU path is ~12x slower and would turn a 20-minute ladder into
        # four hours. This exact omission once made a 12x throughput fix
        # measure as no change at all.
        print("warning: ABALONE_USE_COREML is not 1 — this will be very slow", file=sys.stderr)

    run_dir = REPO_ROOT / "runs" / args.run
    # An archived config, not one being launched: this run 's anchors are
    # precisely what we are here to work around, so validating them would
    # make the tool unable to open the run it was written for.
    cfg = RunConfig.from_yaml(run_dir / "config.yaml", validate=False)
    la = cfg.anchor_ladder
    ckpt = run_dir / "checkpoints"
    out_root = Path(args.out) if args.out else run_dir / "posthoc_ladder"
    out_root.mkdir(parents=True, exist_ok=True)

    report: dict[str, object] = {"run": args.run, "games_per_rung": args.games, "probes": {}}

    for probe in [int(p) for p in args.probes.split(",")]:
        probe_onnx = ckpt / f"gen_{probe:03d}.onnx"
        if not probe_onnx.exists():
            print(f"gen {probe}: no checkpoint at {probe_onnx}, skipping", file=sys.stderr)
            continue

        opponents = [
            LadderOpponent(
                spec=model_spec(ckpt / f"gen_{g:03d}.onnx"),
                games=args.games,
                kind=KIND_FROZEN,
            )
            for g in ANCHOR_GENS
            if g < probe and (ckpt / f"gen_{g:03d}.onnx").exists()
        ]
        print(f"\n=== gen {probe} vs {len(opponents)} anchors, {args.games} games each ===")

        rungs = run_ladder(
            model_onnx=probe_onnx,
            opponents=opponents,
            simulations=args.simulations,
            c_puct=la.c_puct,
            eval_dir=out_root / f"gen_{probe:03d}",
            logs_dir=out_root / f"gen_{probe:03d}" / "logs",
            gen=probe,
            seed=cfg.seed + 7919 * probe,
            batch_size=la.batch_size,
            opening=la.opening,
            random_opening_plies=la.random_opening_plies,
            temperature_plies=la.temperature_plies,
            temperature=la.temperature,
            max_plies=la.max_plies,
            no_progress_plies=la.no_progress_plies,
            threads=la.threads,
            on_rung=lambda r: print(
                f"    {r.label:16} {r.result.wins:3}-{r.result.draws:2}-{r.result.losses:2}"
                f"  score {r.score:.3f}  elo {r.elo_str()}"
            ),
        )
        report["probes"][str(probe)] = {
            "mean_elo": mean_elo(rungs),
            "clamped_fraction": clamped_fraction(rungs),
            "rungs": [
                {
                    "opponent": r.opponent,
                    "wins": r.result.wins,
                    "draws": r.result.draws,
                    "losses": r.result.losses,
                    "score": r.score,
                    "elo": r.result.elo_a,
                    "ci95": [r.result.elo_a_ci95_lo, r.result.elo_a_ci95_hi],
                    "clamped": r.clamped,
                }
                for r in rungs
            ],
        }

    dst = out_root / "summary.json"
    dst.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwrote {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
