"""The outer training loop.

One generation, end to end ([ARCHITECTURE §4](../docs/ARCHITECTURE.md)):

```
  1. self-play      selfplay-batch with the latest ONNX; writes shards
  2. train          (overlaps 1) ingest shards, AdamW steps, EMA update
  3. export         checkpoint .pt + ONNX from the EMA weights
  4. validate       frozen holdout + rolling (this generation's own, unseen)
     games          export reviewable JSON + the data-health line
     curriculum     ratchet `handicap_rate` down (MODEL.md §4)
  5. anchor ladder  every N gens: floor anchors + frozen and trailing
                    checkpoints of this run → Elo, per rung, with CIs
  6. commit         metrics.jsonl, TensorBoard, retention, atomic state.json
```

**Metrics are logged generically.** The measurement layer returns namespaced
dicts (`selfplay/`, `buffer/`, `train/`, `val_frozen/`, `val_rolling/`,
`curriculum/`, `ladder/`, `perf/`) and this module iterates whatever it is
handed into `metrics.jsonl` and TensorBoard. There is deliberately no
whitelist: a metric added downstream must show up here without anyone editing
a list, because the alternative is a new diagnostic that exists in code and
nowhere a human will ever read it. Only the human-readable summary line and
the warnings name specific keys.

**The handicap rate lives in `state.json`, not in the config.** The config's
`handicap_rate` is the initial condition; the live value is annealed down once
per generation by `model.curriculum` and read back from state on resume. It is
deliberately not written back into the config, which is inside `config_hash` —
a run that mutated it would refuse to resume itself the moment the curriculum
moved.

**No gating, no promotion.** Self-play always uses the latest network
(MODEL.md §8). The old 21-game gate was really a 2-game experiment, scored
draws as losses so its 0.55 threshold was unreachable, and gated nothing
because self-play read `current_onnx` regardless. It is replaced by a fixed
anchor ladder, which is the thing anybody actually wanted to know.

**The number to watch is `policy_target_entropy` against
`policy_uniform_entropy`.** The previous run sat at a policy loss of exactly
`ln(62) = ln(branching)` for three generations — search was returning uniform
visits, the policy target carried zero information, and nothing downstream
meant anything. That ratio is on the per-generation summary line and warns
above `validation.entropy_ratio_warn`.

Resume granularity is one generation. `state.json` is written atomically at
each phase transition and carries both the config hash and the git SHA: the
hash covers YAML only, so a change to the ply cap or the plane layout in Rust
invalidates every shard in the buffer without touching it (ARCHITECTURE §5.6).

CLI:
    uv run python -m model.train_loop --config config/dry_run.yaml
    uv run python -m model.train_loop --resume bold-otter-... --config ...
    uv run python -m model.train_loop --resume latest --config ...
"""

from __future__ import annotations

import argparse
import copy
import functools
import json
import math
import os
import random
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from model.abalone_net import (
    BOARD_H,
    BOARD_W,
    INPUT_CHANNELS,
    AbaloneNet,
    build,
)
from model.config import RunConfig
from model.curriculum import AnnealStats, decide_handicap_rate, resolve_initial_rate
from model.encoder import (
    OPP_LOSSES_PLANE,
    OWN_LOSSES_PLANE,
    PLY_PLANE,
    VALID_CELL_MASK,
    VALID_MASK_PLANE,
)
from model.eval import (
    KIND_FLOOR,
    KIND_FROZEN,
    KIND_TRAILING,
    LadderOpponent,
    LadderRung,
    ladder_summary,
    mean_elo_basis,
    resolved_regressions,
    model_spec,
    run_ladder,
    start_self_play,
)
from model.export_game import (
    export_generation,
    format_generation_report,
    selfplay_metrics,
    summarise_generation,
)
from model.export_onnx import export as export_onnx
from model.replay_buffer import ReplayBuffer, find_shards_for_gen
from model.run_id import generate_unique
from model.state import GenRecord, RunState
from model.train_step import LossWeights, StepMetrics, train_step
from model.validate import (
    VAL_FROZEN_PREFIX,
    VAL_ROLLING_PREFIX,
    scalar_metrics,
    validate,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Wall-clock cadence for in-generation progress lines.
PROGRESS_INTERVAL_S = 30.0

#: SGD steps between polls of the shard directory / self-play process.
STEPS_PER_POLL = 8


# --------------------------------------------------------------------------- #
# Pure helpers — everything here is unit-tested in tests/test_train_loop.py     #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ResumePlan:
    """What a resume has to do, derived from `state.json` alone."""

    #: Checkpoint generation to load (`gen_<n>.pt`).
    resume_ckpt_gen: int
    #: First generation to run.
    start_gen: int
    #: Generation whose partial shard directory must be discarded, if any.
    discard_gen: int | None


def plan_resume(current_gen: int, current_phase: str) -> ResumePlan:
    """`current_gen` is the number of *fully completed* generations, so the
    checkpoint to load is always `gen_<current_gen>.pt`. Any phase other than
    `complete` means gen `current_gen + 1` started and did not finish; its
    shards are partial and the generation is redone from self-play."""
    in_progress = current_gen + 1
    return ResumePlan(
        resume_ckpt_gen=current_gen,
        start_gen=in_progress,
        discard_gen=None if current_phase == "complete" else in_progress,
    )


def git_sha_drift(recorded: str, current: str) -> bool:
    """True when both SHAs are known and differ.

    A warning rather than a refusal: most commits do not touch the plane
    layout, and refusing to resume after a docs typo would be its own kind of
    broken. An unknown SHA on either side (a tarball, a detached build) is not
    evidence of drift.
    """
    return bool(recorded) and bool(current) and recorded != current


def should_run_ladder(
    gen: int, total_gens: int, every_gens: int, run_on_final: bool = True
) -> bool:
    """Ladder cadence. `every_gens <= 0` disables it outright — including the
    final-generation run, because "off" should mean off."""
    if every_gens <= 0:
        return False
    if gen % every_gens == 0:
        return True
    return bool(run_on_final and gen >= total_gens)


def should_validate(gen: int, *, enabled: bool, holdout_gen: int, every_gens: int) -> bool:
    """Validation cadence, counted from the first generation *after* the
    holdout. Generation `holdout_gen` itself has no frozen set to score
    against yet — its own data is still being trained on."""
    if not enabled or every_gens <= 0:
        return False
    if gen <= holdout_gen:
        return False
    return (gen - holdout_gen) % every_gens == 0


#: Generations whose near-uniform policy targets are expected rather than a
#: fault. Generation 1 is played by the randomly-initialised network: its priors
#: *are* uniform, so PUCT spreads visits nearly evenly and the ratio sits above
#: any useful threshold by construction. Alarming on it fires on every run that
#: has ever been started, which is the fastest way to teach a reader that this
#: warning means nothing.
ENTROPY_ALARM_FIRST_GEN = 2


def entropy_ratio_alarm(ratio: float | None, threshold: float, gen: int) -> bool:
    """True when the MCTS visit distributions are close enough to uniform that
    the policy target carries no information. At 1.0 the data is worthless and
    every downstream metric is meaningless.

    Silent before `ENTROPY_ALARM_FIRST_GEN`, where a high ratio is the expected
    behaviour of an untrained network rather than evidence of broken search.
    """
    if ratio is None or gen < ENTROPY_ALARM_FIRST_GEN:
        return False
    try:
        r = float(ratio)
    except (TypeError, ValueError):
        return False
    return not math.isnan(r) and r > threshold


#: How far the value head may over-predict draws before it is hedging rather
#: than modelling. A head that emits the same class for everything is the
#: failure this project actually suffered — three generations of a constant
#: output, unnoticed, because aggregate accuracy looked plausible.
DRAW_COLLAPSE_MARGIN = 0.25


def draw_collapse_alarm(
    predicted_rate: float | None, observed_rate: float | None, margin: float
) -> bool:
    """True when the value head predicts draws far more often than they occur.

    **This detects a collapsed head, not collapsed data.** In the original
    failure every game *was* a draw, so a head predicting draws was correct and
    this comparison would have stayed silent — `data_draw_rate` and the
    self-play decisive rate are what catch that, and they do. What this catches
    is the other direction: decisive games in the data and a head that has
    stopped committing to either side, which is what hedging to the majority
    class looks like once the curriculum is working.

    Silent when either rate is missing or `nan`; no measurement is not a bad
    measurement.
    """
    if predicted_rate is None or observed_rate is None:
        return False
    try:
        p, o = float(predicted_rate), float(observed_rate)
    except (TypeError, ValueError):
        return False
    if math.isnan(p) or math.isnan(o):
        return False
    return p - o > margin


def retention_victims(files: list[Path], keep: int, protected: set[Path]) -> list[Path]:
    """Oldest-first files beyond the last `keep`, minus anything protected.

    Protection is by explicit path, not by resolving a symlink: `best.onnx`
    degrades to a plain copy on filesystems without symlinks, and then
    resolving it protects nothing (review §7.3).
    """
    ordered = sorted(files)
    if keep < 0:
        keep = 0
    doomed = ordered[: max(0, len(ordered) - keep)]
    return [f for f in doomed if f not in protected]


def ladder_opponents(
    fixed: list[str],
    frozen_gens: list[int],
    trailing_gens: list[int],
    gen: int,
    ckpt_dir: Path,
    *,
    games: int = 40,
    floor_games: int = 16,
) -> list[LadderOpponent]:
    """Build the rungs for generation `gen`.

    Floor anchors first (cheap, `floor_games` each), then the absolute frozen
    checkpoints ascending, then the trailing `gen − k` checkpoints. Checkpoint
    rungs get the full `games` budget: they are the ones being *measured*, and
    12 games is a ±145 Elo interval.

    Strictly earlier: a generation cannot be its own opponent. A checkpoint that
    retention has collected is silently dropped rather than crashing a ladder
    five hours into a run, and a trailing offset that resolves onto a generation
    already covered by `frozen_gens` is deduplicated rather than played twice.
    """
    out = [LadderOpponent(spec=s, games=floor_games, kind=KIND_FLOOR) for s in fixed]
    seen: set[int] = set()

    def add(g: int, kind: str) -> None:
        if g <= 0 or g >= gen or g in seen:
            return
        path = ckpt_dir / f"gen_{g:03d}.onnx"
        if not path.exists():
            return
        seen.add(g)
        out.append(LadderOpponent(spec=model_spec(path), games=games, kind=kind))

    for g in sorted({int(x) for x in frozen_gens}):
        add(g, KIND_FROZEN)
    # Largest offset first, so the rungs come out in ascending generation order
    # like the frozen ones — a ladder log that reads oldest-to-newest is a
    # progression, and one that reads in config order is a list.
    for k in sorted({int(x) for x in trailing_gens}, reverse=True):
        add(gen - k, KIND_TRAILING)
    return out


def epoch_budget_steps(
    buffer_size: int,
    batch_size: int,
    target_epochs: float | None,
    min_steps: int,
    max_steps: int,
) -> int:
    """SGD steps that amount to `target_epochs` passes over the buffer.

    `steps = epochs × buffer_size / batch_size`, clamped into
    `[min_steps, max_steps]`. `target_epochs is None` keeps the old behaviour of
    running to `max_steps` whenever there is wall-clock to spend.

    The reason this exists: a step budget fixed in absolute terms is an epoch
    budget that floats with how long the games happened to be. The validation
    run's generation 3 spent 1500 steps × 256 over a 19k-position buffer — 20
    raw passes over the same data in one generation — not because anyone chose
    20, but because games had shortened from 148 plies to 67 and the buffer had
    shrunk underneath a constant.
    """
    hi = max(int(max_steps), int(min_steps))
    if target_epochs is None:
        return hi
    if buffer_size <= 0 or batch_size <= 0:
        return int(min_steps)
    want = round(float(target_epochs) * buffer_size / batch_size)
    return max(int(min_steps), min(hi, want))


def epochs_over_buffer(steps: int, batch_size: int, buffer_size: int) -> float:
    """Raw passes over the replay buffer, ignoring D6 augmentation.

    Augmentation multiplies distinct *tensors* by 12 but not distinct
    *positions*, so this is the honest number: it is how many times the network
    saw each recorded position, up to a symmetry.
    """
    if buffer_size <= 0:
        return float("nan")
    return steps * batch_size / buffer_size


def tb_step_offset(history: Sequence[GenRecord]) -> int:
    """TensorBoard global-step base for the next generation: the SGD steps every
    completed generation actually ran.

    This used to be `(gen - 1) * steps_per_gen_min`, which is only correct while
    every generation runs exactly the minimum. The budget is now *derived* from
    `train.target_epochs_per_gen`, so a generation exceeds the minimum as soon
    as the buffer is large enough — and then generation N+1 starts at an offset
    already inside generation N's range. Curves overlay each other and the
    x-axis silently stops being a timeline.

    Accumulated from `state.history`, which `state.json` persists, so it
    survives a resume without a schema change. A generation redone after a crash
    reuses its own offset and overwrites its aborted attempt, which is what a
    redo should do.
    """
    return sum(int(r.train_steps or 0) for r in history)


def epochs_alarm(epochs: float | None, threshold: float) -> bool:
    """True when a single generation made far too many passes over the same
    positions. Overfitting the buffer looks like progress on the training loss
    and like nothing at all on the frozen holdout."""
    if epochs is None:
        return False
    try:
        e = float(epochs)
    except (TypeError, ValueError):
        return False
    return not math.isnan(e) and e > threshold


def overfit_alarm(
    rolling_loss: float | None, train_loss: float | None, delta: float
) -> bool:
    """True when this generation's loss on data it has never seen exceeds its
    loss on the buffer by more than `delta`.

    Both numbers are the same weighted total over the same distribution — the
    rolling holdout is this generation's own shards — so the gap between them
    is memorisation and nothing else. The frozen gen-1 holdout could not show
    this cleanly: a rising gap there is "different from generation 1", which is
    also what *progress* looks like.
    """
    for v in (rolling_loss, train_loss):
        if v is None:
            return False
    try:
        r, t = float(rolling_loss), float(train_loss)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    if math.isnan(r) or math.isnan(t):
        return False
    return (r - t) > delta


def namespaced(prefix: str, metrics: Mapping[str, object]) -> dict[str, object]:
    """`{prefix}/{key}` for every key that is not already namespaced.

    The pass-through matters: the measurement layer is being changed under this
    one to emit its own prefixes, and a consumer that blindly re-prefixed would
    produce `val_frozen/val_frozen/loss_total` the day it landed.
    """
    out: dict[str, object] = {}
    for key, value in metrics.items():
        out[key if "/" in key else f"{prefix}/{key}"] = value
    return out


def _fmt_secs(s: float) -> str:
    if s < 60:
        return f"{s:.0f}s"
    if s < 3600:
        return f"{s / 60:.1f}m"
    return f"{s / 3600:.2f}h"


def _fmt(v: float | None, digits: int = 3) -> str:
    if v is None:
        return "n/a"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "n/a"
    return "nan" if math.isnan(f) else f"{f:.{digits}f}"


# --------------------------------------------------------------------------- #
# Logging                                                                      #
# --------------------------------------------------------------------------- #


def _log(msg: str, *, gen: int | None = None, total_gens: int | None = None) -> None:
    """One timestamped line, flushed so `tail -f` sees it immediately."""
    prefix = f"[{time.strftime('%H:%M:%S')}] "
    if gen is not None and total_gens is not None:
        prefix += f"[gen {gen:>2}/{total_gens}] "
    print(f"{prefix}{msg}", flush=True)


def _warn(msg: str, *, gen: int | None = None, total_gens: int | None = None) -> None:
    _log(f"** WARNING: {msg} **", gen=gen, total_gens=total_gens)


def _section() -> None:
    print("─" * 71, flush=True)


def _git_sha(short: bool = False) -> str:
    """Best-effort commit SHA; empty string outside a repo (which
    `git_sha_drift` reads as "unknown", not "different")."""
    cmd = ["git", "rev-parse"] + (["--short"] if short else []) + ["HEAD"]
    try:
        return subprocess.check_output(
            cmd, cwd=REPO_ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return ""


def _banner(run_id: str, device: torch.device, pid: int, model: AbaloneNet, workers: int) -> None:
    line = "═" * 73
    print(line, flush=True)
    print("  ●○●○●   Abalone — AlphaZero training pipeline   ●○●○●", flush=True)
    print(line, flush=True)
    print(flush=True)
    print(f"  Run:       {run_id}", flush=True)
    print(f"  Started:   {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"  Device:    {device!s:<13}  Workers:  {workers} threads", flush=True)
    print(f"  PID:       {pid!s:<13}  Commit:   {_git_sha(short=True) or 'unknown'}", flush=True)
    arch = f"AbaloneNet ({model.config}, {model.num_parameters() / 1e6:.1f}M params)"
    print(f"  Torch:     {torch.__version__:<13}  Model:    {arch}", flush=True)
    print(flush=True)
    print(line, flush=True)
    print(flush=True)


def _dump_config(cfg: RunConfig, workers_resolved: str) -> None:
    """Print the resolved config so the run log is self-describing."""
    sp, tr, va, la = cfg.self_play, cfg.train, cfg.validation, cfg.anchor_ladder
    p = lambda s: print(s, flush=True)  # noqa: E731
    p("[config]")
    p(f"  gens:                  {cfg.gens}")
    p(f"  seed:                  {cfg.seed}")
    p(f"  net_preset:            {cfg.net_preset}")
    p(f"  runs_root:             {cfg.runs_root}")
    p(f"  inference backend:     {'CoreML (ANE/GPU)' if cfg.use_coreml else 'ORT CPU'}")
    p("  self_play:")
    p(f"    games_per_gen:       {sp.games_per_gen}")
    p(
        f"    sims:                fast {sp.sims_fast} / full {sp.sims_full} "
        f"@ rate {sp.full_search_rate}"
    )
    p(f"    batch_size:          {sp.batch_size}   virtual_loss {sp.virtual_loss}")
    p(f"    c_puct / fpu:        {sp.c_puct} / {sp.fpu_reduction}")
    p(f"    dirichlet:           alpha {sp.dirichlet_alpha}, eps {sp.dirichlet_eps}")
    p(f"    temperature:         {sp.temperature} for {sp.temperature_plies} plies")
    p(
        f"    opening:             {sp.opening}  (+{sp.random_opening_plies} random plies)"
    )
    p(
        f"    handicap:            {sp.handicap_rate:.0%} of games over 0..={sp.handicap_max}"
        "  (initial; annealed in state.json)"
    )
    an = sp.handicap_anneal
    if an.mode == "controller":
        p(
            f"    handicap_anneal:     controller — step {an.step:.2f} × up to "
            f"{an.max_step_multiple:.1f} (proportional to signal/target), down to floor "
            f"{an.floor:.2f}, once natural termination of unseeded games reaches "
            f"{an.target_natural_termination:.0%}"
        )
        p(
            f"                         (needs >= {an.min_unseeded_games} unseeded games/gen "
            f"to act; ~{(1 - sp.handicap_rate) * sp.games_per_gen:.0f} expected at the "
            f"initial rate)"
        )
        if sp.no_progress_plies:
            p(
                "                         INERT: no_progress_plies is on, so natural "
                "termination is not identifiable from the shards"
            )
    elif an.mode == "schedule":
        entries = ", ".join(f"gen>={k}→{an.schedule[k]}" for k in sorted(an.schedule))
        p(f"    handicap_anneal:     schedule — {entries or '(empty: no-op)'}")
    else:
        p(f"    handicap_anneal:     off — fixed at {sp.handicap_rate:.0%} for the run")
    p(
        f"    termination:         max_plies {sp.max_plies}, no_progress "
        f"{sp.no_progress_plies or 'off'}"
    )
    p(f"    capture_gamma:       {sp.capture_gamma}")
    p(f"    shard_games_per_file:{sp.shard_games_per_file}")
    p(f"    worker_threads:      {workers_resolved}")
    p("  train:")
    steps_max = str(tr.steps_per_gen_max) if tr.steps_per_gen_max is not None else "(none)"
    p(f"    steps_per_gen:       min={tr.steps_per_gen_min}, max={steps_max}")
    p(
        "    target_epochs/gen:   "
        + (
            f"{tr.target_epochs_per_gen:g} passes over the buffer "
            f"(warn > {tr.epochs_per_gen_warn:g})"
            if tr.target_epochs_per_gen is not None
            else f"(none — run to max; warn > {tr.epochs_per_gen_warn:g})"
        )
    )
    p(f"    batch_size:          {tr.batch_size}")
    sched = (
        ", ".join(f"gen>={k}→{tr.lr_schedule[k]}" for k in sorted(tr.lr_schedule))
        if tr.lr_schedule
        else "(constant)"
    )
    p(f"    optimizer:           AdamW lr={tr.learning_rate} wd={tr.weight_decay}")
    p(f"    lr_schedule:         {sched}")
    p(
        f"    loss_weights:        value {tr.loss_weights.value}, score "
        f"{tr.loss_weights.score}, capture_map {tr.loss_weights.capture_map}"
    )
    p(
        f"    ema:                 {'on' if tr.ema.enabled else 'off'} "
        f"(decay {tr.ema.decay}, warmup {tr.ema.warmup_steps}) — exported for self-play"
    )
    p(f"    replay_buffer:       {tr.replay_buffer_gens} gens, min {tr.replay_buffer_min_size}")
    p(f"    symmetry_augment:    {str(tr.symmetry_augment).lower()}")
    p("  validation:")
    if va.enabled:
        p(
            f"    frozen:              gen {va.holdout_gen}, every {va.every_gens} gen(s), "
            f"{va.positions} positions/pass, entropy-ratio warn > {va.entropy_ratio_warn}"
        )
        p(
            f"                         ~{va.frozen_holdout_positions()} positions withheld "
            f"by whole game (drift indicator — never gate on it); the rest of gen "
            f"{va.holdout_gen} stays trainable"
        )
        if va.rolling.enabled and va.rolling.every_gens > 0:
            p(
                f"    rolling:             every {va.rolling.every_gens} gen(s), "
                f"{va.rolling.positions} positions from the generation's *own* shards, "
                f"withheld from its own training"
            )
            p(
                f"                         overfit warn when val_rolling loss exceeds "
                f"train loss by > {va.overfit_warn_delta}"
            )
        else:
            p("    rolling:             (disabled)")
    else:
        p("    (disabled)")
    p("  anchor_ladder:")
    if la.every_gens > 0:
        p(
            f"    every {la.every_gens} gen(s) @ {la.simulations} sims, opening "
            f"{la.opening} +{la.random_opening_plies} random plies"
        )
        p(
            f"    floor anchors:       {', '.join(la.opponents) or '(none)'} "
            f"× {la.floor_games} games"
        )
        p(f"    frozen checkpoints:  {la.frozen_gens or '(none)'} × {la.games} games")
        p(
            f"    trailing (gen − k):  {la.trailing_gens or '(none)'} × {la.games} games "
            f"— the rung that does not saturate"
        )
    else:
        p("    (disabled)")
    print(flush=True)


# --------------------------------------------------------------------------- #
# EMA                                                                          #
# --------------------------------------------------------------------------- #


class Ema:
    """Exponential moving average of the model's state dict.

    The EMA is what gets exported to ONNX, so it is what plays self-play and
    every eval match (MODEL.md §8). Cheap variance reduction; the model that
    plays is never the one that just took a noisy step.

    Non-float entries (`num_batches_tracked`) are copied, not averaged. Early
    on, a 0.999 decay is still overwhelmingly the random initialisation, so the
    effective decay is ramped as `min(decay, (1 + n) / (warmup + n))` — the
    standard `num_updates` correction. Without it a 200-step generation would
    export something 82% random.
    """

    def __init__(self, model: torch.nn.Module, decay: float, warmup_steps: int = 10) -> None:
        self.decay = float(decay)
        self.warmup_steps = max(int(warmup_steps), 0)
        self.updates = 0
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def effective_decay(self) -> float:
        if self.warmup_steps <= 0:
            return self.decay
        return min(self.decay, (1.0 + self.updates) / (self.warmup_steps + self.updates))

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        d = self.effective_decay()
        for k, v in model.state_dict().items():
            shadow = self.shadow.get(k)
            if shadow is None or not shadow.is_floating_point():
                self.shadow[k] = v.detach().clone()
                continue
            shadow.mul_(d).add_(v.detach(), alpha=1.0 - d)
        self.updates += 1

    @torch.no_grad()
    def copy_to(self, model: torch.nn.Module) -> None:
        model.load_state_dict({k: v.clone() for k, v in self.shadow.items()})

    def state_dict(self) -> dict:
        return {
            "decay": self.decay,
            "warmup_steps": self.warmup_steps,
            "updates": self.updates,
            "shadow": {k: v.cpu() for k, v in self.shadow.items()},
        }

    def load_state_dict(self, sd: dict, device: torch.device) -> None:
        self.decay = float(sd.get("decay", self.decay))
        self.warmup_steps = int(sd.get("warmup_steps", self.warmup_steps))
        self.updates = int(sd.get("updates", 0))
        self.shadow = {k: v.to(device) for k, v in sd["shadow"].items()}


def _playing_model(model: AbaloneNet, ema: Ema | None) -> AbaloneNet:
    """A detached copy holding the weights that will be exported and played.
    Returns the live model itself when EMA is off."""
    if ema is None:
        return model
    clone = copy.deepcopy(model)
    ema.copy_to(clone)
    clone.eval()
    return clone


# --------------------------------------------------------------------------- #
# Filesystem plumbing                                                          #
# --------------------------------------------------------------------------- #


def _count_complete_shards(gen_dir: Path) -> int:
    """Fully-written shard files. Atomic rename means `.parquet` (never
    `.parquet.tmp`) is exactly the set the trainer may read."""
    return sum(1 for _ in gen_dir.glob("*.parquet")) if gen_dir.exists() else 0


def _count_completed_games(selfplay_log_path: Path) -> tuple[int, float]:
    """Parse completed-game records out of the self-play log.

    Each finished game writes one line:
        `  [tN] game M: P plies, final=Wins(Black), handicap=b/w, score_diff=D`
    The `"] game M: P plies, final="` substring is the contract with
    `selfplay_batch.rs`; trailing fields may be added freely.

    Called at progress cadence (~30 s), not in a hot loop, so re-reading the
    whole file is free. Unparseable lines are skipped rather than fatal — a
    panic backtrace in the stream must not break the counter.

    Returns `(games_done, mean_plies_per_game)`.
    """
    if not selfplay_log_path.exists():
        return 0, 0.0
    games = 0
    total_plies = 0
    try:
        with selfplay_log_path.open() as f:
            for line in f:
                gm = line.find("] game ")
                if gm == -1:
                    continue
                pf = line.find(" plies, final=", gm)
                if pf == -1:
                    continue
                start = pf
                while start > 0 and line[start - 1].isdigit():
                    start -= 1
                if start >= pf:
                    continue
                try:
                    plies = int(line[start:pf])
                except ValueError:
                    continue
                games += 1
                total_plies += plies
    except OSError:
        return 0, 0.0
    return games, (total_plies / games if games else 0.0)


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _save_ckpt(
    model: AbaloneNet,
    optimizer: torch.optim.Optimizer,
    ema: Ema | None,
    preset: str,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "ema": ema.state_dict() if ema is not None else None,
            "preset": preset,
        },
        tmp,
    )
    os.replace(tmp, path)


def _load_ckpt(
    path: Path,
    model: AbaloneNet,
    optimizer: torch.optim.Optimizer,
    ema: Ema | None,
    device: torch.device,
) -> None:
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    if ema is not None and state.get("ema"):
        ema.load_state_dict(state["ema"], device)


def _link_or_copy(src: Path, dst: Path) -> None:
    """Best-effort relative symlink, falling back to a copy."""
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src.name)
    except (OSError, NotImplementedError):
        shutil.copyfile(src, dst)


def _apply_retention(run_dir: Path, cfg: RunConfig, state: RunState, gen: int) -> None:
    """Drop old checkpoints, ONNX exports, shard dirs and exported-game dirs.

    Never collects the checkpoint a resume would need, nor anything
    `state.json` points at.
    """
    ckpt_dir = run_dir / "checkpoints"
    protected = {
        ckpt_dir / f"gen_{gen:03d}.pt",
        ckpt_dir / f"gen_{gen:03d}.onnx",
    }
    for rel in (state.best_onnx, state.current_onnx):
        if rel:
            protected.add(run_dir / rel)
    # Frozen ladder rungs must survive long enough to be played again — and so
    # must every trailing rung the next few ladders will reach back for.
    for g in cfg.anchor_ladder.frozen_gens:
        protected.add(ckpt_dir / f"gen_{int(g):03d}.onnx")
    for k in cfg.anchor_ladder.trailing_gens:
        for future in range(gen, gen + max(cfg.anchor_ladder.every_gens, 1) + 1):
            protected.add(ckpt_dir / f"gen_{future - int(k):03d}.onnx")

    for pattern, keep in (
        ("gen_*.pt", cfg.retention.keep_last_checkpoints),
        ("gen_*.onnx", cfg.retention.keep_last_onnx),
    ):
        for f in retention_victims(sorted(ckpt_dir.glob(pattern)), keep, protected):
            f.unlink(missing_ok=True)

    for root, keep in (
        (run_dir / "shards", cfg.retention.keep_last_shard_gens),
        (run_dir / "games", cfg.retention.keep_last_game_gens),
    ):
        if not root.exists():
            continue
        threshold = gen - keep
        for d in root.glob("gen_*"):
            try:
                n = int(d.name.split("_", 1)[1])
            except (IndexError, ValueError):
                continue
            # The frozen holdout generation is the validation set; it outlives
            # the rolling window by design.
            if n < threshold and n != cfg.validation.holdout_gen:
                shutil.rmtree(d, ignore_errors=True)


def _warmup_bn(model: AbaloneNet, device: torch.device, batches: int = 8) -> None:
    """Populate BatchNorm running statistics before the bootstrap export.

    PyTorch initialises `running_mean = 0, running_var = 1`, so an un-warmed
    `eval()` forward normalises differently from anything `train()` would have
    computed and the generation-1 ONNX behaves unlike the network it came
    from. Synthetic batches follow the real plane layout (MODEL.md §5): binary
    marbles on valid cells, thermometer loss planes, a ply fraction, and the
    valid-cell mask.
    """
    was_training = model.training
    model.train()
    mask = torch.from_numpy(np.ascontiguousarray(VALID_CELL_MASK)).to(device)
    n = 32
    with torch.no_grad():
        for _ in range(batches):
            x = torch.zeros(n, INPUT_CHANNELS, BOARD_H, BOARD_W, device=device)
            x[:, VALID_MASK_PLANE] = mask
            x[:, 0] = (torch.rand(n, BOARD_H, BOARD_W, device=device) < 0.22).float() * mask
            x[:, 1] = (torch.rand(n, BOARD_H, BOARD_W, device=device) < 0.22).float() * mask
            # Thermometers: `>= 1 … >= 5`, so a sample with k losses lights the
            # first k planes. Draw k uniformly in 0..5 per sample and per side.
            for base in (OWN_LOSSES_PLANE, OPP_LOSSES_PLANE):
                k = torch.randint(0, 6, (n, 1, 1), device=device)
                for i in range(5):
                    x[:, base + i] = (k > i).float() * mask
            frac = torch.rand(n, 1, 1, device=device).expand(n, BOARD_H, BOARD_W)
            x[:, PLY_PLANE] = frac * mask
            model(x)
    if not was_training:
        model.eval()


# --------------------------------------------------------------------------- #
# Training phase                                                               #
# --------------------------------------------------------------------------- #


def _train_phase(
    *,
    model: AbaloneNet,
    optimizer: torch.optim.Optimizer,
    ema: Ema | None,
    buffer: ReplayBuffer,
    sp_proc: subprocess.Popen | None,
    selfplay_log_path: Path,
    shards_dir: Path,
    new_gen: int,
    cfg: RunConfig,
    rng: np.random.Generator,
    device: torch.device,
    loss_weights: LossWeights,
    writer: SummaryWriter | None = None,
    phase_start_time: float | None = None,
    hold_rolling: bool = False,
    step_offset: int = 0,
) -> tuple[int, StepMetrics | None, float, float]:
    """Train while self-play runs. The exit rule is dynamic:

      - always at least `train.steps_per_gen_min` steps;
      - the budget above that is `train.target_epochs_per_gen` passes over the
        *current* buffer, recomputed as shards arrive and clamped to
        `steps_per_gen_max` — so a generation that produced short games gets a
        proportionally smaller budget instead of 20 silent epochs;
      - if `target_epochs_per_gen` is null the budget is `steps_per_gen_max`,
        which is the old behaviour;
      - if self-play exits first and we are past the minimum, stop.

    `hold_rolling` withholds a slice of *this* generation for validation. The
    slice can only be drawn once every shard has landed — it is taken by whole
    game and is idempotent, so drawing it early would fix it on whichever two
    games happened to finish first — so this generation is excluded from
    sampling entirely until self-play exits, at which point the slice is marked
    and the other ~90% joins the pool. The held rows are then never sampled
    again for the life of the run.

    Returns `(steps_done, mean_metrics, self_play_seconds, active_train_seconds)`.
    """
    min_steps = cfg.train.steps_per_gen_min
    hard_max = max(cfg.train.steps_per_gen_max or min_steps, min_steps)
    exclude: tuple[int, ...] = (new_gen,) if hold_rolling else ()
    rolling_held = -1

    def mark_rolling() -> None:
        """Fix this generation's rolling slice and release the rest to the
        training pool. Idempotent, both here and in the buffer."""
        nonlocal exclude, rolling_held
        if not hold_rolling or rolling_held >= 0:
            return
        rolling_held = buffer.mark_rolling_holdout(
            new_gen,
            fraction=cfg.validation.rolling.fraction,
            seed=cfg.validation.seed,
        )
        exclude = ()
        _log(
            f"holdout: gen {new_gen} rolling slice of {rolling_held} positions "
            f"({cfg.validation.rolling.fraction:.0%} of its games) — never trained on",
            gen=new_gen,
            total_gens=cfg.gens,
        )

    def budget() -> int:
        """The step cap right now, from the buffer as it currently stands."""
        return epoch_budget_steps(
            buffer.total_size(),
            cfg.train.batch_size,
            cfg.train.target_epochs_per_gen,
            min_steps,
            hard_max,
        )

    max_steps = hard_max
    steps_done = 0
    sums = dict.fromkeys(
        (
            "loss_total",
            "loss_policy",
            "loss_value",
            "loss_score",
            "loss_capture_map",
            "grad_norm",
            "policy_weight_sum",
        ),
        0.0,
    )
    win_loss_total = 0.0
    win_grad_norm = 0.0
    win_steps = 0
    seen_files: set[Path] = set()

    target_games = cfg.self_play.games_per_gen
    per_shard = max(cfg.self_play.shard_games_per_file, 1)
    target_shards = (target_games + per_shard - 1) // per_shard
    phase_start = phase_start_time if phase_start_time is not None else time.time()
    last_progress_log = phase_start
    sp_seconds: float | None = None
    sp_done_announced = False
    first_step_time: float | None = None
    last_step_time: float | None = None

    def ingest_new() -> int:
        ingested = 0
        for p in find_shards_for_gen(shards_dir.parent, new_gen):
            if p in seen_files:
                continue
            try:
                buffer.ingest_shard(p, gen=new_gen)
                seen_files.add(p)
                ingested += 1
            except Exception:
                # Atomic rename makes a partial read rare but not impossible;
                # retry on the next poll rather than dying mid-generation.
                pass
        return ingested

    def emit_progress() -> None:
        nonlocal win_loss_total, win_grad_norm, win_steps
        sp_alive = sp_proc is not None and sp_proc.poll() is None
        games_done, _ = _count_completed_games(selfplay_log_path)
        shards_done = _count_complete_shards(shards_dir)
        pct = 100 * games_done / max(target_games, 1)
        sp_str = (
            f"self-play {games_done:>4}/{target_games} ({pct:4.1f}%)"
            if sp_alive
            else f"self-play {games_done:>4}/{target_games} (done)"
        )
        train_str = f"train {steps_done:>4}/{max_steps}"
        if min_steps < max_steps and steps_done > min_steps:
            train_str += "+"
        if win_steps > 0:
            loss_str = f"loss={win_loss_total / win_steps:.4f}  grad={win_grad_norm / win_steps:.2f}"
        elif steps_done == 0:
            loss_str = "(waiting for buffer)"
        else:
            loss_str = "(no new steps this window)"
        _log(
            f"  progress  ·  {sp_str}  ·  shards {shards_done:>3}/{target_shards}  ·  "
            f"{train_str}  {loss_str}",
            gen=new_gen,
            total_gens=cfg.gens,
        )
        win_loss_total = 0.0
        win_grad_norm = 0.0
        win_steps = 0

    poll_interval = max(0.05, cfg.train.poll_interval_ms / 1000.0)

    while True:
        ingest_new()
        max_steps = budget()

        if not sp_done_announced and sp_proc is not None and sp_proc.poll() is not None:
            sp_seconds = time.time() - phase_start
            sp_done_announced = True
            if sp_proc.returncode == 0:
                ingest_new()
                mark_rolling()
                games_done, avg_plies = _count_completed_games(selfplay_log_path)
                _log(
                    f"self-play complete in {_fmt_secs(sp_seconds)}  ·  {games_done} games  ·  "
                    f"avg {avg_plies:.0f} plies/game",
                    gen=new_gen,
                    total_gens=cfg.gens,
                )
                if steps_done < min_steps:
                    _log(
                        f"phase: training  (drain to {min_steps} SGD steps, "
                        f"batch {cfg.train.batch_size})",
                        gen=new_gen,
                        total_gens=cfg.gens,
                    )

        now = time.time()
        if now - last_progress_log >= PROGRESS_INTERVAL_S:
            emit_progress()
            last_progress_log = now

        threshold = new_gen - cfg.train.replay_buffer_gens + 1
        if threshold > 0:
            buffer.evict_below(threshold)

        sp_alive_now = sp_proc is not None and sp_proc.poll() is None
        if steps_done >= max_steps:
            break
        if steps_done >= min_steps and not sp_alive_now:
            break

        trainable = buffer.total_size() - (buffer.trainable_size(new_gen) if exclude else 0)
        if trainable < cfg.train.replay_buffer_min_size:
            if sp_alive_now:
                time.sleep(poll_interval)
                continue
            if trainable <= 0:
                raise RuntimeError(
                    "self-play produced no trainable data. If "
                    f"validation.holdout_gen ({cfg.validation.holdout_gen}) is the only "
                    "generation in the buffer, everything is held out."
                )
            # Otherwise proceed on a short buffer rather than stall.

        for _ in range(min(STEPS_PER_POLL, max_steps - steps_done)):
            batch = buffer.sample(cfg.train.batch_size, rng, exclude_gens=exclude)
            m = train_step(
                model,
                optimizer,
                batch,
                device=device,
                weights=loss_weights,
                grad_clip=cfg.train.grad_clip,
            )
            if ema is not None:
                ema.update(model)
            md = m.as_dict()
            for k in sums:
                sums[k] += md[k]
            win_loss_total += m.loss_total
            win_grad_norm += m.grad_norm
            win_steps += 1
            steps_done += 1
            now_step = time.time()
            if first_step_time is None:
                first_step_time = now_step
            last_step_time = now_step
            if writer is not None:
                gs = step_offset + steps_done
                writer.add_scalar("train/step_loss_total", m.loss_total, gs)
                writer.add_scalar("train/step_loss_policy", m.loss_policy, gs)
                writer.add_scalar("train/step_loss_value", m.loss_value, gs)
                writer.add_scalar("train/step_loss_score", m.loss_score, gs)
                writer.add_scalar("train/step_loss_capture_map", m.loss_capture_map, gs)
                writer.add_scalar("train/step_grad_norm", m.grad_norm, gs)
            if steps_done >= max_steps:
                break

    # Training may finish before self-play. Keep polling *and* keep emitting
    # progress so a long tail does not look like a hang.
    if sp_proc is not None:
        while sp_proc.poll() is None:
            ingest_new()  # data ingested now seeds the next generation
            now = time.time()
            if now - last_progress_log >= PROGRESS_INTERVAL_S:
                emit_progress()
                last_progress_log = now
            time.sleep(poll_interval)
        if sp_proc.returncode != 0:
            raise RuntimeError(
                f"selfplay-batch exited with status {sp_proc.returncode}; see "
                f"{selfplay_log_path} for details."
            )
        ingest_new()
        if not sp_done_announced:
            sp_seconds = time.time() - phase_start
            games_done, avg_plies = _count_completed_games(selfplay_log_path)
            _log(
                f"self-play complete in {_fmt_secs(sp_seconds)}  ·  {games_done} games  ·  "
                f"avg {avg_plies:.0f} plies/game",
                gen=new_gen,
                total_gens=cfg.gens,
            )

    # Every shard is in by here, whichever branch got us out of the loop. The
    # slice must exist before validation looks for it.
    mark_rolling()

    if steps_done == 0:
        return 0, None, sp_seconds or 0.0, 0.0
    mean = StepMetrics(**{k: v / steps_done for k, v in sums.items()})
    active_train = (
        (last_step_time - first_step_time)
        if first_step_time is not None and last_step_time is not None
        else 0.0
    )
    return steps_done, mean, sp_seconds or (time.time() - phase_start), active_train


# --------------------------------------------------------------------------- #
# Validation phase                                                             #
# --------------------------------------------------------------------------- #


def _score_holdout(
    *,
    model: AbaloneNet,
    buffer: ReplayBuffer,
    cfg: RunConfig,
    kind: str,
    prefix: str,
    positions: int,
    device: torch.device,
    loss_weights: LossWeights,
    gens: list[int] | None = None,
) -> dict[str, object]:
    """One holdout, scored. `{}` when that holdout has nothing in it.

    Both holdouts go through the same function under different namespaces —
    `val_frozen/` and `val_rolling/` — so a metric added to `model.validate`
    appears under both without anything here being touched.
    """
    batches = buffer.iter_holdout_batches(
        kind=kind,
        positions=positions,
        batch_size=cfg.validation.batch_size,
        seed=cfg.validation.seed,
        gens=gens,
    )
    metrics = validate(
        model,
        batches,
        device,
        weights=loss_weights,
        max_plies=cfg.self_play.max_plies,
        prefix=prefix,
    )
    # `positions` is the accumulator's own row count: zero means the iterator
    # yielded nothing, which is "no holdout yet", not "a holdout of zero".
    return {} if not float(metrics.get(f"{prefix}positions", 0) or 0) else metrics


def _prefixed_get(metrics: Mapping[str, object], prefix: str, key: str) -> object:
    return metrics.get(f"{prefix}{key}")


def _validation_phase(
    *,
    model: AbaloneNet,
    buffer: ReplayBuffer,
    cfg: RunConfig,
    new_gen: int,
    device: torch.device,
    loss_weights: LossWeights,
) -> dict[str, object]:
    """Score the exported (EMA) weights on both holdouts.

    The frozen holdout is a fixed ruler and a **drift** indicator: by
    generation 30 its rows come from a network thirty generations weaker, so a
    rising loss there is as likely to be progress as regression. The rolling
    holdout is this generation's own withheld games — same distribution as the
    training data, never trained on — and is the one that can be gated on.
    """
    out: dict[str, object] = {}
    t = time.time()

    frozen = _score_holdout(
        model=model,
        buffer=buffer,
        cfg=cfg,
        kind="frozen",
        prefix=VAL_FROZEN_PREFIX,
        # Against the *frozen slice*, not the whole holdout generation — most
        # of that generation is trainable data and scoring it as held out would
        # report training loss under a validation name.
        positions=min(
            cfg.validation.positions,
            max(buffer.holdout_size(cfg.validation.holdout_gen), 1) * 4,
        ),
        device=device,
        loss_weights=loss_weights,
    )
    rolling = _score_holdout(
        model=model,
        buffer=buffer,
        cfg=cfg,
        kind="rolling",
        prefix=VAL_ROLLING_PREFIX,
        positions=min(
            cfg.validation.rolling.positions,
            max(buffer.rolling_holdout_size(buffer.rolling_holdout_gen), 1) * 4,
        )
        if buffer.rolling_holdout_gen is not None
        else 0,
        device=device,
        loss_weights=loss_weights,
    )
    out.update(frozen)
    out.update(rolling)

    if not out:
        _log(
            "phase: validate  (skipped — neither holdout has rows in the buffer yet)",
            gen=new_gen,
            total_gens=cfg.gens,
        )
        return out

    _log(f"phase: validate  ({_fmt_secs(time.time() - t)})", gen=new_gen, total_gens=cfg.gens)
    for label, prefix, m in (
        (f"frozen gen {cfg.validation.holdout_gen}", VAL_FROZEN_PREFIX, frozen),
        (f"rolling gen {buffer.rolling_holdout_gen}", VAL_ROLLING_PREFIX, rolling),
    ):
        if not m:
            continue
        g = functools.partial(_prefixed_get, m, prefix)
        _log(
            f"  val [{label}]  {int(g('positions') or 0)} pos  ·  loss "
            f"{_fmt(g('loss_total'), 4)}  ·  policy top-1 {_fmt(g('policy_top1_agreement'))}"
            f"  ·  value CE {_fmt(g('value_ce'), 4)} (acc {_fmt(g('value_accuracy'))}, "
            f"ECE {_fmt(g('value_ece'))})",
            gen=new_gen,
            total_gens=cfg.gens,
        )
        _log(
            f"  val [{label}]  score CE {_fmt(g('score_ce'), 4)} (MAE "
            f"{_fmt(g('score_mae'), 2)} marbles)  ·  capture-map AUC "
            f"{_fmt(g('capture_map_auc'))}  ·  policy entropy "
            f"{_fmt(g('data_policy_target_entropy'))} / "
            f"{_fmt(g('data_policy_uniform_entropy'))} = "
            f"{_fmt(g('data_policy_entropy_ratio'))}",
            gen=new_gen,
            total_gens=cfg.gens,
        )

    # ONLY the rolling holdout. Never fall back to the frozen one: its
    # `data_*` metrics describe generation 1's dataset — produced by a random
    # network — and are constant for the life of the run by construction. That
    # fallback fired a "search is producing no information" alarm at generation
    # 2 of a run whose *current* data had just improved from 0.960 to 0.804,
    # which is the precise failure the `data_` prefix exists to prevent.
    #
    # The rolling holdout is a slice of the newest generation, so its target
    # entropy tracks current data and alarming on it is meaningful. When it is
    # unavailable there is no holdout-based alarm — the equivalent check on
    # full self-play data runs separately and is always available.
    predicted_draws = out.get(f"{VAL_ROLLING_PREFIX}value_predicted_draw_rate")
    observed_draws = out.get(f"{VAL_ROLLING_PREFIX}data_draw_rate")
    if draw_collapse_alarm(predicted_draws, observed_draws, DRAW_COLLAPSE_MARGIN):
        _warn(
            f"the value head predicts a draw for {_fmt(predicted_draws)} of held-out "
            f"positions but only {_fmt(observed_draws)} of them actually drew — it is "
            f"hedging to the majority class rather than committing to a side. Check "
            f"value_accuracy_early and the per-class recalls before spending another "
            f"generation",
            gen=new_gen,
            total_gens=cfg.gens,
        )

    ratio = out.get(f"{VAL_ROLLING_PREFIX}data_policy_entropy_ratio")
    if entropy_ratio_alarm(ratio, cfg.validation.entropy_ratio_warn, new_gen):
        _warn(
            f"rolling held-out policy target entropy is {_fmt(ratio)} of ln(mean legal "
            f"moves) — search is producing (near) no information, and no downstream "
            f"metric means anything until that is fixed",
            gen=new_gen,
            total_gens=cfg.gens,
        )
    return out


# --------------------------------------------------------------------------- #
# Ladder phase                                                                 #
# --------------------------------------------------------------------------- #


def _ladder_phase(
    *,
    cfg: RunConfig,
    new_gen: int,
    onnx_path: Path,
    run_dir: Path,
    ckpt_dir: Path,
) -> list[LadderRung]:
    la = cfg.anchor_ladder
    opponents = ladder_opponents(
        list(la.opponents),
        list(la.frozen_gens),
        list(la.trailing_gens),
        new_gen,
        ckpt_dir,
        games=la.games,
        floor_games=la.floor_games,
    )
    if not opponents:
        return []
    total_games = sum(o.games for o in opponents)
    _log(
        f"phase: anchor ladder  ({len(opponents)} rungs, {total_games} games @ "
        f"{la.simulations} sims)",
        gen=new_gen,
        total_gens=cfg.gens,
    )
    for o in opponents:
        _log(f"    rung  {o.label:<26} {o.kind:<9} {o.games} games", gen=new_gen, total_gens=cfg.gens)

    def on_rung(rung: LadderRung) -> None:
        _log(
            f"  vs {rung.label:<26} {rung.result.summary()}",
            gen=new_gen,
            total_gens=cfg.gens,
        )
        if rung.result.suspicious_transcripts:
            _warn(
                f"{rung.result.games} games vs {rung.label} produced only "
                f"{rung.result.distinct_transcripts} distinct transcript(s) — the openings and "
                f"temperature sampling are not randomising the match, so this result is one "
                f"game with a large denominator",
                gen=new_gen,
                total_gens=cfg.gens,
            )

    return run_ladder(
        model_onnx=onnx_path,
        opponents=opponents,
        simulations=la.simulations,
        c_puct=la.c_puct,
        batch_size=la.batch_size,
        opening=la.opening,
        random_opening_plies=la.random_opening_plies,
        temperature_plies=la.temperature_plies,
        temperature=la.temperature,
        max_plies=la.max_plies,
        no_progress_plies=la.no_progress_plies,
        eval_dir=run_dir / "eval",
        logs_dir=run_dir / "logs",
        gen=new_gen,
        seed=(cfg.seed + 7919 * new_gen) & 0xFFFF_FFFF,
        threads=la.threads,
        on_rung=on_rung,
    )


def _report_ladder(rungs: list[LadderRung], gen: int, total_gens: int) -> float | None:
    """Per-rung results, then the headline — never the headline alone.

    The generation-6 ladder of the validation run swept all four rungs 12-0-0
    and reported "+545" as if that were a measurement. It was the sample-size
    bound, four times over. When every rung clamps, the ladder has no
    resolution left and says so loudly enough to be acted on.
    """
    if not rungs:
        return None
    _log("  ladder:", gen=gen, total_gens=total_gens)
    for r in rungs:
        _log(
            f"    {r.label:<26} {r.kind:<9} {r.result.wins_a}-{r.result.wins_b}-"
            f"{r.result.draws}  score {r.score:.3f}±{r.result.score_a_stderr:.3f}  "
            f"elo {r.elo_str()}",
            gen=gen,
            total_gens=total_gens,
        )
    summary = ladder_summary(rungs, gen)
    elo, basis = mean_elo_basis(rungs)
    lo, hi = summary.get("ladder/elo_mean_ci95_lo"), summary.get("ladder/elo_mean_ci95_hi")
    ci = f" [{lo:+.0f}, {hi:+.0f}]" if lo is not None and hi is not None else ""
    frac = summary.get("ladder/clamped_fraction", float("nan"))
    _log(
        f"  ladder Elo (mean over {basis}): {elo:+.0f}{ci}  ·  "
        f"{int(summary['ladder/rungs_clamped'])}/{int(summary['ladder/rungs_measured'])} "
        f"rungs clamped ({frac:.0%})  ·  {_fmt_secs(summary['ladder/seconds'])}",
        gen=gen,
        total_gens=total_gens,
    )
    if summary.get("ladder/all_clamped"):
        _warn(
            "ladder has no resolution — every rung was swept, so every Elo is the "
            "sample-size bound for the game count rather than a measurement of "
            "anything. Add a stronger anchor: raise anchor_ladder.trailing_gens to a "
            "more recent checkpoint, or add a frozen_gens entry from a later "
            "generation. The mean above is a lower bound, not a number",
            gen=gen,
            total_gens=total_gens,
        )
    return None if elo is None or math.isnan(elo) else float(elo)


# --------------------------------------------------------------------------- #
# Process management                                                           #
# --------------------------------------------------------------------------- #


def _setup_signal_handlers(*slots: list[subprocess.Popen | None]) -> None:
    """Terminate tracked subprocesses when the parent dies, so an interrupted
    run does not leave orphaned self-play workers pinning every core."""

    def handler(signum, _frame):
        for slot in slots:
            proc = slot[0] if slot else None
            if proc is not None and proc.poll() is None:
                try:
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                except Exception:
                    pass
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)


def _start_tensorboard(runs_root: Path, port: int) -> subprocess.Popen | None:
    """Serve `runs/` (not just this run) so runs compare side by side."""
    log_path = runs_root / ".tensorboard.log"
    cmd = [
        sys.executable, "-m", "tensorboard.main",
        "--logdir", str(runs_root),
        "--port", str(port),
        "--bind_all",
    ]
    try:
        runs_root.mkdir(parents=True, exist_ok=True)
        proc = subprocess.Popen(
            cmd, stdout=open(log_path, "w"), stderr=subprocess.STDOUT
        )
        _log(f"tensorboard: http://localhost:{port}/  (server log: {log_path})")
        return proc
    except (FileNotFoundError, OSError) as e:
        _log(f"failed to start tensorboard: {e}")
        return None


def _resolve_resume(cfg: RunConfig, resume_arg: str | None) -> tuple[str, RunState | None]:
    """Decide whether to resume an existing run or start fresh."""
    runs_root = REPO_ROOT / cfg.runs_root
    if resume_arg is None:
        return cfg.run_id or generate_unique(runs_root), None

    if resume_arg == "latest":
        # By mtime, not by name: run IDs are `<adjective>-<noun>-<date>-<time>`,
        # so sorting them alphabetically returns whichever adjective sorts last.
        candidates = sorted(
            (p for p in runs_root.glob("*") if (p / "state.json").exists()),
            key=lambda p: (p / "state.json").stat().st_mtime,
        )
        if not candidates:
            raise FileNotFoundError(f"no resumable runs found in {runs_root}")
        run_dir = candidates[-1]
    else:
        run_dir = runs_root / resume_arg
        if not (run_dir / "state.json").exists():
            raise FileNotFoundError(f"no state.json for {resume_arg}")
    state = RunState.load(run_dir / "state.json")

    # Compare against the run's own archived config, field by field, rather
    # than against the fingerprint in state.json.
    #
    # The hash is computed *through* `HASH_EXCLUDED`, so broadening that set
    # changes the hash function and every previously stored fingerprint becomes
    # unreachable — a run nobody touched suddenly cannot be resumed. That is not
    # hypothetical: moving `anchor_ladder` into the exclusion set (it measures,
    # it cannot invalidate a shard) locked `ruby-panther` out of being extended.
    # Both sides of a config-to-config comparison are pruned with today's rules,
    # so the check stays correct across changes to its own definition.
    archived = run_dir / "config.yaml"
    if archived.exists():
        # `validate=False`: the archived config may predate rules added since,
        # and it describes a run that already happened either way.
        prev = RunConfig.from_yaml(archived, validate=False)
        diff = cfg.semantic_diff(prev)
        if diff:
            listing = "\n".join(f"    {p}" for p in diff)
            raise RuntimeError(
                f"config differs from {archived} in fields that change the data "
                f"or the model:\n{listing}\n"
                f"refuse to resume; pass --no-resume to start a fresh run."
            )
    elif state.config_hash != cfg.hash():
        # No archived config (a very old run): fall back to the fingerprint.
        raise RuntimeError(
            f"config hash mismatch on resume:\n"
            f"  state.config_hash  = {state.config_hash}\n"
            f"  current cfg.hash() = {cfg.hash()}\n"
            f"refuse to resume; pass --no-resume to start a fresh run."
        )
    return run_dir.name, state


# --------------------------------------------------------------------------- #
# main                                                                         #
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="path to YAML config")
    parser.add_argument("--resume", type=str, help="run-id to resume, or 'latest'")
    parser.add_argument(
        "--no-resume", action="store_true", help="start fresh even if state exists"
    )
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="spawn a tensorboard server at --tb-port. Event files are always "
        "written to runs/<id>/tb/ regardless of this flag.",
    )
    parser.add_argument("--tb-port", type=int, default=6006)
    args = parser.parse_args(argv)

    if args.config is None:
        parser.error("--config required (e.g. config/validation.yaml or config/dry_run.yaml)")
    cfg = RunConfig.from_yaml(args.config)

    run_id, prev_state = _resolve_resume(cfg, args.resume if not args.no_resume else None)
    cfg.run_id = run_id
    run_dir = REPO_ROOT / cfg.runs_root / run_id
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    cfg.to_yaml(run_dir / "config.yaml")

    random.seed(cfg.seed)
    np.random.seed(cfg.seed % (2**32))
    torch.manual_seed(cfg.seed)

    device = _device()

    # Rust subprocesses inherit our environment; this is the whole CoreML plumb.
    if cfg.use_coreml:
        os.environ["ABALONE_USE_COREML"] = "1"
    else:
        os.environ.pop("ABALONE_USE_COREML", None)

    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model = build(cfg.net_preset).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.learning_rate,
        betas=(cfg.train.adam_beta1, cfg.train.adam_beta2),
        eps=cfg.train.adam_eps,
        weight_decay=cfg.train.weight_decay,
    )
    ema = (
        Ema(model, cfg.train.ema.decay, cfg.train.ema.warmup_steps)
        if cfg.train.ema.enabled
        else None
    )
    loss_weights = LossWeights(**asdict(cfg.train.loss_weights))

    workers_n = cfg.self_play.worker_threads or max((os.cpu_count() or 8) - 1, 1)
    _banner(run_id, device, os.getpid(), model, workers_n)
    _dump_config(
        cfg,
        str(cfg.self_play.worker_threads)
        if cfg.self_play.worker_threads is not None
        else f"null (auto: {workers_n})",
    )
    _log(f"run_dir: {run_dir}")

    git_sha = _git_sha()

    if prev_state is None:
        # Fresh run: bootstrap gen_000 from random weights. BN stats are warmed
        # first so the bootstrap ONNX matches the in-memory forward pass.
        _warmup_bn(model, device)
        if ema is not None:
            ema.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
        _save_ckpt(model, optimizer, ema, cfg.net_preset, ckpt_dir / "gen_000.pt")
        export_onnx(model, ckpt_dir / "gen_000.onnx")
        _link_or_copy(ckpt_dir / "gen_000.onnx", ckpt_dir / "best.onnx")
        state = RunState.fresh(
            run_id=run_id,
            config_hash=cfg.hash(),
            git_sha=git_sha,
            handicap_rate=resolve_initial_rate(None, cfg.self_play.handicap_rate),
        )
        state.current_onnx = "checkpoints/gen_000.onnx"
        state.best_onnx = "checkpoints/gen_000.onnx"
        state.save_atomic(run_dir / "state.json")
    else:
        state = prev_state
        plan = plan_resume(state.current_gen, state.current_phase)
        _log(f"resuming from gen {state.current_gen} (phase={state.current_phase})")
        if state.handicap_rate is None:
            _log(
                "state.json predates the curriculum annealer; seeding handicap_rate from "
                f"config ({cfg.self_play.handicap_rate:.2f})"
            )
        state.handicap_rate = resolve_initial_rate(
            state.handicap_rate, cfg.self_play.handicap_rate
        )
        _log(f"curriculum: resuming at handicap_rate {state.handicap_rate:.2f}")
        if git_sha_drift(state.git_sha, git_sha):
            _warn(
                f"git SHA changed since this run was last advanced: "
                f"{state.git_sha[:12]} → {git_sha[:12]}. `config_hash` covers the YAML "
                f"only — if the commit touched the ply cap, the plane layout or the shard "
                f"schema, every shard already in the buffer is silently invalid"
            )
        last_ckpt = ckpt_dir / f"gen_{plan.resume_ckpt_gen:03d}.pt"
        if not last_ckpt.exists():
            raise FileNotFoundError(
                f"resume requires checkpoint {last_ckpt} but it does not exist. Retention "
                f"may have collected it, or the run directory is corrupt. Use --no-resume."
            )
        _load_ckpt(last_ckpt, model, optimizer, ema, device)
        if plan.discard_gen is not None:
            partial = run_dir / "shards" / f"gen_{plan.discard_gen:03d}"
            if partial.exists():
                _log(f"discarding partial shards for gen {plan.discard_gen}")
                shutil.rmtree(partial, ignore_errors=True)
        state.git_sha = git_sha

    # Replay buffer: preload the shards inside the rolling window, plus the
    # holdout generation, which lives outside it.
    buffer = ReplayBuffer(augment=cfg.train.symmetry_augment)
    holdout_gen = cfg.validation.holdout_gen
    window_start = max(state.current_gen - cfg.train.replay_buffer_gens + 1, 1)
    gens_to_load = set(range(window_start, state.current_gen + 1))
    if cfg.validation.enabled and holdout_gen <= state.current_gen:
        gens_to_load.add(holdout_gen)
    for gen in sorted(gens_to_load):
        for shard in find_shards_for_gen(run_dir / "shards", gen):
            buffer.ingest_shard(shard, gen=gen)
    # Re-take every rolling slice a resumed run had already taken. The
    # selection is deterministic in `(seed, gen)` and the marking is idempotent,
    # so this reproduces the *same* rows — without it, "never trained on" would
    # quietly become "never trained on unless the run was ever restarted", and
    # the one clean generalisation measurement in the system would be silently
    # contaminated by a crash. Driven off `state.history` rather than off the
    # config predicate: a generation whose slice was skipped by the
    # minimum-buffer guard was trained on in full and must stay that way.
    held_before = {
        r.gen: r.rolling_holdout_positions
        for r in state.history
        if r.rolling_holdout_positions
    }
    restored = 0
    for gen in sorted(gens_to_load):
        if gen not in held_before:
            continue
        restored += buffer.mark_rolling_holdout(
            gen, fraction=cfg.validation.rolling.fraction, seed=cfg.validation.seed
        )
    if restored:
        _log(f"rolling holdout: {restored} position(s) restored across resumed generations")
    # Re-take the frozen slice for the same reason the rolling ones are re-taken
    # above: selection is deterministic in `(seed, gen)` and idempotent, so this
    # reproduces exactly the rows the original run withheld.
    if cfg.validation.enabled and holdout_gen <= state.current_gen:
        held = buffer.mark_holdout(
            holdout_gen,
            count=cfg.validation.frozen_holdout_positions(),
            seed=cfg.validation.seed,
        )
        _log(
            f"holdout: gen {holdout_gen} frozen ({held} of "
            f"{buffer.chunk_size(holdout_gen)} positions; the rest is trainable)"
        )

    rng = np.random.default_rng(cfg.seed + state.current_gen)
    sp_state: list[subprocess.Popen | None] = [None]
    tb_state: list[subprocess.Popen | None] = [None]
    _setup_signal_handlers(sp_state, tb_state)

    runs_root = REPO_ROOT / cfg.runs_root
    writer = SummaryWriter(str(run_dir / "tb"))
    if args.tensorboard:
        tb_state[0] = _start_tensorboard(runs_root, port=args.tb_port)
    else:
        _log(f"tensorboard: events at {run_dir / 'tb'}/  (pass --tensorboard to serve them)")

    try:
        return _run_outer_loop(
            cfg=cfg,
            state=state,
            model=model,
            optimizer=optimizer,
            ema=ema,
            loss_weights=loss_weights,
            buffer=buffer,
            rng=rng,
            device=device,
            run_dir=run_dir,
            ckpt_dir=ckpt_dir,
            sp_state=sp_state,
            writer=writer,
            git_sha=git_sha,
        )
    finally:
        writer.close()
        tb_proc = tb_state[0]
        if tb_proc is not None and tb_proc.poll() is None:
            try:
                tb_proc.terminate()
                try:
                    tb_proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    tb_proc.kill()
            except Exception:
                pass


def _run_outer_loop(
    *,
    cfg: RunConfig,
    state: RunState,
    model: AbaloneNet,
    optimizer: torch.optim.Optimizer,
    ema: Ema | None,
    loss_weights: LossWeights,
    buffer: ReplayBuffer,
    rng: np.random.Generator,
    device: torch.device,
    run_dir: Path,
    ckpt_dir: Path,
    sp_state: list[subprocess.Popen | None],
    writer: SummaryWriter,
    git_sha: str,
) -> int:
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    sp = cfg.self_play

    while state.current_gen < cfg.gens:
        new_gen = state.current_gen + 1
        print(flush=True)
        _section()
        gen_t = time.time()
        _log("starting", gen=new_gen, total_gens=cfg.gens)

        lr = cfg.train.learning_rate_at(new_gen)
        for group in optimizer.param_groups:
            group["lr"] = lr

        # ---- 1. self-play ----
        state.current_phase = "self_play"
        state.save_atomic(run_dir / "state.json")
        shards_dir = run_dir / "shards" / f"gen_{new_gen:03d}"
        shards_dir.mkdir(parents=True, exist_ok=True)
        threads_used = sp.worker_threads or max((os.cpu_count() or 8) - 1, 1)
        # The *annealed* rate, from state — not `sp.handicap_rate`, which is only
        # the initial condition and is inside `config_hash` (MODEL.md §4).
        handicap_rate = resolve_initial_rate(state.handicap_rate, sp.handicap_rate)
        state.handicap_rate = handicap_rate
        _log(
            f"phase: self-play  ({sp.games_per_gen} games, sims {sp.sims_fast}/{sp.sims_full} "
            f"@ {sp.full_search_rate:.2f}, batch {sp.batch_size}, {threads_used} threads, "
            f"handicap {handicap_rate:.2f})",
            gen=new_gen,
            total_gens=cfg.gens,
        )
        sp_t = time.time()
        sp_log_path = logs_dir / f"gen_{new_gen:03d}_selfplay.log"
        sp_log_file = open(sp_log_path, "w")
        sp_proc = start_self_play(
            model_onnx=run_dir / state.current_onnx,
            out_dir=shards_dir,
            games=sp.games_per_gen,
            sims_fast=sp.sims_fast,
            sims_full=sp.sims_full,
            full_search_rate=sp.full_search_rate,
            c_puct=sp.c_puct,
            batch_size=sp.batch_size,
            virtual_loss=sp.virtual_loss,
            fpu_reduction=sp.fpu_reduction,
            temperature_plies=sp.temperature_plies,
            temperature=sp.temperature,
            opening=sp.opening,
            handicap_rate=handicap_rate,
            handicap_max=sp.handicap_max,
            random_opening_plies=sp.random_opening_plies,
            max_plies=sp.max_plies,
            no_progress_plies=sp.no_progress_plies,
            capture_gamma=sp.capture_gamma,
            dirichlet_alpha=sp.dirichlet_alpha,
            dirichlet_eps=sp.dirichlet_eps,
            shard_games=sp.shard_games_per_file,
            threads=sp.worker_threads,
            seed=(cfg.seed + 1_000_003 * new_gen) & 0xFFFF_FFFF,
            stdout=sp_log_file,
            stderr=subprocess.STDOUT,
        )
        sp_state[0] = sp_proc

        # ---- 2. training (overlaps self-play) ----
        state.current_phase = "training"
        state.save_atomic(run_dir / "state.json")
        train_t = time.time()
        # Decided here, once, against the buffer as it stands *before* this
        # generation's first shard lands: that is exactly the pool training
        # would be left with while the generation is withheld. Generation 2 of
        # a run whose generation 1 is the frozen holdout has nothing else, and
        # must not withhold anything.
        hold_rolling = (
            cfg.validation.is_rolling_holdout_gen(new_gen)
            and buffer.total_size() >= cfg.train.replay_buffer_min_size
        )
        if cfg.validation.is_rolling_holdout_gen(new_gen) and not hold_rolling:
            _log(
                f"rolling holdout: skipped for gen {new_gen} — only {buffer.total_size()} "
                f"trainable positions outside it, below replay_buffer_min_size "
                f"{cfg.train.replay_buffer_min_size}",
                gen=new_gen,
                total_gens=cfg.gens,
            )
        steps_done, mean_metrics, sp_seconds, active_train_secs = _train_phase(
            model=model,
            optimizer=optimizer,
            ema=ema,
            buffer=buffer,
            sp_proc=sp_proc,
            selfplay_log_path=sp_log_path,
            shards_dir=shards_dir,
            new_gen=new_gen,
            cfg=cfg,
            rng=rng,
            device=device,
            loss_weights=loss_weights,
            writer=writer,
            phase_start_time=sp_t,
            hold_rolling=hold_rolling,
            step_offset=tb_step_offset(state.history),
        )
        sp_state[0] = None
        sp_log_file.close()
        train_seconds = time.time() - train_t
        epochs = epochs_over_buffer(steps_done, cfg.train.batch_size, buffer.total_size())
        if mean_metrics is not None:
            rate = steps_done / active_train_secs if active_train_secs > 0 else 0.0
            _log(
                f"training complete: {steps_done} steps in {_fmt_secs(active_train_secs)} "
                f"({rate:.1f} steps/s) @ lr {lr:g}  ·  {_fmt(epochs, 1)} raw epochs over "
                f"{buffer.total_size()} positions  ·  loss {mean_metrics.loss_total:.4f} "
                f"(policy {mean_metrics.loss_policy:.4f}, value {mean_metrics.loss_value:.4f}, "
                f"score {mean_metrics.loss_score:.4f}, cap {mean_metrics.loss_capture_map:.4f})"
                f"  ·  grad {mean_metrics.grad_norm:.2f}",
                gen=new_gen,
                total_gens=cfg.gens,
            )
        if epochs_alarm(epochs, cfg.train.epochs_per_gen_warn):
            _warn(
                f"{_fmt(epochs, 1)} raw passes over the replay buffer in one generation "
                f"(warn above {cfg.train.epochs_per_gen_warn:g}). The buffer is only as "
                f"diverse as the positions in it — D6 augmentation multiplies tensors, not "
                f"positions. Lower train.target_epochs_per_gen / steps_per_gen_max, or raise "
                f"self_play.games_per_gen: self-play is the cheap half",
                gen=new_gen,
                total_gens=cfg.gens,
            )

        # ---- 3. export (EMA weights) ----
        state.current_phase = "export"
        state.save_atomic(run_dir / "state.json")
        new_pt = ckpt_dir / f"gen_{new_gen:03d}.pt"
        new_onnx = ckpt_dir / f"gen_{new_gen:03d}.onnx"
        _save_ckpt(model, optimizer, ema, cfg.net_preset, new_pt)
        exported = _playing_model(model, ema)
        export_onnx(exported, new_onnx)
        _log(
            f"phase: export  →  checkpoints/gen_{new_gen:03d}.onnx "
            f"({new_onnx.stat().st_size / 1048576:.1f} MB, "
            f"{'EMA' if ema is not None else 'raw'} weights)",
            gen=new_gen,
            total_gens=cfg.gens,
        )
        state.current_onnx = f"checkpoints/gen_{new_gen:03d}.onnx"

        # ---- 4a. exported games + data health ----
        games_dir = run_dir / "games" / f"gen_{new_gen:03d}"
        export_result = export_generation(
            shards_dir,
            games_dir,
            run_id=state.run_id,
            gen=new_gen,
            limit=cfg.export.games_per_gen,
        )
        # Over *every* game in the generation, not just the exported subset —
        # `export.games_per_gen` caps what is written, and a controller fed a
        # 20-of-200 subsample would be measuring the export limit.
        health = export_result.summary
        if health is None:
            health = summarise_generation(export_result.games)
        for line in format_generation_report(export_result):
            _log(f"  {line}", gen=new_gen, total_gens=cfg.gens)

        # ---- 4a′. curriculum ratchet (MODEL.md §4) ----
        # Evaluated after self-play, so it applies from the next generation on.
        anneal_stats = AnnealStats.from_summary(
            health, no_progress_plies=sp.no_progress_plies
        )
        decision = decide_handicap_rate(
            handicap_rate, anneal_stats, sp.handicap_anneal, gen=new_gen
        )
        _log(f"curriculum: {decision.message}", gen=new_gen, total_gens=cfg.gens)
        # `state.handicap_rate` is advanced at commit, not here: a crash between
        # the two would otherwise redo this generation's self-play at the new
        # rate, having thrown away the data that justified it.

        # ---- 4b. held-out validation (frozen ruler + rolling holdout) ----
        val_metrics: dict[str, object] = {}
        if should_validate(
            new_gen,
            enabled=cfg.validation.enabled,
            holdout_gen=cfg.validation.holdout_gen,
            every_gens=cfg.validation.every_gens,
        ):
            state.current_phase = "validate"
            state.save_atomic(run_dir / "state.json")
            val_metrics = _validation_phase(
                model=exported,
                buffer=buffer,
                cfg=cfg,
                new_gen=new_gen,
                device=device,
                loss_weights=loss_weights,
            )

        # ---- 5. anchor ladder ----
        rungs: list[LadderRung] = []
        ladder_elo: float | None = None
        if should_run_ladder(
            new_gen,
            cfg.gens,
            cfg.anchor_ladder.every_gens,
            cfg.anchor_ladder.run_on_final_gen,
        ):
            state.current_phase = "ladder"
            state.save_atomic(run_dir / "state.json")
            rungs = _ladder_phase(
                cfg=cfg,
                new_gen=new_gen,
                onnx_path=new_onnx,
                run_dir=run_dir,
                ckpt_dir=ckpt_dir,
            )
            ladder_elo = _report_ladder(rungs, new_gen, cfg.gens)

        # ---- which checkpoint `best.onnx` points at ----
        #
        # Promote the newest generation unless the ladder caught it *confidently*
        # losing to one of its own recent ancestors. Two rules were rejected on
        # the way here, both for reasons this run demonstrated:
        #
        # "highest ladder Elo" — what this used to do — compares numbers with
        # different bases. Generation 12 recorded +658 as a mean over floor
        # rungs that were all clamped; generation 13 recorded +256 as a mean
        # over trailing rungs. Neither is wrong, they simply are not the same
        # measurement, and comparing them left `best.onnx` on generation 12
        # while generation 13 was beating it head to head.
        #
        # "did it beat the previous generation" is unusable at this resolution.
        # Generation 13 scored 0.547 against generation 12 with a 95% interval
        # of about ±0.17 — a coin flip, and the published model would hop about
        # on noise.
        #
        # So: improvement is the expectation and does not need certifying every
        # generation; a *resolved* regression is the thing worth acting on.
        if rungs:
            regressions = resolved_regressions(rungs)
            if regressions:
                worst = min(regressions, key=lambda r: r.score)
                _warn(
                    f"generation {new_gen} is measurably weaker than {worst.label} "
                    f"(score {worst.score:.3f} ± {worst.result.score_a_stderr:.3f}, "
                    f"upper 95% bound below 0.5). Keeping best.onnx at gen "
                    f"{state.best_gen}",
                    gen=new_gen,
                    total_gens=cfg.gens,
                )
            else:
                previous = (
                    f"gen {state.best_gen}" if state.best_gen else "gen 000 (none yet)"
                )
                state.best_gen = new_gen
                # Informational only — the promotion decision above does not
                # read it. Kept so `report.py` can show what the ladder said at
                # the generation that was promoted.
                state.best_elo = ladder_elo
                state.best_onnx = f"checkpoints/gen_{new_gen:03d}.onnx"
                _link_or_copy(new_onnx, ckpt_dir / "best.onnx")
                _log(
                    f"  best checkpoint → gen {new_gen} (no resolved regression "
                    f"across {len(rungs)} rungs), was {previous}",
                    gen=new_gen,
                    total_gens=cfg.gens,
                )
                if cfg.export.web_export:
                    # `{run_id}` lets a throwaway run keep its artifact to
                    # itself. Without it every config pointed at the same
                    # `web/public/models/best.onnx`, so a six-game dry run
                    # silently replaced the web app's model with a network
                    # trained for ninety seconds.
                    web_dst = REPO_ROOT / cfg.web_export_path.format(run_id=cfg.run_id)
                    web_dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(new_onnx, web_dst)
                    _log(f"  web export → {web_dst}", gen=new_gen, total_gens=cfg.gens)

        # ---- 6. commit ----
        gen_shards = find_shards_for_gen(run_dir / "shards", new_gen)
        gen_positions = buffer.chunk_size(new_gen)
        gen_seconds = time.time() - gen_t

        # Every measurement, under its own namespace, in one dict. Nothing here
        # enumerates metric names: a key added in `model.validate`,
        # `model.export_game` or `model.replay_buffer` lands in metrics.jsonl
        # and TensorBoard without this file being edited.
        metrics: dict[str, object] = {}
        metrics.update(selfplay_metrics(health))
        metrics.update(
            buffer.buffer_metrics(
                gen=new_gen, steps=steps_done, batch_size=cfg.train.batch_size
            )
        )
        metrics.update(
            namespaced(
                "train",
                {
                    "steps": steps_done,
                    "learning_rate": lr,
                    "epochs_over_buffer": epochs,
                    **(asdict_metrics(mean_metrics)),
                },
            )
        )
        metrics.update(val_metrics)
        metrics.update(
            namespaced(
                "curriculum",
                {
                    "handicap_rate": decision.previous,
                    "handicap_rate_next": decision.rate,
                    "stepped": float(decision.stepped),
                    "unseeded_games": anneal_stats.unseeded_games,
                    "natural_termination_rate": anneal_stats.natural_termination_rate,
                },
            )
        )
        if rungs:
            metrics.update(ladder_summary(rungs, new_gen))
        metrics.update(
            namespaced(
                "perf",
                {
                    "self_play_seconds": sp_seconds,
                    "train_seconds": train_seconds,
                    "train_active_seconds": active_train_secs,
                    "gen_seconds": gen_seconds,
                    "shard_count": len(gen_shards),
                },
            )
        )

        record = GenRecord(
            gen=new_gen,
            train_steps=steps_done,
            train_loss_total=_opt(mean_metrics, "loss_total"),
            train_loss_policy=_opt(mean_metrics, "loss_policy"),
            train_loss_value=_opt(mean_metrics, "loss_value"),
            train_loss_score=_opt(mean_metrics, "loss_score"),
            train_loss_capture_map=_opt(mean_metrics, "loss_capture_map"),
            train_grad_norm=_opt(mean_metrics, "grad_norm"),
            train_epochs_over_buffer=_finite(epochs),
            learning_rate=lr,
            rolling_holdout_positions=buffer.rolling_holdout_size(new_gen),
            val_loss_total=_num(metrics, f"{VAL_FROZEN_PREFIX}loss_total"),
            val_policy_top1=_num(metrics, f"{VAL_FROZEN_PREFIX}policy_top1_agreement"),
            val_policy_entropy_ratio=_num(
                metrics, f"{VAL_FROZEN_PREFIX}data_policy_entropy_ratio"
            ),
            val_value_ce=_num(metrics, f"{VAL_FROZEN_PREFIX}value_ce"),
            val_value_accuracy=_num(metrics, f"{VAL_FROZEN_PREFIX}value_accuracy"),
            val_rolling_loss_total=_num(metrics, f"{VAL_ROLLING_PREFIX}loss_total"),
            val_rolling_policy_top1=_num(
                metrics, f"{VAL_ROLLING_PREFIX}policy_top1_agreement"
            ),
            val_rolling_value_ce=_num(metrics, f"{VAL_ROLLING_PREFIX}value_ce"),
            val_rolling_value_accuracy=_num(
                metrics, f"{VAL_ROLLING_PREFIX}value_accuracy"
            ),
            decisive_rate=_num(health, "decisive_rate"),
            mean_plies=_num(health, "mean_plies"),
            mean_abs_score_diff=_num(health, "mean_abs_score_diff"),
            policy_target_entropy=_num(health, "policy_target_entropy"),
            policy_uniform_entropy=_num(health, "policy_uniform_entropy"),
            policy_entropy_gap=_num(health, "policy_entropy_gap"),
            captures_per_100_plies=_num(health, "captures_per_100_plies"),
            handicap_rate=decision.previous,
            handicap_rate_next=decision.rate,
            unseeded_games=anneal_stats.unseeded_games,
            # `None`, not `nan`: "not measured" and "measured as zero" are
            # different facts, and only one of them belongs in a plot.
            natural_termination_rate=_finite(anneal_stats.natural_termination_rate),
            ladder_elo=ladder_elo,
            ladder_elo_ci95_lo=_num(metrics, "ladder/elo_mean_ci95_lo"),
            ladder_elo_ci95_hi=_num(metrics, "ladder/elo_mean_ci95_hi"),
            ladder_clamped_fraction=_num(metrics, "ladder/clamped_fraction"),
            self_play_seconds=sp_seconds,
            train_seconds=train_seconds,
            gen_seconds=gen_seconds,
            buffer_size=buffer.total_size(),
            shard_count=len(gen_shards),
            positions=gen_positions,
        )
        state.append_history(record)

        # metrics.jsonl carries strictly more than `state.history`: every metric
        # every measurement module emitted, plus the non-scalar payloads
        # (`value_calibration`, the handicap histogram, every ladder rung).
        entry: dict[str, object] = {
            "gen": new_gen,
            "git_sha": git_sha,
            **metrics,
            "handicap_distribution": health.get("handicap_distribution"),
            "handicap_anneal_reason": decision.reason,
            "handicap_anneal_stepped": decision.stepped,
        }
        if rungs:
            entry["ladder"] = [
                {
                    "opponent": r.opponent,
                    "kind": r.kind,
                    **{k: v for k, v in r.result.raw.items() if k != "per_game"},
                }
                for r in rungs
            ]
        with (run_dir / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(entry, default=str) + "\n")

        _write_tb(writer, metrics, new_gen)

        # Freeze the holdout generation once it has been produced *and*
        # trained on. From here it is validation-only and survives eviction.
        if (
            cfg.validation.enabled
            and new_gen == cfg.validation.holdout_gen
            and buffer.chunk_size(new_gen) > 0
        ):
            held = buffer.mark_holdout(
                new_gen,
                count=cfg.validation.frozen_holdout_positions(),
                seed=cfg.validation.seed,
            )
            _log(
                f"holdout: gen {new_gen} frozen as the validation set "
                f"({held} of {buffer.chunk_size(new_gen)} positions, by whole game); "
                f"it will not be trained on again. The remaining "
                f"{buffer.trainable_size(new_gen)} stay in the training pool",
                gen=new_gen,
                total_gens=cfg.gens,
            )

        _apply_retention(run_dir, cfg, state, new_gen)

        state.current_gen = new_gen
        state.current_phase = "complete"
        state.git_sha = git_sha
        # The ratchet lands in the same atomic write as the generation it was
        # measured from, so the rate and the completed-generation count can
        # never disagree.
        state.handicap_rate = decision.rate
        state.save_atomic(run_dir / "state.json")

        # ---- the per-generation summary line ----
        # The one place that names specific metrics on purpose: a human reads it.
        ratio = health.get("policy_entropy_ratio")
        gap = health.get("policy_entropy_gap")
        entropy_str = (
            f"entropy {_fmt(health.get('policy_target_entropy'))}"
            f"/{_fmt(health.get('policy_uniform_entropy'))}"
            f" = {_fmt(ratio)} (gap {_fmt(gap)})"
        )
        rolling_loss = _num(metrics, f"{VAL_ROLLING_PREFIX}loss_total")
        _log(
            f"complete in {_fmt_secs(gen_seconds)}  ·  "
            f"loss {_fmt(record.train_loss_total, 4)}"
            + (f" (rolling {_fmt(rolling_loss, 4)})" if rolling_loss is not None else "")
            + f"  ·  decisive {_fmt(health.get('decisive_rate'), 2)}  ·  "
            f"plies {_fmt(health.get('mean_plies'), 0)}  ·  "
            f"nat.term {_fmt(anneal_stats.natural_termination_rate, 2)}  ·  "
            f"handicap {decision.previous:.2f}"
            + (f"→{decision.rate:.2f}" if decision.stepped else "")
            + f"  ·  {entropy_str}"
            + (f"  ·  {_fmt(epochs, 1)} epochs" if not math.isnan(epochs) else "")
            + (f"  ·  elo {ladder_elo:+.0f}" if ladder_elo is not None else ""),
            gen=new_gen,
            total_gens=cfg.gens,
        )
        if entropy_ratio_alarm(ratio, cfg.validation.entropy_ratio_warn, new_gen):
            _warn(
                f"generation {new_gen}'s policy targets are {_fmt(ratio)} of "
                f"ln(mean legal moves) — MCTS visit distributions are ~uniform, so this "
                f"generation's data teaches the policy head nothing. Raise sims_full or "
                f"check the search before spending more compute",
                gen=new_gen,
                total_gens=cfg.gens,
            )
        if overfit_alarm(
            rolling_loss, record.train_loss_total, cfg.validation.overfit_warn_delta
        ):
            _warn(
                f"val_rolling loss {_fmt(rolling_loss, 4)} exceeds this generation's mean "
                f"training loss {_fmt(record.train_loss_total, 4)} by more than "
                f"{cfg.validation.overfit_warn_delta} — the rolling holdout is the same "
                f"distribution as the training data, so the gap is memorisation of the "
                f"replay buffer. Cut train.target_epochs_per_gen, widen "
                f"train.replay_buffer_gens, or make more games per generation",
                gen=new_gen,
                total_gens=cfg.gens,
            )

    _section()
    if state.best_gen:
        _log(
            f"all generations complete. best.onnx → gen {state.best_gen} "
            f"({state.best_onnx}), the newest generation with no resolved "
            f"regression on its ladder."
        )
    else:
        _log(
            "all generations complete. No ladder ever ran, so no checkpoint was "
            f"promoted; the latest export is {state.current_onnx}."
        )
    return 0


def _opt(metrics: StepMetrics | None, name: str) -> float | None:
    return float(getattr(metrics, name)) if metrics is not None else None


def asdict_metrics(metrics: StepMetrics | None) -> dict[str, float]:
    """`StepMetrics` as a plain dict, or `{}` when a generation took no steps.

    Generic on purpose: `StepMetrics` gaining a field must show up in
    `metrics.jsonl` without anyone editing this module.
    """
    return dict(metrics.as_dict()) if metrics is not None else {}


def _finite(value: float | None) -> float | None:
    """`None` for anything that is not a real number, so `state.json` carries a
    JSON `null` rather than a `NaN` token no strict parser will read back."""
    if value is None:
        return None
    v = float(value)
    return None if math.isnan(v) else v


def _num(d: dict | None, key: str) -> float | None:
    """A metric as a plain float, or None when absent/non-numeric."""
    if not d or key not in d:
        return None
    v = d[key]
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return float(v)


def _write_tb(writer: SummaryWriter | None, metrics: Mapping[str, object], gen: int) -> None:
    """Every scalar in `metrics`, under the namespace it arrived with.

    Deliberately not a whitelist. TensorBoard groups by the part before the
    first `/`, which is exactly what the `selfplay/`, `buffer/`, `train/`,
    `val_frozen/`, `val_rolling/`, `curriculum/`, `ladder/` and `perf/`
    prefixes are for — so a metric added anywhere downstream appears in the
    right panel with no change here.
    """
    if writer is None:
        return
    for name, value in scalar_metrics(dict(metrics)).items():
        if not math.isnan(value):
            writer.add_scalar(name, value, gen)
    writer.flush()


if __name__ == "__main__":
    sys.exit(main())
