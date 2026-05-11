"""The outer training loop. Python drives:
  1. Self-play subprocess (`selfplay-batch`) writes shards.
  2. Trainer ingests shards as they arrive, runs SGD steps.
  3. New `.pt` checkpoint and `.onnx` export.
  4. Gating match against the current best (`eval-match`), promote on win.
  5. (Periodically) heuristic anchor + random sanity matches.

Resume: granularity is one generation. After each phase we update
`state.json` (atomically). On startup we read state.json, find the
highest fully-completed gen N, and resume at gen N+1.

CLI:
    uv run python -m model.train_loop --config config/standard.yaml
    uv run python -m model.train_loop --resume bold-otter-...
    uv run python -m model.train_loop --resume latest
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from model.abalone_net import (
    BOARD_H,
    BOARD_W,
    CHANNELS,
    INPUT_CHANNELS,
    NUM_RES_BLOCKS,
    AbaloneNet,
)
from model.config import RunConfig
from model.encoder import VALID_CELL_MASK
from model.eval import run_eval_match, start_self_play
from model.export_onnx import export as export_onnx
from model.replay_buffer import ReplayBuffer, find_shards_for_gen
from model.run_id import generate_unique
from model.state import GenRecord, RunState
from model.train_step import StepMetrics, train_step, value_target_blend_weight

REPO_ROOT = Path(__file__).resolve().parents[1]

# How often (wall-clock seconds) to emit an in-gen progress line during
# the long self-play / training phase. Each gen produces ~ self_play_secs
# / PROGRESS_INTERVAL_S lines.
PROGRESS_INTERVAL_S = 30.0

# Evaluator names recognized in `SelfPlayConfig.evaluator_schedule`.
# Kept in sync with the Rust `Evaluator` enum in `selfplay_batch.rs`.
VALID_EVALUATORS = {"model", "heuristic"}
DEFAULT_EVALUATOR = "model"


def _select_evaluator(new_gen: int, schedule: dict) -> str:
    """Pick the leaf evaluator for `new_gen` from the schedule (a
    `{until_gen: evaluator_name}` dict). Keys are processed in
    ascending order; the first key >= `new_gen` wins. Falls back to
    "model" if nothing matches. Validates names so a typo surfaces
    here, not deep in a Rust subprocess panic."""
    for until_gen in sorted((schedule or {}).keys()):
        try:
            until = int(until_gen)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"evaluator_schedule keys must be int (until_gen); got {until_gen!r}"
            ) from e
        name = str(schedule[until_gen])
        if name not in VALID_EVALUATORS:
            raise ValueError(
                f"unknown evaluator {name!r} in schedule; "
                f"valid: {sorted(VALID_EVALUATORS)}"
            )
        if new_gen <= until:
            return name
    return DEFAULT_EVALUATOR


# ---------- structured logging helpers --------------------------------------


def _log(msg: str, *, gen: int | None = None, total_gens: int | None = None) -> None:
    """Print one timestamped status line, flushed immediately so
    `tail -f` sees it without buffering. Optional `[gen N/M]` prefix
    keeps multi-gen output easy to grep."""
    ts = time.strftime("%H:%M:%S")
    prefix = f"[{ts}] "
    if gen is not None and total_gens is not None:
        prefix += f"[gen {gen:>2}/{total_gens}] "
    print(f"{prefix}{msg}", flush=True)


def _git_short_sha() -> str:
    """Best-effort git commit SHA prefix; 'detached' if outside a repo."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or "detached"
    except (subprocess.SubprocessError, FileNotFoundError):
        return "detached"


def _banner(
    run_id: str,
    device: torch.device,
    pid: int,
    model: AbaloneNet,
    worker_threads: int,
) -> None:
    """Print a self-describing run header. The marbles in the title are
    the two players in Abalone — black ●, white ○. Layout is fixed-width
    so it lines up in any monospace font."""
    line = "═" * 73
    title = "  ●○●○●   Abalone — AlphaZero training pipeline   ●○●○●"
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    n_params = model.num_parameters() / 1e6
    arch = f"AbaloneNet ({NUM_RES_BLOCKS}×{CHANNELS}, {n_params:.1f}M params)"
    torch_v = torch.__version__

    print(line, flush=True)
    print(title, flush=True)
    print(line, flush=True)
    print(flush=True)
    print(f"  Run:       {run_id}", flush=True)
    print(f"  Started:   {started}", flush=True)
    print(
        f"  Device:    {device!s:<13}  Workers:  {worker_threads} threads",
        flush=True,
    )
    print(
        f"  PID:       {pid!s:<13}  Commit:   {_git_short_sha()}",
        flush=True,
    )
    print(
        f"  Torch:     {torch_v:<13}  Model:    {arch}",
        flush=True,
    )
    print(flush=True)
    print(line, flush=True)
    print(flush=True)


def _section() -> None:
    print("─" * 71, flush=True)


def _dump_config(cfg: RunConfig, worker_threads_resolved: str) -> None:
    """Print the resolved config as a structured block, so the run log
    is self-describing — readers don't have to cross-reference config.yaml."""
    sp = cfg.self_play
    tr = cfg.train
    ev = cfg.eval
    print("[config]", flush=True)
    print(f"  gens:                  {cfg.gens}", flush=True)
    print(f"  seed:                  {cfg.seed}", flush=True)
    print(f"  runs_root:             {cfg.runs_root}", flush=True)
    print(
        f"  inference backend:     {'CoreML (ANE/GPU)' if cfg.use_coreml else 'ORT CPU'}",
        flush=True,
    )
    print("  self_play:", flush=True)
    print(f"    games_per_gen:       {sp.games_per_gen}", flush=True)
    print(f"    simulations_per_move:{sp.simulations_per_move}", flush=True)
    print(f"    c_puct:              {sp.c_puct}", flush=True)
    print(f"    temperature_plies:   {sp.temperature_plies}", flush=True)
    print(f"    temperature:         {sp.temperature}", flush=True)
    print(f"    dirichlet_alpha:     {sp.dirichlet_alpha}", flush=True)
    print(f"    dirichlet_eps:       {sp.dirichlet_eps}", flush=True)
    print(f"    shard_games_per_file:{sp.shard_games_per_file}", flush=True)
    print(f"    worker_threads:      {worker_threads_resolved}", flush=True)
    if sp.evaluator_schedule:
        rules = ", ".join(
            f"gen<={k}→{sp.evaluator_schedule[k]}"
            for k in sorted(sp.evaluator_schedule.keys())
        )
        print(f"    evaluator_schedule:  {rules}, then model", flush=True)
    else:
        print("    evaluator_schedule:  (always model)", flush=True)
    print("  train:", flush=True)
    steps_max_str = (
        str(tr.steps_per_gen_max) if tr.steps_per_gen_max is not None else "(none)"
    )
    print(
        f"    steps_per_gen:       min={tr.steps_per_gen_min}, max={steps_max_str}",
        flush=True,
    )
    print(f"    batch_size:          {tr.batch_size}", flush=True)
    print(f"    learning_rate:       {tr.learning_rate}", flush=True)
    print(f"    weight_decay:        {tr.weight_decay}", flush=True)
    print(f"    value_loss_weight:   {tr.value_loss_weight}", flush=True)
    print(
        f"    value_target_blend:  {tr.value_target_blend_start} → "
        f"{tr.value_target_blend_end} by gen {tr.value_target_blend_done_by_gen}",
        flush=True,
    )
    print(f"    replay_buffer_gens:  {tr.replay_buffer_gens}", flush=True)
    print(f"    replay_buffer_min:   {tr.replay_buffer_min_size}", flush=True)
    print(f"    symmetry_augment:    {str(tr.symmetry_augment).lower()}", flush=True)
    print("  eval:", flush=True)
    print(
        f"    gate:        every {ev.gate_every_gens} gen,   "
        f"{ev.gate_games} games, {ev.gate_simulations} sims/move, threshold {ev.gate_threshold}",
        flush=True,
    )
    print(
        f"    heuristic:   every {ev.heuristic_every_gens} gens,  "
        f"{ev.heuristic_games} games, {ev.heuristic_simulations} sims/move",
        flush=True,
    )
    print(
        f"    random:      every {ev.random_every_gens} gens,  "
        f"{ev.random_games} games, {ev.random_simulations} sims/move",
        flush=True,
    )
    print(flush=True)


def _fmt_secs(s: float) -> str:
    if s < 60:
        return f"{s:.0f}s"
    if s < 3600:
        return f"{s/60:.1f}m"
    return f"{s/3600:.2f}h"


def _count_complete_shards(gen_dir: Path) -> int:
    """Number of fully-written shard files in `gen_dir`. Atomic-rename
    means we only count `.parquet` (not `.parquet.tmp`)."""
    if not gen_dir.exists():
        return 0
    return sum(1 for _ in gen_dir.glob("*.parquet"))


def _count_completed_games(selfplay_log_path: Path) -> tuple[int, float]:
    """Parse the self-play subprocess log file for completed-game
    records. Each finished game produces one line of the form
        `  [tN] game M: P plies, final=Wins(Black)|Wins(White)|Draw`
    written by a single `eprintln!` in `selfplay_batch.rs`. The
    "] game M: P plies, final=" substring is the contract.

    Called from `emit_progress` at the main progress cadence (~30s) —
    NOT a hot loop. We re-read the whole file each call; even at 200
    games × ~50 bytes/line = ~10 KB it's nothing. Defensive parsing
    skips lines that don't match instead of erroring, so panics or
    other unrelated output don't break the count.

    Returns `(games_done, avg_plies_per_game)`."""
    if not selfplay_log_path.exists():
        return 0, 0.0
    games = 0
    total_plies = 0
    try:
        with selfplay_log_path.open() as f:
            for line in f:
                # Two stable anchor substrings. The plies value lives
                # between "] game M: " and " plies, final=".
                gm = line.find("] game ")
                if gm == -1:
                    continue
                pf = line.find(" plies, final=", gm)
                if pf == -1:
                    continue
                # The integer ends at pf and starts after a colon-space.
                end = pf
                start = end
                while start > 0 and line[start - 1].isdigit():
                    start -= 1
                if start >= end:
                    continue
                try:
                    plies = int(line[start:end])
                except ValueError:
                    continue
                games += 1
                total_plies += plies
    except OSError:
        return 0, 0.0
    avg = total_plies / games if games > 0 else 0.0
    return games, avg


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _resolve_run_dir(cfg: RunConfig, run_id: str) -> Path:
    return REPO_ROOT / cfg.runs_root / run_id


def _save_ckpt(model: AbaloneNet, optimizer: torch.optim.Optimizer, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        tmp,
    )
    os.replace(tmp, path)


def _load_ckpt(path: Path, model: AbaloneNet, optimizer: torch.optim.Optimizer) -> None:
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])


def _link_or_copy(src: Path, dst: Path) -> None:
    """Best-effort symlink; falls back to copy if symlinks unsupported."""
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src.name)  # relative — same directory
    except (OSError, NotImplementedError):
        shutil.copyfile(src, dst)


def _retain_checkpoints(run_dir: Path, retention: dict, current_gen: int) -> None:
    """Drop old .pt and .onnx files according to retention policy."""
    ckpt_dir = run_dir / "checkpoints"
    keep_pt = retention["keep_last_checkpoints"]
    keep_onnx = retention["keep_last_onnx"]
    for ext, keep in [("pt", keep_pt), ("onnx", keep_onnx)]:
        files = sorted(ckpt_dir.glob(f"gen_*.{ext}"))
        for f in files[: max(0, len(files) - keep)]:
            # Don't delete `best.onnx` (a symlink) or files referenced by it.
            try:
                target = (ckpt_dir / "best.onnx").resolve()
            except OSError:
                target = None
            if f.resolve() != target:
                f.unlink()


def _warmup_bn(model: AbaloneNet, device: torch.device, batches: int = 8) -> None:
    """Populate BN running stats so the bootstrap ONNX produces sane
    activations. PyTorch's default init leaves running_mean=0, running_var=1
    — fine in many cases, but BN's normalization then differs from what
    `model.train()` would have computed, so the first self-play game runs
    with a model whose forward differs from any actual training-mode pass.

    We synthesize small batches resembling real positions (random marbles
    on valid cells, a few constant feature planes) and forward in train()
    mode so the running stats track these activations."""
    was_training = model.training
    model.train()
    mask = torch.from_numpy(VALID_CELL_MASK).to(device)  # (9, 9)
    with torch.no_grad():
        for _ in range(batches):
            x = torch.zeros(32, INPUT_CHANNELS, BOARD_H, BOARD_W, device=device)
            # Plane 5 is the valid-cell mask, always 1 on the 61 valid cells.
            x[:, 5] = mask
            # Planes 0/1: random marbles confined to valid cells.
            own = (torch.rand(32, BOARD_H, BOARD_W, device=device) < 0.2).float() * mask
            opp = (torch.rand(32, BOARD_H, BOARD_W, device=device) < 0.2).float() * mask
            x[:, 0] = own
            x[:, 1] = opp
            # Planes 2/3: capture counts as small constants per sample.
            caps_a = torch.rand(32, 1, 1, device=device) * (3.0 / 6.0)
            caps_b = torch.rand(32, 1, 1, device=device) * (3.0 / 6.0)
            x[:, 2] = caps_a.expand(32, BOARD_H, BOARD_W) * mask
            x[:, 3] = caps_b.expand(32, BOARD_H, BOARD_W) * mask
            # Plane 4: ply / 400, sampled uniformly from [0, 1].
            x[:, 4] = (torch.rand(32, 1, 1, device=device).expand(32, BOARD_H, BOARD_W) * mask)
            model(x)
    if not was_training:
        model.eval()


def _retain_shards(run_dir: Path, retention: dict, current_gen: int) -> None:
    """Drop shard directories below the retention threshold."""
    shards_root = run_dir / "shards"
    if not shards_root.exists():
        return
    threshold = current_gen - retention["keep_last_shard_gens"]
    for d in shards_root.glob("gen_*"):
        try:
            n = int(d.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        if n < threshold:
            shutil.rmtree(d, ignore_errors=True)


def _train_phase(
    *,
    model: AbaloneNet,
    optimizer: torch.optim.Optimizer,
    buffer: ReplayBuffer,
    sp_proc: subprocess.Popen | None,
    selfplay_log_path: Path,
    shards_dir: Path,
    new_gen: int,
    cfg: RunConfig,
    rng: np.random.Generator,
    device: torch.device,
    writer: SummaryWriter | None = None,
    phase_start_time: float | None = None,
) -> tuple[int, StepMetrics | None, float, float]:
    """Train while self-play runs (if `sp_proc` is alive). The exit
    rule is dynamic:

      - Always do at least `train.steps_per_gen_min` SGD steps.
      - If `steps_per_gen_max` is set and self-play is *still running*
        when we hit the min, keep training (consuming otherwise-idle
        wall-clock) up to `steps_per_gen_max` total, then stop.
      - If self-play exits first and we're past the min, stop.

    Returns (steps_done, mean_metrics, self_play_seconds,
    active_train_seconds). `mean_metrics` is the running mean over all
    SGD steps. If `writer` is provided, per-step scalars go to the
    `train/step_*` TB tags. Periodic progress lines fire at
    `PROGRESS_INTERVAL_S` cadence."""

    z_w = value_target_blend_weight(
        new_gen,
        cfg.train.value_target_blend_start,
        cfg.train.value_target_blend_end,
        cfg.train.value_target_blend_done_by_gen,
    )

    min_steps = cfg.train.steps_per_gen_min
    # If max is unset, behave like the old fixed-step mode.
    max_steps = cfg.train.steps_per_gen_max or min_steps
    if max_steps < min_steps:
        max_steps = min_steps
    # TB global step uses the min as the per-gen budget so plots align
    # across gens even when actual step counts vary.
    step_offset = (new_gen - 1) * min_steps

    steps_done = 0
    sum_loss_total = 0.0
    sum_loss_policy = 0.0
    sum_loss_value = 0.0
    sum_grad_norm = 0.0
    # For per-status-line running means (reset after each progress emit).
    win_loss_total = 0.0
    win_grad_norm = 0.0
    win_steps = 0
    seen_files: set[Path] = set()

    target_games = cfg.self_play.games_per_gen
    games_per_shard = max(cfg.self_play.shard_games_per_file, 1)
    target_shards = (target_games + games_per_shard - 1) // games_per_shard
    phase_start = phase_start_time if phase_start_time is not None else time.time()
    last_progress_log = phase_start
    sp_seconds: float | None = None  # set when sp_proc finishes
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
                # Partial files mid-write may fail to read. Atomic-rename
                # on the Rust side makes this rare, but we still tolerate
                # it. Suppress the noise — we'll retry next poll.
                pass
        return ingested

    def emit_progress() -> None:
        nonlocal win_loss_total, win_grad_norm, win_steps
        sp_alive = sp_proc is not None and sp_proc.poll() is None
        # Two distinct signals:
        #  - games_done: parsed from the selfplay log; smooth per-game.
        #    Tells you self-play is alive and making moves.
        #  - shards_done: count of `.parquet` files (atomic-renamed);
        #    lumpy (rotates every shard_games_per_file games per thread)
        #    but this is what training can actually ingest.
        games_done, _ = _count_completed_games(selfplay_log_path)
        shards_done = _count_complete_shards(shards_dir)
        if sp_alive:
            sp_str = (
                f"self-play {games_done:>4}/{target_games} "
                f"({100*games_done/max(target_games,1):4.1f}%)"
            )
        else:
            sp_str = f"self-play {games_done:>4}/{target_games} (done)"
        shard_str = f"shards {shards_done:>3}/{target_shards}"
        # Show steps_done against the upper budget. The `+` mark
        # indicates we're already past the min and into the "extra
        # training while self-play runs" tail.
        train_str = f"train {steps_done:>4}/{max_steps}"
        if min_steps < max_steps and steps_done > min_steps:
            train_str += "+"
        if win_steps > 0:
            loss_str = f"loss={win_loss_total/win_steps:.4f}  grad={win_grad_norm/win_steps:.2f}"
        elif steps_done == 0:
            loss_str = "(waiting for buffer)"
        else:
            loss_str = "(no new steps this window)"
        _log(
            f"  progress  ·  {sp_str}  ·  {shard_str}  ·  "
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

        # Self-play just exited? Emit the one-line summary then continue
        # the training-only tail.
        if (
            not sp_done_announced
            and sp_proc is not None
            and sp_proc.poll() is not None
        ):
            sp_seconds = time.time() - phase_start
            sp_done_announced = True
            # Don't bail on nonzero yet — we still want to drain any final
            # shards. We'll raise after wait() below.
            if sp_proc.returncode == 0:
                games_done, avg_plies = _count_completed_games(selfplay_log_path)
                _log(
                    f"self-play complete in {_fmt_secs(sp_seconds)}  ·  "
                    f"{games_done} games  ·  avg {avg_plies:.0f} plies/game",
                    gen=new_gen,
                    total_gens=cfg.gens,
                )
                # Only announce a training "drain" if we still owe steps.
                if steps_done < min_steps:
                    _log(
                        f"phase: training  (drain to {min_steps} SGD steps, "
                        f"batch {cfg.train.batch_size})",
                        gen=new_gen,
                        total_gens=cfg.gens,
                    )

        # Periodic progress line.
        now = time.time()
        if now - last_progress_log >= PROGRESS_INTERVAL_S:
            emit_progress()
            last_progress_log = now

        # Evict beyond replay window.
        threshold = new_gen - cfg.train.replay_buffer_gens + 1
        if threshold > 0:
            buffer.evict_below(threshold)

        # Stop conditions:
        #  - Always stop at the hard cap.
        #  - Once we've done the min, stop as soon as self-play is done.
        sp_alive_now = sp_proc is not None and sp_proc.poll() is None
        if steps_done >= max_steps:
            break
        if steps_done >= min_steps and not sp_alive_now:
            break

        if buffer.total_size() < cfg.train.replay_buffer_min_size:
            if not sp_alive_now:
                # Self-play done and we still don't have enough data.
                # Sample what we have and proceed; if buffer is empty, bail.
                if buffer.total_size() == 0:
                    raise RuntimeError("self-play produced no data")
                # Continue with reduced buffer.
            else:
                time.sleep(poll_interval)
                continue

        # One small training chunk per poll.
        for _ in range(min(8, max_steps - steps_done)):
            batch = buffer.sample(cfg.train.batch_size, rng)
            m = train_step(
                model,
                optimizer,
                batch,
                device=device,
                value_loss_weight=cfg.train.value_loss_weight,
                z_weight=z_w,
            )
            sum_loss_total += m.loss_total
            sum_loss_policy += m.loss_policy
            sum_loss_value += m.loss_value
            sum_grad_norm += m.grad_norm
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
                writer.add_scalar("train/step_grad_norm", m.grad_norm, gs)
            if steps_done >= max_steps:
                break

        # Yield to self-play subprocess if it's still running.
        if sp_proc is not None and sp_proc.poll() is None:
            time.sleep(0)  # cooperative

    # If we exited training (either by hitting the cap or finishing
    # the min after sp completed) while self-play is still running, we
    # need to wait for it — BUT keep emitting progress lines so the log
    # tail doesn't go silent. Previously we just called sp_proc.wait()
    # here, which blocked silently for the rest of self-play.
    if sp_proc is not None:
        while sp_proc.poll() is None:
            ingest_new()  # data ingested now seeds next gen's buffer
            now = time.time()
            if now - last_progress_log >= PROGRESS_INTERVAL_S:
                emit_progress()
                last_progress_log = now
            time.sleep(poll_interval)
        if sp_proc.returncode != 0:
            raise RuntimeError(
                f"selfplay-batch exited with non-zero status {sp_proc.returncode}; "
                f"see runs/<id>/logs/gen_NNN_selfplay.log for details."
            )
        # Drain final shards: trainer may have exited steps loop before
        # the last shard rotated to disk.
        ingest_new()
        if not sp_done_announced:
            sp_seconds = time.time() - phase_start
            games_done, avg_plies = _count_completed_games(selfplay_log_path)
            _log(
                f"self-play complete in {_fmt_secs(sp_seconds)}  ·  "
                f"{games_done} games  ·  avg {avg_plies:.0f} plies/game",
                gen=new_gen,
                total_gens=cfg.gens,
            )

    if steps_done == 0:
        return 0, None, sp_seconds or 0.0, 0.0
    mean_metrics = StepMetrics(
        loss_total=sum_loss_total / steps_done,
        loss_policy=sum_loss_policy / steps_done,
        loss_value=sum_loss_value / steps_done,
        grad_norm=sum_grad_norm / steps_done,
    )
    # `active_train_seconds` is the wall-clock from the first SGD step to
    # the last — excludes the leading "waiting on buffer" stretch so the
    # steps/sec rate reflects real training throughput.
    active_train = (
        (last_step_time - first_step_time)
        if first_step_time is not None and last_step_time is not None
        else 0.0
    )
    return (
        steps_done,
        mean_metrics,
        sp_seconds or (time.time() - phase_start),
        active_train,
    )


def _run_one_eval(
    *,
    kind: str,
    player_a_label: str,
    player_b_label: str,
    player_a,
    player_b,
    games: int,
    simulations: int,
    c_puct: float,
    out_json: Path,
    log_path: Path,
    seed: int,
    new_gen: int,
    total_gens: int,
    promoted_marker: bool = False,
    threshold: float | None = None,
) -> float:
    """Spawn one eval match, redirect its noisy stdout to `log_path`,
    log a one-line pre/post status. Returns `winrate_a`."""
    _log(
        f"phase: eval-{kind}  ({player_a_label} vs {player_b_label}, "
        f"{games} games, {simulations} sims/move)",
        gen=new_gen,
        total_gens=total_gens,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t = time.time()
    with open(log_path, "w") as logf:
        result = run_eval_match(
            player_a=player_a,
            player_b=player_b,
            games=games,
            simulations=simulations,
            c_puct=c_puct,
            out_json=out_json,
            seed=seed,
            stdout=logf,
            stderr=subprocess.STDOUT,
        )
    elapsed = time.time() - t
    marker = ""
    if promoted_marker and threshold is not None and result.winrate_a >= threshold:
        marker = "  ✓ PROMOTED"
    _log(
        f"  {kind:<9}  won {result.wins_a:>2}-{result.wins_b:<2} "
        f"({result.draws} draws)  winrate {result.winrate_a:.3f}{marker}  "
        f"({_fmt_secs(elapsed)})",
        gen=new_gen,
        total_gens=total_gens,
    )
    return result.winrate_a


def _maybe_eval(
    *,
    run_dir: Path,
    new_onnx: Path,
    best_onnx: Path,
    new_gen: int,
    cfg: RunConfig,
) -> tuple[float | None, float | None, float | None]:
    """Run gating + (optionally) heuristic + (optionally) random matches.
    Returns (gate_winrate, heuristic_winrate, random_winrate). Any may
    be None if not run this gen. Logs each match individually so a tail
    of the run log shows the eval phase clearly."""
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    gate_winrate = None
    if new_gen % cfg.eval.gate_every_gens == 0:
        gate_winrate = _run_one_eval(
            kind="gate",
            player_a_label=f"gen_{new_gen:03d}",
            player_b_label="best",
            player_a=new_onnx,
            player_b=best_onnx,
            games=cfg.eval.gate_games,
            simulations=cfg.eval.gate_simulations,
            c_puct=cfg.self_play.c_puct,
            out_json=eval_dir / f"gen_{new_gen:03d}_gate.json",
            log_path=logs_dir / f"gen_{new_gen:03d}_gate.log",
            seed=new_gen,
            new_gen=new_gen,
            total_gens=cfg.gens,
            promoted_marker=True,
            threshold=cfg.eval.gate_threshold,
        )

    heuristic_winrate = None
    if (
        cfg.eval.heuristic_every_gens > 0
        and new_gen % cfg.eval.heuristic_every_gens == 0
    ):
        heuristic_winrate = _run_one_eval(
            kind="heuristic",
            player_a_label=f"gen_{new_gen:03d}",
            player_b_label="heuristic",
            player_a=new_onnx,
            player_b="heuristic",
            games=cfg.eval.heuristic_games,
            simulations=cfg.eval.heuristic_simulations,
            c_puct=cfg.self_play.c_puct,
            out_json=eval_dir / f"gen_{new_gen:03d}_heuristic.json",
            log_path=logs_dir / f"gen_{new_gen:03d}_heuristic.log",
            seed=new_gen + 100_000,
            new_gen=new_gen,
            total_gens=cfg.gens,
        )

    random_winrate = None
    if cfg.eval.random_every_gens > 0 and new_gen % cfg.eval.random_every_gens == 0:
        random_winrate = _run_one_eval(
            kind="random",
            player_a_label=f"gen_{new_gen:03d}",
            player_b_label="random",
            player_a=new_onnx,
            player_b="random",
            games=cfg.eval.random_games,
            simulations=cfg.eval.random_simulations,
            c_puct=cfg.self_play.c_puct,
            out_json=eval_dir / f"gen_{new_gen:03d}_random.json",
            log_path=logs_dir / f"gen_{new_gen:03d}_random.log",
            seed=new_gen + 200_000,
            new_gen=new_gen,
            total_gens=cfg.gens,
        )

    return gate_winrate, heuristic_winrate, random_winrate


def _setup_signal_handlers(*slots: list[subprocess.Popen | None]) -> None:
    """Make sure tracked subprocesses die when the parent does. Each
    slot is a single-element mutable list containing the current Popen
    (or None). Used for both the self-play subprocess and the TB server."""

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
    """Spawn `tensorboard --logdir <runs_root>` so the user can browse
    training scalars live at http://localhost:<port>/.

    Pointing at `runs_root` (not just the current run dir) lets TB show
    all runs side-by-side for comparison. Invoked via `python -m
    tensorboard.main` to avoid PATH dependence on the venv binary.
    Returns the Popen; None if spawning failed."""
    log_path = runs_root / ".tensorboard.log"
    cmd = [
        sys.executable,
        "-m",
        "tensorboard.main",
        "--logdir",
        str(runs_root),
        "--port",
        str(port),
        "--bind_all",
    ]
    try:
        runs_root.mkdir(parents=True, exist_ok=True)
        log_file = open(log_path, "w")
        proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
        _log(f"tensorboard: http://localhost:{port}/  (server log: {log_path})")
        return proc
    except (FileNotFoundError, OSError) as e:
        _log(f"failed to start tensorboard: {e}")
        return None


def _resolve_resume(
    cfg: RunConfig, resume_arg: str | None
) -> tuple[str, RunState | None]:
    """Decide whether to resume an existing run or start fresh."""
    runs_root = REPO_ROOT / cfg.runs_root
    if resume_arg is not None:
        if resume_arg == "latest":
            candidates = sorted(
                p for p in runs_root.glob("*") if (p / "state.json").exists()
            )
            if not candidates:
                raise FileNotFoundError(f"no resumable runs found in {runs_root}")
            run_dir = candidates[-1]
            run_id = run_dir.name
        else:
            run_dir = runs_root / resume_arg
            if not (run_dir / "state.json").exists():
                raise FileNotFoundError(f"no state.json for {resume_arg}")
            run_id = resume_arg
        state = RunState.load(run_dir / "state.json")

        # Config-hash drift check.
        if state.config_hash != cfg.hash():
            raise RuntimeError(
                f"config hash mismatch on resume:\n"
                f"  state.config_hash = {state.config_hash}\n"
                f"  current cfg.hash() = {cfg.hash()}\n"
                f"refuse to resume; pass --no-resume to start a fresh run."
            )
        return run_id, state

    # Fresh run.
    run_id = cfg.run_id or generate_unique(runs_root)
    return run_id, None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, help="path to YAML config")
    parser.add_argument(
        "--resume",
        type=str,
        help="run-id to resume, or 'latest' for the most recent",
    )
    parser.add_argument(
        "--no-resume", action="store_true", help="start fresh even if state exists"
    )
    parser.add_argument(
        "--tensorboard", action="store_true",
        help="spawn a tensorboard server at --tb-port. Event files are "
        "always written to runs/<id>/tb/ regardless of this flag.",
    )
    parser.add_argument(
        "--tb-port", type=int, default=6006,
        help="port for the tensorboard server when --tensorboard is set (default 6006)",
    )
    args = parser.parse_args(argv)

    if args.config is None:
        parser.error(
            "--config required (use config/standard.yaml or config/dry_run.yaml)"
        )
    cfg = RunConfig.from_yaml(args.config)

    run_id, prev_state = _resolve_resume(
        cfg, args.resume if not args.no_resume else None
    )
    cfg.run_id = run_id
    run_dir = _resolve_run_dir(cfg, run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)

    # Freeze the resolved config.
    cfg.to_yaml(run_dir / "config.yaml")

    # Seed Python/numpy/torch RNGs deterministically (resume keeps drifting
    # but that's OK per agreed contract).
    random.seed(cfg.seed)
    np.random.seed(cfg.seed % (2**32))
    torch.manual_seed(cfg.seed)

    device = _device()

    # Plumb the CoreML toggle to Rust subprocesses via env var. Children
    # inherit our environment so this is all that's needed.
    if cfg.use_coreml:
        os.environ["ABALONE_USE_COREML"] = "1"
    else:
        os.environ.pop("ABALONE_USE_COREML", None)

    # Bootstrap or resume model.
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model = AbaloneNet().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
    )

    # Banner + structured config dump so the log is self-describing.
    workers_n = (
        cfg.self_play.worker_threads
        if cfg.self_play.worker_threads is not None
        else max((os.cpu_count() or 8) - 1, 1)
    )
    _banner(run_id, device, os.getpid(), model, workers_n)
    threads_resolved = (
        f"{cfg.self_play.worker_threads}"
        if cfg.self_play.worker_threads is not None
        else f"null (auto: {workers_n})"
    )
    _dump_config(cfg, threads_resolved)
    _log(f"run_dir: {run_dir}")
    if prev_state is not None:
        _log(
            f"resuming from gen {prev_state.current_gen} "
            f"(phase={prev_state.current_phase})"
        )

    if prev_state is None:
        # Fresh: bootstrap gen_000 (random weights) and start at gen 1.
        # Warm BN running stats before export so the bootstrap ONNX
        # behaves like the in-memory train()-mode forward pass.
        _warmup_bn(model, device)
        _save_ckpt(model, optimizer, ckpt_dir / "gen_000.pt")
        export_onnx(model, ckpt_dir / "gen_000.onnx")
        _link_or_copy(ckpt_dir / "gen_000.onnx", ckpt_dir / "best.onnx")
        state = RunState.fresh(run_id=run_id, config_hash=cfg.hash())
        state.current_onnx = "checkpoints/gen_000.onnx"
        state.best_onnx = "checkpoints/gen_000.onnx"
        state.current_gen = 0  # 0 fully-completed gens
        state.current_phase = "complete"  # no gen in progress
        state.save_atomic(run_dir / "state.json")
    else:
        # Resume. `current_gen` is the number of fully-completed gens, so
        # the checkpoint to load is always `gen_<current_gen>.pt`.
        state = prev_state
        last_ckpt = ckpt_dir / f"gen_{state.current_gen:03d}.pt"
        if not last_ckpt.exists():
            raise FileNotFoundError(
                f"resume requires checkpoint {last_ckpt} but it doesn't exist. "
                f"Retention may have purged it, or the run directory is corrupt. "
                f"Use --no-resume to start a fresh run."
            )
        _load_ckpt(last_ckpt, model, optimizer)
        # If we crashed mid-gen (phase != "complete"), the partial shard
        # dir for the in-progress gen is stale; nuke it and we'll redo
        # the whole gen from self_play.
        if state.current_phase != "complete":
            in_progress_gen = state.current_gen + 1
            partial = run_dir / "shards" / f"gen_{in_progress_gen:03d}"
            if partial.exists():
                shutil.rmtree(partial, ignore_errors=True)

    # Replay buffer: pre-load existing shards within the window.
    buffer = ReplayBuffer(augment=cfg.train.symmetry_augment)
    threshold = state.current_gen - cfg.train.replay_buffer_gens + 1
    for gen in range(max(threshold, 1), state.current_gen + 1):
        for shard in find_shards_for_gen(run_dir / "shards", gen):
            buffer.ingest_shard(shard, gen=gen)

    rng = np.random.default_rng(cfg.seed + state.current_gen)
    sp_state: list[subprocess.Popen | None] = [None]
    tb_state: list[subprocess.Popen | None] = [None]
    _setup_signal_handlers(sp_state, tb_state)

    # TensorBoard: SummaryWriter writes events to runs/<id>/tb/. The
    # server (if not disabled) is spawned at runs/ root so cross-run
    # comparison works in the UI.
    runs_root = REPO_ROOT / cfg.runs_root
    writer = SummaryWriter(str(run_dir / "tb"))
    if args.tensorboard:
        tb_state[0] = _start_tensorboard(runs_root, port=args.tb_port)
    else:
        _log(
            f"tensorboard: events at {run_dir / 'tb'}/  "
            f"(pass --tensorboard to spawn the server here)"
        )

    try:
        return _run_outer_loop(
            cfg=cfg,
            state=state,
            model=model,
            optimizer=optimizer,
            buffer=buffer,
            rng=rng,
            device=device,
            run_dir=run_dir,
            ckpt_dir=ckpt_dir,
            sp_state=sp_state,
            writer=writer,
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
    buffer: ReplayBuffer,
    rng: np.random.Generator,
    device: torch.device,
    run_dir: Path,
    ckpt_dir: Path,
    sp_state: list[subprocess.Popen | None],
    writer: SummaryWriter,
) -> int:
    # Outer loop.
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    while state.current_gen < cfg.gens:
        new_gen = state.current_gen + 1
        print(flush=True)
        _section()
        gen_t = time.time()
        _log("starting", gen=new_gen, total_gens=cfg.gens)

        # ---- self-play phase ----
        state.current_phase = "self_play"
        state.save_atomic(run_dir / "state.json")
        shards_dir = run_dir / "shards" / f"gen_{new_gen:03d}"
        shards_dir.mkdir(parents=True, exist_ok=True)
        threads_used = (
            cfg.self_play.worker_threads
            if cfg.self_play.worker_threads is not None
            else max(os.cpu_count() - 1, 1) if os.cpu_count() else 8
        )
        evaluator_name = _select_evaluator(
            new_gen, cfg.self_play.evaluator_schedule
        )
        _log(
            f"phase: self-play  ({cfg.self_play.games_per_gen} games, "
            f"{cfg.self_play.simulations_per_move} sims/move, "
            f"{threads_used} threads, evaluator={evaluator_name})",
            gen=new_gen,
            total_gens=cfg.gens,
        )
        sp_t = time.time()
        sp_log_path = logs_dir / f"gen_{new_gen:03d}_selfplay.log"
        sp_log_file = open(sp_log_path, "w")
        # Both stdout and stderr go to the per-gen log file. The Python
        # driver re-reads this file at progress-tick cadence to count
        # completed games (`_count_completed_games`).
        sp_proc = start_self_play(
            model_onnx=run_dir / state.current_onnx,
            out_dir=shards_dir,
            games=cfg.self_play.games_per_gen,
            simulations=cfg.self_play.simulations_per_move,
            c_puct=cfg.self_play.c_puct,
            temperature_plies=cfg.self_play.temperature_plies,
            temperature=cfg.self_play.temperature,
            dirichlet_alpha=cfg.self_play.dirichlet_alpha,
            dirichlet_eps=cfg.self_play.dirichlet_eps,
            shard_games=cfg.self_play.shard_games_per_file,
            threads=cfg.self_play.worker_threads,
            seed=(cfg.seed + new_gen) & 0xFFFF_FFFF,
            evaluator=evaluator_name,
            stdout=sp_log_file,
            stderr=subprocess.STDOUT,
        )
        sp_state[0] = sp_proc

        # ---- training phase (overlaps self-play) ----
        state.current_phase = "training"
        state.save_atomic(run_dir / "state.json")
        train_t = time.time()
        steps_done, last_metrics, sp_seconds, active_train_secs = _train_phase(
            model=model,
            optimizer=optimizer,
            buffer=buffer,
            sp_proc=sp_proc,
            selfplay_log_path=sp_log_path,
            shards_dir=shards_dir,
            new_gen=new_gen,
            cfg=cfg,
            rng=rng,
            device=device,
            writer=writer,
            phase_start_time=sp_t,
        )
        sp_state[0] = None
        sp_log_file.close()
        train_seconds = time.time() - train_t
        if last_metrics is not None:
            # Use `active_train_secs` (first SGD step to last) for the
            # throughput metric — excludes the leading wait-for-buffer
            # stretch so the rate is comparable across gens. `steps_done`
            # may exceed `steps_per_gen_min` if we used idle wall-clock
            # to keep training while self-play kept running.
            rate = steps_done / active_train_secs if active_train_secs > 0 else 0.0
            _log(
                f"training complete: {steps_done} steps in "
                f"{_fmt_secs(active_train_secs)} ({rate:.1f} steps/s)  ·  "
                f"mean loss={last_metrics.loss_total:.4f}  "
                f"mean grad={last_metrics.grad_norm:.2f}",
                gen=new_gen,
                total_gens=cfg.gens,
            )

        # ---- export ----
        state.current_phase = "export"
        state.save_atomic(run_dir / "state.json")
        new_pt = ckpt_dir / f"gen_{new_gen:03d}.pt"
        new_onnx = ckpt_dir / f"gen_{new_gen:03d}.onnx"
        _save_ckpt(model, optimizer, new_pt)
        export_onnx(model, new_onnx)
        onnx_mb = new_onnx.stat().st_size / (1024 * 1024)
        _log(
            f"phase: export  →  checkpoints/gen_{new_gen:03d}.onnx ({onnx_mb:.1f} MB)",
            gen=new_gen,
            total_gens=cfg.gens,
        )

        # ---- gate + heuristic + random ----
        state.current_phase = "gate"
        state.save_atomic(run_dir / "state.json")
        best_onnx_path = run_dir / state.best_onnx
        gate, heur, rand = _maybe_eval(
            run_dir=run_dir,
            new_onnx=new_onnx,
            best_onnx=best_onnx_path,
            new_gen=new_gen,
            cfg=cfg,
        )

        # ---- promotion decision ----
        promoted = False
        if gate is not None and gate >= cfg.eval.gate_threshold:
            promoted = True
            state.best_gen = new_gen
            state.best_onnx = f"checkpoints/gen_{new_gen:03d}.onnx"
            _link_or_copy(new_onnx, ckpt_dir / "best.onnx")
            if cfg.retention.web_export_on_promotion:
                web_dst = REPO_ROOT / cfg.web_export_path
                web_dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(new_onnx, web_dst)

        state.current_onnx = f"checkpoints/gen_{new_gen:03d}.onnx"

        # Per-gen diagnostics: shard count, total positions, avg game length.
        gen_shards = find_shards_for_gen(run_dir / "shards", new_gen)
        shard_count = len(gen_shards)
        gen_positions = buffer.chunk_size(new_gen)
        games_per_gen = cfg.self_play.games_per_gen
        plies_per_game_avg = (
            float(gen_positions) / games_per_gen if games_per_gen > 0 else None
        )

        state.append_history(
            GenRecord(
                gen=new_gen,
                promoted=promoted,
                train_loss_total=last_metrics.loss_total if last_metrics else None,
                train_loss_policy=last_metrics.loss_policy if last_metrics else None,
                train_loss_value=last_metrics.loss_value if last_metrics else None,
                train_grad_norm=last_metrics.grad_norm if last_metrics else None,
                gate_winrate=gate,
                heuristic_winrate=heur,
                random_winrate=rand,
                self_play_seconds=sp_seconds,
                train_seconds=train_seconds,
                buffer_size=buffer.total_size(),
                shard_count=shard_count,
                plies_per_game_avg=plies_per_game_avg,
            )
        )

        # Per-gen metrics jsonl entry (for plotting / wandb later).
        record = state.history[-1]
        metrics_path = run_dir / "metrics.jsonl"
        with metrics_path.open("a") as f:
            f.write(json.dumps(asdict(record)) + "\n")

        # Per-gen TB scalars. Mirrors metrics.jsonl fields with `gen/`
        # prefix so they sort cleanly alongside the per-step `train/`
        # scalars in TensorBoard.
        if writer is not None:
            tb_fields = {
                "gen/loss_total": record.train_loss_total,
                "gen/loss_policy": record.train_loss_policy,
                "gen/loss_value": record.train_loss_value,
                "gen/grad_norm": record.train_grad_norm,
                "gen/gate_winrate": record.gate_winrate,
                "gen/heuristic_winrate": record.heuristic_winrate,
                "gen/random_winrate": record.random_winrate,
                "perf/self_play_seconds": record.self_play_seconds,
                "perf/train_seconds": record.train_seconds,
                "data/buffer_size": record.buffer_size,
                "data/shard_count": record.shard_count,
                "data/plies_per_game_avg": record.plies_per_game_avg,
            }
            for tag, v in tb_fields.items():
                if v is not None:
                    writer.add_scalar(tag, float(v), new_gen)
            writer.add_scalar("gen/promoted", 1.0 if promoted else 0.0, new_gen)
            writer.flush()

        # Retention.
        _retain_checkpoints(run_dir, asdict(cfg.retention), new_gen)
        _retain_shards(run_dir, asdict(cfg.retention), new_gen)

        # Mark gen complete: advance `current_gen` and set phase to
        # "complete" (= no gen in progress). One atomic save.
        state.current_gen = new_gen
        state.current_phase = "complete"
        state.save_atomic(run_dir / "state.json")

        gen_secs = time.time() - gen_t
        loss_str = f"{last_metrics.loss_total:.4f}" if last_metrics else "N/A"
        promoted_str = "✓" if promoted else "✗"
        _log(
            f"complete in {_fmt_secs(gen_secs)}  ·  loss={loss_str}  ·  promoted {promoted_str}",
            gen=new_gen,
            total_gens=cfg.gens,
        )

    _section()
    _log("all generations complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
