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

from model.abalone_net import BOARD_H, BOARD_W, INPUT_CHANNELS, AbaloneNet
from model.config import RunConfig
from model.encoder import VALID_CELL_MASK
from model.eval import run_eval_match, start_self_play
from model.export_onnx import export as export_onnx
from model.replay_buffer import ReplayBuffer, find_shards_for_gen
from model.run_id import generate_unique
from model.state import GenRecord, RunState
from model.train_step import StepMetrics, train_step, value_target_blend_weight

REPO_ROOT = Path(__file__).resolve().parents[1]


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
    shards_dir: Path,
    new_gen: int,
    cfg: RunConfig,
    rng: np.random.Generator,
    device: torch.device,
) -> tuple[int, StepMetrics | None]:
    """Train while self-play runs (if `sp_proc` is alive). Continue
    training after self-play exits until we've done `steps_per_gen` SGD
    steps total. Returns (steps_done, mean_metrics).

    `mean_metrics` is the running mean over all SGD steps in this gen
    (loss_total/policy/value, grad_norm), or None if we never trained."""

    z_w = value_target_blend_weight(
        new_gen,
        cfg.train.value_target_blend_start,
        cfg.train.value_target_blend_end,
        cfg.train.value_target_blend_done_by_gen,
    )

    steps_done = 0
    sum_loss_total = 0.0
    sum_loss_policy = 0.0
    sum_loss_value = 0.0
    sum_grad_norm = 0.0
    seen_files: set[Path] = set()

    def ingest_new() -> int:
        ingested = 0
        for p in find_shards_for_gen(shards_dir.parent, new_gen):
            if p in seen_files:
                continue
            try:
                buffer.ingest_shard(p, gen=new_gen)
                seen_files.add(p)
                ingested += 1
            except Exception as e:
                # Partial files mid-write may fail to read; skip and try again next poll.
                print(f"  (skipping {p.name}: {e})", file=sys.stderr)
        return ingested

    poll_interval = max(0.05, cfg.train.poll_interval_ms / 1000.0)

    while steps_done < cfg.train.steps_per_gen:
        ingest_new()

        # Evict beyond replay window.
        threshold = new_gen - cfg.train.replay_buffer_gens + 1
        if threshold > 0:
            buffer.evict_below(threshold)

        if buffer.total_size() < cfg.train.replay_buffer_min_size:
            if sp_proc is None or sp_proc.poll() is not None:
                # Self-play done and we still don't have enough data.
                # Sample what we have and proceed; if buffer is empty, bail.
                if buffer.total_size() == 0:
                    raise RuntimeError("self-play produced no data")
                # Continue with reduced buffer.
            else:
                time.sleep(poll_interval)
                continue

        # One small training chunk per poll.
        for _ in range(min(8, cfg.train.steps_per_gen - steps_done)):
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
            steps_done += 1
            if steps_done >= cfg.train.steps_per_gen:
                break

        # Yield to self-play subprocess if it's still running.
        if sp_proc is not None and sp_proc.poll() is None:
            time.sleep(0)  # cooperative

    # Drain: pick up any final shards self-play wrote after we exited the loop.
    if sp_proc is not None:
        sp_proc.wait()
        if sp_proc.returncode != 0:
            raise RuntimeError(
                f"selfplay-batch exited with non-zero status {sp_proc.returncode}; "
                f"see subprocess stderr above for cause."
            )
        ingest_new()

    if steps_done == 0:
        return 0, None
    mean_metrics = StepMetrics(
        loss_total=sum_loss_total / steps_done,
        loss_policy=sum_loss_policy / steps_done,
        loss_value=sum_loss_value / steps_done,
        grad_norm=sum_grad_norm / steps_done,
    )
    return steps_done, mean_metrics


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
    be None if not run this gen."""
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    gate_winrate = None
    if new_gen % cfg.eval.gate_every_gens == 0:
        out = eval_dir / f"gen_{new_gen:03d}_gate.json"
        result = run_eval_match(
            player_a=new_onnx,
            player_b=best_onnx,
            games=cfg.eval.gate_games,
            simulations=cfg.eval.gate_simulations,
            c_puct=cfg.self_play.c_puct,
            out_json=out,
            seed=new_gen,
        )
        gate_winrate = result.winrate_a

    heuristic_winrate = None
    if (
        cfg.eval.heuristic_every_gens > 0
        and new_gen % cfg.eval.heuristic_every_gens == 0
    ):
        out = eval_dir / f"gen_{new_gen:03d}_heuristic.json"
        result = run_eval_match(
            player_a=new_onnx,
            player_b="heuristic",
            games=cfg.eval.heuristic_games,
            simulations=cfg.eval.heuristic_simulations,
            c_puct=cfg.self_play.c_puct,
            out_json=out,
            seed=new_gen + 100_000,
        )
        heuristic_winrate = result.winrate_a

    random_winrate = None
    if cfg.eval.random_every_gens > 0 and new_gen % cfg.eval.random_every_gens == 0:
        out = eval_dir / f"gen_{new_gen:03d}_random.json"
        result = run_eval_match(
            player_a=new_onnx,
            player_b="random",
            games=cfg.eval.random_games,
            simulations=cfg.eval.random_simulations,
            c_puct=cfg.self_play.c_puct,
            out_json=out,
            seed=new_gen + 200_000,
        )
        random_winrate = result.winrate_a

    return gate_winrate, heuristic_winrate, random_winrate


def _setup_signal_handlers(state: list[subprocess.Popen | None]) -> None:
    """Make sure subprocesses die when the parent does. `state[0]` is
    the currently-running self-play subprocess (if any)."""

    def handler(signum, _frame):
        proc = state[0]
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
    print(f"[train_loop] run_id = {run_id}")
    print(f"[train_loop] run_dir = {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    # Freeze the resolved config.
    cfg.to_yaml(run_dir / "config.yaml")

    # Seed Python/numpy/torch RNGs deterministically (resume keeps drifting
    # but that's OK per agreed contract).
    random.seed(cfg.seed)
    np.random.seed(cfg.seed % (2**32))
    torch.manual_seed(cfg.seed)

    device = _device()
    print(f"[train_loop] device = {device}")

    # Bootstrap or resume model.
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model = AbaloneNet().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
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
    _setup_signal_handlers(sp_state)

    # Outer loop.
    while state.current_gen < cfg.gens:
        new_gen = state.current_gen + 1
        print(f"\n[train_loop] === GEN {new_gen}/{cfg.gens} ===")

        # ---- self-play phase ----
        state.current_phase = "self_play"
        state.save_atomic(run_dir / "state.json")
        shards_dir = run_dir / "shards" / f"gen_{new_gen:03d}"
        shards_dir.mkdir(parents=True, exist_ok=True)
        sp_t = time.time()
        print("[train_loop] launching self-play subprocess")
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
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        sp_state[0] = sp_proc

        # ---- training phase (overlaps self-play) ----
        state.current_phase = "training"
        state.save_atomic(run_dir / "state.json")
        train_t = time.time()
        _, last_metrics = _train_phase(
            model=model,
            optimizer=optimizer,
            buffer=buffer,
            sp_proc=sp_proc,
            shards_dir=shards_dir,
            new_gen=new_gen,
            cfg=cfg,
            rng=rng,
            device=device,
        )
        sp_state[0] = None
        train_seconds = time.time() - train_t
        sp_seconds = time.time() - sp_t

        # ---- export ----
        state.current_phase = "export"
        state.save_atomic(run_dir / "state.json")
        new_pt = ckpt_dir / f"gen_{new_gen:03d}.pt"
        new_onnx = ckpt_dir / f"gen_{new_gen:03d}.onnx"
        _save_ckpt(model, optimizer, new_pt)
        export_onnx(model, new_onnx)

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
        metrics_path = run_dir / "metrics.jsonl"
        with metrics_path.open("a") as f:
            f.write(json.dumps(asdict(state.history[-1])) + "\n")

        # Retention.
        _retain_checkpoints(run_dir, asdict(cfg.retention), new_gen)
        _retain_shards(run_dir, asdict(cfg.retention), new_gen)

        # Mark gen complete: advance `current_gen` and set phase to
        # "complete" (= no gen in progress). One atomic save.
        state.current_gen = new_gen
        state.current_phase = "complete"
        state.save_atomic(run_dir / "state.json")

        loss_str = f"{last_metrics.loss_total:.4f}" if last_metrics else "N/A"
        print(
            f"[train_loop] gen {new_gen} done. promoted={promoted}, gate={gate}, "
            f"heuristic={heur}, random={rand}, loss={loss_str}"
        )

    print("[train_loop] all generations complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
