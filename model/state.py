"""Run state: atomic JSON file at `runs/<run-id>/state.json`.

Resume granularity is one generation.

State machine:
- `current_gen` = number of fully-completed gens (0 at bootstrap).
- `current_phase` describes progress on gen `current_gen + 1`:
    * `"complete"`  — no gen in progress; ready to start `current_gen + 1`.
    * `"self_play"` / `"training"` / `"export"` / `"gate"` — that phase of
      gen `current_gen + 1` has started but not yet finished.

We save state.json (atomically, fsync'd) at the start of every phase and
again when a gen wraps up. On crash, resume:
  - Loads `gen_<current_gen>.pt` (the last fully-completed checkpoint).
  - If `current_phase != "complete"`, nukes the partial shards for
    `gen_<current_gen + 1>` and restarts that gen from `self_play`.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

Phase = Literal[
    "self_play", "training", "export", "gate", "heuristic_eval", "random_eval", "complete"
]



@dataclass
class GenRecord:
    """One row of the per-generation history log."""

    gen: int
    promoted: bool
    train_loss_policy: float | None = None
    train_loss_value: float | None = None
    train_loss_total: float | None = None
    train_grad_norm: float | None = None
    gate_winrate: float | None = None
    heuristic_winrate: float | None = None
    random_winrate: float | None = None
    self_play_seconds: float | None = None
    train_seconds: float | None = None
    buffer_size: int | None = None
    shard_count: int | None = None
    plies_per_game_avg: float | None = None


@dataclass
class RunState:
    schema_version: int = 1
    run_id: str = ""
    config_hash: str = ""
    # Number of fully-completed generations.
    current_gen: int = 0
    # Progress on gen `current_gen + 1`. `"complete"` means no gen is in
    # progress (we just finished `current_gen` cleanly, or we just
    # bootstrapped). Anything else means that phase of gen
    # `current_gen + 1` started but didn't finish.
    current_phase: Phase = "complete"
    best_gen: int = 0
    best_onnx: str = ""  # path relative to runs_root/<run-id>/
    current_onnx: str = ""
    history: list[GenRecord] = field(default_factory=list)

    @classmethod
    def fresh(cls, run_id: str, config_hash: str) -> RunState:
        return cls(run_id=run_id, config_hash=config_hash)

    @classmethod
    def load(cls, path: Path) -> RunState:
        raw = json.loads(Path(path).read_text())
        history_raw = raw.pop("history", [])
        st = cls(**raw)
        st.history = [GenRecord(**r) for r in history_raw]
        return st

    def save_atomic(self, path: Path, fsync: bool = True) -> None:
        """Write to `<path>.tmp`, fsync, rename. Survives crashes:
        either the old contents or fully-new contents are visible."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        d = asdict(self)
        text = json.dumps(d, indent=2, sort_keys=False)
        tmp_fd, tmp_name = tempfile.mkstemp(
            dir=path.parent, prefix=path.name + ".", suffix=".tmp"
        )
        try:
            with os.fdopen(tmp_fd, "w") as f:
                f.write(text)
                if fsync:
                    f.flush()
                    os.fsync(f.fileno())
            os.replace(tmp_name, path)
        except Exception:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise

    def append_history(self, record: GenRecord) -> None:
        self.history.append(record)
