"""Run state: atomic JSON file at `runs/<run-id>/state.json`.

Resume granularity is one generation. After each completed generation
phase (`self_play`, `training`, `export`, `gate`, `heuristic_eval`,
`random_eval`), we update `state.json` and fsync. If we crash before
the fsync, on resume we re-run that phase. After fsync, we resume at
the next phase.
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
    gate_winrate: float | None = None
    heuristic_winrate: float | None = None
    random_winrate: float | None = None
    self_play_seconds: float | None = None
    train_seconds: float | None = None


@dataclass
class RunState:
    schema_version: int = 1
    run_id: str = ""
    config_hash: str = ""
    # Highest generation whose phase work *has begun* (could be partway).
    current_gen: int = 0
    # The next phase to execute within `current_gen`. After all phases
    # of a generation complete, `current_gen` increments and phase
    # resets to `self_play`.
    current_phase: Phase = "self_play"
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

    def begin_next_gen(self) -> None:
        """Call after the current gen's `complete` phase to move on."""
        self.current_gen += 1
        self.current_phase = "self_play"

    def append_history(self, record: GenRecord) -> None:
        self.history.append(record)
