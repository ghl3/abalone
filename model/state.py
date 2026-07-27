"""Run state: atomic JSON file at `runs/<run-id>/state.json`.

Resume granularity is one generation.

State machine:
- `current_gen` = number of fully-completed gens (0 at bootstrap).
- `current_phase` describes progress on gen `current_gen + 1`:
    * `"complete"`  — no gen in progress; ready to start `current_gen + 1`.
    * `"self_play"` / `"training"` / `"export"` / `"validate"` / `"ladder"` —
      that phase of gen `current_gen + 1` started but has not finished.

State is saved (atomically, fsync'd) at the start of every phase and again when
a gen wraps up. On crash, resume:
  - Loads `gen_<current_gen>.pt` (the last fully-completed checkpoint).
  - If `current_phase != "complete"`, discards the partial shards for
    `gen_<current_gen + 1>` and redoes that gen from `self_play`.

`git_sha` records the commit the run was last advanced with. `config_hash`
covers the YAML only, so changing the ply cap or the plane layout in Rust
silently invalidates every shard in the buffer with no detection
([ARCHITECTURE §5.6](../docs/ARCHITECTURE.md#56-enforcement)). The SHA is a
warning, not a refusal: most commits are harmless and refusing to resume after
a docs typo would be its own kind of broken.

Loading is tolerant of unknown keys so a state file written by a newer build
does not hard-fail an older one; missing keys take their defaults.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Literal

Phase = Literal[
    "self_play", "training", "export", "validate", "ladder", "complete"
]


def _filtered(cls: type, raw: dict[str, Any]) -> dict[str, Any]:
    """`raw` restricted to `cls`'s fields."""
    names = {f.name for f in fields(cls)}
    return {k: v for k, v in raw.items() if k in names}


@dataclass
class GenRecord:
    """One row of the per-generation history log. Mirrored, with more detail,
    into `metrics.jsonl`."""

    gen: int

    # -- training --
    train_steps: int | None = None
    train_loss_total: float | None = None
    train_loss_policy: float | None = None
    train_loss_value: float | None = None
    train_loss_score: float | None = None
    train_loss_capture_map: float | None = None
    train_grad_norm: float | None = None
    learning_rate: float | None = None

    # -- held-out validation (MODEL.md §8.1) --
    val_loss_total: float | None = None
    val_policy_top1: float | None = None
    val_policy_entropy_ratio: float | None = None
    val_value_ce: float | None = None
    val_value_accuracy: float | None = None

    # -- data health, from the exported games --
    decisive_rate: float | None = None
    mean_plies: float | None = None
    mean_abs_score_diff: float | None = None
    policy_target_entropy: float | None = None
    policy_uniform_entropy: float | None = None

    # -- strength --
    ladder_elo: float | None = None

    # -- accounting --
    self_play_seconds: float | None = None
    train_seconds: float | None = None
    gen_seconds: float | None = None
    buffer_size: int | None = None
    shard_count: int | None = None
    positions: int | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> GenRecord:
        return cls(**_filtered(cls, raw))


@dataclass
class RunState:
    schema_version: int = 2
    run_id: str = ""
    config_hash: str = ""
    #: Commit the run was last advanced with. Warned about on resume mismatch.
    git_sha: str = ""
    # Number of fully-completed generations.
    current_gen: int = 0
    # Progress on gen `current_gen + 1`.
    current_phase: Phase = "complete"
    #: Best generation *by anchor-ladder Elo*. With gating deleted this is not
    #: a promotion — self-play always uses `current_onnx` — it only selects the
    #: web export.
    best_gen: int = 0
    best_elo: float | None = None
    best_onnx: str = ""  # path relative to runs/<run-id>/
    current_onnx: str = ""
    history: list[GenRecord] = field(default_factory=list)

    @classmethod
    def fresh(cls, run_id: str, config_hash: str, git_sha: str = "") -> RunState:
        return cls(run_id=run_id, config_hash=config_hash, git_sha=git_sha)

    @classmethod
    def load(cls, path: Path) -> RunState:
        raw = json.loads(Path(path).read_text())
        history_raw = raw.pop("history", [])
        st = cls(**_filtered(cls, raw))
        st.history = [GenRecord.from_dict(r) for r in history_raw]
        return st

    def save_atomic(self, path: Path, fsync: bool = True) -> None:
        """Write to `<path>.tmp`, fsync, rename. Survives crashes: either the
        old contents or fully-new contents are visible, never a mix."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(asdict(self), indent=2, sort_keys=False)
        tmp_fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
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
