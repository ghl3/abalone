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
from typing import Any, Literal, NamedTuple

Phase = Literal[
    "self_play", "training", "export", "validate", "ladder", "complete"
]


def _filtered(cls: type, raw: dict[str, Any]) -> dict[str, Any]:
    """`raw` restricted to `cls`'s fields."""
    names = {f.name for f in fields(cls)}
    return {k: v for k, v in raw.items() if k in names}


class Totals(NamedTuple):
    """Run-to-date counters, summed over `RunState.history`.

    Two independent axes, and neither is recoverable from the other.

    `games`/`positions` are **experience**: how much Abalone the run has
    actually seen. `train_steps` is **optimisation**: how much gradient descent
    has been done on it. Doubling `replay_buffer_gens` or `steps_per_gen_max`
    moves the second without producing a single new position; doubling
    `games_per_gen` moves the first without taking a single extra step.

    Positions are rows, not distinct board states. Measured on `ruby-panther`,
    the two differ by 0.28% at generation 2 and 1.26% at generation 14 — the
    rate rises as the policy sharpens and revisits lines — so treating rows as
    distinct positions is accurate to about a percent, but it is an
    approximation and not a definition.

    Samples fed to the optimiser is `train_steps × train.batch_size`, which is
    larger than `positions` by the number of times the buffer is resampled
    (6.2× over `ruby-panther`'s 14 generations). It is deliberately not a field
    here: it depends on a config value this class does not have, and it counts
    D6-augmented views rather than anything unique.
    """

    games: int
    positions: int
    train_steps: int


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
    #: Raw passes over the replay buffer this generation, D6 augmentation
    #: ignored. 20 of these in one generation is overfitting by construction.
    train_epochs_over_buffer: float | None = None

    # -- held-out validation (MODEL.md §8.1) --
    # `val_*` is the FROZEN holdout: a fixed ruler, and therefore a drift
    # indicator — by generation 30 it scores the network on positions a
    # thirty-generation-weaker network produced. `val_rolling_*` is this
    # generation's own withheld games, never trained on: same distribution as
    # the training data, which is what makes it the one worth gating on.
    val_loss_total: float | None = None
    val_policy_top1: float | None = None
    val_policy_entropy_ratio: float | None = None
    val_value_ce: float | None = None
    val_value_accuracy: float | None = None
    #: Rows this generation withheld from its own training and never sampled
    #: again. `None`/0 means no slice was taken — the guard fires when the pool
    #: outside the generation is below `train.replay_buffer_min_size`. A resume
    #: re-takes the slice only for generations that recorded one here, so a
    #: generation that was trained on in full stays that way.
    rolling_holdout_positions: int | None = None
    val_rolling_loss_total: float | None = None
    val_rolling_policy_top1: float | None = None
    val_rolling_value_ce: float | None = None
    val_rolling_value_accuracy: float | None = None

    # -- data health, from the exported games --
    decisive_rate: float | None = None
    mean_plies: float | None = None
    mean_abs_score_diff: float | None = None
    policy_target_entropy: float | None = None
    policy_uniform_entropy: float | None = None
    policy_entropy_gap: float | None = None
    #: Captures per 100 plies. Falling while `mean_plies` rises is competent
    #: defence emerging (MODEL.md §8.2); the reverse is a brawl.
    captures_per_100_plies: float | None = None

    # -- curriculum (MODEL.md §4) --
    #: Seeding rate self-play actually used for this generation.
    handicap_rate: float | None = None
    #: Rate after this generation's ratchet — what gen+1 will use.
    handicap_rate_next: float | None = None
    #: The control signal and its sample size, for this generation.
    unseeded_games: int | None = None
    natural_termination_rate: float | None = None

    # -- strength --
    #: Mean Elo over the *fixed-reference* rungs (floor anchors and absolute
    #: frozen checkpoints). Never quoted without its interval and the clamped
    #: fraction: when every rung is swept the "Elo" is the sample-size bound.
    ladder_elo: float | None = None
    ladder_elo_ci95_lo: float | None = None
    ladder_elo_ci95_hi: float | None = None
    ladder_clamped_fraction: float | None = None

    # -- accounting --
    self_play_seconds: float | None = None
    train_seconds: float | None = None
    gen_seconds: float | None = None
    buffer_size: int | None = None
    shard_count: int | None = None
    positions: int | None = None
    #: Games self-play completed this generation. Normally
    #: `self_play.games_per_gen`, but recorded rather than assumed because a
    #: crashed generation is redone and the shards are what get counted.
    games: int | None = None

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
    #: **Informational only.** The ladder Elo recorded at the generation
    #: that was promoted — it does NOT drive promotion. `elo_mean`'s basis
    #: changes with the panel (fixed-reference rungs when any resolved,
    #: trailing rungs otherwise), so comparing it across generations
    #: compares different measurements. Promotion is decided by
    #: `eval.resolved_regressions`; see the block in `train_loop`.
    best_elo: float | None = None
    best_onnx: str = ""  # path relative to runs/<run-id>/
    current_onnx: str = ""
    #: Live capture-handicap seeding rate (MODEL.md §4). Initialised from
    #: `self_play.handicap_rate` and ratcheted down by `model.curriculum`; it
    #: lives here rather than in the config precisely *because* it changes —
    #: `config_hash` covers the YAML, so an annealed rate written back into the
    #: config would make every resume refuse. `None` means "never set": a state
    #: file written before the annealer existed, which falls back to the config.
    handicap_rate: float | None = None
    history: list[GenRecord] = field(default_factory=list)

    @classmethod
    def fresh(
        cls,
        run_id: str,
        config_hash: str,
        git_sha: str = "",
        handicap_rate: float | None = None,
    ) -> RunState:
        return cls(
            run_id=run_id,
            config_hash=config_hash,
            git_sha=git_sha,
            handicap_rate=handicap_rate,
        )

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

    def totals(self, games_per_gen: int | None = None) -> Totals:
        """Run-to-date [`Totals`][model.state.Totals], summed over `history`.

        Derived rather than carried as counters, so it is correct for runs that
        predate it and cannot drift out of step with the per-generation records
        it sums. `history` gains a row only when a generation commits, so a
        generation redone after a crash is counted once, not twice.

        `games_per_gen` backfills records written before the `games` field
        existed. It is inside `config_hash`, so it cannot change within a run
        and the reconstruction is exact — worth doing, because extending a
        finished run is a workflow this project explicitly wants and a silently
        short total would misreport every generation after the resume.
        `positions` and `train_steps` need no such fallback: both have been
        recorded since the first schema.
        """
        return Totals(
            games=sum(
                r.games if r.games is not None else (games_per_gen or 0)
                for r in self.history
            ),
            positions=sum(r.positions or 0 for r in self.history),
            train_steps=sum(r.train_steps or 0 for r in self.history),
        )
