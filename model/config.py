"""Run configuration: dataclasses + YAML loader.

Schema is intentionally narrow — only knobs we plan to actually tune.
Model architecture, MCTS internals, and loss formulation are hardcoded
in the source files where they're used; if we want to A/B them we'll
edit code, not config.

A given run's config is frozen at startup and saved to
`runs/<run-id>/config.yaml`. Resume validates by hashing this file
and comparing to `state.json`'s recorded hash; mismatch refuses to
resume (use `--no-resume` to override).
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_origin

import yaml


@dataclass
class SelfPlayConfig:
    games_per_gen: int = 200
    shard_games_per_file: int = 8
    simulations_per_move: int = 200
    c_puct: float = 1.4
    temperature_plies: int = 50
    temperature: float = 1.0
    dirichlet_alpha: float = 0.3
    dirichlet_eps: float = 0.25
    worker_threads: int | None = None  # null = (cores - 1)


@dataclass
class TrainConfig:
    steps_per_gen: int = 1000
    batch_size: int = 256
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    value_loss_weight: float = 0.5
    # Value target blends MCTS-Q with terminal-z. At gen 0 we lean
    # heavily on Q (denser signal); by `value_target_blend_done_by_gen`
    # we've ramped to fully terminal-z (less biased).
    value_target_blend_start: float = 0.5  # weight on terminal-z at gen 0
    value_target_blend_end: float = 1.0
    value_target_blend_done_by_gen: int = 20
    replay_buffer_gens: int = 20
    replay_buffer_min_size: int = 1000
    symmetry_augment: bool = True
    poll_interval_ms: int = 250


@dataclass
class EvalConfig:
    gate_every_gens: int = 1
    gate_games: int = 21
    gate_simulations: int = 200
    gate_threshold: float = 0.55
    heuristic_every_gens: int = 5
    heuristic_games: int = 21
    heuristic_simulations: int = 200
    random_every_gens: int = 10
    random_games: int = 11
    random_simulations: int = 200


@dataclass
class RetentionConfig:
    keep_last_checkpoints: int = 5
    keep_last_onnx: int = 25
    keep_last_shard_gens: int = 25
    web_export_on_promotion: bool = True


@dataclass
class RunConfig:
    """Top-level config. Each named field is one subgroup; the few
    top-level scalars are run identity and outer-loop limits."""

    run_id: str | None = None  # auto-generated if None
    seed: int = 0
    gens: int = 50
    runs_root: str = "runs"
    web_export_path: str = "web/public/models/best.onnx"
    # Infra knob: route Rust ORT inference through Apple's CoreML
    # execution provider (Neural Engine / GPU) instead of CPU. On our
    # 7M-param model with parallel workers, CPU is faster; enable only
    # if you've scaled up the model and benchmarked the difference.
    # Excluded from `config_hash` so flipping this on resume is allowed.
    use_coreml: bool = False
    self_play: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    retention: RetentionConfig = field(default_factory=RetentionConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> RunConfig:
        text = Path(path).read_text()
        raw = yaml.safe_load(text) or {}
        return _from_dict(cls, raw)

    def to_yaml(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        text = yaml.safe_dump(asdict(self), sort_keys=False, default_flow_style=False)
        path.write_text(text)

    def hash(self) -> str:
        """Stable SHA-256 hex of the resolved config. Used by the
        resume-time validation to detect drift since last run.

        Excludes `run_id` (auto-generated, not a config choice) and `gens`
        (the outer loop bound — bumping it on resume should be allowed so
        you can extend a finished run)."""
        d = asdict(self)
        d.pop("run_id", None)
        d.pop("gens", None)
        d.pop("use_coreml", None)
        # Stable serialization: yaml with sorted keys, default flow.
        canonical = yaml.safe_dump(d, sort_keys=True, default_flow_style=False)
        return hashlib.sha256(canonical.encode()).hexdigest()


def _from_dict(target: type, raw: dict[str, Any]) -> Any:
    """Recursively populate a dataclass tree from a plain dict.
    Unknown keys are rejected (typo'd configs should fail loudly)."""
    if not is_dataclass(target):
        return raw
    fmap = {f.name: f for f in fields(target)}
    unknown = set(raw.keys()) - set(fmap.keys())
    if unknown:
        raise ValueError(
            f"Unknown config keys for {target.__name__}: {sorted(unknown)}. "
            f"Allowed: {sorted(fmap.keys())}"
        )
    kwargs: dict[str, Any] = {}
    for name, f in fmap.items():
        if name not in raw:
            continue  # use default
        v = raw[name]
        ftype = f.type
        # Handle nested dataclasses by inspecting the annotation.
        if isinstance(ftype, type) and is_dataclass(ftype):
            kwargs[name] = _from_dict(ftype, v or {})
        elif get_origin(ftype) is None and is_dataclass(_resolve(ftype, target)):
            kwargs[name] = _from_dict(_resolve(ftype, target), v or {})
        else:
            kwargs[name] = v
    return target(**kwargs)


def _resolve(annotation: Any, owner: type) -> Any:
    """Best-effort resolution of a field annotation that may be a string
    (when `from __future__ import annotations` is in effect on the
    declaring module). Falls back to the raw annotation."""
    if isinstance(annotation, str):
        # Look up the class in the owner module's globals.
        import sys

        mod = sys.modules.get(owner.__module__)
        if mod is not None:
            return getattr(mod, annotation, annotation)
    return annotation
