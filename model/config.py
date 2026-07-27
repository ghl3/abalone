"""Run configuration: dataclasses + YAML loader.

The schema is intentionally narrow — only knobs we plan to actually tune.
Network internals, the move-index space and the loss *formulation* are
hardcoded where they are used; if we want to A/B those we edit code, not YAML.

Three groups mirror something outside Python and must not drift from it:

* `self_play` mirrors `SelfPlayConfig::default()` in
  [`crates/selfplay/src/lib.rs`]. Every scalar here becomes a
  `selfplay-batch` flag. `tests/test_config.py` pins the defaults against the
  Rust source so a change on either side fails loudly. The one exception is
  `self_play.handicap_anneal`, which has no Rust counterpart: Rust is told a
  rate per invocation and knows nothing about how that rate is chosen.
* `train.loss_weights` mirrors `model.train_step.LossWeights`
  (MODEL.md §6.6: `w_v = 1.0`, `w_s = 0.15`, `w_c = 0.15`).
* `anchor_ladder.opponents` are `eval-match` player specs.

**Unknown keys are rejected.** A typo'd knob that silently keeps the default is
how a run quietly measures the wrong thing for four hours.

A run's config is frozen at startup into `runs/<run-id>/config.yaml`. Resume
hashes it and compares against `state.json`; a mismatch refuses to resume.
`config_hash` covers only what changes the *meaning* of the data or the model —
see `HASH_EXCLUDED`. It does **not** cover the code, which is why `state.json`
also records the git SHA (ARCHITECTURE.md §5.6).

Deleted deliberately, do not add back:

* `eval.gate_*` / promotion — self-play always uses the latest network
  (MODEL.md §8). Progress is measured by the anchor ladder.
* `train.value_target_blend_*` — the value head trains on the game outcome
  alone (ARCHITECTURE.md §5.5).
* `self_play.evaluator_schedule` — `selfplay-batch` rejects
  `--evaluator heuristic`; the curriculum in MODEL.md §4 replaced the
  heuristic bootstrap.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_origin

import yaml

#: Openings `selfplay-batch` and `eval-match` both accept, spelled identically.
OPENINGS = ("standard", "belgian")

#: Trunk presets in `model.abalone_net.PRESETS`. Duplicated as plain strings so
#: importing the config does not drag torch in; a test pins the two together.
NET_PRESETS = ("small", "base", "large")

#: `no_progress_plies: 0` means "rule off". The Rust side spells that
#: `NO_PROGRESS_DISABLED = u32::MAX`; `selfplay-batch` does the translation.
NO_PROGRESS_OFF = 0

#: How `handicap_rate` moves during a run (MODEL.md §4). `controller` is the
#: closed loop on natural termination; `schedule` is a hand-written `{gen: rate}`
#: ladder; `off` pins the rate at its configured value for the whole run.
ANNEAL_MODES = ("controller", "schedule", "off")


@dataclass
class HandicapAnnealConfig:
    """How the capture-handicap curriculum retires itself (MODEL.md §4).

    The seeded fraction is a crutch: it puts terminal states inside the search
    horizon before the network can reach them on its own. Left static it is a
    distortion — at generation 40 a `handicap_rate` of 0.7 still spends 70% of
    the compute on artificially-seeded positions instead of the ones real play
    produces. This group is what takes the crutch away.

    **The control signal is the natural termination rate of *unseeded* games**:
    the fraction of `handicap == (0, 0)` games that reach six captures before
    the ply cap. It is 0.0 under random play and only rises when the network can
    actually close a game out — which is exactly the condition under which the
    endgame curriculum has done its job.

    **Decisive rate is not the signal and must not be substituted for it.**
    Belgian Daisy with adjudication and no handicap measures 63% decisive under
    *uniformly random* play (mean margin 0.89 — one lucky push at the ply cap).
    Any threshold on decisive rate fires in generation one against a random
    network and pulls the crutch before it has done anything.
    """

    mode: str = "controller"
    #: Step down once this fraction of unseeded games ends on captures.
    target_natural_termination: float = 0.25
    #: Absolute decrement per firing. One step per generation, at most.
    step: float = 0.05
    #: The rate never goes below this, in any mode. Deliberately non-zero:
    #: positions at 5–4 captures are strategically critical and essentially
    #: never visited by self-play from a fresh start (MODEL.md §4), so a
    #: permanent trickle of seeded games is worth keeping purely as coverage.
    floor: float = 0.10
    #: Below this many unseeded games the termination estimate is noise and the
    #: controller holds rather than acting on it.
    min_unseeded_games: int = 20
    #: `{gen: rate}` — "from this generation onward, use this rate". Used only
    #: when `mode == "schedule"`. Empty is a no-op.
    schedule: dict = field(default_factory=dict)

    def validate(self) -> None:
        _require(
            self.mode in ANNEAL_MODES,
            f"self_play.handicap_anneal.mode must be one of {list(ANNEAL_MODES)}, "
            f"got {self.mode!r}",
        )
        _require(
            0.0 <= self.target_natural_termination <= 1.0,
            "self_play.handicap_anneal.target_natural_termination must be in [0, 1]",
        )
        _require(self.step > 0.0, "self_play.handicap_anneal.step must be > 0")
        _require(0.0 <= self.floor <= 1.0, "self_play.handicap_anneal.floor must be in [0, 1]")
        _require(
            self.min_unseeded_games >= 0,
            "self_play.handicap_anneal.min_unseeded_games must be >= 0",
        )
        for k, v in self.schedule.items():
            _require(
                isinstance(k, int) and not isinstance(k, bool),
                f"self_play.handicap_anneal.schedule keys must be generation ints, got {k!r}",
            )
            _require(
                0.0 <= float(v) <= 1.0,
                f"self_play.handicap_anneal.schedule[{k}] must be in [0, 1]",
            )


@dataclass
class SelfPlayConfig:
    """One `selfplay-batch` invocation. Defaults track `SelfPlayConfig::default()`."""

    # ---- volume ----
    games_per_gen: int = 200
    #: Games per parquet file. Smaller = training sees fresh data sooner.
    shard_games_per_file: int = 8
    worker_threads: int | None = None  # null = (cores - 1)

    # ---- search (MODEL.md §7) ----
    #: Simulations for an ordinary move; these positions carry no policy target.
    sims_fast: int = 200
    #: Simulations for a full-search move. Only these produce policy targets.
    sims_full: int = 800
    #: Probability a move runs the full budget — playout cap randomisation.
    full_search_rate: float = 0.25
    c_puct: float = 1.4
    #: Leaves per network call. The dominant throughput lever (MODEL.md §7.1);
    #: also the fixed width the CoreML path zero-pads to.
    batch_size: int = 16
    virtual_loss: float = 1.0
    fpu_reduction: float = 0.25
    #: alpha ≈ 10 / branching, branching ≈ 60.
    dirichlet_alpha: float = 0.2
    dirichlet_eps: float = 0.25

    # ---- move selection ----
    temperature_plies: int = 30
    temperature: float = 1.0

    # ---- game setup / curriculum (MODEL.md §4) ----
    opening: str = "standard"
    #: Fraction of games seeded with a capture handicap. This is what makes
    #: terminals reachable in generation one; it replaced the heuristic teacher.
    #:
    #: **Initial value only.** The live rate is annealed down during a run and
    #: lives in `state.json` (`RunState.handicap_rate`). Nothing may write the
    #: annealed value back here: this field is inside `config_hash`, so a run
    #: that mutated it would refuse to resume itself the moment the curriculum
    #: moved.
    handicap_rate: float = 0.7
    handicap_max: int = 5
    handicap_anneal: HandicapAnnealConfig = field(default_factory=HandicapAnnealConfig)
    #: Uniformly-random plies before search takes over. Decorrelates a
    #: generation's games; not searched and not recorded.
    random_opening_plies: int = 2
    max_plies: int = 200
    #: Adjudicate after this many plies without a capture. 0 = off.
    no_progress_plies: int = NO_PROGRESS_OFF

    # ---- targets ----
    #: Per-ply discount for the capture-map target.
    capture_gamma: float = 0.98

    def validate(self) -> None:
        _require(self.games_per_gen > 0, "self_play.games_per_gen must be > 0")
        _require(self.shard_games_per_file > 0, "self_play.shard_games_per_file must be > 0")
        _require(self.sims_fast > 0 and self.sims_full > 0, "self_play sims must be > 0")
        _require(
            self.sims_full >= self.sims_fast,
            f"self_play.sims_full ({self.sims_full}) must be >= sims_fast ({self.sims_fast})",
        )
        _require(
            0.0 <= self.full_search_rate <= 1.0, "self_play.full_search_rate must be in [0, 1]"
        )
        _require(0.0 <= self.handicap_rate <= 1.0, "self_play.handicap_rate must be in [0, 1]")
        _require(
            0 <= self.handicap_max <= 5,
            "self_play.handicap_max must be in [0, 5]; 6 would seed a finished game",
        )
        self.handicap_anneal.validate()
        _require(
            self.handicap_anneal.floor <= self.handicap_rate,
            f"self_play.handicap_anneal.floor ({self.handicap_anneal.floor}) is above the "
            f"initial self_play.handicap_rate ({self.handicap_rate}); the curriculum could "
            f"never move",
        )
        _require(self.temperature > 0.0, "self_play.temperature must be > 0")
        _require(self.max_plies > 0, "self_play.max_plies must be > 0")
        _require(self.batch_size > 0, "self_play.batch_size must be > 0")
        _require(
            self.opening in OPENINGS,
            f"self_play.opening must be one of {list(OPENINGS)}, got {self.opening!r}",
        )


@dataclass
class LossWeightsConfig:
    """Head weights in the total loss (MODEL.md §6.6). `policy` is fixed at 1.0
    by construction — it is the scale the others are expressed against."""

    value: float = 1.0
    score: float = 0.15
    capture_map: float = 0.15


@dataclass
class EmaConfig:
    """Exponential moving average of the weights. The EMA is what gets exported
    to ONNX and therefore what plays self-play and eval (MODEL.md §8)."""

    enabled: bool = True
    decay: float = 0.999
    #: Early on, a 0.999 EMA is still mostly the random initialisation. The
    #: effective decay is ramped as `min(decay, (1 + n) / (warmup + n))` over
    #: the first steps, the standard TF `num_updates` correction.
    warmup_steps: int = 10


@dataclass
class TrainConfig:
    # Minimum SGD steps per generation; training always does at least this many.
    steps_per_gen_min: int = 1000
    # Cap when self-play is still running after the minimum: keep consuming
    # otherwise-idle wall-clock up to here. null behaves like a fixed budget.
    steps_per_gen_max: int | None = None
    batch_size: int = 256
    learning_rate: float = 1.0e-3
    #: Step decay keyed to generation milestones: `{gen: lr}`, meaning "from
    #: this generation onward, use this LR". A constant LR for a whole run
    #: leaves strength on the table (MODEL.md §8). Empty = constant.
    lr_schedule: dict = field(default_factory=dict)
    # ---- AdamW (decoupled weight decay; no L2 term in the loss) ----
    weight_decay: float = 1.0e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1.0e-8
    grad_clip: float | None = 1.0
    loss_weights: LossWeightsConfig = field(default_factory=LossWeightsConfig)
    ema: EmaConfig = field(default_factory=EmaConfig)
    # ---- replay buffer ----
    replay_buffer_gens: int = 20
    replay_buffer_min_size: int = 1000
    symmetry_augment: bool = True
    poll_interval_ms: int = 250

    def learning_rate_at(self, gen: int) -> float:
        """LR for generation `gen`: the base rate, overridden by the highest
        milestone at or below `gen`."""
        lr = self.learning_rate
        for milestone in sorted(int(k) for k in self.lr_schedule):
            if gen >= milestone:
                lr = float(self.lr_schedule[milestone])
        return lr

    def validate(self) -> None:
        _require(self.steps_per_gen_min > 0, "train.steps_per_gen_min must be > 0")
        _require(
            self.steps_per_gen_max is None
            or self.steps_per_gen_max >= self.steps_per_gen_min,
            "train.steps_per_gen_max must be >= steps_per_gen_min",
        )
        _require(self.batch_size > 0, "train.batch_size must be > 0")
        _require(self.learning_rate > 0, "train.learning_rate must be > 0")
        _require(self.replay_buffer_gens > 0, "train.replay_buffer_gens must be > 0")
        _require(0.0 < self.ema.decay < 1.0, "train.ema.decay must be in (0, 1)")
        for k, v in self.lr_schedule.items():
            _require(
                isinstance(k, int) and not isinstance(k, bool),
                f"train.lr_schedule keys must be generation ints, got {k!r}",
            )
            _require(float(v) > 0, f"train.lr_schedule[{k}] must be > 0")


@dataclass
class ValidationConfig:
    """Held-out validation (MODEL.md §8.1) — the feedback loop the failed run
    lacked. `holdout_gen` is frozen once produced and never trained on again.

    Honest caveat: generation `holdout_gen` *is* sampled during its own
    generation's training, because a generation with nothing to train on is
    worse than a slightly warm validation set. It is frozen from the following
    generation onward, which is where the curve starts meaning something.
    """

    enabled: bool = True
    holdout_gen: int = 1
    every_gens: int = 1
    #: Positions per validation pass. Sampled with a fixed seed, so the same
    #: rows come back every generation and the curve is comparable.
    positions: int = 4096
    batch_size: int = 512
    seed: int = 20260727
    #: WARN when `policy_target_entropy / policy_uniform_entropy` exceeds this.
    #: At 1.0 search produced zero information and nothing downstream means
    #: anything — that is exactly how the previous run died.
    entropy_ratio_warn: float = 0.95


@dataclass
class AnchorLadderConfig:
    """Fixed opponents, run every N generations, converted to Elo (MODEL.md
    §8.1). Fixed anchors give a monotone curve; self-play gating cannot."""

    every_gens: int = 5  # 0 disables the ladder entirely
    #: Always run on the last generation of a run, whatever the cadence — the
    #: final number is the one anybody quotes.
    run_on_final_gen: bool = True
    games: int = 40
    #: Fallback sims; `@N` in an opponent spec overrides it for that opponent.
    simulations: int = 400
    opponents: list = field(
        default_factory=lambda: ["random", "heuristic@100", "heuristic@800"]
    )
    #: Earlier generations to freeze as extra rungs. A rung is skipped when the
    #: generation is not strictly earlier than the current one, or its ONNX has
    #: been collected by retention.
    frozen_gens: list = field(default_factory=lambda: [1])
    batch_size: int = 32
    c_puct: float = 1.4
    # Openings and early-ply sampling are what make N games N samples rather
    # than one game replayed N times (review §3.2).
    opening: str = "standard"
    random_opening_plies: int = 2
    temperature_plies: int = 10
    temperature: float = 1.0
    max_plies: int = 200
    no_progress_plies: int = NO_PROGRESS_OFF
    threads: int | None = None

    def validate(self) -> None:
        _require(
            self.opening in OPENINGS,
            f"anchor_ladder.opening must be one of {list(OPENINGS)}",
        )
        _require(self.games > 0, "anchor_ladder.games must be > 0")
        for spec in self.opponents:
            _require(
                isinstance(spec, str), f"anchor_ladder.opponents must be strings, got {spec!r}"
            )


@dataclass
class ExportConfig:
    """Reviewable-game JSON and the web artifact."""

    #: Games exported to `games/gen_NNN/` per generation. null = all of them.
    games_per_gen: int | None = 20
    #: Copy the best-by-ladder-Elo ONNX to `web_export_path`. With gating gone
    #: there is no promotion event, so "best" means best measured Elo.
    web_export: bool = True


@dataclass
class RetentionConfig:
    keep_last_checkpoints: int = 5
    keep_last_onnx: int = 25
    keep_last_shard_gens: int = 25
    keep_last_game_gens: int = 25


@dataclass
class RunConfig:
    """Top-level config. Each named field is one subgroup; the top-level
    scalars are run identity, outer-loop bounds and infrastructure."""

    run_id: str | None = None  # auto-generated if None
    seed: int = 0
    gens: int = 50
    runs_root: str = "runs"
    web_export_path: str = "web/public/models/best.onnx"
    #: Trunk size, from `model.abalone_net.PRESETS`.
    net_preset: str = "base"
    # Route Rust ORT inference through Apple's CoreML execution provider.
    # Measured self-play throughput: 0.9 pos/s on CPU vs 29.5 pos/s on CoreML
    # at batch 32 — the batched search plus fixed-width padding inverted the
    # old CPU-wins benchmark. Excluded from `config_hash`: it changes speed,
    # not meaning, so flipping it on resume is allowed.
    use_coreml: bool = False
    self_play: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    anchor_ladder: AnchorLadderConfig = field(default_factory=AnchorLadderConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    retention: RetentionConfig = field(default_factory=RetentionConfig)

    # -- loading ---------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: Path) -> RunConfig:
        raw = yaml.safe_load(Path(path).read_text()) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: top level must be a mapping, got {type(raw).__name__}")
        cfg = _from_dict(cls, raw)
        cfg.validate()
        return cfg

    def to_yaml(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(asdict(self), sort_keys=False, default_flow_style=False)
        )

    def validate(self) -> None:
        _require(self.gens > 0, "gens must be > 0")
        _require(
            self.net_preset in NET_PRESETS,
            f"net_preset must be one of {list(NET_PRESETS)}, got {self.net_preset!r}",
        )
        self.self_play.validate()
        self.train.validate()
        self.anchor_ladder.validate()
        _require(
            self.validation.holdout_gen >= 1,
            "validation.holdout_gen must be >= 1 (generations are 1-indexed)",
        )

    # -- hashing ---------------------------------------------------------------

    def hash(self) -> str:
        """Stable SHA-256 of the parts of the config that change what the run
        *means*. Resume compares this against `state.json`.

        `HASH_EXCLUDED` lists what is deliberately outside it: run identity,
        outer-loop bounds, and pure infrastructure that can be retuned mid-run
        without invalidating a single shard.

        This hash covers YAML only. Changing the ply cap in Rust, or the plane
        layout, invalidates every shard in the buffer and this would not
        notice — which is why `state.json` also carries the git SHA.
        """
        d = _prune(asdict(self), HASH_EXCLUDED)
        canonical = yaml.safe_dump(d, sort_keys=True, default_flow_style=False)
        return hashlib.sha256(canonical.encode()).hexdigest()


#: Dotted paths excluded from `config_hash`. Everything here is either run
#: identity, an outer-loop bound, or infrastructure — none of it changes the
#: distribution of the data or the shape of the model, so changing it on resume
#: is legitimate and must not invalidate the run.
HASH_EXCLUDED: frozenset[str] = frozenset(
    {
        "run_id",
        "gens",
        "runs_root",
        "web_export_path",
        "use_coreml",
        "self_play.worker_threads",
        "self_play.shard_games_per_file",
        "train.steps_per_gen_max",
        "train.poll_interval_ms",
        "anchor_ladder.threads",
        "export",
        "retention",
    }
)


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def _prune(d: dict[str, Any], excluded: frozenset[str], prefix: str = "") -> dict[str, Any]:
    """Copy of `d` without the dotted paths in `excluded`."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        path = f"{prefix}{k}"
        if path in excluded:
            continue
        out[k] = _prune(v, excluded, f"{path}.") if isinstance(v, dict) else v
    return out


def _from_dict(target: type, raw: dict[str, Any]) -> Any:
    """Recursively populate a dataclass tree from a plain dict.
    Unknown keys are rejected — a typo'd config must fail loudly."""
    if not is_dataclass(target):
        return raw
    if not isinstance(raw, dict):
        raise ValueError(f"{target.__name__} expects a mapping, got {type(raw).__name__}")
    fmap = {f.name: f for f in fields(target)}
    unknown = set(raw.keys()) - set(fmap.keys())
    if unknown:
        raise ValueError(
            f"Unknown config keys for {target.__name__}: {sorted(map(str, unknown))}. "
            f"Allowed: {sorted(fmap.keys())}"
        )
    kwargs: dict[str, Any] = {}
    for name, f in fmap.items():
        if name not in raw:
            continue  # use default
        v = raw[name]
        ftype = f.type
        if isinstance(ftype, type) and is_dataclass(ftype):
            kwargs[name] = _from_dict(ftype, v or {})
        elif get_origin(ftype) is None and is_dataclass(_resolve(ftype, target)):
            kwargs[name] = _from_dict(_resolve(ftype, target), v or {})
        else:
            kwargs[name] = v
    return target(**kwargs)


def _resolve(annotation: Any, owner: type) -> Any:
    """Best-effort resolution of a field annotation that may be a string
    (`from __future__ import annotations` is in effect here). Falls back to the
    raw annotation."""
    if isinstance(annotation, str):
        import sys

        mod = sys.modules.get(owner.__module__)
        if mod is not None:
            return getattr(mod, annotation, annotation)
    return annotation


__all__ = [
    "ANNEAL_MODES",
    "HASH_EXCLUDED",
    "NET_PRESETS",
    "NO_PROGRESS_OFF",
    "OPENINGS",
    "AnchorLadderConfig",
    "EmaConfig",
    "ExportConfig",
    "HandicapAnnealConfig",
    "LossWeightsConfig",
    "RetentionConfig",
    "RunConfig",
    "SelfPlayConfig",
    "TrainConfig",
    "ValidationConfig",
]
