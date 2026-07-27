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
    #: Base decrement per firing, at a signal sitting exactly on the target.
    step: float = 0.05
    #: The step is *proportional*: `step × clamp(signal / target, 1, K)`, where
    #: K is this. A fixed step is too timid once the signal saturates — the
    #: six-generation validation run measured `natural_termination_rate = 1.00`
    #: (four times the 0.25 target) from generation 4 and still crawled down by
    #: 0.05 a generation, needing eight more generations to reach the floor.
    #: At K = 4 a saturated signal retires the crutch in three.
    max_step_multiple: float = 4.0
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
        _require(
            self.max_step_multiple >= 1.0,
            "self_play.handicap_anneal.max_step_multiple must be >= 1 (1 is the old "
            "fixed-step behaviour; below 1 would shrink the configured step)",
        )
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
    #: Preferred step budget expressed as *passes over the replay buffer*:
    #: `steps = epochs × buffer_size / batch_size`, clamped into
    #: `[steps_per_gen_min, steps_per_gen_max]`. A step count fixed in absolute
    #: terms silently becomes an epoch count that depends on how long the games
    #: happened to be: the validation run's generation 3 took 1500 steps × 256
    #: over a 19k-position buffer — **20 raw epochs in one generation** — purely
    #: because games had shortened from 148 plies to 67. null = old behaviour
    #: (run to `steps_per_gen_max` whenever self-play is still going).
    target_epochs_per_gen: float | None = None
    #: WARN above this many raw epochs over the buffer in one generation.
    #: "Raw" ignores D6 augmentation, which multiplies distinct samples by 12;
    #: the buffer is still only as diverse as the positions in it.
    epochs_per_gen_warn: float = 8.0
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
        _require(
            self.target_epochs_per_gen is None or self.target_epochs_per_gen > 0,
            "train.target_epochs_per_gen must be > 0 (null to disable)",
        )
        _require(
            self.epochs_per_gen_warn > 0, "train.epochs_per_gen_warn must be > 0"
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
class RollingHoldoutConfig:
    """The *current-distribution* held-out set (MODEL.md §8.1).

    `ValidationConfig.holdout_gen` is frozen forever, which is what makes its
    curve comparable — and also what makes it progressively less relevant: by
    generation 30 it is scoring the network on positions produced by a network
    thirty generations weaker. It cannot cleanly show overfitting to the replay
    buffer, because "worse on generation 1" and "different from generation 1"
    look the same.

    The rolling holdout fixes that without a second self-play run:
    `ReplayBuffer.mark_rolling_holdout` withholds `fraction` of generation `N`'s
    own games — **by whole game**, since positions inside a game share `z` and
    are near-duplicates — and those rows are never sampled into a training
    batch, ever. They are the newest positions in existence and the network has
    provably never seen them, so `val_rolling/loss_total` against the
    generation's mean training loss is memorisation and nothing else.

    Skipped automatically whenever the trainable pool outside the generation is
    below `train.replay_buffer_min_size` (generation 2, where generation 1 is
    already the frozen holdout, is the case that matters).
    """

    enabled: bool = True
    #: Cadence, counted from `ValidationConfig.holdout_gen`. 1 = every
    #: generation, which is the point — a holdout that is stale is the problem
    #: this group exists to solve. 0 disables.
    every_gens: int = 1
    #: Fraction of the generation's games withheld. The cost is real and
    #: permanent — at `every_gens: 1` the run trains on 90% of its data — and it
    #: buys the only clean overfitting signal in the system.
    fraction: float = 0.10
    #: Positions per scoring pass. Sampled with replacement from the slice.
    positions: int = 4096


@dataclass
class ValidationConfig:
    """Held-out validation (MODEL.md §8.1) — the feedback loop the failed run
    lacked. `holdout_gen` is frozen once produced and never trained on again.

    Honest caveat: generation `holdout_gen` *is* sampled during its own
    generation's training, because a generation with nothing to train on is
    worse than a slightly warm validation set. It is frozen from the following
    generation onward, which is where the curve starts meaning something.

    Two holdouts, two jobs. The frozen one (`val_frozen/…`) is a fixed ruler:
    same rows, same seed, every generation, so its curve is comparable end to
    end. The rolling one (`val_rolling/…`, see `RollingHoldoutConfig`) is
    always current-distribution, which is the only way to see the network
    overfitting the replay buffer rather than merely drifting away from
    generation 1.
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
    #: WARN when `val_rolling/loss_total` exceeds the generation's mean training
    #: loss by more than this. Both are the same weighted total over the same
    #: distribution — one on data the network trained on, one on data it has
    #: never seen — so a widening gap is overfitting to the buffer and nothing
    #: else. Absolute, in nats, because the total loss is a sum of CEs.
    overfit_warn_delta: float = 0.35
    rolling: RollingHoldoutConfig = field(default_factory=RollingHoldoutConfig)

    def is_rolling_holdout_gen(self, gen: int) -> bool:
        """True when generation `gen`'s own shards are withheld from its own
        training. Never the frozen holdout generation — that one is already
        special-cased and has nothing else to train on."""
        r = self.rolling
        if not (self.enabled and r.enabled and r.every_gens > 0):
            return False
        if gen <= self.holdout_gen:
            return False
        return (gen - self.holdout_gen) % r.every_gens == 0


@dataclass
class AnchorLadderConfig:
    """The measurement of strength (MODEL.md §8.1).

    **The rungs are mostly the network's own past selves.** The
    six-generation validation run swept every rung of the old ladder 12-0-0 by
    generation 6 — `random`, `heuristic@100`, `heuristic@800` and frozen
    `gen_001` — and every one of them clamped at the same +545 Elo sample-size
    bound. A mean over four clamped rungs is not a measurement, and averaging
    it into `best.onnx` selection made that selection arbitrary. Anchors have
    to span the strength range, and the only opponents guaranteed to keep pace
    with the network are earlier copies of it.

    Three kinds of rung, and they are not interchangeable:

    * **floor** (`opponents`) — `random` and one cheap `heuristic@N`. A sanity
      check: losing to these is a bug, beating them is not an achievement.
      Deliberately fewer games (`floor_games`) because the answer is binary.
    * **frozen** (`frozen_gens`) — absolute generations, kept for the whole
      run. Fixed reference points, so their Elo curve is comparable end to end
      and is what the headline number averages.
    * **trailing** (`trailing_gens`) — `gen − k`. A *moving* reference: it
      improves as the network does, so it never saturates and it is the rung
      that still has resolution at generation 40. Excluded from the headline
      mean precisely because it moves.

    `heuristic@800` is gone. Its evaluator is `tanh(6.0·capture_diff + …)` and
    `tanh(6.0) = 0.99999`, so it cannot tell being one marble up from four: one
    capture ahead it believes it has won, one behind it believes it has lost.
    Its weights were also tuned under the old 400-ply draw-at-cap rules. It is
    not a yardstick, and MODEL.md §8.2 no longer treats beating it as a
    milestone.
    """

    every_gens: int = 5  # 0 disables the ladder entirely
    #: Always run on the last generation of a run, whatever the cadence — the
    #: final number is the one anybody quotes.
    run_on_final_gen: bool = True
    #: Games per *checkpoint* rung (frozen and trailing). 12 was far too few:
    #: ±0.5/√12 on the score is a ±145 Elo confidence interval, wider than any
    #: generation-to-generation gain the ladder exists to detect.
    games: int = 40
    #: Games per *floor* rung. A sanity check needs a fraction of the games a
    #: measurement does, and the floor rungs are the ones that clamp anyway.
    floor_games: int = 16
    #: Fallback sims; `@N` in an opponent spec overrides it for that opponent.
    #: 200 rather than 400: ladder cost is games × plies × sims, and games buy
    #: resolution while sims buy an absolute strength nobody reads off a ladder.
    simulations: int = 200
    #: Floor anchors only. Checkpoint rungs come from `frozen_gens` /
    #: `trailing_gens` and are resolved against the run's own checkpoint dir.
    opponents: list = field(default_factory=lambda: ["random", "heuristic@100"])
    #: Absolute earlier generations to freeze as rungs. A rung is skipped when
    #: the generation is not strictly earlier than the current one, or its ONNX
    #: has been collected by retention.
    frozen_gens: list = field(default_factory=lambda: [1])
    #: Offsets back from the current generation: `[5]` plays `gen − 5`. The
    #: rung that keeps its resolution once the fixed anchors are all swept.
    trailing_gens: list = field(default_factory=lambda: [5])
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
        _require(self.floor_games > 0, "anchor_ladder.floor_games must be > 0")
        for spec in self.opponents:
            _require(
                isinstance(spec, str), f"anchor_ladder.opponents must be strings, got {spec!r}"
            )
        for k in self.trailing_gens:
            _require(
                isinstance(k, int) and not isinstance(k, bool) and k > 0,
                f"anchor_ladder.trailing_gens are positive generation offsets, got {k!r}",
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
        _require(
            0.0 <= self.validation.rolling.fraction < 1.0,
            "validation.rolling.fraction must be in [0, 1); 1.0 would withhold the "
            "entire generation from its own training",
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
        "train.epochs_per_gen_warn",
        "train.poll_interval_ms",
        "validation.overfit_warn_delta",
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
    "RollingHoldoutConfig",
    "RunConfig",
    "SelfPlayConfig",
    "TrainConfig",
    "ValidationConfig",
]
