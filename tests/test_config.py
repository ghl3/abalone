"""Config schema tests.

Three properties are worth pinning here, and only one of them is obvious:

1. **Every YAML in `config/` loads and round-trips.** A config that fails to
   parse is caught the moment someone runs it, so this is the cheap half.
2. **Unknown keys are rejected, at every level of the tree.** This is the half
   that matters. A knob renamed on one side of the fence and left stale in the
   YAML silently keeps its default, and the run measures something nobody
   asked for. It is also the guard against the deleted knobs
   (`evaluator_schedule`, `value_target_blend_*`, the whole `eval` gating
   group) creeping back in through a copy-pasted file.
3. **Defaults agree with the Rust `SelfPlayConfig::default()`.** Python owns
   the CLI invocation, Rust owns the semantics, and nothing but a test keeps
   them the same. The Rust source is parsed rather than transcribed, so the
   test cannot rot into a copy of what the defaults used to be.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from model.config import (
    HASH_EXCLUDED,
    NET_PRESETS,
    OPENINGS,
    AnchorLadderConfig,
    RunConfig,
    SelfPlayConfig,
    TrainConfig,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "config"
RUST_LIB = REPO_ROOT / "crates" / "selfplay" / "src" / "lib.rs"

CONFIG_FILES = sorted(CONFIG_DIR.glob("*.yaml"))

#: Config files that must exist by name. `validation.yaml` is the roadmap's
#: "is the signal back?" run and is referenced from the docs.
REQUIRED_CONFIGS = {"dry_run.yaml", "validation.yaml", "medium.yaml", "standard.yaml"}


def _ids(paths: list[Path]) -> list[str]:
    return [p.name for p in paths]


# --------------------------------------------------------------------------- #
# The files on disk                                                            #
# --------------------------------------------------------------------------- #


def test_required_configs_exist() -> None:
    assert REQUIRED_CONFIGS <= {p.name for p in CONFIG_FILES}


@pytest.mark.parametrize("path", CONFIG_FILES, ids=_ids(CONFIG_FILES))
def test_every_config_loads(path: Path) -> None:
    cfg = RunConfig.from_yaml(path)
    assert cfg.gens > 0
    assert cfg.net_preset in NET_PRESETS
    assert cfg.self_play.opening in OPENINGS
    assert cfg.anchor_ladder.opening in OPENINGS


@pytest.mark.parametrize("path", CONFIG_FILES, ids=_ids(CONFIG_FILES))
def test_every_config_round_trips(path: Path, tmp_path: Path) -> None:
    """`to_yaml` → `from_yaml` is the identity, which is what makes the frozen
    `runs/<id>/config.yaml` a faithful record of what actually ran."""
    cfg = RunConfig.from_yaml(path)
    out = tmp_path / "frozen.yaml"
    cfg.to_yaml(out)
    assert RunConfig.from_yaml(out) == cfg
    assert RunConfig.from_yaml(out).hash() == cfg.hash()


@pytest.mark.parametrize("path", CONFIG_FILES, ids=_ids(CONFIG_FILES))
def test_configs_are_internally_consistent(path: Path) -> None:
    cfg = RunConfig.from_yaml(path)
    assert cfg.self_play.sims_full >= cfg.self_play.sims_fast
    # A holdout beyond the run is a validation group that can never fire.
    if cfg.validation.enabled:
        assert cfg.validation.holdout_gen < cfg.gens, (
            f"{path.name}: holdout_gen {cfg.validation.holdout_gen} is not before the "
            f"end of a {cfg.gens}-generation run, so validation would never run"
        )
    # Frozen ladder rungs have to be reachable and retained.
    for g in cfg.anchor_ladder.frozen_gens:
        assert 0 < g < cfg.gens, f"{path.name}: frozen ladder gen {g} outside the run"


@pytest.mark.parametrize("path", CONFIG_FILES, ids=_ids(CONFIG_FILES))
def test_deleted_knobs_are_gone(path: Path) -> None:
    """The gate, the z/q blend and the evaluator schedule were deleted, not
    deprecated. `selfplay-batch` now rejects `--evaluator heuristic` outright,
    so a stale schedule is a crash five minutes into a run."""
    raw = yaml.safe_load(path.read_text()) or {}
    assert "eval" not in raw, f"{path.name}: gating group `eval` is gone"
    assert "evaluator_schedule" not in raw.get("self_play", {})
    train = raw.get("train", {})
    for dead in (
        "value_target_blend_start",
        "value_target_blend_end",
        "value_target_blend_done_by_gen",
        "value_loss_weight",
    ):
        assert dead not in train, f"{path.name}: `train.{dead}` was deleted"


# --------------------------------------------------------------------------- #
# Unknown-key rejection                                                        #
# --------------------------------------------------------------------------- #


def _write(tmp_path: Path, data: dict) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(data))
    return p


def test_unknown_top_level_key_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown config keys for RunConfig"):
        RunConfig.from_yaml(_write(tmp_path, {"gens": 3, "gensss": 4}))


@pytest.mark.parametrize(
    ("group", "key"),
    [
        ("self_play", "simulations_per_move"),  # renamed to sims_fast/sims_full
        ("self_play", "evaluator_schedule"),
        ("train", "value_target_blend_start"),
        ("train", "value_loss_weight"),
        ("validation", "holdout_generation"),
        ("anchor_ladder", "opponent"),
        ("retention", "web_export_on_promotion"),
        ("export", "games"),
    ],
)
def test_unknown_nested_key_rejected(tmp_path: Path, group: str, key: str) -> None:
    with pytest.raises(ValueError, match="Unknown config keys"):
        RunConfig.from_yaml(_write(tmp_path, {group: {key: 1}}))


def test_unknown_key_in_deeply_nested_group_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown config keys for LossWeightsConfig"):
        RunConfig.from_yaml(_write(tmp_path, {"train": {"loss_weights": {"policy": 1.0}}}))
    with pytest.raises(ValueError, match="Unknown config keys for EmaConfig"):
        RunConfig.from_yaml(_write(tmp_path, {"train": {"ema": {"momentum": 0.9}}}))


def test_gating_group_is_not_silently_accepted(tmp_path: Path) -> None:
    """The single most likely stale config: a copy of the old file."""
    with pytest.raises(ValueError, match="Unknown config keys for RunConfig"):
        RunConfig.from_yaml(_write(tmp_path, {"eval": {"gate_games": 21}}))


# --------------------------------------------------------------------------- #
# Validation of values                                                         #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("data", "match"),
    [
        ({"net_preset": "enormous"}, "net_preset"),
        ({"gens": 0}, "gens"),
        ({"self_play": {"opening": "belgian-daisy"}}, "opening"),
        ({"self_play": {"sims_fast": 800, "sims_full": 400}}, "sims_full"),
        ({"self_play": {"handicap_max": 6}}, "handicap_max"),
        ({"self_play": {"handicap_rate": 1.5}}, "handicap_rate"),
        ({"self_play": {"full_search_rate": -0.1}}, "full_search_rate"),
        ({"train": {"steps_per_gen_min": 100, "steps_per_gen_max": 10}}, "steps_per_gen_max"),
        ({"train": {"ema": {"decay": 1.0}}}, "decay"),
        ({"validation": {"holdout_gen": 0}}, "holdout_gen"),
    ],
)
def test_invalid_values_rejected(tmp_path: Path, data: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        RunConfig.from_yaml(_write(tmp_path, data))


def test_handicap_max_5_is_the_boundary(tmp_path: Path) -> None:
    """6 would seed an already-finished game — the same assertion the Rust
    `SelfPlayConfig::validate` makes."""
    RunConfig.from_yaml(_write(tmp_path, {"self_play": {"handicap_max": 5}}))


# --------------------------------------------------------------------------- #
# Agreement with the Rust defaults                                             #
# --------------------------------------------------------------------------- #


def _rust_selfplay_defaults() -> dict[str, str]:
    """Parse the `impl Default for SelfPlayConfig` block out of the Rust source.

    Parsed, not transcribed: a transcription becomes a second copy of the
    defaults and drifts in exactly the way this test exists to prevent.
    """
    src = RUST_LIB.read_text()
    m = re.search(
        r"impl Default for SelfPlayConfig \{.*?Self \{(.*?)\n        \}", src, re.S
    )
    assert m, "could not find `impl Default for SelfPlayConfig` in crates/selfplay/src/lib.rs"
    out: dict[str, str] = {}
    for line in m.group(1).splitlines():
        line = line.split("//")[0].strip()
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        out[key.strip()] = value.strip().rstrip(",")
    return out


#: Rust field name → Python field name. Only `capture_gamma` differs, and only
#: because the CLI flag is `--gamma`.
_SHARED_NUMERIC_FIELDS = [
    "sims_fast",
    "sims_full",
    "full_search_rate",
    "c_puct",
    "batch_size",
    "virtual_loss",
    "fpu_reduction",
    "dirichlet_alpha",
    "dirichlet_eps",
    "temperature_plies",
    "temperature",
    "handicap_rate",
    "random_opening_plies",
]


def test_rust_defaults_parse() -> None:
    defaults = _rust_selfplay_defaults()
    assert set(_SHARED_NUMERIC_FIELDS) <= set(defaults), (
        f"missing from the Rust default block: "
        f"{sorted(set(_SHARED_NUMERIC_FIELDS) - set(defaults))}"
    )


@pytest.mark.parametrize("field", _SHARED_NUMERIC_FIELDS)
def test_selfplay_defaults_match_rust(field: str) -> None:
    """Python builds the argv; Rust defines what the flags mean. If the two
    disagree about a default, a config that omits the knob silently gets a
    different value than the one documented on either side."""
    rust = _rust_selfplay_defaults()[field]
    expected = float(rust)
    actual = float(getattr(SelfPlayConfig(), field))
    assert actual == pytest.approx(expected), (
        f"self_play.{field}: Python default {actual} != Rust default {expected}"
    )


def test_symbolic_defaults_match_rust() -> None:
    """The three defaults the Rust side spells as constants rather than
    literals: the ply cap, the no-progress sentinel and the handicap ceiling."""
    rust = _rust_selfplay_defaults()
    assert rust["max_plies"] == "DEFAULT_MAX_PLIES"
    assert rust["no_progress_plies"] == "NO_PROGRESS_DISABLED"
    assert rust["handicap_max"] == "MAX_HANDICAP"
    assert rust["opening"] == "Opening::default()"
    assert rust["capture_gamma"] == "DEFAULT_CAPTURE_GAMMA"

    lib_rs = RUST_LIB.read_text()
    game_rs = (REPO_ROOT / "crates" / "game" / "src" / "game.rs").read_text()
    max_plies = int(
        re.search(r"pub const DEFAULT_MAX_PLIES: u32 = (\d+);", game_rs).group(1)
    )
    max_handicap = int(re.search(r"pub const MAX_HANDICAP: u8 = (\d+);", lib_rs).group(1))
    gamma = float(
        re.search(r"pub const DEFAULT_CAPTURE_GAMMA: f32 = ([\d.]+);", lib_rs).group(1)
    )

    d = SelfPlayConfig()
    assert d.max_plies == max_plies
    # 0 is the Python spelling of NO_PROGRESS_DISABLED; the binary translates.
    assert d.no_progress_plies == 0
    assert d.handicap_max == max_handicap
    assert d.capture_gamma == pytest.approx(gamma)
    # `Opening::default()` is `#[default] Standard`.
    assert d.opening == "standard"


def test_loss_weight_defaults_match_train_step() -> None:
    """MODEL.md §6.6: `w_v = 1.0`, `w_s = 0.15`, `w_c = 0.15`. The config must
    not quietly disagree with the module that applies them."""
    from model.train_step import DEFAULT_LOSS_WEIGHTS

    lw = TrainConfig().loss_weights
    assert (lw.value, lw.score, lw.capture_map) == (
        DEFAULT_LOSS_WEIGHTS.value,
        DEFAULT_LOSS_WEIGHTS.score,
        DEFAULT_LOSS_WEIGHTS.capture_map,
    )


def test_net_presets_match_abalone_net() -> None:
    from model.abalone_net import PRESETS

    assert set(NET_PRESETS) == set(PRESETS)


def test_anchor_ladder_defaults_are_the_documented_ladder() -> None:
    """MODEL.md §8.1 names the anchors explicitly: fixed opponents are the only
    thing that gives a monotone curve."""
    assert AnchorLadderConfig().opponents == ["random", "heuristic@100", "heuristic@800"]


# --------------------------------------------------------------------------- #
# The config hash                                                              #
# --------------------------------------------------------------------------- #


def test_hash_is_stable_across_instances() -> None:
    assert RunConfig().hash() == RunConfig().hash()


def test_hash_is_stable_across_key_order(tmp_path: Path) -> None:
    a = _write(tmp_path / "a", {"seed": 1, "self_play": {"c_puct": 1.5, "sims_fast": 10}})
    b = _write(tmp_path / "b", {"self_play": {"sims_fast": 10, "c_puct": 1.5}, "seed": 1})
    assert RunConfig.from_yaml(a).hash() == RunConfig.from_yaml(b).hash()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: setattr(c, "seed", 99),
        lambda c: setattr(c, "net_preset", "large"),
        lambda c: setattr(c.self_play, "sims_full", 1600),
        lambda c: setattr(c.self_play, "max_plies", 400),
        lambda c: setattr(c.self_play, "opening", "belgian"),
        lambda c: setattr(c.self_play, "handicap_rate", 0.1),
        lambda c: setattr(c.train, "batch_size", 512),
        lambda c: setattr(c.train, "learning_rate", 1e-4),
        lambda c: setattr(c.train.loss_weights, "score", 0.5),
        lambda c: setattr(c.train.ema, "decay", 0.99),
        lambda c: setattr(c.validation, "holdout_gen", 3),
        lambda c: setattr(c.anchor_ladder, "games", 100),
    ],
)
def test_semantic_changes_change_the_hash(mutate) -> None:
    """Anything that changes what the data means, or what the model is, must
    invalidate a resume."""
    base = RunConfig()
    changed = RunConfig()
    mutate(changed)
    assert changed.hash() != base.hash()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: setattr(c, "run_id", "some-run-20260727-1200"),
        lambda c: setattr(c, "gens", 999),
        lambda c: setattr(c, "runs_root", "/tmp/elsewhere"),
        lambda c: setattr(c, "web_export_path", "web/public/models/other.onnx"),
        lambda c: setattr(c, "use_coreml", True),
        lambda c: setattr(c.self_play, "worker_threads", 3),
        lambda c: setattr(c.self_play, "shard_games_per_file", 64),
        lambda c: setattr(c.train, "steps_per_gen_max", 12345),
        lambda c: setattr(c.train, "poll_interval_ms", 1000),
        lambda c: setattr(c.anchor_ladder, "threads", 2),
        lambda c: setattr(c.export, "games_per_gen", 1),
        lambda c: setattr(c.retention, "keep_last_onnx", 1),
    ],
)
def test_infrastructure_changes_do_not_change_the_hash(mutate) -> None:
    """Retuning threads, retention or the outer-loop bound mid-run is
    legitimate and must not invalidate a single shard."""
    base = RunConfig()
    changed = RunConfig()
    mutate(changed)
    assert changed.hash() == base.hash()


def test_hash_excluded_paths_all_exist() -> None:
    """A typo in `HASH_EXCLUDED` fails open — the path silently stays in the
    hash and nobody notices until a resume is refused for no reason."""
    from dataclasses import asdict

    flat = asdict(RunConfig())

    def resolve(path: str) -> bool:
        node = flat
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return False
            node = node[part]
        return True

    missing = sorted(p for p in HASH_EXCLUDED if not resolve(p))
    assert not missing, f"HASH_EXCLUDED names non-existent config paths: {missing}"


def test_learning_rate_schedule_is_a_step_decay() -> None:
    tr = TrainConfig(learning_rate=1e-3, lr_schedule={20: 3e-4, 35: 1e-4})
    assert tr.learning_rate_at(1) == pytest.approx(1e-3)
    assert tr.learning_rate_at(19) == pytest.approx(1e-3)
    assert tr.learning_rate_at(20) == pytest.approx(3e-4)
    assert tr.learning_rate_at(34) == pytest.approx(3e-4)
    assert tr.learning_rate_at(35) == pytest.approx(1e-4)
    assert tr.learning_rate_at(1000) == pytest.approx(1e-4)


def test_learning_rate_schedule_is_order_independent() -> None:
    a = TrainConfig(lr_schedule={35: 1e-4, 20: 3e-4})
    b = TrainConfig(lr_schedule={20: 3e-4, 35: 1e-4})
    assert [a.learning_rate_at(g) for g in range(1, 40)] == [
        b.learning_rate_at(g) for g in range(1, 40)
    ]


def test_empty_lr_schedule_is_constant() -> None:
    tr = TrainConfig(learning_rate=7e-4)
    assert all(tr.learning_rate_at(g) == pytest.approx(7e-4) for g in range(1, 100))
