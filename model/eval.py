"""Subprocess wrappers around the Rust eval-match and selfplay-batch
binaries. These do not run inference themselves; they just shell out
and parse JSON results.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _bin(name: str) -> Path:
    """Resolve a Rust release-build binary by name. Raises if missing
    so we fail loudly instead of silently shelling out to nothing."""
    candidates = [
        REPO_ROOT / "target" / "release" / name,
        REPO_ROOT / "target" / "debug" / name,
    ]
    for p in candidates:
        if p.exists():
            return p
    if shutil.which(name) is not None:
        return Path(shutil.which(name))
    raise FileNotFoundError(
        f"could not find Rust binary `{name}`. Build with "
        f"`cargo build --release -p abalone-selfplay --bin {name}`."
    )


@dataclass
class MatchResult:
    player_a: str
    player_b: str
    games: int
    simulations: int
    wins_a: int
    wins_b: int
    draws: int
    winrate_a: float
    winrate_a_excluding_draws: float
    elapsed_seconds: float

    @classmethod
    def from_json(cls, path: Path) -> MatchResult:
        d = json.loads(Path(path).read_text())
        return cls(**d)


def start_self_play(
    *,
    model_onnx: Path,
    out_dir: Path,
    games: int,
    simulations: int,
    c_puct: float,
    temperature_plies: int,
    temperature: float,
    dirichlet_alpha: float,
    dirichlet_eps: float,
    shard_games: int,
    threads: int | None,
    seed: int,
    stdout=None,
    stderr=None,
) -> subprocess.Popen:
    """Spawn `selfplay-batch` as a non-blocking child process. Returns
    the `Popen` so the caller can poll/wait while doing other work
    (the training loop streams shard files as they appear)."""
    cmd = [
        str(_bin("selfplay-batch")),
        "--model", str(model_onnx),
        "--out-dir", str(out_dir),
        "--games", str(games),
        "--simulations", str(simulations),
        "--c-puct", str(c_puct),
        "--temperature-plies", str(temperature_plies),
        "--temperature", str(temperature),
        "--dirichlet-alpha", str(dirichlet_alpha),
        "--dirichlet-eps", str(dirichlet_eps),
        "--shard-games", str(shard_games),
        "--seed", str(seed),
    ]
    if threads is not None:
        cmd += ["--threads", str(threads)]
    return subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=stdout, stderr=stderr)


def _format_player(spec: str | Path) -> str:
    """Convert a player spec to the CLI string the eval-match binary expects."""
    if isinstance(spec, Path):
        return f"model:{spec}"
    if spec in ("heuristic", "random"):
        return spec
    if spec.startswith("model:"):
        return spec
    raise ValueError(f"unsupported player spec: {spec!r}")


def run_eval_match(
    *,
    player_a: str | Path,
    player_b: str | Path,
    games: int,
    simulations: int,
    c_puct: float,
    out_json: Path,
    seed: int = 0,
) -> MatchResult:
    """Spawn `eval-match` and parse its JSON result."""
    cmd = [
        str(_bin("eval-match")),
        "--player-a", _format_player(player_a),
        "--player-b", _format_player(player_b),
        "--games", str(games),
        "--simulations", str(simulations),
        "--c-puct", str(c_puct),
        "--out-json", str(out_json),
        "--seed", str(seed),
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    return MatchResult.from_json(Path(out_json))
