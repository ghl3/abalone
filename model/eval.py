"""Subprocess wrappers around the Rust `selfplay-batch` and `eval-match`
binaries. Nothing here runs inference; it builds argv, shells out, and parses
JSON.

Two things in this module are load-bearing and easy to get subtly wrong:

**`winrate_a` changed meaning.** It is now the standard score
`(wins + 0.5·draws) / games`. The old `wins_a / games` — which scored a draw as
a loss and put a 0.55 threshold out of mathematical reach in a drawish game —
survives as `wins_only_rate_a`. Anything comparing against a historical
`winrate_a` number is comparing two different quantities.

**`MatchResult` parsing is tolerant.** `eval-match` grew ~25 fields and will
grow more; `cls(**d)` on a fixed dataclass turns every future field into a
`TypeError` at the worst possible moment (after the match has been played).
Known fields are promoted to attributes, everything else stays reachable
through `raw` / `get()` / `[]`.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

#: A player spec: `random`, `heuristic`, `heuristic@800`, `model:p.onnx`,
#: `model:p.onnx@400`. The `@N` suffix is a per-player simulation override and
#: is the whole point of the ladder — `heuristic@100` and `heuristic@800` are
#: different opponents, and "beats heuristic@800 at equal sims" is the
#: milestone (MODEL.md §8.2).
_SPEC_RE = re.compile(r"^(?:random|heuristic|model:.+?)(?:@(\d+))?$")

#: A match of more than this many games that produced no more distinct
#: transcripts than this is one game wearing a large denominator. This is the
#: regression signal for the determinism bug of review §3.2, where a 21-game
#: gate was really 2 games.
DETERMINISM_GAMES_FLOOR = 2


def _bin(name: str) -> Path:
    """Resolve a Rust binary by name. Raises if missing so we fail loudly
    instead of silently shelling out to nothing."""
    for p in (REPO_ROOT / "target" / "release" / name, REPO_ROOT / "target" / "debug" / name):
        if p.exists():
            return p
    found = shutil.which(name)
    if found is not None:
        return Path(found)
    raise FileNotFoundError(
        f"could not find Rust binary `{name}`. Build with "
        f"`cargo build --release -p abalone-selfplay --bin {name}`."
    )


# --------------------------------------------------------------------------- #
# Match results                                                                #
# --------------------------------------------------------------------------- #


@dataclass
class MatchResult:
    """One `eval-match` result. Unknown keys are preserved, not fatal."""

    player_a: str = ""
    player_b: str = ""
    games: int = 0
    simulations: int = 0
    simulations_a: int = 0
    simulations_b: int = 0

    # -- tallies --
    wins_a: int = 0
    wins_b: int = 0
    draws: int = 0

    # -- rates --
    #: `(wins + 0.5·draws) / games`, the primary metric.
    score_a: float = float("nan")
    score_a_stderr: float = float("nan")
    #: Alias of `score_a`. It used to mean `wins_a / games`; it does not now.
    winrate_a: float = float("nan")
    #: The pre-fix definition, `wins_a / games`.
    wins_only_rate_a: float = float("nan")
    winrate_a_excluding_draws: float = float("nan")
    decisive_rate: float = float("nan")

    # -- strength --
    elo_a: float = float("nan")
    elo_a_stderr: float = float("nan")
    elo_a_ci95_lo: float = float("nan")
    elo_a_ci95_hi: float = float("nan")
    #: True when the score hit 0 or 1 and the Elo is a sample-size bound.
    elo_a_clamped: bool = False

    # -- diagnostics --
    mean_plies: float = float("nan")
    mean_score_diff_a: float = float("nan")
    mean_abs_score_diff: float = float("nan")
    distinct_transcripts: int = 0
    elapsed_seconds: float = float("nan")

    #: Every key the binary emitted, including `per_game` and anything added
    #: after this dataclass was last touched.
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MatchResult:
        known = {f.name for f in cls.__dataclass_fields__.values()} - {"raw"}
        kwargs = {k: v for k, v in d.items() if k in known}
        return cls(**kwargs, raw=dict(d))

    @classmethod
    def from_json(cls, path: Path) -> MatchResult:
        return cls.from_dict(json.loads(Path(path).read_text()))

    # -- access to whatever we did not model --
    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def __contains__(self, key: str) -> bool:
        return key in self.raw

    @property
    def per_game(self) -> list[dict[str, Any]]:
        return list(self.raw.get("per_game") or [])

    @property
    def suspicious_transcripts(self) -> bool:
        """True when the match is really one game replayed N times."""
        return (
            self.games > DETERMINISM_GAMES_FLOOR
            and self.distinct_transcripts <= DETERMINISM_GAMES_FLOOR
        )

    def summary(self) -> str:
        """One line, the numbers that matter, safe against missing fields."""
        ci = ""
        if not math.isnan(self.elo_a_ci95_lo) and not math.isnan(self.elo_a_ci95_hi):
            ci = f" [{self.elo_a_ci95_lo:+.0f}, {self.elo_a_ci95_hi:+.0f}]"
        bounded = " (bounded)" if self.elo_a_clamped else ""
        return (
            f"{self.wins_a}-{self.wins_b}-{self.draws} (W-L-D)  "
            f"score {self.score_a:.3f}±{self.score_a_stderr:.3f}  "
            f"elo {self.elo_a:+.0f}{ci}{bounded}  "
            f"decisive {self.decisive_rate:.2f}  plies {self.mean_plies:.0f}  "
            f"transcripts {self.distinct_transcripts}/{self.games}"
        )


@dataclass
class LadderRung:
    """One anchor-ladder match: the opponent spec and what it measured."""

    opponent: str
    result: MatchResult
    #: False when the rung was skipped (e.g. a frozen checkpoint not on disk).
    played: bool = True

    @property
    def elo(self) -> float:
        return self.result.elo_a

    @property
    def score(self) -> float:
        return self.result.score_a


# --------------------------------------------------------------------------- #
# Player specs                                                                 #
# --------------------------------------------------------------------------- #


def _format_player(spec: str | Path) -> str:
    """Normalise a player spec to the string `eval-match` expects.

    Accepts `Path` (an ONNX file), and the string forms `random`,
    `heuristic`, `heuristic@N`, `model:<path>`, `model:<path>@N`. The `@N`
    per-player simulation override passes straight through — it is how the
    ladder distinguishes `heuristic@100` from `heuristic@800`.
    """
    if isinstance(spec, Path):
        return f"model:{spec}"
    if not isinstance(spec, str):
        raise TypeError(f"player spec must be str or Path, got {type(spec).__name__}")
    if not _SPEC_RE.match(spec):
        raise ValueError(
            f"unsupported player spec: {spec!r}. Expected random | heuristic[@N] | "
            f"model:<path.onnx>[@N]"
        )
    return spec


def model_spec(onnx: Path | str, simulations: int | None = None) -> str:
    """`model:<path>` with an optional `@N` simulation override."""
    return f"model:{onnx}" + (f"@{int(simulations)}" if simulations else "")


# --------------------------------------------------------------------------- #
# Self-play                                                                    #
# --------------------------------------------------------------------------- #


def start_self_play(
    *,
    model_onnx: Path,
    out_dir: Path,
    games: int,
    sims_fast: int,
    sims_full: int,
    full_search_rate: float,
    c_puct: float,
    batch_size: int,
    virtual_loss: float,
    fpu_reduction: float,
    temperature_plies: int,
    temperature: float,
    opening: str,
    handicap_rate: float,
    handicap_max: int,
    random_opening_plies: int,
    max_plies: int,
    no_progress_plies: int,
    capture_gamma: float,
    dirichlet_alpha: float,
    dirichlet_eps: float,
    shard_games: int,
    threads: int | None,
    seed: int,
    stdout=None,
    stderr=None,
) -> subprocess.Popen:
    """Spawn `selfplay-batch` as a non-blocking child. Returns the `Popen` so
    the caller can poll it while training on the shards it is producing.

    There is exactly one evaluator: the network. `--evaluator heuristic` is
    rejected by the binary — the hand-written evaluator is retired from the
    training loop (MODEL.md §1) and survives only as a ladder opponent.

    `no_progress_plies=0` means "rule off"; the binary maps 0 onto its
    `NO_PROGRESS_DISABLED` sentinel.
    """
    if model_onnx is None:
        raise ValueError("start_self_play requires model_onnx: self-play is NN-driven")
    cmd = [
        str(_bin("selfplay-batch")),
        "--evaluator", "model",
        "--model", str(model_onnx),
        "--out-dir", str(out_dir),
        "--games", str(games),
        "--sims-fast", str(sims_fast),
        "--sims-full", str(sims_full),
        "--full-search-rate", str(full_search_rate),
        "--batch-size", str(batch_size),
        "--virtual-loss", str(virtual_loss),
        "--fpu-reduction", str(fpu_reduction),
        "--c-puct", str(c_puct),
        "--temperature-plies", str(temperature_plies),
        "--temperature", str(temperature),
        "--opening", opening,
        "--handicap-rate", str(handicap_rate),
        "--handicap-max", str(handicap_max),
        "--random-opening-plies", str(random_opening_plies),
        "--max-plies", str(max_plies),
        "--no-progress-plies", str(no_progress_plies),
        "--gamma", str(capture_gamma),
        "--dirichlet-alpha", str(dirichlet_alpha),
        "--dirichlet-eps", str(dirichlet_eps),
        "--shard-games", str(shard_games),
        "--seed", str(seed),
    ]
    if threads is not None:
        cmd += ["--threads", str(threads)]
    return subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=stdout, stderr=stderr)


# --------------------------------------------------------------------------- #
# Eval matches                                                                 #
# --------------------------------------------------------------------------- #


def run_eval_match(
    *,
    player_a: str | Path,
    player_b: str | Path,
    games: int,
    simulations: int,
    c_puct: float,
    out_json: Path,
    batch_size: int = 32,
    opening: str = "standard",
    random_opening_plies: int = 2,
    temperature_plies: int = 10,
    temperature: float = 1.0,
    max_plies: int = 200,
    no_progress_plies: int = 0,
    seed: int = 0,
    threads: int | None = None,
    stdout=None,
    stderr=None,
) -> MatchResult:
    """Play a match and parse the JSON summary.

    `random_opening_plies` and `temperature_plies` are not optional garnish:
    MCTS over a deterministic evaluator from a fixed start is a pure function
    of the position, so without them N games is one game replayed N times.
    Check `MatchResult.suspicious_transcripts` on the way out.

    `no_progress_plies=0` means "off"; the flag is simply not passed, because
    `eval-match` takes the raw ply count with no zero sentinel.
    """
    cmd = [
        str(_bin("eval-match")),
        "--player-a", _format_player(player_a),
        "--player-b", _format_player(player_b),
        "--games", str(games),
        "--simulations", str(simulations),
        "--c-puct", str(c_puct),
        "--batch-size", str(batch_size),
        "--opening", opening,
        "--random-opening-plies", str(random_opening_plies),
        "--temperature-plies", str(temperature_plies),
        "--temperature", str(temperature),
        "--max-plies", str(max_plies),
        "--out-json", str(out_json),
        "--seed", str(seed),
    ]
    if no_progress_plies:
        cmd += ["--no-progress-plies", str(no_progress_plies)]
    if threads is not None:
        cmd += ["--threads", str(threads)]
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True, stdout=stdout, stderr=stderr)
    return MatchResult.from_json(Path(out_json))


def run_ladder(
    *,
    model_onnx: Path,
    opponents: list[str],
    games: int,
    simulations: int,
    c_puct: float,
    eval_dir: Path,
    logs_dir: Path,
    gen: int,
    seed: int,
    batch_size: int = 32,
    opening: str = "standard",
    random_opening_plies: int = 2,
    temperature_plies: int = 10,
    temperature: float = 1.0,
    max_plies: int = 200,
    no_progress_plies: int = 0,
    threads: int | None = None,
    on_rung=None,
) -> list[LadderRung]:
    """Play `model_onnx` against each opponent in turn and return the rungs.

    Fixed anchors are the whole point: they give a monotone Elo curve that
    self-play gating cannot (MODEL.md §8.1). Each match gets its own seed
    derived from `(seed, rung index)` so rungs are independent but the ladder
    as a whole is reproducible. `on_rung(rung)` is called after each match so
    the caller can log progress without waiting for the whole ladder.
    """
    eval_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    rungs: list[LadderRung] = []
    for i, opponent in enumerate(opponents):
        tag = _slug(opponent)
        log_path = logs_dir / f"gen_{gen:03d}_ladder_{tag}.log"
        with open(log_path, "w") as logf:
            result = run_eval_match(
                player_a=model_spec(model_onnx),
                player_b=opponent,
                games=games,
                simulations=simulations,
                c_puct=c_puct,
                batch_size=batch_size,
                opening=opening,
                random_opening_plies=random_opening_plies,
                temperature_plies=temperature_plies,
                temperature=temperature,
                max_plies=max_plies,
                no_progress_plies=no_progress_plies,
                out_json=eval_dir / f"gen_{gen:03d}_ladder_{tag}.json",
                seed=seed + 1013 * i,
                threads=threads,
                stdout=logf,
                stderr=subprocess.STDOUT,
            )
        rung = LadderRung(opponent=opponent, result=result)
        rungs.append(rung)
        if on_rung is not None:
            on_rung(rung)
    return rungs


def _slug(spec: str) -> str:
    """Filesystem-safe tag for a player spec (`model:a/b.onnx@400` → `b_400`)."""
    if spec.startswith("model:"):
        head, _, sims = spec[len("model:") :].partition("@")
        stem = Path(head).stem
        return f"{stem}_{sims}" if sims else stem
    return re.sub(r"[^A-Za-z0-9]+", "_", spec).strip("_") or "player"


def mean_elo(rungs: list[LadderRung], *, exclude_prefix: str = "model:") -> float:
    """Mean Elo over the *fixed* rungs — the ones whose strength does not move
    between generations.

    Frozen earlier checkpoints are excluded by default: they are useful as a
    "did we beat our past self" check, but averaging against a moving reference
    would make the tracked best-checkpoint number meaningless.
    """
    vals = [
        r.elo
        for r in rungs
        if r.played and not r.opponent.startswith(exclude_prefix) and not math.isnan(r.elo)
    ]
    return sum(vals) / len(vals) if vals else float("nan")


__all__ = [
    "DETERMINISM_GAMES_FLOOR",
    "LadderRung",
    "MatchResult",
    "mean_elo",
    "model_spec",
    "run_eval_match",
    "run_ladder",
    "start_self_play",
]
