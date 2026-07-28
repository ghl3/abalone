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

**A clamped rung is a bound, not a measurement.** When a match ends 12-0-0 the
score is 1.0, the Elo estimate is `+inf`, and `eval-match` reports the
sample-size bound instead with `elo_a_clamped: true`. Every rung of the
six-generation validation run's generation-6 ladder came back clamped at the
same +545, and the mean over them was reported as if it were a number. Anything
that aggregates rungs here also reports `clamped_fraction`, and `ladder_summary`
emits `ladder/all_clamped` so the loop can say out loud that the ladder has run
out of resolution.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

#: A player spec: `random`, `heuristic`, `heuristic@800`, `model:p.onnx`,
#: `model:p.onnx@400`. The `@N` suffix is a per-player simulation override:
#: `heuristic@100` and `heuristic@800` are different opponents, and a frozen
#: checkpoint played at the ladder's own simulation budget is a fair rung.
#: Beating `heuristic@800` is no longer a milestone (MODEL.md §8.2) — a 1.1M
#: network cleared it after 180 games — but the spec grammar is unchanged.
_SPEC_RE = re.compile(r"^(?:random|heuristic|model:.+?)(?:@(\d+))?$")

#: A match of more than this many games that produced no more distinct
#: transcripts than this is one game wearing a large denominator. This is the
#: regression signal for the determinism bug of review §3.2, where a 21-game
#: gate was really 2 games.
DETERMINISM_GAMES_FLOOR = 2

#: The three kinds of ladder rung (see `AnchorLadderConfig`). Only the kind
#: decides whether a rung's Elo belongs in the headline mean: `FLOOR` and
#: `FROZEN` are fixed references, `TRAILING` moves with the network.
KIND_FLOOR = "floor"
KIND_FROZEN = "frozen"
KIND_TRAILING = "trailing"
LADDER_KINDS = (KIND_FLOOR, KIND_FROZEN, KIND_TRAILING)

#: Kinds whose strength does not change between generations, and which may
#: therefore be averaged into a curve that is comparable end to end.
FIXED_REFERENCE_KINDS = frozenset({KIND_FLOOR, KIND_FROZEN})


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


def opponent_label(spec: str) -> str:
    """Short display name for a ladder rung. Frozen checkpoints are passed to
    `eval-match` as absolute paths, and an absolute path in a log line buries
    the numbers it is there to show."""
    if spec.startswith("model:"):
        head, sep, sims = spec[len("model:") :].partition("@")
        return f"model:{Path(head).name}" + (sep + sims if sep else "")
    return spec


@dataclass(frozen=True)
class LadderOpponent:
    """One rung to play: what, how many games, and what role it plays.

    Games are per-rung rather than per-ladder because the rungs are not the
    same kind of question. A floor anchor answers "is anything catastrophically
    broken", which needs a handful of games; a checkpoint rung answers "how much
    stronger am I than N generations ago", which needs enough games for the
    confidence interval to exclude zero.
    """

    spec: str
    games: int
    kind: str = KIND_FLOOR

    @property
    def label(self) -> str:
        return opponent_label(self.spec)

    @property
    def is_fixed_reference(self) -> bool:
        return self.kind in FIXED_REFERENCE_KINDS


@dataclass
class LadderRung:
    """One anchor-ladder match: the opponent spec and what it measured."""

    opponent: str
    result: MatchResult
    #: False when the rung was skipped (e.g. a frozen checkpoint not on disk).
    played: bool = True
    #: One of `LADDER_KINDS`. Defaults to `floor` so a bare
    #: `LadderRung(opponent, result)` is still meaningful.
    kind: str = KIND_FLOOR

    @property
    def elo(self) -> float:
        return self.result.elo_a

    @property
    def score(self) -> float:
        return self.result.score_a

    @property
    def clamped(self) -> bool:
        """True when the score hit 0 or 1 and the Elo is a sample-size bound
        rather than an estimate."""
        return bool(self.result.elo_a_clamped)

    @property
    def label(self) -> str:
        return opponent_label(self.opponent)

    @property
    def is_fixed_reference(self) -> bool:
        return self.kind in FIXED_REFERENCE_KINDS

    def elo_str(self) -> str:
        """Elo with its 95% CI, always — an Elo without one is an opinion."""
        r = self.result
        if math.isnan(r.elo_a):
            return "n/a"
        ci = ""
        if not math.isnan(r.elo_a_ci95_lo) and not math.isnan(r.elo_a_ci95_hi):
            ci = f" [{r.elo_a_ci95_lo:+.0f}, {r.elo_a_ci95_hi:+.0f}]"
        return f"{r.elo_a:+.0f}{ci}" + (" (bound)" if self.clamped else "")


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
    opponents: Sequence[LadderOpponent],
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

    Each match gets its own seed derived from `(seed, rung index)` so rungs are
    independent but the ladder as a whole is reproducible. `on_rung(rung)` is
    called after each match so the caller can log progress — and cost — without
    waiting for the whole ladder, which is the single most expensive phase of a
    generation.
    """
    eval_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    rungs: list[LadderRung] = []
    for i, opponent in enumerate(opponents):
        tag = _slug(opponent.spec)
        log_path = logs_dir / f"gen_{gen:03d}_ladder_{tag}.log"
        with open(log_path, "w") as logf:
            result = run_eval_match(
                player_a=model_spec(model_onnx),
                player_b=opponent.spec,
                games=opponent.games,
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
        rung = LadderRung(opponent=opponent.spec, result=result, kind=opponent.kind)
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


def _measured(rungs: Sequence[LadderRung]) -> list[LadderRung]:
    return [r for r in rungs if r.played and not math.isnan(r.elo)]


def mean_elo(rungs: Sequence[LadderRung]) -> float:
    """Mean Elo over the rungs with a *fixed* reference strength — the floor
    anchors and the absolute frozen checkpoints.

    Trailing rungs (`gen − k`) are excluded: they get stronger every time the
    network does, so averaging them in would flatten by construction the very
    curve the ladder exists to draw.

    Clamped rungs are *included*, because a clamp is a lower bound on strength
    and dropping the rungs that were swept would bias the mean downward exactly
    when the network is doing well. That makes the mean a lower bound too, and
    is precisely why `ladder_summary` reports `clamped_fraction` beside it and
    the loop refuses to print one without the other.

    **With no frozen rungs there is nothing fixed left but the floors**, and the
    floors are swept from about generation 3 onwards — the mean would be a
    constant sample-size bound reported as a strength, which is the exact
    failure the `clamped` machinery exists to expose. A trailing-only ladder is
    a deliberate configuration (it is the only shape that survives a run being
    extended later), so in that case this falls back to the trailing rungs and
    the number means "how far ahead of my recent past selves am I". That is not
    comparable across generations, and nothing should treat it as if it were:
    `ladder/score_vs_gen_minus_k` is the per-generation signal to read.
    """
    elo, _ = mean_elo_basis(rungs)
    return elo


def mean_elo_basis(rungs: Sequence[LadderRung]) -> tuple[float, str]:
    """`mean_elo` plus the name of what it actually averaged.

    Callers that print the number must print the basis with it. "Mean over
    fixed-reference rungs" is a lie when the fixed references were all swept and
    the value came from the trailing rungs instead, and a mislabelled metric is
    the failure mode this project has paid for repeatedly.
    """
    measured = _measured(rungs)
    fixed = [r.elo for r in measured if r.is_fixed_reference]
    if any(not r.clamped for r in measured if r.is_fixed_reference):
        return sum(fixed) / len(fixed), "fixed-reference rungs"
    trailing = [r.elo for r in measured if r.kind == KIND_TRAILING]
    if trailing:
        return sum(trailing) / len(trailing), "trailing rungs (no fixed rung resolved)"
    if fixed:
        return sum(fixed) / len(fixed), "fixed-reference rungs, all clamped"
    return float("nan"), "nothing measured"


def resolved_regressions(rungs: Sequence[LadderRung]) -> list[LadderRung]:
    """Rungs where the network is *confidently* worse than the opponent.

    "Confidently" is the whole point: a rung is only counted when the upper end
    of its 95% interval on the score sits below 0.5, so ordinary
    generation-to-generation noise cannot trigger it.

    This exists because the obvious promotion rule — "did I beat my immediate
    predecessor" — is unusable at the resolution we have. Generation 13 scored
    0.547 against generation 12 with a 95% interval of roughly ±0.17: a
    coin-flip dressed as a decision, and keying `best.onnx` on it would make the
    published model hop about at random. The ladder's job here is to catch a
    *regression*, not to certify every increment.

    Floor rungs count too. Losing to `random` is not a subtle signal.
    """
    out: list[LadderRung] = []
    for r in _measured(rungs):
        se = r.result.score_a_stderr
        if math.isnan(r.score) or math.isnan(se):
            continue
        if r.score + 1.96 * se < 0.5:
            out.append(r)
    return out


def clamped_fraction(rungs: Sequence[LadderRung]) -> float:
    """Fraction of the measured rungs whose Elo is a sample-size bound.

    At 1.0 the ladder measured nothing: every opponent was swept, and the
    "+545" that comes out is a property of the game count, not of the network.
    """
    measured = _measured(rungs)
    if not measured:
        return float("nan")
    return sum(1 for r in measured if r.clamped) / len(measured)


def _trailing_offsets(
    rungs: Sequence[LadderRung], gen: int | None
) -> list[tuple[LadderRung, int]]:
    """Trailing rungs paired with the offset `k` they represent.

    The offset is not stored on the rung — `ladder_opponents` resolves `gen − k`
    to a filename and the rung only carries that path — so it is recovered as
    `gen − (generation in the filename)`.

    `gen` must be supplied by the caller and is not inferred from the rungs.
    Anchoring on the newest trailing opponent instead would be right only when
    `1` is among the offsets, and silently off by a constant otherwise: a
    gauntlet of `[2, 4, 8]` at generation 12 would label its rungs 0, 2 and 6.
    A mislabelled offset is worse than a missing one, because the series looks
    plottable and answers a different question every generation.
    """
    if gen is None:
        return []
    out: list[tuple[LadderRung, int]] = []
    for rung in rungs:
        if rung.kind != KIND_TRAILING:
            continue
        m = re.search(r"gen_(\d+)\.onnx", rung.opponent)
        if m:
            offset = gen - int(m.group(1))
            if offset > 0:
                out.append((rung, offset))
    return out


def ladder_summary(rungs: Sequence[LadderRung], gen: int | None = None) -> dict[str, float]:
    """Flat, already-namespaced `ladder/…` metrics for one ladder.

    Per-rung numbers are always emitted — the headline mean is a convenience,
    not the result. Every Elo comes with its 95% interval, and every rung with
    the flag saying whether that interval is real.
    """
    measured = _measured(rungs)
    out: dict[str, float] = {
        "ladder/rungs": float(len(rungs)),
        "ladder/rungs_measured": float(len(measured)),
        "ladder/rungs_clamped": float(sum(1 for r in measured if r.clamped)),
        "ladder/clamped_fraction": clamped_fraction(rungs),
        "ladder/all_clamped": float(bool(measured) and all(r.clamped for r in measured)),
        "ladder/elo_mean": mean_elo(rungs),
        "ladder/seconds": float(
            sum(
                r.result.elapsed_seconds
                for r in rungs
                if not math.isnan(r.result.elapsed_seconds)
            )
        ),
    }
    # The mean's own interval: independent matches, so the standard errors add
    # in quadrature. Reported so nobody quotes a mean Elo bare.
    ses = [
        r.result.elo_a_stderr
        for r in measured
        if r.is_fixed_reference and not math.isnan(r.result.elo_a_stderr)
    ]
    if ses:
        se = math.sqrt(sum(s * s for s in ses)) / len(ses)
        out["ladder/elo_mean_stderr"] = se
        out["ladder/elo_mean_ci95_lo"] = out["ladder/elo_mean"] - 1.96 * se
        out["ladder/elo_mean_ci95_hi"] = out["ladder/elo_mean"] + 1.96 * se

    for kind in LADDER_KINDS:
        vals = [r.elo for r in measured if r.kind == kind]
        if vals:
            out[f"ladder/elo_mean_{kind}"] = sum(vals) / len(vals)

    # The per-generation signal, keyed by *offset* rather than by opponent name.
    #
    # `ladder/score/gen_007` is a different series every generation, so it
    # cannot be plotted as a trend. `ladder/score_vs_gen_minus_2` is the same
    # question asked at every generation — "how do I do against the network I
    # was two generations ago" — and it is stationary in a way absolute Elo is
    # not. A fixed anchor saturates once beaten; `gen − k` improves as fast as
    # the network does, so a flat 0.75 here means the learning rate is holding
    # and a drift toward 0.50 means it has stalled.
    #
    # It also survives a run being extended. Nothing here depends on any other
    # generation's ladder having been run, so adding five more generations to a
    # finished run yields five more comparable points and changes none of the
    # earlier ones.
    for rung, offset in _trailing_offsets(rungs, gen):
        out[f"ladder/score_vs_gen_minus_{offset}"] = rung.result.score_a
        out[f"ladder/elo_vs_gen_minus_{offset}"] = rung.result.elo_a
        out[f"ladder/clamped_vs_gen_minus_{offset}"] = float(rung.clamped)

    for rung in rungs:
        tag = _slug(rung.opponent)
        r = rung.result
        out[f"ladder/elo/{tag}"] = r.elo_a
        out[f"ladder/elo_ci95_lo/{tag}"] = r.elo_a_ci95_lo
        out[f"ladder/elo_ci95_hi/{tag}"] = r.elo_a_ci95_hi
        out[f"ladder/score/{tag}"] = r.score_a
        out[f"ladder/score_stderr/{tag}"] = r.score_a_stderr
        out[f"ladder/clamped/{tag}"] = float(rung.clamped)
        out[f"ladder/games/{tag}"] = float(r.games)
        out[f"ladder/mean_plies/{tag}"] = r.mean_plies
        out[f"ladder/distinct_transcripts/{tag}"] = float(r.distinct_transcripts)
        out[f"ladder/seconds/{tag}"] = r.elapsed_seconds
    return out


__all__ = [
    "DETERMINISM_GAMES_FLOOR",
    "FIXED_REFERENCE_KINDS",
    "KIND_FLOOR",
    "KIND_FROZEN",
    "KIND_TRAILING",
    "LADDER_KINDS",
    "LadderOpponent",
    "LadderRung",
    "MatchResult",
    "clamped_fraction",
    "ladder_summary",
    "mean_elo",
    "mean_elo_basis",
    "resolved_regressions",
    "model_spec",
    "opponent_label",
    "run_eval_match",
    "run_ladder",
    "start_self_play",
]
