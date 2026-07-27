"""Game export — self-play shards to reviewable JSON.

Turns raw parquet shards (`docs/ARCHITECTURE.md` §5.4) into the game format the
review UI loads (§7.4), one JSON file per game under
`runs/<id>/games/gen_NNN/game_NNNN.json`.

**Why this exists.** Every self-play position already carries the full MCTS visit
distribution and root value from the moment it was generated. Surfacing that is a
training-pipeline debugger as much as a product feature: it makes "the policy
target is uniform" something you can *see* rather than infer from a loss curve.
The previous run sat at a policy loss of exactly `ln(62)` — the visit
distributions carried zero information — and nobody noticed for three
generations. `summarise_generation` reports `policy_target_entropy` beside
`policy_uniform_entropy` (= `ln(mean legal moves)`) so that comparison is one
line of CLI output, and the names match `model/validate.py` deliberately.

`summarise_generation` is also where the **curriculum control signal** is
measured: `natural_termination_rate`, the fraction of unseeded games that reach
six captures before the ply cap, plus the seeded/unseeded split of *every* rate
it reports, which a human reads to sanity-check it. `model/curriculum.py` turns
that into a `handicap_rate` (MODEL.md §4).

`selfplay_metrics(summary)` re-emits the whole block under the **`selfplay/`**
namespace as plain numbers — properties of THIS generation's games, as opposed
to `val_rolling/` and `val_frozen/` (model quality on held-out positions) and
`buffer/` (what the training window contains). `summarise_generation` itself
keeps bare keys because `model/curriculum.py` reads two of them directly.

**Sign conventions — the easiest thing here to get wrong and the hardest to
notice.** The shard stores `z`, `score_diff` and `q` POV-relative to the side to
move *at that position*, so raw values alternate sign down a trajectory. This
module handles that in exactly two ways and nowhere else:

* `result.score_diff` and `result.outcome` are normalised to a FIXED POV
  (**Black**), by multiplying each row's POV-relative label by `+1` for a
  Black-to-move row and `-1` for a White-to-move row. Every row of a game must
  agree once normalised; disagreement is a corrupt shard and raises `ShardError`.
* Per-move `q` / `z` / `score_diff` stay POV-relative (that is what search
  actually reported) and each move carries an explicit `value_pov` naming the
  side they belong to, so the UI cannot mis-sign the eval bar.

**Notation is not our job.** Only the flat move `idx` is emitted; the WASM engine
owns the canonical `Move` display and renders notation client-side. Reimplementing
it in Python would be a fifth cross-language contract to keep in sync.

The reader is written against the schema in ARCHITECTURE §5.4 rather than against
whatever the Rust writer happens to emit — extra columns are ignored, missing
ones are a skip with a clear reason, and move indices are range-checked against
the canonical `MOVE_SPACE`.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from model.encoder import MOVE_SPACE

#: The full v2 shard schema, in declaration order. Documentation and a fixture
#: reference; the reader only *requires* `REQUIRED_COLUMNS`.
SHARD_COLUMNS: tuple[str, ...] = (
    "game_id",
    "seed",
    "opening",
    "handicap_black",
    "handicap_white",
    "own_bb_lo",
    "own_bb_hi",
    "opp_bb_lo",
    "opp_bb_hi",
    "black_losses",
    "white_losses",
    "turn",
    "ply",
    "max_plies",
    "move_played",
    "is_full_search",
    "z",
    "score_diff",
    "q",
    "child_move_idxs",
    "child_visits",
    "cap_map_idx",
    "cap_map_val",
)

#: Columns the export actually reads. Projecting to these keeps the (large)
#: capture-map and bitboard columns off the wire entirely.
#:
#: `black_losses` / `white_losses` are here for `captures_per_100_plies`: the
#: capture *differential* in `score_diff` cannot recover the capture *total*,
#: and the total is the bloodbath indicator.
REQUIRED_COLUMNS: tuple[str, ...] = (
    "game_id",
    "seed",
    "opening",
    "handicap_black",
    "handicap_white",
    "black_losses",
    "white_losses",
    "turn",
    "ply",
    "max_plies",
    "move_played",
    "is_full_search",
    "z",
    "score_diff",
    "q",
    "child_move_idxs",
    "child_visits",
)

BLACK = 0
WHITE = 1

TURN_NAMES: dict[int, str] = {BLACK: "black", WHITE: "white"}
OPENING_NAMES: dict[int, str] = {0: "standard", 1: "belgian_daisy"}

BLACK_WINS = "black_wins"
WHITE_WINS = "white_wins"
DRAW = "draw"

#: `q` is float32; six decimals is lossless enough for a display value and keeps
#: the JSON free of 0.18000000715255737.
_Q_DIGITS = 6

_GEN_DIR_RE = re.compile(r"^gen_(\d+)$")

NAN = float("nan")


class ShardError(Exception):
    """A shard file or a game within one is unusable. Callers skip and report."""


@dataclass(frozen=True)
class Skipped:
    """One thing that could not be exported, and why."""

    source: str
    reason: str

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.source}: {self.reason}"


@dataclass
class ExportResult:
    """What one generation's export produced.

    `games` holds the games actually written to disk, which `export.games_per_gen`
    caps. `summary` is computed over **every** game in the generation's shards,
    capped or not: it feeds the curriculum controller (MODEL.md §4), and a
    controller reading a 20-of-200 subsample would be measuring the export
    limit as much as the network.
    """

    gen: int
    games: list[dict[str, Any]] = field(default_factory=list)
    paths: list[Path] = field(default_factory=list)
    skipped: list[Skipped] = field(default_factory=list)
    shards_read: int = 0
    #: `summarise_generation` over all games; `None` only if it never ran.
    summary: dict[str, Any] | None = None


# --------------------------------------------------------------------------- #
# POV                                                                          #
# --------------------------------------------------------------------------- #


def pov_sign(turn: int) -> int:
    """Multiplier taking a value POV-relative to `turn` into Black's POV.

    Shard `z`, `score_diff` and `q` alternate sign down a trajectory because they
    are relative to whoever is to move. This is the only place that fact is
    encoded.
    """
    if turn == BLACK:
        return 1
    if turn == WHITE:
        return -1
    raise ShardError(f"turn must be 0 (black) or 1 (white), got {turn!r}")


def outcome_name(z_black: int) -> str:
    """Map a Black-POV outcome in {+1, 0, -1} to the exported outcome string."""
    if z_black > 0:
        return BLACK_WINS
    if z_black < 0:
        return WHITE_WINS
    return DRAW


# --------------------------------------------------------------------------- #
# Reading                                                                      #
# --------------------------------------------------------------------------- #


def read_shard(path: str | Path) -> pa.Table:
    """Read one shard, projected to `REQUIRED_COLUMNS`.

    Raises `ShardError` for anything unreadable, truncated, or missing a column
    the export needs — the caller's cue to skip the file and carry on.
    """
    path = Path(path)
    try:
        schema = pq.read_schema(path)
    except Exception as exc:  # any pyarrow/IO failure is a skip, not a crash
        raise ShardError(f"unreadable parquet ({type(exc).__name__}: {exc})") from exc

    missing = [c for c in REQUIRED_COLUMNS if c not in schema.names]
    if missing:
        raise ShardError(f"missing columns: {', '.join(missing)}")

    try:
        return pq.read_table(path, columns=list(REQUIRED_COLUMNS))
    except Exception as exc:  # truncated or corrupt body
        raise ShardError(f"unreadable parquet ({type(exc).__name__}: {exc})") from exc


def split_games(table: pa.Table | Iterable[Mapping[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    """Group shard rows into games by `game_id`, each sorted by `ply`.

    Rows for one game are contiguous and ply-ordered in practice; nothing here
    relies on it. A single shard may hold several games, and one game may be
    split across shards — see `collect_games`.
    """
    rows: Iterable[Mapping[str, Any]]
    rows = table.to_pylist() if isinstance(table, pa.Table) else table

    games: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        games.setdefault(int(row["game_id"]), []).append(dict(row))
    return {
        gid: sorted(rs, key=lambda r: int(r["ply"])) for gid, rs in sorted(games.items())
    }


def find_shards(shards_dir: str | Path) -> list[Path]:
    """Shard files for one generation, in a stable order. Empty if absent."""
    shards_dir = Path(shards_dir)
    if not shards_dir.is_dir():
        return []
    return sorted(p for p in shards_dir.glob("*.parquet") if p.is_file())


def find_generations(shards_root: str | Path) -> list[int]:
    """Generation numbers with a `gen_NNN` shard directory, ascending."""
    shards_root = Path(shards_root)
    if not shards_root.is_dir():
        return []
    gens = []
    for entry in shards_root.iterdir():
        m = _GEN_DIR_RE.match(entry.name)
        if m and entry.is_dir():
            gens.append(int(m.group(1)))
    return sorted(gens)


def collect_games(
    paths: Iterable[str | Path],
) -> tuple[dict[int, list[dict[str, Any]]], list[Skipped]]:
    """Read every shard and merge rows into games keyed by `game_id`.

    Merging across files matters: `game_id` is unique within a (run, generation),
    so a game whose rows straddle two shard files still comes back whole. Files
    that fail to read are reported, not raised.
    """
    merged: dict[int, list[dict[str, Any]]] = {}
    skipped: list[Skipped] = []
    for path in paths:
        path = Path(path)
        try:
            table = read_shard(path)
        except ShardError as exc:
            skipped.append(Skipped(str(path), str(exc)))
            continue
        for gid, rows in split_games(table).items():
            merged.setdefault(gid, []).extend(rows)
    ordered = {
        gid: sorted(rows, key=lambda r: int(r["ply"])) for gid, rows in sorted(merged.items())
    }
    return ordered, skipped


# --------------------------------------------------------------------------- #
# Export                                                                       #
# --------------------------------------------------------------------------- #


def _unique(rows: Sequence[Mapping[str, Any]], column: str) -> int:
    """One per-game constant, asserted constant across the game's rows."""
    values = {int(r[column]) for r in rows}
    if len(values) != 1:
        raise ShardError(f"{column} is not constant within the game: {sorted(values)}")
    return values.pop()


def _move_index(value: Any, what: str) -> int:
    idx = int(value)
    if not 0 <= idx < MOVE_SPACE:
        raise ShardError(f"{what} {idx} outside move space [0, {MOVE_SPACE})")
    return idx


def _visit_pairs(row: Mapping[str, Any]) -> list[list[int]]:
    """`(move_idx, visits)` pairs sorted by visits desc, move index asc."""
    idxs = list(row["child_move_idxs"] or [])
    visits = list(row["child_visits"] or [])
    if len(idxs) != len(visits):
        raise ShardError(
            f"child_move_idxs/child_visits length mismatch ({len(idxs)} vs {len(visits)})"
        )
    pairs = [
        (_move_index(i, "child move index"), int(v))
        for i, v in zip(idxs, visits, strict=True)
    ]
    pairs.sort(key=lambda p: (-p[1], p[0]))
    return [[i, v] for i, v in pairs]


def export_game(
    rows: Sequence[Mapping[str, Any]],
    run_id: str,
    gen: int,
) -> dict[str, Any]:
    """Build the review-UI JSON for one game from its shard rows.

    `rows` need not be sorted. Everything the UI needs beyond the board itself is
    here; the board is replayed from `opening` + `handicap` through the moves by
    the WASM engine.

    See the module docstring for the POV contract: `result` is Black-relative,
    per-move values are relative to `value_pov`.
    """
    if not rows:
        raise ShardError("no rows for game")
    rows = sorted(rows, key=lambda r: int(r["ply"]))

    game_id = _unique(rows, "game_id")
    seed = _unique(rows, "seed")
    opening = _unique(rows, "opening")
    handicap_black = _unique(rows, "handicap_black")
    handicap_white = _unique(rows, "handicap_white")
    max_plies = _unique(rows, "max_plies")

    # Normalise the per-position labels to Black's POV. Every row must agree once
    # the alternating sign is removed — if they do not, the shard is corrupt and
    # a fixed-POV `result` would be a coin flip.
    z_black = {int(r["z"]) * pov_sign(int(r["turn"])) for r in rows}
    if len(z_black) != 1:
        raise ShardError(f"z disagrees across the game once POV-normalised: {sorted(z_black)}")
    score_black = {int(r["score_diff"]) * pov_sign(int(r["turn"])) for r in rows}
    if len(score_black) != 1:
        raise ShardError(
            f"score_diff disagrees across the game once POV-normalised: {sorted(score_black)}"
        )

    moves = []
    for row in rows:
        turn = int(row["turn"])
        pov_sign(turn)  # validates the encoding
        pairs = _visit_pairs(row)
        moves.append(
            {
                "ply": int(row["ply"]),
                "turn": TURN_NAMES[turn],
                "idx": _move_index(row["move_played"], "move_played"),
                "is_full_search": bool(row["is_full_search"]),
                "q": round(float(row["q"]), _Q_DIGITS),
                "z": int(row["z"]),
                "score_diff": int(row["score_diff"]),
                "value_pov": TURN_NAMES[turn],
                "visits": pairs,
                "total_visits": sum(v for _, v in pairs),
            }
        )

    plies = int(rows[-1]["ply"]) + 1
    captures, captures_in_play = _capture_counts(rows, plies, max_plies)

    return {
        "run_id": str(run_id),
        "gen": int(gen),
        "game_id": game_id,
        "opening": OPENING_NAMES.get(opening, f"unknown_{opening}"),
        "handicap": [handicap_black, handicap_white],
        "seed": seed,
        "max_plies": max_plies,
        "result": {
            "outcome": outcome_name(next(iter(z_black))),
            "score_diff": next(iter(score_black)),
            "plies": plies,
            # Marbles off the board at the end, and how many of those were
            # earned rather than handed over by the curriculum. See
            # `_capture_counts`.
            "captures": captures,
            "captures_in_play": captures_in_play,
        },
        "moves": moves,
    }


def _capture_counts(
    rows: Sequence[Mapping[str, Any]], plies: int, max_plies: int
) -> tuple[int, int]:
    """`(total captures at the end, captures made during play)`.

    The shard records `black_losses` / `white_losses` **before** each move, so
    the last row is the position the final move was played from and the final
    move's own capture is in no row. Two facts close that gap:

    * a move pushes off at most one marble (`Game::apply_with_capture` returns
      at most one cell), so the miss is 0 or 1;
    * a game that ended short of its ply cap ended *because* the sixth capture
      landed, so on those the final move captured exactly one.

    A capped game's final move is genuinely unknowable, and is counted as no
    capture — an undercount of at most one marble across ~200 plies.

    "During play" excludes the handicap the curriculum granted at seeding: a
    game that starts at (3, 2) was given five captures, and counting those
    would make every seeded game look like a bloodbath.
    """
    first = int(rows[0]["black_losses"]) + int(rows[0]["white_losses"])
    last = int(rows[-1]["black_losses"]) + int(rows[-1]["white_losses"])
    natural = max_plies > 0 and plies < max_plies
    total = last + (1 if natural else 0)
    return total, total - first


def game_filename(game_id: int) -> str:
    return f"game_{int(game_id):04d}.json"


def export_generation(
    shards_dir: str | Path,
    out_dir: str | Path,
    run_id: str,
    gen: int,
    limit: int | None = None,
) -> ExportResult:
    """Export every game under `shards_dir` to `out_dir/game_NNNN.json`.

    A shard or a game that cannot be read is recorded in `ExportResult.skipped`
    and the batch continues — one truncated file (a self-play worker killed
    mid-write is the common case) must not cost a generation's export.

    `limit` caps what is *written*, not what is *measured*: every game is built
    and folded into `result.summary`, and only the first `limit` are kept and
    written out. The generator below is what keeps that from doubling peak
    memory on a 200-game generation — a game that is not being written is
    summarised and dropped in the same step.
    """
    shards_dir = Path(shards_dir)
    out_dir = Path(out_dir)

    paths = find_shards(shards_dir)
    result = ExportResult(gen=int(gen), shards_read=len(paths))
    if not paths:
        result.skipped.append(Skipped(str(shards_dir), "no shard files found"))
        result.summary = summarise_generation(())
        return result

    games, skipped = collect_games(paths)
    result.skipped.extend(skipped)

    def build_and_write() -> Iterable[dict[str, Any]]:
        for game_id, rows in games.items():
            try:
                game = export_game(rows, run_id, gen)
            except ShardError as exc:
                result.skipped.append(Skipped(f"game {game_id}", str(exc)))
                continue
            if limit is None or len(result.games) < limit:
                path = out_dir / game_filename(game_id)
                try:
                    out_dir.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(game), encoding="utf-8")
                    result.games.append(game)
                    result.paths.append(path)
                except OSError as exc:
                    # A failed write costs the file, not the statistics.
                    result.skipped.append(Skipped(str(path), f"write failed ({exc})"))
            yield game

    result.summary = summarise_generation(build_and_write())
    return result


def load_exported_games(games_dir: str | Path) -> list[dict[str, Any]]:
    """Read back `game_NNNN.json` files, skipping any that will not parse."""
    games_dir = Path(games_dir)
    if not games_dir.is_dir():
        return []
    games = []
    for path in sorted(games_dir.glob("game_*.json")):
        try:
            games.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return games


# --------------------------------------------------------------------------- #
# Summary                                                                      #
# --------------------------------------------------------------------------- #


def visit_entropy(visits: Iterable[int]) -> float:
    """Shannon entropy (nats) of a visit distribution. `nan` if it is empty.

    Zero-visit children contribute nothing, so a uniform distribution over `n`
    visited children returns exactly `ln(n)` — the pathological value, and the
    thing `summarise_generation` exists to make visible.
    """
    counts = [int(v) for v in visits]
    total = sum(counts)
    if total <= 0:
        return NAN
    entropy = 0.0
    for v in counts:
        if v > 0:
            p = v / total
            entropy -= p * math.log(p)
    return entropy


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else NAN


def is_unseeded(game: Mapping[str, Any]) -> bool:
    """True when the game was played from the plain opening.

    By the *recorded* handicap, not by whether self-play chose the game for
    seeding: the two handicaps are drawn independently per side, so a game
    selected for seeding legitimately comes out at `(0, 0)` and is, in every way
    that matters to the measurement, an unseeded game.
    """
    try:
        black, white = game["handicap"][0], game["handicap"][1]
    except (KeyError, IndexError, TypeError):
        return False
    return int(black) == 0 and int(white) == 0


def terminated_naturally(game: Mapping[str, Any]) -> bool:
    """True when the game ended on six captures rather than at the ply cap.

    With the no-progress rule off (the default), `Game::state()` has exactly two
    exits: a side reaching `WIN_THRESHOLD` captures, or `ply >= max_plies`, which
    is adjudicated on capture differential. A recorded ply count short of the
    game's own cap therefore means the six captures happened — no separate flag
    needed, and none exists in the shard schema.

    The reading is conservative in one corner: a game whose sixth capture lands
    on the very move that reaches the cap counts as capped. That errs toward
    holding the curriculum, which is the safe direction.

    **This is only sound while `no_progress_plies` is 0.** With the rule on, a
    third exit stops the game early without six captures and this returns a false
    positive; `model.curriculum.AnnealStats.from_summary` refuses to use the
    statistic in that case.
    """
    try:
        plies = int(game["result"]["plies"])
        cap = int(game["max_plies"])
    except (KeyError, TypeError, ValueError):
        return False
    return cap > 0 and plies < cap


def captures_in_play(game: Mapping[str, Any]) -> float:
    """Marbles pushed off during play, excluding the seeded head start.

    `nan` for a game exported before `result.captures_in_play` existed, so an
    old `runs/` directory reads back as "not measured" rather than as zero
    captures.
    """
    try:
        return float(game["result"]["captures_in_play"])
    except (KeyError, TypeError, ValueError):
        return NAN


@dataclass
class _Split:
    """One half of the seeded / unseeded split of a generation's games.

    Every rate the summary reports is tracked on both sides. Seeded games start
    a capture from the end and look nothing like unseeded ones; a pooled rate
    is a weighted average of two different populations whose weights move with
    the curriculum, which makes the pooled number move even when neither
    population does.
    """

    games: int = 0
    decisive: int = 0
    draws: int = 0
    black_wins: int = 0
    white_wins: int = 0
    natural: int = 0
    plies: int = 0
    abs_score: int = 0
    positions: int = 0
    policy_rows: int = 0
    #: Captures and plies over the games whose capture count is known, so a
    #: mixture of new and pre-`captures_in_play` exports cannot silently
    #: deflate the rate.
    captures: int = 0
    capture_games: int = 0
    capture_plies: int = 0

    def add(
        self,
        *,
        outcome: str,
        natural: bool,
        plies: int,
        abs_score: int,
        captures: float,
        positions: int,
        policy_rows: int,
    ) -> None:
        self.games += 1
        self.decisive += int(outcome in (BLACK_WINS, WHITE_WINS))
        self.draws += int(outcome == DRAW)
        self.black_wins += int(outcome == BLACK_WINS)
        self.white_wins += int(outcome == WHITE_WINS)
        self.natural += int(natural)
        self.plies += plies
        self.abs_score += abs_score
        self.positions += positions
        self.policy_rows += policy_rows
        if not math.isnan(captures):
            self.captures += int(captures)
            self.capture_games += 1
            self.capture_plies += plies

    def rates(self) -> dict[str, Any]:
        return {
            "games": self.games,
            "positions": self.positions,
            "decisive_rate": _ratio(self.decisive, self.games),
            "draw_rate": _ratio(self.draws, self.games),
            "black_win_rate": _ratio(self.black_wins, self.games),
            "white_win_rate": _ratio(self.white_wins, self.games),
            "natural_termination_rate": _ratio(self.natural, self.games),
            "mean_plies": _ratio(self.plies, self.games),
            "mean_abs_score_diff": _ratio(self.abs_score, self.games),
            "mean_captures": _ratio(self.captures, self.capture_games),
            "captures_per_100_plies": _ratio(100.0 * self.captures, self.capture_plies),
            "full_search_rate": _ratio(self.policy_rows, self.positions),
        }


def summarise_generation(games: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate health stats over a generation's games, in one streaming pass.

    The pair that matters for the *training signal* is `policy_target_entropy`
    against `policy_uniform_entropy` (= `ln(mean_legal_moves)`). Their ratio at
    1.0 means search returned roughly uniform visits over legal moves: the policy
    target carries zero information and no downstream metric means anything. That
    is precisely the failure that ran unnoticed for three generations.

    The number that drives the *curriculum* is `natural_termination_rate`: the
    fraction of **unseeded** games (`handicap == (0, 0)`) that reached six
    captures before the ply cap. See `model.curriculum` for why that, and not
    the decisive rate, is the signal — under uniformly random play the decisive
    rate is already 63% while natural termination is 0%.

    The seeded/unseeded split of `decisive_rate` and `mean_abs_score_diff` is
    reported beside it as the human sanity check: seeded games start one capture
    from the end and should look nothing like unseeded ones.

    `mean_legal_moves` is the mean number of root children in the search result,
    which for a fully-expanded root is the legal-move count.

    **`captures_per_100_plies` is the positional-maturity proxy.** Six captures
    inside ~30 moves is one every five moves — a bloodbath, and evidence that
    *both* sides are defending badly rather than that either is attacking well.
    As skill rises this should FALL while `mean_plies` rises. The two moving
    together in the wrong direction (short games, high capture rate) is the
    signature of a network that has learned to end games without learning to
    play them.

    Every rate is reported three times: pooled, `unseeded_*` and `seeded_*`. The
    pooled figure is a mixture whose weights are set by the curriculum, so it
    moves when the handicap rate moves even if neither population changed.
    """
    n_games = 0
    handicaps: Counter[str] = Counter()

    unseeded = _Split()
    seeded = _Split()
    overall = _Split()

    entropy_total = 0.0
    entropy_n = 0
    legal_total = 0
    legal_n = 0

    for game in games:
        n_games += 1
        result = game["result"]
        outcome = str(result["outcome"])
        plies = int(result["plies"])
        abs_score = abs(int(result["score_diff"]))
        handicaps[f"{int(game['handicap'][0])}-{int(game['handicap'][1])}"] += 1

        game_positions = 0
        game_policy_rows = 0
        for move in game["moves"]:
            game_positions += 1
            if not move["is_full_search"]:
                continue
            game_policy_rows += 1
            counts = [int(v) for _, v in move["visits"]]
            entropy = visit_entropy(counts)
            if not math.isnan(entropy):
                entropy_total += entropy
                entropy_n += 1
            legal_total += len(counts)
            legal_n += 1

        stats = {
            "outcome": outcome,
            "natural": terminated_naturally(game),
            "plies": plies,
            "abs_score": abs_score,
            "captures": captures_in_play(game),
            "positions": game_positions,
            "policy_rows": game_policy_rows,
        }
        overall.add(**stats)
        (unseeded if is_unseeded(game) else seeded).add(**stats)

    mean_legal = _ratio(legal_total, legal_n)
    target_entropy = _ratio(entropy_total, entropy_n)
    uniform_entropy = math.log(mean_legal) if mean_legal and mean_legal > 0 else NAN
    ratio = (
        target_entropy / uniform_entropy
        if uniform_entropy and uniform_entropy > 0 and not math.isnan(target_entropy)
        else NAN
    )
    # Reported directly because the success criterion is stated as a gap
    # (MODEL.md §8.2: "policy-target entropy well below ln(branching)"), and a
    # reader should not have to subtract two logged numbers to check it.
    gap = (
        uniform_entropy - target_entropy
        if not (math.isnan(uniform_entropy) or math.isnan(target_entropy))
        else NAN
    )

    pooled = overall.rates()
    unseeded_rates = unseeded.rates()
    seeded_rates = seeded.rates()
    out: dict[str, Any] = {
        "games": n_games,
        "positions": overall.positions,
        "policy_rows": overall.policy_rows,
        "handicap_distribution": dict(sorted(handicaps.items())),
        # ---- the curriculum control signal (MODEL.md §4) ----
        "unseeded_games": unseeded.games,
        "seeded_games": seeded.games,
        "natural_terminations": unseeded.natural,
        "natural_termination_rate": unseeded_rates["natural_termination_rate"],
        # ---- the training signal ----
        "policy_target_entropy": target_entropy,
        "policy_uniform_entropy": uniform_entropy,
        "policy_entropy_ratio": ratio,
        "policy_entropy_gap": gap,
        "mean_legal_moves": mean_legal,
    }
    # Pooled rates keep their bare names; the two splits are the same key set
    # under `unseeded_` / `seeded_`, so nothing is reported on one side only.
    # `natural_termination_rate` is the exception: bare, it has always meant the
    # UNSEEDED rate, because that is the curriculum control signal and a pooled
    # one would be dominated by seeded games that finish trivially.
    for key, value in pooled.items():
        if key not in ("games", "positions", "natural_termination_rate"):
            out[key] = value
    for prefix, rates in (("unseeded_", unseeded_rates), ("seeded_", seeded_rates)):
        for key, value in rates.items():
            if key == "games":
                continue
            out[f"{prefix}{key}"] = value
    return out


#: Namespace for `summarise_generation`: properties of THIS generation's
#: self-play data. Distinct from `val_*` (model quality on held-out positions)
#: and `buffer/` (what the training window contains).
SELFPLAY_PREFIX = "selfplay/"


def selfplay_metrics(
    summary: Mapping[str, Any], prefix: str = SELFPLAY_PREFIX
) -> dict[str, float]:
    """`summarise_generation` output as a flat, namespaced block of numbers.

    Every value is a plain `int` or `float` and every key carries `prefix`, so a
    caller can log the whole dict generically. `handicap_distribution` is the
    only entry dropped — it is a histogram, and it belongs in `metrics.jsonl`
    rather than in a scalar series.

    `summarise_generation` itself keeps bare keys: `model/curriculum.py` reads
    `natural_termination_rate` and `unseeded_games` straight off it.
    """
    return {
        f"{prefix}{k}": float(v)
        for k, v in summary.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #


def _fmt(value: float, digits: int = 2) -> str:
    return "n/a" if value is None or math.isnan(value) else f"{value:.{digits}f}"


def _pct(value: float) -> str:
    return "n/a" if value is None or math.isnan(value) else f"{100.0 * value:.1f}%"


#: Above this the visit distribution is indistinguishable from uniform.
UNIFORM_ENTROPY_WARN_RATIO = 0.98


def format_generation_report(result: ExportResult, max_skipped: int = 5) -> list[str]:
    """Human-readable lines for one generation's export.

    Reports `result.summary` — every game in the generation — falling back to the
    written subset for an `ExportResult` built by hand.
    """
    summary = result.summary
    if summary is None:
        summary = summarise_generation(result.games)
    lines = [
        f"gen {result.gen:03d}  {summary['games']} games / {summary['positions']} positions"
        f"  from {result.shards_read} shard(s)"
    ]
    if summary["games"]:
        lines.append(
            f"  decisive {_pct(summary['decisive_rate'])}"
            f"   draw {_pct(summary['draw_rate'])}"
            f"   mean plies {_fmt(summary['mean_plies'], 1)}"
            f"   mean |score| {_fmt(summary['mean_abs_score_diff'])}"
            f"   full-search {_pct(summary['full_search_rate'])}"
        )
        # Should FALL as plies rise. Both moving the other way is a network
        # that has learned to end games without learning to play them.
        lines.append(
            f"  captures/100 plies {_fmt(summary.get('captures_per_100_plies'), 2)}"
            f"   (unseeded {_fmt(summary.get('unseeded_captures_per_100_plies'), 2)}"
            f"   seeded {_fmt(summary.get('seeded_captures_per_100_plies'), 2)})"
            f"   mean captures in play {_fmt(summary.get('mean_captures'), 2)}"
        )
        # The split the curriculum controller reads. Seeded games start a capture
        # from the end and look nothing like unseeded ones; averaging the two
        # together is what makes a raw decisive rate unreadable.
        lines.append(
            f"  unseeded {summary['unseeded_games']:>3} games"
            f"  ·  natural termination {_pct(summary['natural_termination_rate'])}"
            f"   decisive {_pct(summary['unseeded_decisive_rate'])}"
            f"   mean |score| {_fmt(summary['unseeded_mean_abs_score_diff'])}"
        )
        lines.append(
            f"  seeded   {summary['seeded_games']:>3} games"
            f"  ·  natural termination {_pct(summary['seeded_natural_termination_rate'])}"
            f"   decisive {_pct(summary['seeded_decisive_rate'])}"
            f"   mean |score| {_fmt(summary['seeded_mean_abs_score_diff'])}"
        )
        handicaps = summary["handicap_distribution"]
        if handicaps:
            lines.append(
                "  handicap  " + "  ".join(f"{k}×{v}" for k, v in handicaps.items())
            )
        lines.append(
            f"  policy target entropy {_fmt(summary['policy_target_entropy'], 3)}"
            f"   vs ln(mean legal {_fmt(summary['mean_legal_moves'], 1)})"
            f" = {_fmt(summary['policy_uniform_entropy'], 3)}"
            f"   gap {_fmt(summary.get('policy_entropy_gap'), 3)}"
            f"   ratio {_fmt(summary['policy_entropy_ratio'], 3)}"
        )
        ratio = summary["policy_entropy_ratio"]
        if not math.isnan(ratio) and ratio >= UNIFORM_ENTROPY_WARN_RATIO:
            lines.append(
                "  ** WARNING: visit distributions are ~uniform over legal moves —"
                " the policy target carries no information **"
            )
    for skip in result.skipped[:max_skipped]:
        lines.append(f"  skipped {skip}")
    if len(result.skipped) > max_skipped:
        lines.append(f"  ... and {len(result.skipped) - max_skipped} more skipped")
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export self-play shards to reviewable game JSON."
    )
    parser.add_argument("--run-dir", required=True, help="runs/<run-id>")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--gen", type=int, help="generation to export")
    selection.add_argument(
        "--all-gens", action="store_true", help="export every generation with shards"
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="root for exported games; default <run-dir>/games. Each generation "
        "lands in <out-dir>/gen_NNN/",
    )
    parser.add_argument("--limit", type=int, default=None, help="max games per generation")
    parser.add_argument(
        "--run-id", default=None, help="run id recorded in the JSON; default <run-dir> name"
    )
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    run_id = args.run_id or run_dir.name
    shards_root = run_dir / "shards"
    out_root = Path(args.out_dir) if args.out_dir else run_dir / "games"

    gens = find_generations(shards_root) if args.all_gens else [int(args.gen)]
    if not gens:
        print(f"no generations found under {shards_root}")
        return 1

    total_games = 0
    total_skipped = 0
    for gen in gens:
        result = export_generation(
            shards_root / f"gen_{gen:03d}",
            out_root / f"gen_{gen:03d}",
            run_id,
            gen,
            limit=args.limit,
        )
        for line in format_generation_report(result):
            print(line)
        total_games += len(result.games)
        total_skipped += len(result.skipped)

    suffix = f", {total_skipped} skipped" if total_skipped else ""
    print(f"exported {total_games} game(s) to {out_root}{suffix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
