//! Eval match between two players: the per-generation gate and the fixed
//! **anchor ladder** of `docs/MODEL.md` §8.1.
//!
//! # Player specs
//!
//! ```text
//!   random                     uniform-random legal move, no search
//!   heuristic                  heuristic-MCTS at --simulations
//!   heuristic@800              ... at 800 simulations
//!   model:<path.onnx>          ONNX-driven MCTS at --simulations
//!   model:<path.onnx>@400      ... at 400 simulations
//! ```
//!
//! The `@N` suffix may also be written `:N` (`heuristic:800`,
//! `model:best.onnx:400`). Per-player simulation counts are the whole
//! point of the ladder: `heuristic@100` and `heuristic@800` are
//! different opponents, and "beats `heuristic@800` at equal simulation
//! count" is the milestone. Prefer the `@` form for models — a path
//! whose final `:`-separated component is all digits would otherwise be
//! read as an override.
//!
//! The hand-written heuristic is retired from the *training* loop
//! (`docs/MODEL.md`) but deliberately kept here. A frozen yardstick is
//! not a teacher, and only a fixed opponent yields a monotone Elo curve.
//!
//! # Why every game must actually differ
//!
//! MCTS over a deterministic evaluator with no root noise is a pure
//! function of the position, so a match played from one fixed start is
//! one game replayed N times. That is not hypothetical: a 21-game gate
//! in the logged run produced exactly two distinct games, 11 of one and
//! 10 of the other (`docs/2026-07-27-architecture-review.md` §3.2). Two
//! sources of variation fix it:
//!
//!   * `--random-opening-plies N` — N uniformly-random legal plies
//!     before the players take over. Openings are **paired**: games
//!     `2k` and `2k+1` share a start and swap colours, so a lopsided
//!     random opening is played from both sides and cancels out of the
//!     score instead of adding variance to it.
//!   * `--temperature-plies N` / `--temperature T` — for the first N
//!     searched plies the move is sampled from the root visit counts
//!     with `P(i) ∝ N_i^(1/T)`; argmax thereafter.
//!
//! Per-game seeds are derived from `--seed` and the game index, so the
//! match as a whole is reproducible while its games differ from each
//! other. The JSON reports `distinct_transcripts`; anything ≤ 2 for a
//! match of more than 2 games means the randomisation is not working
//! and the result is one game wearing a large denominator. That check
//! is asserted in the tests at the bottom of this file, against a
//! deliberately-defective configuration that makes it fail.
//!
//! # Scoring
//!
//! The primary metric is the standard score `(wins + 0.5·draws) /
//! games`. `wins / games` scores a draw as a loss, which puts the 0.55
//! gate threshold out of mathematical reach in a drawish game
//! (`docs/2026-07-27-architecture-review.md` §3.3). `winrate_a` now
//! carries the standard score; `wins_only_rate_a` preserves the old
//! definition for anyone who wants it.
//!
//! Run:
//!   eval-match --player-a model:checkpoints/gen_002.onnx \
//!              --player-b heuristic@800 \
//!              --games 21 --simulations 200 --threads 8 \
//!              --out-json runs/foo/eval/gen_002_gate.json

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Mutex;
use std::thread;
use std::time::Instant;

use abalone_game::{
    encode, Game, GameState, Move, Opening, Side, DEFAULT_MAX_PLIES,
    NO_PROGRESS_DISABLED,
};
use abalone_mcts::{heuristic, search, search_batched, LeafEval, SearchConfig};
use abalone_selfplay::ort_eval::OrtEvaluator;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use serde::Serialize;

const USAGE: &str = "\
eval-match — play a match between two players and write a JSON summary.

  --player-a SPEC            required; see specs below
  --player-b SPEC            required
  --games N                  games to play                  [21]
  --simulations N            fallback sims per move         [200]
  --c-puct F                 PUCT exploration constant      [1.4]
  --batch-size N             leaves per NN call (model)     [32]
  --opening standard|belgian starting layout                [standard]
  --random-opening-plies N   random plies before play       [2]
  --temperature-plies N      plies sampled from visits      [10]
  --temperature T            sampling temperature           [1.0]
  --max-plies N              adjudicate on captures at      [200]
  --no-progress-plies N      adjudicate after N quiet plies [off]
  --out-json PATH            summary destination
  --seed N                   match seed                     [0]
  --threads N                worker threads                 [cores-1]

Player specs:
  random | heuristic | heuristic@800 | model:p.onnx | model:p.onnx@400
";

// ---------------------------------------------------------------------
// Player specs
// ---------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq)]
enum PlayerKind {
    Model(PathBuf),
    Heuristic,
    Random,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct PlayerSpec {
    kind: PlayerKind,
    /// Per-player simulation override; `None` falls back to
    /// `--simulations`. This is what makes `heuristic@100` and
    /// `heuristic@800` distinct rungs of the anchor ladder.
    simulations: Option<u32>,
}

/// Split a trailing `@N` or `:N` simulation override off a spec.
/// `@` is tried first so that a model path containing a colon is not
/// mangled when the user has been explicit.
fn split_sim_override(s: &str) -> (&str, Option<u32>) {
    for sep in ['@', ':'] {
        if let Some((head, tail)) = s.rsplit_once(sep) {
            if let Ok(n) = tail.parse::<u32>() {
                if n > 0 && !head.is_empty() {
                    return (head, Some(n));
                }
            }
        }
    }
    (s, None)
}

impl PlayerSpec {
    fn parse(s: &str) -> Result<Self, String> {
        let (head, simulations) = split_sim_override(s);
        let kind = if let Some(path) = head.strip_prefix("model:") {
            if path.is_empty() {
                return Err(format!("model spec needs a path: {s}"));
            }
            PlayerKind::Model(PathBuf::from(path))
        } else if head == "heuristic" {
            PlayerKind::Heuristic
        } else if head == "random" {
            PlayerKind::Random
        } else {
            return Err(format!(
                "unknown player spec: {s} (want random | heuristic[@N] | \
                 model:<path.onnx>[@N])"
            ));
        };
        Ok(PlayerSpec { kind, simulations })
    }

    /// Simulations this player will actually use.
    fn effective_simulations(&self, default: u32) -> u32 {
        match self.kind {
            // `random` does not search; reporting a sim count for it
            // would be a lie on the ladder.
            PlayerKind::Random => 0,
            _ => self.simulations.unwrap_or(default),
        }
    }

    /// Ladder name. Searching players always carry their simulation
    /// count so that `heuristic@100` and `heuristic@800` never collapse
    /// into one label in a results table.
    fn label(&self, default_sims: u32) -> String {
        let n = self.effective_simulations(default_sims);
        match &self.kind {
            PlayerKind::Model(p) => format!("model:{}@{}", p.display(), n),
            PlayerKind::Heuristic => format!("heuristic@{n}"),
            PlayerKind::Random => "random".to_string(),
        }
    }
}

fn parse_opening(s: &str) -> Result<Opening, String> {
    match s.to_ascii_lowercase().as_str() {
        "standard" => Ok(Opening::Standard),
        "belgian" | "belgian-daisy" | "belgiandaisy" | "daisy" => {
            Ok(Opening::BelgianDaisy)
        }
        _ => Err(format!("unknown opening: {s} (want standard|belgian)")),
    }
}

fn opening_name(o: Opening) -> &'static str {
    match o {
        Opening::Standard => "standard",
        Opening::BelgianDaisy => "belgian",
    }
}

// ---------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------

#[derive(Clone, Debug)]
struct Args {
    a: PlayerSpec,
    b: PlayerSpec,
    games: u32,
    simulations: u32,
    c_puct: f32,
    batch_size: usize,
    rules: MatchRules,
    out_json: PathBuf,
    seed: u64,
    threads: usize,
}

impl Args {
    fn parse() -> Self {
        let mut args = std::env::args().skip(1);
        let mut a_spec: Option<PlayerSpec> = None;
        let mut b_spec: Option<PlayerSpec> = None;
        let mut a = Args {
            a: PlayerSpec {
                kind: PlayerKind::Random,
                simulations: None,
            },
            b: PlayerSpec {
                kind: PlayerKind::Random,
                simulations: None,
            },
            games: 21,
            simulations: 200,
            c_puct: 1.4,
            batch_size: 32,
            rules: MatchRules::default(),
            out_json: PathBuf::from("/tmp/eval-match.json"),
            seed: 0,
            threads: num_cpus_or(8),
        };
        while let Some(k) = args.next() {
            let mut nxt = || args.next().expect("missing value");
            match k.as_str() {
                "--help" | "-h" => {
                    print!("{USAGE}");
                    std::process::exit(0);
                }
                "--player-a" => a_spec = Some(spec_or_die(&nxt())),
                "--player-b" => b_spec = Some(spec_or_die(&nxt())),
                "--games" => a.games = nxt().parse().unwrap(),
                "--simulations" => a.simulations = nxt().parse().unwrap(),
                "--c-puct" => a.c_puct = nxt().parse().unwrap(),
                "--batch-size" => a.batch_size = nxt().parse().unwrap(),
                "--opening" => {
                    a.rules.opening =
                        parse_opening(&nxt()).unwrap_or_else(|e| panic!("{e}"));
                }
                "--random-opening-plies" => {
                    a.rules.random_opening_plies = nxt().parse().unwrap();
                }
                "--temperature-plies" => {
                    a.rules.temperature_plies = nxt().parse().unwrap();
                }
                "--temperature" => a.rules.temperature = nxt().parse().unwrap(),
                "--max-plies" => a.rules.max_plies = nxt().parse().unwrap(),
                "--no-progress-plies" => {
                    a.rules.no_progress_plies = nxt().parse().unwrap();
                }
                "--out-json" => a.out_json = PathBuf::from(nxt()),
                "--seed" => a.seed = nxt().parse().unwrap(),
                "--threads" => a.threads = nxt().parse().unwrap(),
                _ => {
                    eprint!("{USAGE}");
                    panic!("unknown arg: {k}");
                }
            }
        }
        a.a = a_spec.expect("--player-a required");
        a.b = b_spec.expect("--player-b required");
        a.batch_size = a.batch_size.max(1);
        a
    }
}

fn spec_or_die(s: &str) -> PlayerSpec {
    PlayerSpec::parse(s).unwrap_or_else(|e| panic!("{e}"))
}

fn num_cpus_or(default: usize) -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get().saturating_sub(1).max(1))
        .unwrap_or(default)
}

// ---------------------------------------------------------------------
// Seeds
// ---------------------------------------------------------------------

/// Stream ids keep the opening RNG independent of each player's move
/// RNG, so changing `--random-opening-plies` does not reshuffle the
/// sampling decisions inside a game.
const STREAM_OPENING: u64 = 1;
const STREAM_A: u64 = 2;
const STREAM_B: u64 = 3;

/// SplitMix64 finalizer.
fn mix64(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    x
}

/// Deterministic per-game, per-stream seed. The whole match is a pure
/// function of `--seed`, but every game draws from a different, well
/// separated part of the generator's state space.
fn derive_seed(base: u64, index: u64, stream: u64) -> u64 {
    mix64(
        base ^ index.wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ stream.wrapping_mul(0xD1B5_4A32_D192_ED03),
    )
}

// ---------------------------------------------------------------------
// Move selection
// ---------------------------------------------------------------------

/// Below this, sampling is indistinguishable from argmax (and `1/T`
/// starts to overflow the exponent).
const ARGMAX_TEMPERATURE: f32 = 1e-3;

/// Ties break to the **last** maximum, which is what `Iterator::max_by_key`
/// does — and therefore what `SearchResult::best` does. Matching it keeps
/// `select_move(v, 0.0)` identical to the move MCTS would have named.
fn argmax_move(visits: &[(Move, u32)]) -> Move {
    visits
        .iter()
        .max_by_key(|&&(_, n)| n)
        .map(|&(mv, _)| mv)
        .expect("non-empty visit list")
}

/// Pick a move from root visit counts: argmax, or for `temperature > 0`
/// a sample with `P(i) ∝ N_i^(1/T)`.
///
/// Weights are taken relative to the largest visit count, so
/// `(N_i / N_max)^(1/T)` stays in `[0, 1]` however small `T` gets. The
/// distribution therefore collapses onto the argmax as `T → 0` instead
/// of overflowing to `inf/inf = NaN`.
fn select_move<R: Rng + ?Sized>(
    visits: &[(Move, u32)],
    temperature: f32,
    rng: &mut R,
) -> Move {
    if temperature.is_nan() || temperature <= ARGMAX_TEMPERATURE {
        return argmax_move(visits);
    }
    let n_max = visits.iter().map(|&(_, n)| n).max().unwrap_or(0);
    if n_max == 0 {
        // No simulation reached any child (e.g. `simulations = 0`);
        // fall back rather than divide by zero.
        return argmax_move(visits);
    }
    let inv_t = 1.0 / f64::from(temperature);
    let n_max = f64::from(n_max);
    let mut weights: Vec<f64> = Vec::with_capacity(visits.len());
    let mut total = 0.0f64;
    for &(_, n) in visits {
        let w = (f64::from(n) / n_max).powf(inv_t);
        let w = if w.is_finite() { w } else { 0.0 };
        weights.push(w);
        total += w;
    }
    if !total.is_finite() || total <= 0.0 {
        return argmax_move(visits);
    }
    let mut u = rng.gen::<f64>() * total;
    for (i, w) in weights.iter().enumerate() {
        u -= w;
        if u <= 0.0 {
            return visits[i].0;
        }
    }
    visits[visits.len() - 1].0
}

// ---------------------------------------------------------------------
// Players
// ---------------------------------------------------------------------

/// Anything that can produce root visit counts. A trait so that the
/// game loop — where openings, temperature and transcript hashing live
/// — is unit-testable without an ONNX file on disk.
trait MovePicker {
    /// `(legal move, visit count)` for every child of the root. `None`
    /// means "does not search"; the caller then plays a uniform-random
    /// legal move.
    fn root_visits(
        &mut self,
        g: &Game,
        rng: &mut SmallRng,
    ) -> Option<Vec<(Move, u32)>>;
}

// Per-thread players: each worker holds its own `OrtEvaluator` so all
// inference runs in parallel. Cost is one ONNX load per thread.
enum Player {
    Model {
        evaluator: Box<OrtEvaluator>,
        cfg: SearchConfig,
    },
    Heuristic {
        cfg: SearchConfig,
    },
    Random,
}

impl Player {
    fn from_spec(spec: &PlayerSpec, base: &SearchConfig, default: u32) -> Self {
        let cfg = SearchConfig {
            simulations: spec.effective_simulations(default),
            ..base.clone()
        };
        match &spec.kind {
            PlayerKind::Model(p) => Player::Model {
                evaluator: {
                    let mut e = OrtEvaluator::from_onnx(p).expect("load onnx model");
                    // Pad every forward to one fixed width. ORT's CoreML
                    // provider compiles a separate model per input shape, and
                    // a search's final batch is almost always partial, so
                    // without this every match thrashes the compiler.
                    //
                    // Self-play has always done this; eval-match did not, and
                    // the cost was not subtle. A 32-game rung at 200 sims ran
                    // 492,800 evaluations in 2417s — 204 evals/s against
                    // self-play's ~16,000/s on the same machine, 78x slower,
                    // and 40 minutes of a 53.7-minute ladder.
                    e.set_fixed_batch(Some(cfg.batch_size.max(1)));
                    Box::new(e)
                },
                cfg,
            },
            PlayerKind::Heuristic => Player::Heuristic {
                // A hand-written eval costs nothing per leaf, so
                // batching buys nothing; `batch_size = 1` keeps the
                // search exactly sequential.
                cfg: SearchConfig {
                    batch_size: 1,
                    ..cfg
                },
            },
            PlayerKind::Random => Player::Random,
        }
    }
}

impl MovePicker for Player {
    fn root_visits(
        &mut self,
        g: &Game,
        rng: &mut SmallRng,
    ) -> Option<Vec<(Move, u32)>> {
        match self {
            Player::Model { evaluator, cfg } => {
                let seed = rng.gen::<u64>();
                let res = search_batched(g, cfg, seed, |batch| {
                    evaluate_batch(evaluator, batch)
                })?;
                Some(res.visits)
            }
            Player::Heuristic { cfg } => {
                Some(search(g, cfg, rng, heuristic)?.visits)
            }
            Player::Random => None,
        }
    }
}

/// Batched leaf evaluation for the model player: one ORT `run()` for
/// the whole batch, which is where the throughput is (`docs/MODEL.md`
/// §7.1). `ort_eval.rs` owns the tensor packing and the collapse of the
/// 3-way value head to the scalar search wants.
fn evaluate_batch(
    evaluator: &mut OrtEvaluator,
    batch: &[Game],
) -> Vec<LeafEval> {
    evaluator.evaluate_batch(batch).expect("ort evaluate batch")
}

// ---------------------------------------------------------------------
// One game
// ---------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
struct MatchRules {
    opening: Opening,
    max_plies: u32,
    no_progress_plies: u32,
    random_opening_plies: u32,
    temperature_plies: u32,
    temperature: f32,
}

impl Default for MatchRules {
    fn default() -> Self {
        Self {
            opening: Opening::Standard,
            max_plies: DEFAULT_MAX_PLIES,
            no_progress_plies: NO_PROGRESS_DISABLED,
            random_opening_plies: 2,
            temperature_plies: 10,
            temperature: 1.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
enum Outcome {
    A,
    B,
    Draw,
}

#[derive(Clone, Copy, Debug, Serialize)]
struct GameRecord {
    index: u32,
    a_is_black: bool,
    outcome: Outcome,
    plies: u32,
    /// Captures made minus captures conceded, from A's point of view.
    /// Distinguishes "won by a hair after adjudication" from "won
    /// decisively", which matters now that a capped game is adjudicated
    /// on capture differential rather than drawn.
    score_diff_a: i32,
    /// FNV-1a over the encoded move sequence. Colour assignment is
    /// deliberately excluded: two games are the same transcript iff the
    /// same moves were played in the same order.
    transcript: u64,
}

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn fnv_step(h: u64, x: u16) -> u64 {
    (h ^ u64::from(x)).wrapping_mul(FNV_PRIME)
}

fn play_one_game(
    index: u32,
    base_seed: u64,
    rules: &MatchRules,
    player_a: &mut dyn MovePicker,
    player_b: &mut dyn MovePicker,
) -> GameRecord {
    let a_is_black = index.is_multiple_of(2);
    // Paired openings: games 2k and 2k+1 share a start and swap
    // colours, so a lopsided random opening is played from both sides
    // and cancels out of the score.
    let mut open_rng = SmallRng::seed_from_u64(derive_seed(
        base_seed,
        u64::from(index / 2),
        STREAM_OPENING,
    ));
    let mut rng_a = SmallRng::seed_from_u64(derive_seed(
        base_seed,
        u64::from(index),
        STREAM_A,
    ));
    let mut rng_b = SmallRng::seed_from_u64(derive_seed(
        base_seed,
        u64::from(index),
        STREAM_B,
    ));

    let mut g =
        Game::new(rules.opening, rules.max_plies, rules.no_progress_plies);
    let mut transcript = FNV_OFFSET;

    for _ in 0..rules.random_opening_plies {
        if g.is_terminal() {
            break;
        }
        let moves = g.legal_moves();
        if moves.is_empty() {
            break;
        }
        let mv = moves[open_rng.gen_range(0..moves.len())];
        transcript = fnv_step(transcript, encode(mv));
        g.apply(mv);
    }

    // Temperature counts searched plies, not board plies: the random
    // opening is already maximally noisy and should not eat the budget.
    let mut searched = 0u32;
    while !g.is_terminal() {
        let a_to_move = (g.turn == Side::Black) == a_is_black;
        let (player, rng): (&mut dyn MovePicker, &mut SmallRng) = if a_to_move {
            (&mut *player_a, &mut rng_a)
        } else {
            (&mut *player_b, &mut rng_b)
        };
        let temperature = if searched < rules.temperature_plies {
            rules.temperature
        } else {
            0.0
        };
        let rooted = player.root_visits(&g, rng);
        let mv = match rooted {
            Some(v) if !v.is_empty() => select_move(&v, temperature, rng),
            _ => {
                let moves = g.legal_moves();
                if moves.is_empty() {
                    break;
                }
                moves[rng.gen_range(0..moves.len())]
            }
        };
        transcript = fnv_step(transcript, encode(mv));
        g.apply(mv);
        searched += 1;
    }

    let a_side = if a_is_black { Side::Black } else { Side::White };
    let outcome = match g.state() {
        GameState::Wins(s) if s == a_side => Outcome::A,
        GameState::Wins(_) => Outcome::B,
        // `InProgress` is only reachable via the move-less break above,
        // which Abalone's rules make unreachable in practice.
        GameState::Draw | GameState::InProgress => Outcome::Draw,
    };
    GameRecord {
        index,
        a_is_black,
        outcome,
        plies: g.ply,
        score_diff_a: g.score_diff(a_side),
        transcript,
    }
}

// ---------------------------------------------------------------------
// Scoring
// ---------------------------------------------------------------------

/// Standard score for A: a draw is half a point, not a loss.
fn standard_score(wins_a: u32, wins_b: u32, draws: u32) -> f64 {
    let games = wins_a + wins_b + draws;
    if games == 0 {
        return 0.5;
    }
    (f64::from(wins_a) + 0.5 * f64::from(draws)) / f64::from(games)
}

/// The pre-fix metric, `wins_a / games`, kept only so the JSON can
/// still report it. This is the formula that scored a draw as a loss
/// and put the 0.55 gate threshold out of reach.
fn wins_only_rate(wins_a: u32, games: u32) -> f64 {
    if games == 0 {
        0.0
    } else {
        f64::from(wins_a) / f64::from(games)
    }
}

/// Standard error of the mean per-game score. Per-game scores take only
/// the values `1`, `0.5`, `0`, so the variance is exact rather than a
/// normal approximation to something else.
fn score_stderr(wins_a: u32, wins_b: u32, draws: u32) -> f64 {
    let games = wins_a + wins_b + draws;
    if games == 0 {
        return 0.0;
    }
    let n = f64::from(games);
    let s = standard_score(wins_a, wins_b, draws);
    let sum_sq = f64::from(wins_a) + 0.25 * f64::from(draws);
    let var = (sum_sq / n - s * s).max(0.0);
    (var / n).sqrt()
}

/// Elo difference implied by a score, guarded at the extremes.
///
/// `-400·log10(1/s − 1)` diverges at `s ∈ {0, 1}`, and a clean sweep is
/// exactly the case a small match produces. The score is first clamped
/// to `[1/(2n), 1 − 1/(2n)]` — "no better than half a game short of
/// perfect" — which ties the bound to the sample size rather than to an
/// arbitrary constant. Returns `(elo, clamped)`.
fn elo_from_score(s: f64, games: u32) -> (f64, bool) {
    let n = f64::from(games.max(1));
    let bound = 1.0 / (2.0 * n);
    let c = s.clamp(bound, 1.0 - bound);
    ((-400.0) * (1.0 / c - 1.0).log10(), (c - s).abs() > 1e-12)
}

/// Delta-method standard error of the Elo estimate:
/// `d(elo)/ds = (400/ln 10) / (s(1−s))`.
fn elo_stderr(s: f64, se: f64, games: u32) -> f64 {
    let n = f64::from(games.max(1));
    let bound = 1.0 / (2.0 * n);
    let c = s.clamp(bound, 1.0 - bound);
    (400.0 / std::f64::consts::LN_10) * se / (c * (1.0 - c))
}

#[derive(Clone, Debug, Default)]
struct Aggregate {
    wins_a: u32,
    wins_b: u32,
    draws: u32,
    mean_plies: f64,
    mean_score_diff_a: f64,
    mean_abs_score_diff: f64,
    distinct_transcripts: u32,
}

fn aggregate(records: &[GameRecord]) -> Aggregate {
    let mut agg = Aggregate::default();
    if records.is_empty() {
        return agg;
    }
    let mut seen: HashSet<u64> = HashSet::with_capacity(records.len());
    let (mut plies, mut diff, mut abs_diff) = (0f64, 0f64, 0f64);
    for r in records {
        match r.outcome {
            Outcome::A => agg.wins_a += 1,
            Outcome::B => agg.wins_b += 1,
            Outcome::Draw => agg.draws += 1,
        }
        plies += f64::from(r.plies);
        diff += f64::from(r.score_diff_a);
        abs_diff += f64::from(r.score_diff_a.abs());
        seen.insert(r.transcript);
    }
    let n = records.len() as f64;
    agg.mean_plies = plies / n;
    agg.mean_score_diff_a = diff / n;
    agg.mean_abs_score_diff = abs_diff / n;
    agg.distinct_transcripts = seen.len() as u32;
    agg
}

// ---------------------------------------------------------------------
// JSON
// ---------------------------------------------------------------------

#[derive(Serialize)]
struct MatchResult {
    // --- configuration ---
    player_a: String,
    player_b: String,
    games: u32,
    /// Fallback simulation count (`--simulations`). Per-player counts,
    /// which may differ, are `simulations_a` / `simulations_b`.
    simulations: u32,
    simulations_a: u32,
    simulations_b: u32,
    c_puct: f32,
    batch_size: usize,
    opening: &'static str,
    random_opening_plies: u32,
    temperature_plies: u32,
    temperature: f32,
    max_plies: u32,
    no_progress_plies: u32,
    seed: u64,
    threads: usize,

    // --- outcome tallies ---
    wins_a: u32,
    wins_b: u32,
    draws: u32,

    // --- rates ---
    /// `(wins + 0.5·draws) / games`. The primary metric.
    score_a: f64,
    score_a_stderr: f64,
    /// Alias of `score_a`. NOTE: this used to be `wins_a / games`,
    /// which scored a draw as a loss; see `wins_only_rate_a`.
    winrate_a: f64,
    /// The pre-fix definition, `wins_a / games`.
    wins_only_rate_a: f64,
    /// `wins_a / (wins_a + wins_b)`; `0.5` when nothing was decisive.
    winrate_a_excluding_draws: f64,
    /// `(wins_a + wins_b) / games`.
    decisive_rate: f64,

    // --- strength estimate ---
    elo_a: f64,
    elo_a_stderr: f64,
    elo_a_ci95_lo: f64,
    elo_a_ci95_hi: f64,
    /// `true` when the score hit `0` or `1` and the Elo figure is a
    /// sample-size-derived bound rather than an estimate.
    elo_a_clamped: bool,

    // --- diagnostics ---
    mean_plies: f64,
    mean_score_diff_a: f64,
    /// Mean `|captures made − captures conceded|` at termination.
    mean_abs_score_diff: f64,
    /// Distinct move sequences among the games played. `≤ 2` for a
    /// match of more than 2 games means the randomisation is broken.
    distinct_transcripts: u32,
    elapsed_seconds: f32,
    per_game: Vec<GameRecord>,
}

// ---------------------------------------------------------------------
// main
// ---------------------------------------------------------------------

fn main() {
    let args = Args::parse();
    let sims_a = args.a.effective_simulations(args.simulations);
    let sims_b = args.b.effective_simulations(args.simulations);
    eprintln!(
        "eval-match: {} vs {} | {} games | opening={} \
         random-opening-plies={} temperature={}@{}plies \
         batch={} threads={} seed={}",
        args.a.label(args.simulations),
        args.b.label(args.simulations),
        args.games,
        opening_name(args.rules.opening),
        args.rules.random_opening_plies,
        args.rules.temperature,
        args.rules.temperature_plies,
        args.batch_size,
        args.threads,
        args.seed,
    );

    let base_cfg = SearchConfig {
        simulations: args.simulations,
        c_puct: args.c_puct,
        batch_size: args.batch_size,
        // No root noise in evaluation: variation comes from the random
        // opening and the temperature-sampled early plies, which is
        // measurable and reproducible. Noise in the priors would blur
        // the strength being measured.
        dirichlet_eps: 0.0,
        ..Default::default()
    };

    let games = args.games;
    let threads = args.threads.clamp(1, games.max(1) as usize);
    let next_game = AtomicU32::new(0);
    let results: Mutex<Vec<GameRecord>> =
        Mutex::new(Vec::with_capacity(games as usize));
    let next_game = &next_game;
    let results = &results;
    let t = Instant::now();

    thread::scope(|s| {
        for _tid in 0..threads {
            let a_spec = args.a.clone();
            let b_spec = args.b.clone();
            let cfg = base_cfg.clone();
            let rules = args.rules;
            let seed = args.seed;
            let default_sims = args.simulations;
            s.spawn(move || {
                let mut player_a =
                    Player::from_spec(&a_spec, &cfg, default_sims);
                let mut player_b =
                    Player::from_spec(&b_spec, &cfg, default_sims);
                loop {
                    let i = next_game.fetch_add(1, Ordering::Relaxed);
                    if i >= games {
                        break;
                    }
                    let rec = play_one_game(
                        i,
                        seed,
                        &rules,
                        &mut player_a,
                        &mut player_b,
                    );
                    eprintln!(
                        "  game {}: a_is_black={} outcome={:?} plies={} \
                         score_diff_a={:+}",
                        rec.index,
                        rec.a_is_black,
                        rec.outcome,
                        rec.plies,
                        rec.score_diff_a,
                    );
                    results.lock().expect("results mutex").push(rec);
                }
            });
        }
    });

    // Games are claimed by whichever worker is free, so the completion
    // order is nondeterministic; the *content* of game `i` is not.
    let mut records = results.lock().expect("results mutex").clone();
    records.sort_by_key(|r| r.index);

    let agg = aggregate(&records);
    let (wins_a, wins_b, draws) = (agg.wins_a, agg.wins_b, agg.draws);
    let decisive = wins_a + wins_b;
    let score = standard_score(wins_a, wins_b, draws);
    let se = score_stderr(wins_a, wins_b, draws);
    let (elo, elo_clamped) = elo_from_score(score, games);
    let (lo, _) = elo_from_score(score - 1.96 * se, games);
    let (hi, _) = elo_from_score(score + 1.96 * se, games);

    let result = MatchResult {
        player_a: args.a.label(args.simulations),
        player_b: args.b.label(args.simulations),
        games,
        simulations: args.simulations,
        simulations_a: sims_a,
        simulations_b: sims_b,
        c_puct: args.c_puct,
        batch_size: args.batch_size,
        opening: opening_name(args.rules.opening),
        random_opening_plies: args.rules.random_opening_plies,
        temperature_plies: args.rules.temperature_plies,
        temperature: args.rules.temperature,
        max_plies: args.rules.max_plies,
        no_progress_plies: args.rules.no_progress_plies,
        seed: args.seed,
        threads,
        wins_a,
        wins_b,
        draws,
        score_a: score,
        score_a_stderr: se,
        winrate_a: score,
        wins_only_rate_a: wins_only_rate(wins_a, games),
        winrate_a_excluding_draws: if decisive == 0 {
            0.5
        } else {
            f64::from(wins_a) / f64::from(decisive)
        },
        decisive_rate: if games == 0 {
            0.0
        } else {
            f64::from(decisive) / f64::from(games)
        },
        elo_a: elo,
        elo_a_stderr: elo_stderr(score, se, games),
        elo_a_ci95_lo: lo,
        elo_a_ci95_hi: hi,
        elo_a_clamped: elo_clamped,
        mean_plies: agg.mean_plies,
        mean_score_diff_a: agg.mean_score_diff_a,
        mean_abs_score_diff: agg.mean_abs_score_diff,
        distinct_transcripts: agg.distinct_transcripts,
        elapsed_seconds: t.elapsed().as_secs_f32(),
        per_game: records,
    };

    if let Some(parent) = args.out_json.parent() {
        std::fs::create_dir_all(parent).expect("create out-json parent");
    }
    let s = serde_json::to_string_pretty(&result).expect("serialize");
    std::fs::write(&args.out_json, s).expect("write out-json");
    eprintln!("wrote {}", args.out_json.display());
    eprintln!(
        "  wins_a={wins_a} wins_b={wins_b} draws={draws}  \
         score_a={score:.3} ±{se:.3}  elo={elo:+.0} \
         [{lo:+.0}, {hi:+.0}]{}  n={games}",
        if elo_clamped { " (bounded)" } else { "" },
    );
    eprintln!(
        "  decisive={:.3}  mean_plies={:.1}  mean|score_diff|={:.2}  \
         distinct_transcripts={}/{}",
        result.decisive_rate,
        agg.mean_plies,
        agg.mean_abs_score_diff,
        agg.distinct_transcripts,
        games,
    );

    // The Defect-1 guard, live. A deterministic evaluator from a fixed
    // start replays one game per colour assignment; if that is what we
    // just did, the sample size is a fiction and the caller must know.
    if games > 2 && agg.distinct_transcripts <= 2 {
        eprintln!(
            "WARNING: {games} games produced only {} distinct transcript(s). \
             Openings/temperature are not randomising the match — this is \
             {} game(s) replayed, not {games} samples. Check \
             --random-opening-plies and --temperature-plies.",
            agg.distinct_transcripts, agg.distinct_transcripts,
        );
    }
}

// ---------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---------- player specs ----------

    #[test]
    fn parses_plain_specs() {
        assert_eq!(
            PlayerSpec::parse("random").unwrap(),
            PlayerSpec {
                kind: PlayerKind::Random,
                simulations: None
            }
        );
        assert_eq!(
            PlayerSpec::parse("heuristic").unwrap(),
            PlayerSpec {
                kind: PlayerKind::Heuristic,
                simulations: None
            }
        );
        assert_eq!(
            PlayerSpec::parse("model:checkpoints/best.onnx").unwrap(),
            PlayerSpec {
                kind: PlayerKind::Model(PathBuf::from("checkpoints/best.onnx")),
                simulations: None
            }
        );
    }

    #[test]
    fn parses_simulation_overrides() {
        for (s, sims) in [
            ("heuristic@800", 800u32),
            ("heuristic:100", 100),
            ("random@50", 50),
        ] {
            let p = PlayerSpec::parse(s).unwrap();
            assert_eq!(p.simulations, Some(sims), "{s}");
        }
        let m = PlayerSpec::parse("model:ckpt/gen_002.onnx@400").unwrap();
        assert_eq!(
            m.kind,
            PlayerKind::Model(PathBuf::from("ckpt/gen_002.onnx"))
        );
        assert_eq!(m.simulations, Some(400));

        let m = PlayerSpec::parse("model:ckpt/gen_002.onnx:400").unwrap();
        assert_eq!(
            m.kind,
            PlayerKind::Model(PathBuf::from("ckpt/gen_002.onnx"))
        );
        assert_eq!(m.simulations, Some(400));
    }

    #[test]
    fn a_path_with_a_colon_is_not_mistaken_for_an_override() {
        // The suffix is only an override when it parses as a positive
        // integer, so ordinary paths survive.
        let m =
            PlayerSpec::parse("model:runs/2026-07-27:00/best.onnx").unwrap();
        assert_eq!(
            m.kind,
            PlayerKind::Model(PathBuf::from("runs/2026-07-27:00/best.onnx"))
        );
        assert_eq!(m.simulations, None);
    }

    #[test]
    fn rejects_unknown_specs() {
        assert!(PlayerSpec::parse("mcts").is_err());
        assert!(PlayerSpec::parse("model:").is_err());
        assert!(PlayerSpec::parse("").is_err());
    }

    #[test]
    fn labels_carry_the_simulation_count_for_searching_players() {
        let h = PlayerSpec::parse("heuristic@800").unwrap();
        assert_eq!(h.label(200), "heuristic@800");
        let h = PlayerSpec::parse("heuristic").unwrap();
        assert_eq!(h.label(100), "heuristic@100");
        // Distinct rungs of the ladder must not share a label.
        assert_ne!(
            PlayerSpec::parse("heuristic@100").unwrap().label(200),
            PlayerSpec::parse("heuristic@800").unwrap().label(200)
        );
        // `random` does not search, so no count.
        assert_eq!(PlayerSpec::parse("random").unwrap().label(200), "random");
    }

    #[test]
    fn parses_openings() {
        assert_eq!(parse_opening("standard").unwrap(), Opening::Standard);
        assert_eq!(parse_opening("Belgian").unwrap(), Opening::BelgianDaisy);
        assert_eq!(
            parse_opening("belgian-daisy").unwrap(),
            Opening::BelgianDaisy
        );
        assert!(parse_opening("nonsense").is_err());
    }

    // ---------- scoring ----------

    #[test]
    fn a_draw_is_half_a_point_not_a_loss() {
        // Defect 2, exactly: the logged "21 draws" gate scored 0.0
        // under `wins / games`, making the 0.55 threshold unreachable.
        assert_eq!(standard_score(0, 0, 21), 0.5);
        assert_eq!(standard_score(21, 0, 0), 1.0);
        assert_eq!(standard_score(0, 21, 0), 0.0);
        assert_eq!(standard_score(10, 5, 6), (10.0 + 3.0) / 21.0);
        // 11 wins, 10 draws clears a 0.55 gate; the old wins/games
        // (0.524) would not.
        assert!(standard_score(11, 0, 10) > 0.55);
        assert!(wins_only_rate(11, 21) < 0.55);
        // The logged failure mode: 21 draws was reported as 0.000.
        assert_eq!(wins_only_rate(0, 21), 0.0);
        assert_eq!(standard_score(0, 0, 21), 0.5);
        assert_eq!(wins_only_rate(0, 0), 0.0);
    }

    #[test]
    fn score_of_an_empty_match_is_even() {
        assert_eq!(standard_score(0, 0, 0), 0.5);
        assert_eq!(score_stderr(0, 0, 0), 0.0);
    }

    #[test]
    fn all_draws_has_no_variance() {
        assert_eq!(score_stderr(0, 0, 21), 0.0);
    }

    #[test]
    fn stderr_shrinks_with_sample_size() {
        let small = score_stderr(5, 5, 0);
        let large = score_stderr(50, 50, 0);
        assert!(large < small, "{large} !< {small}");
        // 10 games at 50%: sd of a fair coin is 0.5, so se = 0.5/sqrt(10).
        assert!((small - 0.5 / 10f64.sqrt()).abs() < 1e-9);
    }

    // ---------- elo ----------

    #[test]
    fn elo_is_zero_at_an_even_score() {
        let (e, clamped) = elo_from_score(0.5, 100);
        assert!(e.abs() < 1e-9);
        assert!(!clamped);
    }

    #[test]
    fn elo_is_antisymmetric() {
        for s in [0.55f64, 0.6, 0.75, 0.9] {
            let (a, _) = elo_from_score(s, 100);
            let (b, _) = elo_from_score(1.0 - s, 100);
            assert!((a + b).abs() < 1e-9, "s={s}: {a} vs {b}");
            assert!(a > 0.0);
        }
    }

    #[test]
    fn elo_matches_the_known_value_at_75_percent() {
        let (e, _) = elo_from_score(0.75, 1000);
        assert!((e - 190.848).abs() < 0.01, "{e}");
    }

    #[test]
    fn elo_guards_the_extremes() {
        // A clean sweep would be +inf unguarded.
        let (e, clamped) = elo_from_score(1.0, 21);
        assert!(e.is_finite() && e > 0.0, "{e}");
        assert!(clamped);
        let (e, clamped) = elo_from_score(0.0, 21);
        assert!(e.is_finite() && e < 0.0, "{e}");
        assert!(clamped);
        // The bound tightens as the sample grows: a 100-0 sweep is
        // stronger evidence than a 21-0 sweep.
        let (small, _) = elo_from_score(1.0, 21);
        let (large, _) = elo_from_score(1.0, 100);
        assert!(large > small, "{large} !> {small}");
        // Out-of-range inputs (a CI endpoint) stay finite too.
        assert!(elo_from_score(1.4, 21).0.is_finite());
        assert!(elo_from_score(-0.4, 21).0.is_finite());
    }

    #[test]
    fn elo_stderr_is_positive_and_scales_with_uncertainty() {
        let se = score_stderr(11, 10, 0);
        let s = standard_score(11, 10, 0);
        let e = elo_stderr(s, se, 21);
        assert!(e > 0.0 && e.is_finite(), "{e}");
        // A 100-game match of the same score has a tighter band.
        let se2 = score_stderr(52, 48, 0);
        let s2 = standard_score(52, 48, 0);
        assert!(elo_stderr(s2, se2, 100) < e);
        // Zero variance (all draws) gives a zero-width band, not NaN.
        assert_eq!(elo_stderr(0.5, 0.0, 21), 0.0);
    }

    // ---------- seeds ----------

    #[test]
    fn per_game_seeds_are_deterministic_and_distinct() {
        let mut seen = HashSet::new();
        for i in 0..1000u64 {
            for stream in [STREAM_OPENING, STREAM_A, STREAM_B] {
                assert!(
                    seen.insert(derive_seed(12345, i, stream)),
                    "collision at game {i} stream {stream}"
                );
            }
        }
        // Reproducible for a fixed base...
        assert_eq!(derive_seed(7, 3, STREAM_A), derive_seed(7, 3, STREAM_A));
        // ...and a different match seed moves every game.
        for i in 0..64u64 {
            assert_ne!(
                derive_seed(7, i, STREAM_A),
                derive_seed(8, i, STREAM_A)
            );
        }
        // Adjacent games get well-separated seeds, not `base + 1`.
        assert_ne!(
            derive_seed(0, 0, STREAM_A).wrapping_add(1),
            derive_seed(0, 1, STREAM_A)
        );
    }

    #[test]
    fn paired_games_share_an_opening_seed_but_not_a_move_seed() {
        // Games 2k and 2k+1 replay the same random opening from
        // opposite colours.
        assert_eq!(
            derive_seed(9, u64::from(4u32 / 2), STREAM_OPENING),
            derive_seed(9, u64::from(5u32 / 2), STREAM_OPENING)
        );
        assert_ne!(
            derive_seed(9, 4, STREAM_A),
            derive_seed(9, 5, STREAM_A),
            "but the sampling stream must still differ"
        );
    }

    // ---------- temperature sampling ----------

    fn visits(counts: &[u32]) -> Vec<(Move, u32)> {
        counts
            .iter()
            .enumerate()
            .map(|(i, &n)| (abalone_game::decode(i as u16), n))
            .collect()
    }

    #[test]
    fn zero_temperature_is_argmax() {
        let v = visits(&[3, 40, 7, 12]);
        let mut rng = SmallRng::seed_from_u64(0);
        for _ in 0..64 {
            assert_eq!(select_move(&v, 0.0, &mut rng), v[1].0);
        }
    }

    #[test]
    fn temperature_reduces_to_argmax_as_it_approaches_zero() {
        let v = visits(&[3, 40, 7, 12]);
        let mut rng = SmallRng::seed_from_u64(1);
        for t in [1e-4f32, 1e-3, 0.01, 0.02] {
            for _ in 0..200 {
                assert_eq!(
                    select_move(&v, t, &mut rng),
                    v[1].0,
                    "t={t} must be argmax"
                );
            }
        }
        // NaN is treated as "no temperature" rather than propagating.
        assert_eq!(select_move(&v, f32::NAN, &mut rng), v[1].0);
    }

    #[test]
    fn temperature_one_samples_proportional_to_visits() {
        let v = visits(&[10, 30, 60, 0]);
        let mut rng = SmallRng::seed_from_u64(2);
        let mut hits = [0u32; 4];
        for _ in 0..20_000 {
            let mv = select_move(&v, 1.0, &mut rng);
            let i = v.iter().position(|&(m, _)| m == mv).unwrap();
            hits[i] += 1;
        }
        assert_eq!(hits[3], 0, "a zero-visit move must never be sampled");
        for (i, want) in [0.1f64, 0.3, 0.6].iter().enumerate() {
            let got = f64::from(hits[i]) / 20_000.0;
            assert!((got - want).abs() < 0.02, "i={i}: {got} vs {want}");
        }
    }

    #[test]
    fn high_temperature_flattens_the_distribution() {
        let v = visits(&[1, 100, 1, 1]);
        let mut rng = SmallRng::seed_from_u64(3);
        let mut argmax_hits = 0u32;
        for _ in 0..2000 {
            if select_move(&v, 8.0, &mut rng) == v[1].0 {
                argmax_hits += 1;
            }
        }
        // At T=1 this would be ~97%; flattened it must be far lower.
        assert!(argmax_hits < 1400, "{argmax_hits}/2000");
        assert!(argmax_hits > 400, "{argmax_hits}/2000");
    }

    #[test]
    fn all_zero_visits_falls_back_to_argmax() {
        // `simulations = 0`, or a root every one of whose descents
        // resolved terminally: no division by zero, no NaN weights.
        let v = visits(&[0, 0, 0]);
        let mut rng = SmallRng::seed_from_u64(4);
        for _ in 0..32 {
            // Ties break to the last maximum, as `SearchResult::best`
            // does; the point is that it is a legal move and stable.
            assert_eq!(select_move(&v, 1.0, &mut rng), v[2].0);
        }
    }

    #[test]
    fn argmax_agrees_with_the_search_engine_tie_break() {
        // `select_move(.., 0.0)` must name the same move MCTS would.
        let g = Game::new_standard();
        let cfg = SearchConfig {
            simulations: 64,
            batch_size: 8,
            ..Default::default()
        };
        let res = search_batched(&g, &cfg, 5, |b| {
            b.iter().map(|_| LeafEval::from_value(0.0)).collect()
        })
        .unwrap();
        let mut rng = SmallRng::seed_from_u64(0);
        assert_eq!(select_move(&res.visits, 0.0, &mut rng), res.best);
    }

    // ---------- aggregation ----------

    fn rec(
        index: u32,
        outcome: Outcome,
        diff: i32,
        transcript: u64,
    ) -> GameRecord {
        GameRecord {
            index,
            a_is_black: index.is_multiple_of(2),
            outcome,
            plies: 100 + index,
            score_diff_a: diff,
            transcript,
        }
    }

    #[test]
    fn aggregate_tallies_and_means() {
        let rs = [
            rec(0, Outcome::A, 6, 1),
            rec(1, Outcome::B, -2, 2),
            rec(2, Outcome::Draw, 0, 3),
            rec(3, Outcome::A, 1, 3),
        ];
        let a = aggregate(&rs);
        assert_eq!((a.wins_a, a.wins_b, a.draws), (2, 1, 1));
        assert_eq!(a.distinct_transcripts, 3);
        assert!((a.mean_score_diff_a - 1.25).abs() < 1e-9);
        // Mean *absolute* differential separates a hair-thin
        // adjudication from a decisive push-off.
        assert!((a.mean_abs_score_diff - 2.25).abs() < 1e-9);
        assert!((a.mean_plies - 101.5).abs() < 1e-9);
    }

    #[test]
    fn aggregate_of_nothing_is_empty() {
        let a = aggregate(&[]);
        assert_eq!((a.wins_a, a.wins_b, a.draws), (0, 0, 0));
        assert_eq!(a.distinct_transcripts, 0);
    }

    // ---------- the Defect-1 regression guard ----------

    /// A cheap stand-in for an engine: visit counts are a pure function
    /// of the position and the player's `salt`, exactly like MCTS over
    /// a deterministic evaluator. Two different salts are two different
    /// "engines", which is what the logged 21-game gate actually had
    /// (a model on one side, the heuristic on the other).
    struct SyntheticPlayer {
        salt: u64,
    }

    impl MovePicker for SyntheticPlayer {
        fn root_visits(
            &mut self,
            g: &Game,
            _rng: &mut SmallRng,
        ) -> Option<Vec<(Move, u32)>> {
            let moves = g.legal_moves();
            if moves.is_empty() {
                return None;
            }
            let base = mix64(
                self.salt
                    ^ (g.board.marbles[0] as u64)
                    ^ ((g.board.marbles[1] >> 64) as u64)
                    ^ u64::from(g.ply),
            );
            Some(
                moves
                    .iter()
                    .enumerate()
                    .map(|(i, &m)| {
                        let h = mix64(base ^ (i as u64));
                        (m, 1 + (h % 64) as u32)
                    })
                    .collect(),
            )
        }
    }

    fn run_match(games: u32, rules: &MatchRules, seed: u64) -> Vec<GameRecord> {
        let mut a = SyntheticPlayer { salt: 0xA11CE };
        let mut b = SyntheticPlayer { salt: 0xB0B };
        (0..games)
            .map(|i| play_one_game(i, seed, rules, &mut a, &mut b))
            .collect()
    }

    /// The bug, reproduced. Deterministic players + a fixed start +
    /// no temperature = one game per colour assignment. This is the
    /// state of the world the guard exists to catch, and it shows the
    /// guard is not vacuous: the very same assertion used in
    /// `randomisation_makes_n_games_n_samples` fails here.
    #[test]
    fn without_randomisation_a_match_is_two_games_replayed() {
        let rules = MatchRules {
            random_opening_plies: 0,
            temperature_plies: 0,
            max_plies: 60,
            ..MatchRules::default()
        };
        let records = run_match(12, &rules, 99);
        let agg = aggregate(&records);
        assert_eq!(
            agg.distinct_transcripts, 2,
            "a deterministic match from a fixed start must collapse to \
             one game per colour assignment — if this ever exceeds 2 the \
             guard below has stopped being meaningful"
        );
        // And therefore the guard's condition holds — it can fail.
        assert!(records.len() as u32 > 2 && agg.distinct_transcripts <= 2);
    }

    /// The fix. Randomised openings + temperature-sampled early plies
    /// turn N games into N samples.
    #[test]
    fn randomisation_makes_n_games_n_samples() {
        let rules = MatchRules {
            random_opening_plies: 2,
            temperature_plies: 10,
            temperature: 1.0,
            max_plies: 60,
            ..MatchRules::default()
        };
        let games = 12u32;
        let records = run_match(games, &rules, 99);
        let agg = aggregate(&records);
        assert!(
            agg.distinct_transcripts > 2,
            "N games must be more than 2 distinct transcripts, got {}",
            agg.distinct_transcripts
        );
        // In practice every game should differ.
        assert_eq!(
            agg.distinct_transcripts, games,
            "expected all {games} games to differ"
        );
    }

    /// Either source of variation is sufficient on its own, so a caller
    /// who disables one is not silently back at 2 games.
    #[test]
    fn random_openings_alone_diversify_a_match() {
        let rules = MatchRules {
            random_opening_plies: 3,
            temperature_plies: 0,
            max_plies: 60,
            ..MatchRules::default()
        };
        let agg = aggregate(&run_match(12, &rules, 5));
        assert!(agg.distinct_transcripts > 2, "{:?}", agg);
    }

    #[test]
    fn temperature_alone_diversifies_a_match() {
        let rules = MatchRules {
            random_opening_plies: 0,
            temperature_plies: 10,
            temperature: 1.0,
            max_plies: 60,
            ..MatchRules::default()
        };
        let agg = aggregate(&run_match(12, &rules, 5));
        assert!(agg.distinct_transcripts > 2, "{:?}", agg);
    }

    /// A player that never searches: the `random` rung of the ladder.
    struct RandomPlayer;

    impl MovePicker for RandomPlayer {
        fn root_visits(
            &mut self,
            _g: &Game,
            _rng: &mut SmallRng,
        ) -> Option<Vec<(Move, u32)>> {
            None
        }
    }

    #[test]
    fn random_players_produce_distinct_games() {
        let rules = MatchRules {
            max_plies: 60,
            ..MatchRules::default()
        };
        let (mut a, mut b) = (RandomPlayer, RandomPlayer);
        let records: Vec<GameRecord> = (0..10)
            .map(|i| play_one_game(i, 3, &rules, &mut a, &mut b))
            .collect();
        let agg = aggregate(&records);
        assert_eq!(agg.distinct_transcripts, 10);
        assert_eq!(agg.wins_a + agg.wins_b + agg.draws, 10);
    }

    // ---------- match reproducibility ----------

    #[test]
    fn a_match_is_reproducible_from_its_seed() {
        let rules = MatchRules {
            max_plies: 60,
            ..MatchRules::default()
        };
        let first = run_match(6, &rules, 2024);
        let again = run_match(6, &rules, 2024);
        for (x, y) in first.iter().zip(again.iter()) {
            assert_eq!(x.transcript, y.transcript);
            assert_eq!(x.outcome, y.outcome);
            assert_eq!(x.score_diff_a, y.score_diff_a);
        }
        // A different match seed gives a different match.
        let other = run_match(6, &rules, 2025);
        assert!(first
            .iter()
            .zip(other.iter())
            .any(|(x, y)| x.transcript != y.transcript));
    }

    #[test]
    fn colours_alternate_and_score_diff_follows_the_right_side() {
        let rules = MatchRules {
            max_plies: 40,
            ..MatchRules::default()
        };
        let records = run_match(4, &rules, 11);
        assert_eq!(
            records.iter().map(|r| r.a_is_black).collect::<Vec<_>>(),
            vec![true, false, true, false]
        );
        for r in &records {
            // A win for A means a non-negative differential for A, and
            // vice versa — under adjudication these are the same fact.
            match r.outcome {
                Outcome::A => assert!(r.score_diff_a >= 0, "{r:?}"),
                Outcome::B => assert!(r.score_diff_a <= 0, "{r:?}"),
                Outcome::Draw => {}
            }
        }
    }

    #[test]
    fn opening_plies_are_actually_played() {
        let rules = MatchRules {
            random_opening_plies: 4,
            temperature_plies: 0,
            max_plies: 4,
            ..MatchRules::default()
        };
        // With the cap equal to the opening length the players never
        // move, so the transcript is the opening alone — and pairs of
        // games share it.
        let records = run_match(4, &rules, 77);
        assert!(records.iter().all(|r| r.plies == 4));
        assert_eq!(records[0].transcript, records[1].transcript);
        assert_ne!(records[0].transcript, records[2].transcript);
    }
}
