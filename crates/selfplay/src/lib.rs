//! Self-play trajectory generation.
//!
//! One [`play_game`] call plays a single game with batched MCTS and returns a
//! [`GameOutcome`]: every searched position, its visit distribution, the move
//! played, and the training targets that can only be computed once the game is
//! over (outcome, final score differential, discounted capture map). The
//! [`shard`] module writes those to parquet.
//!
//! Three things here are worth reading the rationale for:
//!
//! **Batched search.** Search is driven through the [`abalone_mcts::Search`]
//! coroutine, so a move costs `simulations / batch_size` network calls instead
//! of `simulations`. That is the single largest throughput lever in the system
//! ([MODEL §7.1](../../../docs/MODEL.md)) and it is why the evaluator signature
//! is `FnMut(&[Game]) -> Vec<LeafEval>` rather than a per-position callback.
//!
//! **Curriculum seeding** ([MODEL §4](../../../docs/MODEL.md)). A fraction of
//! games start with each side having already conceded `0..=handicap_max`
//! marbles, drawn independently per side. The win condition is a counter, so
//! starting near the threshold puts terminals inside the search horizon
//! immediately — real signal in generation one, with no hand-written evaluator
//! and nothing to unlearn. Which marbles are removed is uniformly random, so no
//! positional judgment enters.
//!
//! **Playout cap randomisation** ([MODEL §7.2](../../../docs/MODEL.md)). Most
//! moves run at `sims_fast`; a random `full_search_rate` fraction run at
//! `sims_full` and *only those* set [`TrajectoryEntry::is_full_search`], i.e.
//! only those carry a policy target. Every position still carries value, score
//! and capture-map targets, which is what makes the trade profitable.

/// Plane encoding, re-exported so `abalone_selfplay::encoder` still resolves.
/// It lives in its own crate because the browser build needs it too and cannot
/// link this crate's `ort`/`parquet` dependencies.
pub use abalone_encoder as encoder;

pub mod ort_eval;
pub mod shard;

use abalone_game::{
    encode, Cell, Game, GameState, Move, Opening, Side, DEFAULT_MAX_PLIES, NO_PROGRESS_DISABLED,
};
use abalone_mcts::{LeafEval, Search, SearchConfig};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

/// Cells in one 9x9 capture-map channel (includes the 20 off-board slots, so
/// the flat index is the raw [`Cell`] value — same convention as the encoder).
pub const CAP_MAP_CELLS: usize = 81;
/// Capture-map channels: 0 = marbles the side to move loses, 1 = marbles the
/// opponent loses.
pub const CAP_MAP_CHANNELS: usize = 2;
/// Flat size of the capture-map target, `channel * 81 + cell`.
pub const CAP_MAP_SIZE: usize = CAP_MAP_CHANNELS * CAP_MAP_CELLS;

/// Discount applied per ply of distance to a future capture.
pub const DEFAULT_CAPTURE_GAMMA: f32 = 0.98;
/// Largest handicap either side may be given. Six is a win, so five is the cap.
pub const MAX_HANDICAP: u8 = 5;

/// Self-play hyperparameters.
///
/// The search knobs come in a fast/full pair because of playout cap
/// randomisation; everything else is per-game setup.
#[derive(Clone, Debug)]
pub struct SelfPlayConfig {
    // ---- search ----
    /// Simulations for an ordinary move. No policy target is recorded.
    pub sims_fast: u32,
    /// Simulations for a full-search move. These are the policy targets.
    pub sims_full: u32,
    /// Probability a given move runs the full simulation count.
    pub full_search_rate: f32,
    pub c_puct: f32,
    /// Leaves collected per network call. `1` reproduces sequential search.
    pub batch_size: usize,
    pub virtual_loss: f32,
    pub fpu_reduction: f32,
    /// Root Dirichlet noise. A [`SearchConfig`] field — never an evaluator
    /// side effect.
    pub dirichlet_alpha: f32,
    pub dirichlet_eps: f32,

    // ---- move selection ----
    /// Plies (from the start of the game) during which the played move is
    /// sampled from the visit distribution rather than taken as the argmax.
    pub temperature_plies: u32,
    pub temperature: f32,

    // ---- game setup / curriculum ----
    pub opening: Opening,
    /// Fraction of games given a capture handicap.
    pub handicap_rate: f32,
    /// Handicaps are drawn from `0..=handicap_max`, independently per side.
    pub handicap_max: u8,
    /// Plies played uniformly at random before search takes over, to
    /// decorrelate the games in a generation. Not searched and not recorded:
    /// they are a diversity device, not data.
    pub random_opening_plies: u32,
    pub max_plies: u32,
    /// [`NO_PROGRESS_DISABLED`] switches the no-progress rule off.
    pub no_progress_plies: u32,

    // ---- targets ----
    /// Per-ply discount for the capture-map target.
    pub capture_gamma: f32,
}

impl Default for SelfPlayConfig {
    fn default() -> Self {
        Self {
            sims_fast: 200,
            sims_full: 800,
            full_search_rate: 0.25,
            c_puct: 1.4,
            batch_size: 16,
            virtual_loss: 1.0,
            fpu_reduction: 0.25,
            // alpha ~ 10 / branching, branching ~ 60.
            dirichlet_alpha: 0.2,
            dirichlet_eps: 0.25,
            temperature_plies: 30,
            temperature: 1.0,
            opening: Opening::default(),
            handicap_rate: 0.7,
            handicap_max: MAX_HANDICAP,
            random_opening_plies: 2,
            max_plies: DEFAULT_MAX_PLIES,
            no_progress_plies: NO_PROGRESS_DISABLED,
            capture_gamma: DEFAULT_CAPTURE_GAMMA,
        }
    }
}

impl SelfPlayConfig {
    /// The [`SearchConfig`] for a move with the given simulation budget.
    pub fn search_config(&self, simulations: u32) -> SearchConfig {
        SearchConfig {
            simulations,
            c_puct: self.c_puct,
            batch_size: self.batch_size.max(1),
            virtual_loss: self.virtual_loss,
            fpu_reduction: self.fpu_reduction,
            dirichlet_alpha: self.dirichlet_alpha,
            dirichlet_eps: self.dirichlet_eps,
        }
    }

    fn validate(&self) {
        assert!(
            self.handicap_max <= MAX_HANDICAP,
            "handicap_max must be <= {MAX_HANDICAP}; {} would seed a finished game",
            self.handicap_max
        );
        assert!(self.sims_fast > 0 && self.sims_full > 0, "simulations must be > 0");
        assert!(
            (0.0..=1.0).contains(&self.full_search_rate),
            "full_search_rate must be a probability"
        );
        assert!(
            (0.0..=1.0).contains(&self.handicap_rate),
            "handicap_rate must be a probability"
        );
        assert!(self.temperature > 0.0, "temperature must be > 0");
    }
}

/// A marble leaving the board: when, from which cell, and whose it was.
///
/// `cell` is where the marble stood at the moment it was pushed off, which for
/// a multi-marble push is *not* what a before/after bitboard diff reports — see
/// [`abalone_game::Board::apply_with_capture`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct CaptureEvent {
    /// Ply of the position the capturing move was made *from*.
    pub ply: u32,
    pub cell: Cell,
    /// The side that lost the marble.
    pub lost_by: Side,
}

/// One searched position in a self-play trajectory.
#[derive(Clone, Debug)]
pub struct TrajectoryEntry {
    pub state: Game,
    /// Parallel arrays: `(move_idx, visit_count)` for each legal child of the
    /// root. `move_idx` is the flat 0..2562 index from `abalone_game::encode`.
    /// Only a policy target when [`is_full_search`](Self::is_full_search).
    pub child_visits: Vec<(u16, u32)>,
    /// Move index actually played from this position.
    pub move_played: u16,
    /// Visit-weighted average Q at the root from this position's `to_move`
    /// POV. Diagnostics and the review UI; not a loss term.
    pub q: f32,
    /// True iff this position was searched at `sims_full`. Playout cap
    /// randomisation: only these positions contribute to the policy loss.
    pub is_full_search: bool,
    /// Sparse capture-map target: `(channel * 81 + cell, weight)`, ascending by
    /// index, weights in `(0, 1]`. Channel is POV-relative, cells are absolute.
    pub capture_map: Vec<(u16, f32)>,
}

#[derive(Clone, Debug)]
pub struct GameOutcome {
    /// Source game within a (run, generation). Carried into the shard so rows
    /// can be grouped without inferring boundaries from `ply` resets.
    pub game_id: u32,
    /// RNG seed; the whole game replays from it given the same evaluator.
    pub seed: u64,
    pub opening: Opening,
    /// Marbles Black conceded at curriculum seeding (0 if unseeded).
    pub handicap_black: u8,
    /// Marbles White conceded at curriculum seeding (0 if unseeded).
    pub handicap_white: u8,
    pub trajectory: Vec<TrajectoryEntry>,
    pub final_state: Game,
    /// Every marble pushed off during the game, in ply order.
    pub captures: Vec<CaptureEvent>,
}

impl GameOutcome {
    /// Final result from the perspective of `to_move` at the given entry.
    /// `+1` win, `-1` loss, `0` draw.
    pub fn z_for(&self, entry: &TrajectoryEntry) -> f32 {
        f32::from(self.z_class_for(entry))
    }

    /// [`z_for`](Self::z_for) as the `i8` the shard stores.
    pub fn z_class_for(&self, entry: &TrajectoryEntry) -> i8 {
        let pov = entry.state.turn;
        match self.final_state.state() {
            GameState::Wins(s) => {
                if s == pov {
                    1
                } else {
                    -1
                }
            }
            _ => 0,
        }
    }

    /// Final capture differential from `pov`'s point of view, in `[-6, 6]`.
    /// This is the game's own score, not an evaluation.
    pub fn final_score_diff(&self, pov: Side) -> i32 {
        self.final_state.score_diff(pov)
    }

    /// [`final_score_diff`](Self::final_score_diff) from the entry's POV, as
    /// the `i8` the shard stores.
    pub fn score_diff_for(&self, entry: &TrajectoryEntry) -> i8 {
        self.final_score_diff(entry.state.turn) as i8
    }
}

/// Accumulate the discounted capture-map target for a position at ply `t` whose
/// side to move is `pov`.
///
/// Every capture at ply `t' >= t` that removed a marble of side `X` from cell
/// `c` contributes `gamma^(t'-t)` to channel `0` if `X == pov` else channel `1`,
/// at cell `c`. Cells are absolute board indices — only the *channel* is
/// POV-relative, matching the own/opponent convention of the input planes.
/// Weights are clamped to `[0, 1]`; only non-zero entries are returned, sorted
/// by flat index.
pub fn capture_map_target(
    captures: &[CaptureEvent],
    ply: u32,
    pov: Side,
    gamma: f32,
) -> Vec<(u16, f32)> {
    let mut dense = [0f32; CAP_MAP_SIZE];
    for ev in captures {
        if ev.ply < ply {
            continue;
        }
        let channel = if ev.lost_by == pov { 0 } else { 1 };
        let idx = channel * CAP_MAP_CELLS + ev.cell as usize;
        debug_assert!(idx < CAP_MAP_SIZE, "capture cell {} out of range", ev.cell);
        dense[idx] += gamma.powi((ev.ply - ply) as i32);
    }
    dense
        .iter()
        .enumerate()
        .filter(|(_, &w)| w > 0.0)
        .map(|(i, &w)| (i as u16, w.clamp(0.0, 1.0)))
        .collect()
}

/// Sample an index into `visits` weighted by `N_i^(1/temp)`.
fn sample_by_visits<R: Rng + ?Sized>(visits: &[(Move, u32)], temp: f32, rng: &mut R) -> usize {
    debug_assert!(!visits.is_empty());
    let weights: Vec<f64> = visits
        .iter()
        .map(|&(_, v)| (v as f64).powf(1.0 / temp as f64))
        .collect();
    let total: f64 = weights.iter().sum();
    if total <= 0.0 {
        // Degenerate case (all-zero visits): fall back to uniform.
        return rng.gen_range(0..visits.len());
    }
    let mut r = rng.gen_range(0.0..total);
    for (i, w) in weights.iter().enumerate() {
        r -= *w;
        if r <= 0.0 {
            return i;
        }
    }
    visits.len() - 1
}

fn argmax(visits: &[(Move, u32)]) -> usize {
    visits
        .iter()
        .enumerate()
        .max_by_key(|(_, &(_, v))| v)
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Draw this game's starting position: opening, optional capture handicap, and
/// a few uniformly-random plies. Returns the game plus the handicap actually
/// applied so it can be recorded in the shard.
fn seed_position<R: Rng>(cfg: &SelfPlayConfig, rng: &mut R) -> (Game, u8, u8) {
    let (hb, hw) = if rng.gen::<f32>() < cfg.handicap_rate {
        // Independently per side: (5, 5), (0, 3) and (2, 0) are all reachable,
        // which is the point — asymmetric material is exactly the regime
        // self-play from the standard start never visits.
        (
            rng.gen_range(0..=cfg.handicap_max),
            rng.gen_range(0..=cfg.handicap_max),
        )
    } else {
        (0, 0)
    };
    let mut g = Game::with_handicap(
        cfg.opening,
        cfg.max_plies,
        cfg.no_progress_plies,
        hb,
        hw,
        rng,
    );
    for _ in 0..cfg.random_opening_plies {
        if g.is_terminal() {
            break;
        }
        let moves = g.legal_moves();
        if moves.is_empty() {
            break;
        }
        let pick = rng.gen_range(0..moves.len());
        g.apply(moves[pick]);
    }
    (g, hb, hw)
}

/// Play one self-play game. `eval_fn` is called once per batch of leaf
/// positions and must return one [`LeafEval`] per position, in order.
///
/// The game is a pure function of `cfg`, `seed` and `eval_fn`; `game_id` is
/// carried through to the shard and does not affect play.
pub fn play_game<F>(cfg: &SelfPlayConfig, game_id: u32, seed: u64, mut eval_fn: F) -> GameOutcome
where
    F: FnMut(&[Game]) -> Vec<LeafEval>,
{
    cfg.validate();
    let mut rng = SmallRng::seed_from_u64(seed);
    let (mut g, handicap_black, handicap_white) = seed_position(cfg, &mut rng);

    let mut traj: Vec<TrajectoryEntry> = Vec::new();
    let mut captures: Vec<CaptureEvent> = Vec::new();

    // Tree reuse. `Search` bakes its simulation budget in at construction and a
    // rerooted tree carries that budget with it, so we can only re-root into a
    // move whose budget matches. Mixing them would make the recorded visit
    // counts mean different things from move to move; when the budget flips we
    // pay for a fresh tree instead.
    let mut carried: Option<(Search, u32)> = None;

    while !g.is_terminal() {
        let is_full_search = rng.gen::<f32>() < cfg.full_search_rate;
        let budget = if is_full_search {
            cfg.sims_full
        } else {
            cfg.sims_fast
        };

        let mut search = match carried.take() {
            Some((s, b)) if b == budget => s,
            _ => Search::begin(&g, &cfg.search_config(budget), rng.gen()),
        };
        loop {
            let batch = search.next_batch();
            if batch.is_empty() {
                break;
            }
            let evals = eval_fn(batch);
            search.submit(&evals);
        }
        let res = search
            .result()
            .expect("a non-terminal root always has at least one legal move");

        let chosen_idx = if g.ply < cfg.temperature_plies {
            sample_by_visits(&res.visits, cfg.temperature, &mut rng)
        } else {
            argmax(&res.visits)
        };
        let chosen_mv = res.visits[chosen_idx].0;

        // Visit-weighted average Q at the root, from this position's POV.
        let total_visits: u64 = res.visits.iter().map(|&(_, v)| v as u64).sum();
        let q = if total_visits == 0 {
            0.0
        } else {
            res.q_parent_pov
                .iter()
                .zip(res.visits.iter())
                .map(|(&q, &(_, v))| q * (v as f32))
                .sum::<f32>()
                / (total_visits as f32)
        };

        traj.push(TrajectoryEntry {
            state: g,
            child_visits: res.visits.iter().map(|&(mv, v)| (encode(mv), v)).collect(),
            move_played: encode(chosen_mv),
            q,
            is_full_search,
            // Filled in once the game is over and all captures are known.
            capture_map: Vec::new(),
        });

        let mover = g.turn;
        let ply = g.ply;
        if let Some(cell) = g.apply_with_capture(chosen_mv) {
            captures.push(CaptureEvent {
                ply,
                cell,
                lost_by: mover.other(),
            });
        }
        // Re-rooting copies the retained subtree, so there is no point paying
        // for it on the move that ended the game.
        carried = (!g.is_terminal()).then(|| (search.reroot(chosen_mv), budget));
    }

    for entry in traj.iter_mut() {
        entry.capture_map = capture_map_target(
            &captures,
            entry.state.ply,
            entry.state.turn,
            cfg.capture_gamma,
        );
    }

    GameOutcome {
        game_id,
        seed,
        opening: cfg.opening,
        handicap_black,
        handicap_white,
        trajectory: traj,
        final_state: g,
        captures,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use abalone_game::cell::parse;
    use abalone_mcts::heuristic;

    /// Batched wrapper around the heuristic evaluator: no network, no RNG, and
    /// deterministic, so it isolates the self-play mechanics under test.
    fn heuristic_batch(games: &[Game]) -> Vec<LeafEval> {
        let mut rng = SmallRng::seed_from_u64(0);
        games.iter().map(|g| heuristic(g, &mut rng)).collect()
    }

    /// A cheap config for tests: short games, small budgets, no noise.
    fn test_cfg() -> SelfPlayConfig {
        SelfPlayConfig {
            sims_fast: 8,
            sims_full: 24,
            full_search_rate: 0.25,
            batch_size: 1,
            temperature_plies: 4,
            dirichlet_eps: 0.0,
            opening: Opening::BelgianDaisy,
            handicap_rate: 0.0,
            random_opening_plies: 0,
            max_plies: 24,
            ..Default::default()
        }
    }

    // ---------- basic trajectory shape ----------

    #[test]
    fn play_game_produces_consistent_trajectory() {
        let cfg = test_cfg();
        let outcome = play_game(&cfg, 7, 0, heuristic_batch);
        assert!(outcome.final_state.is_terminal());
        assert_eq!(outcome.game_id, 7);
        assert_eq!(outcome.seed, 0);
        assert_eq!(outcome.opening, Opening::BelgianDaisy);
        // With `random_opening_plies = 0` every ply is searched and recorded.
        for (i, entry) in outcome.trajectory.iter().enumerate() {
            assert_eq!(entry.state.ply as usize, i);
            assert_eq!(entry.state.max_plies, cfg.max_plies);
            let legal = entry.state.legal_moves();
            let played = legal
                .iter()
                .any(|&m| encode(m) == entry.move_played);
            assert!(played, "move_played must be legal at ply {i}");
            assert_eq!(entry.child_visits.len(), legal.len());
        }
        for entry in &outcome.trajectory {
            let z = outcome.z_for(entry);
            assert!(z == 0.0 || z == 1.0 || z == -1.0);
            assert!((-6..=6).contains(&outcome.score_diff_for(entry)));
        }
    }

    #[test]
    fn z_alternates_sign_along_trajectory() {
        let cfg = test_cfg();
        let outcome = play_game(&cfg, 0, 1, heuristic_batch);
        if let GameState::Wins(_) = outcome.final_state.state() {
            for w in outcome.trajectory.windows(2) {
                assert_eq!(outcome.z_for(&w[0]), -outcome.z_for(&w[1]));
                assert_eq!(
                    outcome.score_diff_for(&w[0]),
                    -outcome.score_diff_for(&w[1])
                );
            }
        }
    }

    #[test]
    fn random_opening_plies_skip_the_first_positions() {
        let cfg = SelfPlayConfig {
            random_opening_plies: 2,
            ..test_cfg()
        };
        let outcome = play_game(&cfg, 0, 5, heuristic_batch);
        assert_eq!(
            outcome.trajectory[0].state.ply, 2,
            "random opening plies are played but not recorded"
        );
    }

    // ---------- curriculum seeding ----------

    #[test]
    fn handicap_seeding_hits_the_requested_rate_and_range() {
        let cfg = SelfPlayConfig {
            handicap_rate: 0.5,
            handicap_max: 5,
            ..test_cfg()
        };
        const N: u32 = 2000;
        let mut seeded = 0u32;
        let mut hist = [[0u32; 6]; 2];
        for i in 0..N {
            let mut rng = SmallRng::seed_from_u64(u64::from(i));
            let (g, hb, hw) = seed_position(&cfg, &mut rng);
            if hb > 0 || hw > 0 {
                seeded += 1;
            }
            hist[0][hb as usize] += 1;
            hist[1][hw as usize] += 1;
            // The board must stay perfectly consistent with the counters.
            assert_eq!(g.board.count(Side::Black), 14 - u32::from(hb));
            assert_eq!(g.board.count(Side::White), 14 - u32::from(hw));
            assert_eq!(g.board.lost(Side::Black), hb);
            assert_eq!(g.board.lost(Side::White), hw);
            assert_eq!(g.score_diff(Side::Black), i32::from(hw) - i32::from(hb));
            assert!(!g.is_terminal());
        }
        // P(no handicap | seeded) = (1/6)^2, so the observed "looks seeded"
        // rate is rate * (1 - 1/36).
        let expected = 0.5 * (1.0 - 1.0 / 36.0) * f64::from(N);
        let observed = f64::from(seeded);
        assert!(
            (observed - expected).abs() < 0.1 * expected,
            "seeded {observed} games, expected ~{expected}"
        );
        // Uniform over 0..=5 on each side: ~N/2/6 each, generous band.
        for (side, counts) in hist.iter().enumerate() {
            for (level, &c) in counts.iter().enumerate() {
                assert!(c > 0, "handicap level {level} never drawn for side {side}");
            }
            let seeded_side: u32 = counts[1..].iter().sum();
            let per_level = f64::from(seeded_side) / 5.0;
            for (level, &c) in counts.iter().enumerate().skip(1) {
                let c = f64::from(c);
                assert!(
                    (c - per_level).abs() < 0.4 * per_level,
                    "handicap level {level} drawn {c} times, expected ~{per_level}"
                );
            }
        }
    }

    #[test]
    fn handicap_rate_zero_never_seeds() {
        let cfg = SelfPlayConfig {
            handicap_rate: 0.0,
            ..test_cfg()
        };
        for i in 0..200 {
            let mut rng = SmallRng::seed_from_u64(i);
            let (g, hb, hw) = seed_position(&cfg, &mut rng);
            assert_eq!((hb, hw), (0, 0));
            assert_eq!(g.board.pushed_off, [0, 0]);
        }
    }

    #[test]
    fn handicap_seeded_games_terminate_and_are_mostly_decisive() {
        let cfg = SelfPlayConfig {
            handicap_rate: 1.0,
            handicap_max: 5,
            max_plies: 60,
            ..test_cfg()
        };
        let mut decisive = 0;
        const N: u32 = 12;
        for i in 0..N {
            let outcome = play_game(&cfg, i, u64::from(i), heuristic_batch);
            assert!(outcome.final_state.is_terminal(), "game {i} did not finish");
            assert!(outcome.final_state.ply <= cfg.max_plies);
            // The seeded handicap is visible in the very first recorded
            // position's counters, and captures only add to it.
            let first = &outcome.trajectory[0].state;
            assert_eq!(first.board.lost(Side::Black), outcome.handicap_black);
            assert_eq!(first.board.lost(Side::White), outcome.handicap_white);
            if !matches!(outcome.final_state.state(), GameState::Draw) {
                decisive += 1;
            }
        }
        assert!(
            decisive * 2 >= N,
            "handicap seeding should make most games decisive, got {decisive}/{N}"
        );
    }

    // ---------- playout cap randomisation ----------

    #[test]
    fn playout_cap_randomisation_splits_and_budgets_correctly() {
        let cfg = SelfPlayConfig {
            sims_fast: 10,
            sims_full: 40,
            full_search_rate: 0.25,
            temperature_plies: 200,
            max_plies: 60,
            handicap_rate: 1.0,
            ..test_cfg()
        };
        let mut full = 0u32;
        let mut total = 0u32;
        for i in 0..12u32 {
            let outcome = play_game(&cfg, i, u64::from(i) + 100, heuristic_batch);
            for e in &outcome.trajectory {
                let sum: u32 = e.child_visits.iter().map(|(_, v)| v).sum();
                let budget = if e.is_full_search {
                    cfg.sims_full
                } else {
                    cfg.sims_fast
                };
                // A re-rooted tree contributes its retained visits to the
                // budget, and the visit that first expanded the retained node
                // descended no further — so the children can be one short.
                assert!(
                    sum == budget || sum == budget - 1,
                    "expected {budget} visits (or one fewer after tree reuse), got {sum}"
                );
                if e.is_full_search {
                    full += 1;
                }
                total += 1;
            }
        }
        let rate = f64::from(full) / f64::from(total);
        assert!(
            (rate - 0.25).abs() < 0.1,
            "full-search rate {rate:.3} over {total} positions, expected ~0.25"
        );
    }

    #[test]
    fn full_search_rate_extremes() {
        let never = SelfPlayConfig {
            full_search_rate: 0.0,
            ..test_cfg()
        };
        let o = play_game(&never, 0, 3, heuristic_batch);
        assert!(o.trajectory.iter().all(|e| !e.is_full_search));
        assert!(o
            .trajectory
            .iter()
            .all(|e| e.child_visits.iter().map(|(_, v)| v).sum::<u32>() >= never.sims_fast - 1));

        let always = SelfPlayConfig {
            full_search_rate: 1.0,
            ..test_cfg()
        };
        let o = play_game(&always, 0, 3, heuristic_batch);
        assert!(o.trajectory.iter().all(|e| e.is_full_search));
        assert!(o
            .trajectory
            .iter()
            .all(|e| e.child_visits.iter().map(|(_, v)| v).sum::<u32>() >= always.sims_full - 1));
    }

    // ---------- capture-map targets ----------

    fn ev(ply: u32, cell: &str, lost_by: Side) -> CaptureEvent {
        CaptureEvent {
            ply,
            cell: parse(cell).unwrap(),
            lost_by,
        }
    }

    #[test]
    fn capture_map_discounts_by_distance_and_flips_channel() {
        let gamma = 0.98f32;
        let c = parse("A1").unwrap();
        let events = [ev(10, "A1", Side::Black)];

        // Black to move at ply 10: Black loses the marble -> channel 0, weight 1.
        let m = capture_map_target(&events, 10, Side::Black, gamma);
        assert_eq!(m.len(), 1);
        assert_eq!(m[0].0, u16::from(c));
        assert!((m[0].1 - 1.0).abs() < 1e-6);

        // Four plies earlier, still Black to move: gamma^4.
        let m = capture_map_target(&events, 6, Side::Black, gamma);
        assert_eq!(m[0].0, u16::from(c));
        assert!((m[0].1 - gamma.powi(4)).abs() < 1e-6);

        // Same ply, White to move: the marble is the OPPONENT's -> channel 1.
        let m = capture_map_target(&events, 10, Side::White, gamma);
        assert_eq!(m.len(), 1);
        assert_eq!(m[0].0, CAP_MAP_CELLS as u16 + u16::from(c));
        assert!((m[0].1 - 1.0).abs() < 1e-6);

        // Positions after the capture see nothing.
        assert!(capture_map_target(&events, 11, Side::Black, gamma).is_empty());
    }

    #[test]
    fn capture_map_accumulates_and_clamps() {
        let gamma = 1.0f32;
        // Three captures on the same cell, same victim: 3.0 before clamping.
        let events = [
            ev(4, "E5", Side::White),
            ev(6, "E5", Side::White),
            ev(8, "E5", Side::White),
        ];
        let m = capture_map_target(&events, 0, Side::White, gamma);
        assert_eq!(m.len(), 1);
        assert_eq!(m[0].1, 1.0, "weights are clamped to [0, 1]");

        // With discounting the same three stay under 1 further back.
        let m = capture_map_target(&events, 0, Side::White, 0.5);
        let expect = 0.5f32.powi(4) + 0.5f32.powi(6) + 0.5f32.powi(8);
        assert!((m[0].1 - expect).abs() < 1e-6);
    }

    #[test]
    fn capture_map_separates_channels_and_sorts_by_index() {
        let gamma = 0.98f32;
        let events = [
            ev(3, "I9", Side::White),
            ev(5, "A1", Side::Black),
            ev(7, "E5", Side::White),
        ];
        let m = capture_map_target(&events, 0, Side::Black, gamma);
        assert_eq!(m.len(), 3);
        for w in m.windows(2) {
            assert!(w[0].0 < w[1].0, "sparse entries must be index-sorted");
        }
        // Black's own loss (A1, ply 5) is channel 0; the two White losses are
        // channel 1.
        let a1 = u16::from(parse("A1").unwrap());
        assert!(m.iter().any(|&(i, w)| i == a1 && (w - gamma.powi(5)).abs() < 1e-6));
        let i9 = CAP_MAP_CELLS as u16 + u16::from(parse("I9").unwrap());
        assert!(m.iter().any(|&(i, w)| i == i9 && (w - gamma.powi(3)).abs() < 1e-6));
        assert!(m.iter().all(|&(i, _)| i != CAP_MAP_CELLS as u16 + a1));
    }

    #[test]
    fn capture_map_is_empty_without_captures() {
        assert!(capture_map_target(&[], 0, Side::Black, 0.98).is_empty());
    }

    #[test]
    fn capture_map_targets_match_the_game_that_produced_them() {
        // Play a real game with handicaps so captures actually happen, then
        // recompute every target from the recorded events and compare.
        let cfg = SelfPlayConfig {
            handicap_rate: 1.0,
            max_plies: 40,
            ..test_cfg()
        };
        let mut saw_captures = false;
        for i in 0..8u32 {
            let o = play_game(&cfg, i, u64::from(i) + 900, heuristic_batch);
            if !o.captures.is_empty() {
                saw_captures = true;
            }
            for ev in &o.captures {
                assert!(ev.ply < o.final_state.ply);
            }
            for e in &o.trajectory {
                let expect = capture_map_target(
                    &o.captures,
                    e.state.ply,
                    e.state.turn,
                    cfg.capture_gamma,
                );
                assert_eq!(e.capture_map, expect);
                for &(idx, w) in &e.capture_map {
                    assert!((idx as usize) < CAP_MAP_SIZE);
                    assert!(w > 0.0 && w <= 1.0);
                }
            }
            // The last position before a decisive capture must carry weight 1.
            if let Some(last) = o.captures.last() {
                let entry = o
                    .trajectory
                    .iter()
                    .find(|e| e.state.ply == last.ply)
                    .expect("the capturing position is in the trajectory");
                let channel = if last.lost_by == entry.state.turn { 0 } else { 1 };
                let idx = (channel * CAP_MAP_CELLS + last.cell as usize) as u16;
                let w = entry
                    .capture_map
                    .iter()
                    .find(|&&(i, _)| i == idx)
                    .map(|&(_, w)| w)
                    .expect("the imminent capture must be in the map");
                assert!(w >= 0.99, "an immediate capture has weight ~1, got {w}");
            }
        }
        assert!(saw_captures, "expected captures in at least one seeded game");
    }

    // ---------- batching ----------

    #[test]
    fn batched_and_sequential_search_agree_on_game_shape() {
        // Same seed, same evaluator, different batch size. The trees differ
        // (virtual loss changes the descent order), but both must produce a
        // legal, terminating game of comparable length.
        let base = SelfPlayConfig {
            sims_fast: 32,
            sims_full: 32,
            full_search_rate: 0.5,
            max_plies: 30,
            handicap_rate: 1.0,
            ..test_cfg()
        };
        for seed in 0..3u64 {
            let seq = play_game(&SelfPlayConfig { batch_size: 1, ..base.clone() }, 0, seed, heuristic_batch);
            let bat = play_game(&SelfPlayConfig { batch_size: 8, ..base.clone() }, 0, seed, heuristic_batch);
            for o in [&seq, &bat] {
                assert!(o.final_state.is_terminal());
                for e in &o.trajectory {
                    let legal = e.state.legal_moves();
                    assert!(legal.iter().any(|&m| encode(m) == e.move_played));
                    let sum: u32 = e.child_visits.iter().map(|(_, v)| v).sum();
                    assert!(sum == 32 || sum == 31, "batched search must respect the budget, got {sum}");
                }
            }
            // Identical simulation budgets on both sides mean the same number of
            // decisions gets made; the seeded position is identical too.
            assert_eq!(seq.handicap_black, bat.handicap_black);
            assert_eq!(seq.handicap_white, bat.handicap_white);
            assert_eq!(seq.trajectory[0].state, bat.trajectory[0].state);
        }
    }

    /// Self-play throughput at several batch sizes.
    ///
    /// ```text
    /// cargo test --release -p abalone-selfplay -- --ignored --nocapture throughput
    /// ```
    ///
    /// Set `ABALONE_BENCH_MODEL=/path/to.onnx` to measure the real network
    /// path; without it the evaluator is synthetic (planes are still encoded,
    /// so the search and encoding costs are real — only the matmuls are not).
    /// Old checkpoints under `runs/*/checkpoints/` are the previous 6-plane,
    /// 2-output contract and will not load; export a fresh one with
    /// `model/export_onnx.py`.
    #[test]
    #[ignore = "measurement; run explicitly with --ignored --nocapture"]
    fn measure_selfplay_throughput() {
        use std::time::Instant;

        let model = std::env::var("ABALONE_BENCH_MODEL").ok();
        let mut ort = model.as_ref().map(|p| {
            crate::ort_eval::OrtEvaluator::from_onnx(p).expect("load ABALONE_BENCH_MODEL")
        });
        let fixed_batch = crate::ort_eval::use_coreml();
        println!(
            "\nself-play throughput ({})",
            match &model {
                Some(p) => format!("ONNX {p}"),
                None => "synthetic evaluator".to_string(),
            }
        );

        const GAMES: u32 = 4;
        for &batch_size in &[1usize, 8, 32] {
            let cfg = SelfPlayConfig {
                sims_fast: 100,
                sims_full: 400,
                full_search_rate: 0.25,
                batch_size,
                opening: Opening::BelgianDaisy,
                handicap_rate: 0.7,
                max_plies: 120,
                ..Default::default()
            };
            if let Some(e) = ort.as_mut() {
                e.set_fixed_batch(fixed_batch.then_some(batch_size));
            }
            let (mut calls, mut positions, mut entries) = (0u64, 0u64, 0u64);
            let t = Instant::now();
            for i in 0..GAMES {
                let o = play_game(&cfg, i, u64::from(i), |b| {
                    calls += 1;
                    positions += b.len() as u64;
                    match ort.as_mut() {
                        Some(e) => e.evaluate_batch(b).expect("ort evaluate"),
                        None => synthetic_batch(b),
                    }
                });
                entries += o.trajectory.len() as u64;
            }
            let secs = t.elapsed().as_secs_f64();
            println!(
                "  batch {batch_size:>3}: {:>7.1} positions/sec  {:>8.1} NN evals/sec  \
                 {:>7.1} NN calls/sec  (mean batch {:.1}, {entries} positions in {secs:.1}s)",
                entries as f64 / secs,
                positions as f64 / secs,
                calls as f64 / secs,
                positions as f64 / calls as f64,
            );
        }
        println!();
    }

    /// Stand-in for the network: does the real encoding work, then returns a
    /// cheap deterministic value and uniform priors.
    fn synthetic_batch(games: &[Game]) -> Vec<LeafEval> {
        let mut buf = vec![0f32; crate::encoder::PLANE_SIZE];
        games
            .iter()
            .map(|g| {
                crate::encoder::encode_planes(g, &mut buf);
                let n = g.legal_moves().len();
                let v = f32::from(g.score_diff(g.turn) as i8) / 6.0;
                LeafEval {
                    value: v,
                    priors: Some(vec![1.0 / n.max(1) as f32; n]),
                }
            })
            .collect()
    }

    #[test]
    fn batching_reduces_evaluator_calls() {
        // The ply cap has to leave room for a real tree: within a few plies of
        // the cap every descent resolves terminally and needs no evaluation at
        // all, which would make this measure nothing.
        let cfg = SelfPlayConfig {
            sims_fast: 64,
            sims_full: 64,
            max_plies: 24,
            handicap_rate: 0.0,
            ..test_cfg()
        };
        let mut calls_1 = 0u32;
        let mut positions_1 = 0u32;
        play_game(&SelfPlayConfig { batch_size: 1, ..cfg.clone() }, 0, 2, |b| {
            calls_1 += 1;
            positions_1 += b.len() as u32;
            heuristic_batch(b)
        });
        let mut calls_32 = 0u32;
        let mut positions_32 = 0u32;
        play_game(&SelfPlayConfig { batch_size: 32, ..cfg.clone() }, 0, 2, |b| {
            calls_32 += 1;
            positions_32 += b.len() as u32;
            heuristic_batch(b)
        });
        assert_eq!(
            calls_1, positions_1,
            "batch_size 1 means one position per call"
        );
        assert!(
            calls_32 < calls_1,
            "batch 32 must need fewer calls: {calls_32} vs {calls_1}"
        );
        // The honest measure: positions per network call. Tree reuse shrinks
        // both totals (later searches inherit most of their budget), so the
        // ratio of call counts understates the batching win — this does not.
        let mean_batch = f64::from(positions_32) / f64::from(calls_32);
        assert!(
            mean_batch >= 4.0,
            "mean batch {mean_batch:.1} at batch_size 32; batching is not engaging"
        );
    }
}
