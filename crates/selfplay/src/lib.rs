//! Self-play game generation. Drives MCTS with a pluggable leaf
//! evaluator, produces (state, child_visits, z, q) trajectories, and
//! writes them to a parquet shard format.

pub mod encoder;
pub mod ort_eval;
pub mod shard;

use abalone_game::{encode, Game, GameState, Move};
use abalone_mcts::{search, LeafEval, SearchConfig};
use rand::Rng;
use rand_distr::{Distribution, Gamma};

/// Self-play hyperparameters.
#[derive(Clone, Debug)]
pub struct SelfPlayConfig {
    pub simulations: u32,
    pub c_puct: f32,
    pub temperature_plies: u32,
    pub temperature: f32,
    /// Dirichlet noise mixing parameters at the root of each search.
    /// Set `dirichlet_eps` to 0 to disable. Standard AlphaZero values
    /// for ~50-move branching: alpha = 0.3, eps = 0.25.
    pub dirichlet_alpha: f32,
    pub dirichlet_eps: f32,
}

impl Default for SelfPlayConfig {
    fn default() -> Self {
        Self {
            simulations: 200,
            c_puct: 1.4,
            temperature_plies: 50,
            temperature: 1.0,
            dirichlet_alpha: 0.3,
            dirichlet_eps: 0.25,
        }
    }
}

/// Mix `dirichlet_eps` of Dirichlet(`alpha`) noise into `priors` in place.
/// Standard AlphaZero exploration trick applied at the root only; lets
/// search visit moves the network underweighted but might be learnable.
pub fn mix_dirichlet<R: Rng + ?Sized>(priors: &mut [f32], alpha: f32, eps: f32, rng: &mut R) {
    if priors.is_empty() || eps <= 0.0 {
        return;
    }
    let gamma = Gamma::new(alpha as f64, 1.0).expect("alpha > 0");
    let samples: Vec<f32> = (0..priors.len())
        .map(|_| gamma.sample(rng) as f32)
        .collect();
    let total: f32 = samples.iter().sum();
    if total <= 0.0 {
        return;
    }
    for (p, g) in priors.iter_mut().zip(samples.iter()) {
        let dirichlet = g / total;
        *p = (1.0 - eps) * (*p) + eps * dirichlet;
    }
}

/// One position in a self-play trajectory: the state before a move, the
/// MCTS visit distribution at the root, the move that was actually
/// played (chosen by temperature sampling or argmax), and the
/// visit-weighted root Q-value (used as a value-target bootstrap during
/// training).
#[derive(Clone, Debug)]
pub struct TrajectoryEntry {
    pub state: Game,
    /// Parallel arrays: `(move_idx, visit_count)` for each legal child.
    /// `move_idx` is the flat 0..2562 index from `abalone_game::encode`.
    pub child_visits: Vec<(u16, u32)>,
    /// Move index actually played from this position.
    pub move_played: u16,
    /// Visit-weighted average Q at the root from this position's `to_move`
    /// POV. Densifies the value signal for training.
    pub q: f32,
}

#[derive(Clone, Debug)]
pub struct GameOutcome {
    pub trajectory: Vec<TrajectoryEntry>,
    pub final_state: Game,
}

impl GameOutcome {
    /// Final result from the perspective of `to_move` at the given entry.
    /// `+1` win, `-1` loss, `0` draw.
    pub fn z_for(&self, entry: &TrajectoryEntry) -> f32 {
        let pov = entry.state.turn;
        match self.final_state.state() {
            GameState::Wins(s) => {
                if s == pov {
                    1.0
                } else {
                    -1.0
                }
            }
            _ => 0.0,
        }
    }
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

/// Play one self-play game from the standard opening. Returns the full
/// trajectory and final state.
pub fn play_game<R, F>(
    cfg: &SelfPlayConfig,
    rng: &mut R,
    mut eval_fn: F,
) -> GameOutcome
where
    R: Rng + ?Sized,
    F: FnMut(&Game, &mut R) -> LeafEval,
{
    let mut g = Game::new_standard();
    let mut traj: Vec<TrajectoryEntry> = Vec::new();

    let search_cfg = SearchConfig {
        simulations: cfg.simulations,
        c_puct: cfg.c_puct,
    };

    while !g.is_terminal() {
        // Wrap the eval_fn so the FIRST call (root expansion) gets
        // Dirichlet noise mixed into its priors. Subsequent calls
        // (leaf expansions) are passed through unchanged.
        let alpha = cfg.dirichlet_alpha;
        let eps = cfg.dirichlet_eps;
        let mut first_call = true;
        let res = {
            let mut wrapped = |state: &Game, rng_inner: &mut R| -> LeafEval {
                let mut e = eval_fn(state, rng_inner);
                if first_call && eps > 0.0 {
                    first_call = false;
                    if let Some(p) = e.priors.as_mut() {
                        mix_dirichlet(p, alpha, eps, rng_inner);
                    }
                }
                e
            };
            search(&g, &search_cfg, rng, &mut wrapped)
                .expect("non-terminal => search returns Some")
        };

        let chosen_idx = if g.ply < cfg.temperature_plies {
            sample_by_visits(&res.visits, cfg.temperature, rng)
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
            child_visits: res
                .visits
                .iter()
                .map(|&(mv, v)| (encode(mv), v))
                .collect(),
            move_played: encode(chosen_mv),
            q,
        });

        g.apply(chosen_mv);
    }

    GameOutcome {
        trajectory: traj,
        final_state: g,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use abalone_mcts::heuristic;
    use rand::rngs::SmallRng;
    use rand::SeedableRng;

    #[test]
    fn play_game_produces_consistent_trajectory() {
        let cfg = SelfPlayConfig {
            simulations: 20,
            c_puct: 1.4,
            temperature_plies: 4,
            temperature: 1.0,
            dirichlet_alpha: 0.3,
            dirichlet_eps: 0.0,
        };
        let mut rng = SmallRng::seed_from_u64(0);
        let outcome = play_game(&cfg, &mut rng, heuristic);
        assert!(outcome.final_state.is_terminal());
        // Each entry's `state.ply` equals its index (we apply moves in order).
        for (i, entry) in outcome.trajectory.iter().enumerate() {
            assert_eq!(entry.state.ply as usize, i);
        }
        // Visit counts in each entry sum to `simulations`.
        for entry in &outcome.trajectory {
            let sum: u32 = entry.child_visits.iter().map(|(_, v)| v).sum();
            assert_eq!(sum, cfg.simulations);
        }
        // z is consistent: terminal → ±1 or 0, no NaN.
        for entry in &outcome.trajectory {
            let z = outcome.z_for(entry);
            assert!(z == 0.0 || z == 1.0 || z == -1.0);
        }
    }

    #[test]
    fn z_alternates_sign_along_trajectory() {
        // Within a single game, two successive entries are from opposite
        // POVs (turn alternates), so `z_for` returns opposite signs unless
        // the game is a draw.
        let cfg = SelfPlayConfig {
            simulations: 15,
            c_puct: 1.4,
            temperature_plies: 4,
            temperature: 1.0,
            dirichlet_alpha: 0.3,
            dirichlet_eps: 0.0,
        };
        let mut rng = SmallRng::seed_from_u64(1);
        let outcome = play_game(&cfg, &mut rng, heuristic);
        if let GameState::Wins(_) = outcome.final_state.state() {
            for w in outcome.trajectory.windows(2) {
                let z0 = outcome.z_for(&w[0]);
                let z1 = outcome.z_for(&w[1]);
                assert_eq!(z0, -z1, "consecutive z should alternate sign");
            }
        }
    }
}
