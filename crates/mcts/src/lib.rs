//! Monte Carlo Tree Search with PUCT selection. The leaf evaluator is
//! pluggable: pass a closure `Fn(&Game, &mut R) -> LeafEval` that returns
//! a value in `[-1, 1]` from the leaf's `to_move` perspective, plus an
//! optional vector of priors (parallel to `game.legal_moves()` at the
//! leaf). Three batteries-included evaluators live alongside:
//!
//!   * [`random_rollout`] — simulate to terminal with uniform-random moves.
//!     Uniform priors.
//!   * [`heuristic`] — hand-crafted positional eval (see [`mod@eval`]).
//!     Uniform priors.
//!   * (NN evaluator: provided by callers via the closure path; the
//!     `selfplay` and `wasm` crates wire one up.)
//!
//! # Conventions
//!
//! Each node stores `total_value` from its own `to_move`'s perspective.
//! Selection at a parent computes `score = -Q(child) + U(child)` (negate
//! because child's POV is opposite of parent's). Backprop walks leaf →
//! root, alternating sign at each step because `to_move` flips ply by ply.
//!
//! # PUCT
//!
//! ```text
//! score(child) = -Q(child) + c_puct * P(child) * sqrt(N(parent)) / (1 + N(child))
//! ```
//!
//! Priors `P(child)` come from the leaf evaluator (or are uniform). The
//! `sqrt(N(parent))` term uses `max(N, 1)` to avoid the all-zero score on
//! the first traversal after expansion.

pub mod eval;

use abalone_game::{Game, GameState, Move, Side};
use rand::Rng;

type NodeId = u32;

/// What a leaf evaluator returns. Value is in `[-1, 1]` from the leaf's
/// to_move POV. Priors, if present, must be parallel to the move list
/// produced by `game.legal_moves()` at the leaf and sum to 1 (the
/// caller is responsible for normalisation; we don't re-softmax).
#[derive(Clone, Debug, Default)]
pub struct LeafEval {
    pub value: f32,
    pub priors: Option<Vec<f32>>,
}

impl LeafEval {
    pub fn from_value(value: f32) -> Self {
        Self {
            value,
            priors: None,
        }
    }
}

#[derive(Clone, Debug)]
struct Node {
    state: Game,
    n_visits: u32,
    total_value: f32,
    /// PUCT prior `P(this | parent)`. Set when the parent expands. The
    /// root has `prior = 0.0` (never used in selection).
    prior: f32,
    children: Vec<(Move, NodeId)>,
    expanded: bool,
}

#[derive(Clone, Debug)]
pub struct SearchConfig {
    pub simulations: u32,
    pub c_puct: f32,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            simulations: 800,
            c_puct: 1.4,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SearchResult {
    pub best: Move,
    /// `(move, visit_count)` for every legal child of the root.
    pub visits: Vec<(Move, u32)>,
    /// Q-value of each child from the root player's POV, parallel to
    /// `visits`. Sign convention: positive = good for the side to move at
    /// the root. Children with zero visits are reported as `0.0`.
    pub q_parent_pov: Vec<f32>,
}

/// Run MCTS from `game` for `cfg.simulations` iterations using `eval_fn`
/// to evaluate non-terminal leaves. Returns `None` if `game` is terminal.
pub fn search<R, F>(
    game: &Game,
    cfg: &SearchConfig,
    rng: &mut R,
    mut eval_fn: F,
) -> Option<SearchResult>
where
    R: Rng + ?Sized,
    F: FnMut(&Game, &mut R) -> LeafEval,
{
    if game.is_terminal() {
        return None;
    }
    let legal = game.legal_moves();
    if legal.is_empty() {
        return None;
    }

    let mut nodes: Vec<Node> = Vec::with_capacity((cfg.simulations as usize).saturating_mul(4));
    nodes.push(Node {
        state: *game,
        n_visits: 0,
        total_value: 0.0,
        prior: 0.0,
        children: Vec::new(),
        expanded: false,
    });
    let root: NodeId = 0;

    // Expand the root using `eval_fn`'s priors (if any). Root's value is
    // unused (we never query Q on the root for selection).
    let root_eval = eval_fn(game, rng);
    expand(&mut nodes, root, root_eval.priors.as_deref());

    for _ in 0..cfg.simulations {
        // ----- Selection -----
        let mut path: Vec<NodeId> = Vec::with_capacity(16);
        path.push(root);
        let mut current = root;
        loop {
            let n = &nodes[current as usize];
            if !n.expanded || n.state.is_terminal() {
                break;
            }
            let chosen = select_child(&nodes, current, cfg.c_puct);
            path.push(chosen);
            current = chosen;
        }

        // ----- Evaluation (+ optional expansion) -----
        let leaf_state = nodes[current as usize].state;
        let v = if leaf_state.is_terminal() {
            outcome_from_pov(leaf_state.turn, &leaf_state)
        } else {
            let leaf_eval = eval_fn(&leaf_state, rng);
            expand(&mut nodes, current, leaf_eval.priors.as_deref());
            leaf_eval.value
        };

        // ----- Backprop -----
        let mut sign = 1.0f32;
        for &nid in path.iter().rev() {
            let n = &mut nodes[nid as usize];
            n.n_visits += 1;
            n.total_value += v * sign;
            sign = -sign;
        }
    }

    let root_children = &nodes[root as usize].children;
    let visits: Vec<(Move, u32)> = root_children
        .iter()
        .map(|&(mv, cid)| (mv, nodes[cid as usize].n_visits))
        .collect();
    let q_parent_pov: Vec<f32> = root_children
        .iter()
        .map(|&(_, cid)| {
            let c = &nodes[cid as usize];
            if c.n_visits == 0 {
                0.0
            } else {
                // child stores from child's POV; root sees the negation.
                -(c.total_value / c.n_visits as f32)
            }
        })
        .collect();
    let best = visits
        .iter()
        .max_by_key(|&&(_, n)| n)
        .map(|&(mv, _)| mv)
        .expect("non-empty children after expansion");
    Some(SearchResult {
        best,
        visits,
        q_parent_pov,
    })
}

/// Random-rollout evaluator: simulate to terminal with uniform-random
/// moves and return the outcome from `game.turn`'s POV. Uniform priors.
pub fn random_rollout<R: Rng + ?Sized>(game: &Game, rng: &mut R) -> LeafEval {
    LeafEval::from_value(simulate_random(*game, rng))
}

/// Heuristic evaluator with the default weights. Uniform priors. For
/// custom weights pass a closure that calls [`eval::evaluate`] directly.
pub fn heuristic<R: Rng + ?Sized>(game: &Game, _rng: &mut R) -> LeafEval {
    LeafEval::from_value(eval::evaluate(
        &game.board,
        game.turn,
        &eval::Weights::default(),
    ))
}

fn expand(nodes: &mut Vec<Node>, parent: NodeId, priors: Option<&[f32]>) {
    let state = nodes[parent as usize].state;
    let moves = state.legal_moves();
    let n_children = moves.len();
    let uniform = 1.0 / (n_children as f32);
    let mut child_ids: Vec<(Move, NodeId)> = Vec::with_capacity(n_children);
    for (i, &mv) in moves.iter().enumerate() {
        let mut child_state = state;
        child_state.apply(mv);
        let cid = nodes.len() as NodeId;
        let prior = match priors {
            Some(p) if p.len() == n_children => p[i],
            _ => uniform,
        };
        nodes.push(Node {
            state: child_state,
            n_visits: 0,
            total_value: 0.0,
            prior,
            children: Vec::new(),
            expanded: false,
        });
        child_ids.push((mv, cid));
    }
    let p = &mut nodes[parent as usize];
    p.children = child_ids;
    p.expanded = true;
}

fn select_child(nodes: &[Node], parent: NodeId, c_puct: f32) -> NodeId {
    let p = &nodes[parent as usize];
    let n_parent = p.n_visits.max(1) as f32;
    let sqrt_n_parent = n_parent.sqrt();

    let mut best_idx = 0usize;
    let mut best_score = f32::NEG_INFINITY;
    for (i, &(_, cid)) in p.children.iter().enumerate() {
        let c = &nodes[cid as usize];
        let q_child_pov = if c.n_visits == 0 {
            0.0
        } else {
            c.total_value / c.n_visits as f32
        };
        let q_parent_pov = -q_child_pov;
        let u = c_puct * c.prior * sqrt_n_parent / (1.0 + c.n_visits as f32);
        let score = q_parent_pov + u;
        if score > best_score {
            best_score = score;
            best_idx = i;
        }
    }
    p.children[best_idx].1
}

fn simulate_random<R: Rng + ?Sized>(state: Game, rng: &mut R) -> f32 {
    let leaf_pov = state.turn;
    let mut g = state;
    while !g.is_terminal() {
        let moves = g.legal_moves();
        if moves.is_empty() {
            break;
        }
        let i = rng.gen_range(0..moves.len());
        g.apply(moves[i]);
    }
    outcome_from_pov(leaf_pov, &g)
}

fn outcome_from_pov(pov: Side, g: &Game) -> f32 {
    match g.state() {
        GameState::InProgress => 0.0,
        GameState::Wins(s) => {
            if s == pov {
                1.0
            } else {
                -1.0
            }
        }
        GameState::Draw => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use abalone_game::game::MAX_PLIES;
    use rand::rngs::SmallRng;
    use rand::SeedableRng;

    #[test]
    fn terminal_returns_none() {
        let mut g = Game::new_standard();
        g.ply = MAX_PLIES;
        assert!(g.is_terminal());
        let mut rng = SmallRng::seed_from_u64(0);
        assert!(search(&g, &SearchConfig::default(), &mut rng, random_rollout).is_none());
    }

    #[test]
    fn search_returns_a_legal_move() {
        let g = Game::new_standard();
        let cfg = SearchConfig {
            simulations: 30,
            c_puct: 1.4,
        };
        let mut rng = SmallRng::seed_from_u64(7);
        let r = search(&g, &cfg, &mut rng, random_rollout).unwrap();
        let legal: Vec<Move> = g.legal_moves().iter().copied().collect();
        assert!(legal.contains(&r.best));
        let total_visits: u32 = r.visits.iter().map(|&(_, n)| n).sum();
        assert_eq!(total_visits, cfg.simulations);
    }

    #[test]
    fn seeded_search_is_deterministic() {
        let g = Game::new_standard();
        let cfg = SearchConfig {
            simulations: 60,
            c_puct: 1.4,
        };
        let mut rng1 = SmallRng::seed_from_u64(42);
        let mut rng2 = SmallRng::seed_from_u64(42);
        let r1 = search(&g, &cfg, &mut rng1, random_rollout).unwrap();
        let r2 = search(&g, &cfg, &mut rng2, random_rollout).unwrap();
        assert_eq!(r1.best, r2.best);
        assert_eq!(r1.visits, r2.visits);
    }

    #[test]
    fn plays_several_moves_without_panicking() {
        let mut g = Game::new_standard();
        let cfg = SearchConfig {
            simulations: 25,
            c_puct: 1.4,
        };
        let mut rng = SmallRng::seed_from_u64(1);
        for _ in 0..6 {
            if g.is_terminal() {
                break;
            }
            let r = search(&g, &cfg, &mut rng, random_rollout).unwrap();
            g.apply(r.best);
        }
        assert_eq!(g.ply, 6);
    }

    #[test]
    fn root_visits_sum_to_simulations() {
        let g = Game::new_standard();
        let cfg = SearchConfig {
            simulations: 40,
            c_puct: 1.4,
        };
        let mut rng = SmallRng::seed_from_u64(99);
        let r = search(&g, &cfg, &mut rng, random_rollout).unwrap();
        let sum: u32 = r.visits.iter().map(|&(_, n)| n).sum();
        assert_eq!(sum, cfg.simulations);
    }

    #[test]
    fn heuristic_search_runs() {
        let g = Game::new_standard();
        let cfg = SearchConfig {
            simulations: 30,
            c_puct: 1.4,
        };
        let mut rng = SmallRng::seed_from_u64(0);
        let r = search(&g, &cfg, &mut rng, heuristic).unwrap();
        let legal: Vec<Move> = g.legal_moves().iter().copied().collect();
        assert!(legal.contains(&r.best));
    }

    #[test]
    fn nonuniform_priors_are_used() {
        // With one move strongly preferred by priors and zero-value at every
        // leaf, MCTS should visit the preferred move overwhelmingly more.
        let g = Game::new_standard();
        let n_legal = g.legal_moves().len();
        // priors that put 0.9 mass on move 0 and the rest uniform.
        let preferred = 0;
        let other_mass = (1.0 - 0.9) / (n_legal as f32 - 1.0);
        let priors: Vec<f32> = (0..n_legal)
            .map(|i| if i == preferred { 0.9 } else { other_mass })
            .collect();
        let cfg = SearchConfig {
            simulations: 200,
            c_puct: 1.4,
        };
        let mut rng = SmallRng::seed_from_u64(0);
        let r = search(&g, &cfg, &mut rng, |_g, _r| LeafEval {
            value: 0.0,
            priors: Some(priors.clone()),
        })
        .unwrap();
        let visits_preferred = r.visits[preferred].1;
        let visits_others_max = r
            .visits
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != preferred)
            .map(|(_, &(_, n))| n)
            .max()
            .unwrap();
        assert!(
            visits_preferred > visits_others_max,
            "preferred {} <= max other {}",
            visits_preferred,
            visits_others_max
        );
    }
}
