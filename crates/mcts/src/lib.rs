//! Monte Carlo Tree Search with PUCT selection and uniform-random playout
//! evaluation. Designed so the eventual neural-network swap (NN-supplied
//! priors + value) replaces only the leaf evaluator; the tree code stays
//! unchanged.
//!
//! # Conventions
//!
//! Each node stores `total_value` from its own `to_move`'s perspective. So
//! `Q(child) = child.total_value / child.n_visits` is the average return
//! the player at `child` would expect — which means *the parent player*
//! wants to maximise `-Q(child)` (their opponent's expected return is
//! their loss). Selection negates Q accordingly.
//!
//! Backprop walks leaf → root, alternating sign at each step because
//! `to_move` flips ply by ply.
//!
//! # PUCT
//!
//! ```text
//! score(child) = -Q(child) + c_puct * P(child) * sqrt(N(parent)) / (1 + N(child))
//! ```
//!
//! With uniform priors `P(child) = 1 / num_legal_children_at_parent`. The
//! `sqrt(N(parent))` term uses `max(N, 1)` to avoid the all-zero degenerate
//! score on the very first traversal after expansion.

use abalone_engine::{Game, GameState, Move, Side};
use rand::Rng;

type NodeId = u32;

#[derive(Clone, Debug)]
struct Node {
    state: Game,
    n_visits: u32,
    total_value: f32,
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
}

pub fn search<R: Rng + ?Sized>(
    game: &Game,
    cfg: &SearchConfig,
    rng: &mut R,
) -> Option<SearchResult> {
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
        children: Vec::new(),
        expanded: false,
    });
    let root: NodeId = 0;

    expand(&mut nodes, root);

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
            let v = rollout(leaf_state, rng);
            expand(&mut nodes, current);
            v
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

    // ----- Pick the most-visited move at the root -----
    let visits: Vec<(Move, u32)> = nodes[root as usize]
        .children
        .iter()
        .map(|&(mv, cid)| (mv, nodes[cid as usize].n_visits))
        .collect();
    let best = visits
        .iter()
        .max_by_key(|&&(_, n)| n)
        .map(|&(mv, _)| mv)
        .expect("non-empty children after expansion");
    Some(SearchResult { best, visits })
}

fn expand(nodes: &mut Vec<Node>, parent: NodeId) {
    let state = nodes[parent as usize].state;
    let moves = state.legal_moves();
    let mut child_ids: Vec<(Move, NodeId)> = Vec::with_capacity(moves.len());
    for &mv in moves.iter() {
        let mut child_state = state;
        child_state.apply(mv);
        let cid = nodes.len() as NodeId;
        nodes.push(Node {
            state: child_state,
            n_visits: 0,
            total_value: 0.0,
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
    let inv_legal = 1.0 / (p.children.len() as f32);
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
        let u = c_puct * inv_legal * sqrt_n_parent / (1.0 + c.n_visits as f32);
        let score = q_parent_pov + u;
        if score > best_score {
            best_score = score;
            best_idx = i;
        }
    }
    p.children[best_idx].1
}

fn rollout<R: Rng + ?Sized>(state: Game, rng: &mut R) -> f32 {
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
    use abalone_engine::game::MAX_PLIES;
    use rand::rngs::SmallRng;
    use rand::SeedableRng;

    #[test]
    fn terminal_returns_none() {
        let mut g = Game::new_standard();
        g.ply = MAX_PLIES; // forces Draw
        assert!(g.is_terminal());
        let mut rng = SmallRng::seed_from_u64(0);
        assert!(search(&g, &SearchConfig::default(), &mut rng).is_none());
    }

    #[test]
    fn search_returns_a_legal_move() {
        let g = Game::new_standard();
        let cfg = SearchConfig {
            simulations: 30,
            c_puct: 1.4,
        };
        let mut rng = SmallRng::seed_from_u64(7);
        let r = search(&g, &cfg, &mut rng).expect("non-terminal => some move");
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
        let r1 = search(&g, &cfg, &mut rng1).unwrap();
        let r2 = search(&g, &cfg, &mut rng2).unwrap();
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
            let r = search(&g, &cfg, &mut rng).unwrap();
            g.apply(r.best);
        }
        assert_eq!(g.ply, 6);
    }

    #[test]
    fn child_visits_sum_matches_simulations_minus_one() {
        // Each simulation traverses to a leaf and increments every node on
        // the path — including a child of the root. So the root's children's
        // visit counts must sum to exactly `simulations` (every iteration
        // descends through exactly one root-child).
        let g = Game::new_standard();
        let cfg = SearchConfig {
            simulations: 40,
            c_puct: 1.4,
        };
        let mut rng = SmallRng::seed_from_u64(99);
        let r = search(&g, &cfg, &mut rng).unwrap();
        let sum: u32 = r.visits.iter().map(|&(_, n)| n).sum();
        assert_eq!(sum, cfg.simulations);
    }
}
