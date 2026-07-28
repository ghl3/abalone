//! Monte Carlo Tree Search with PUCT selection and **batched** leaf
//! evaluation.
//!
//! The leaf evaluator is pluggable. Two shapes are supported:
//!
//! * batched — `FnMut(&[Game]) -> Vec<LeafEval>`, one call per 16–64 leaves
//!   ([`search_batched`]). This is the shape neural-network inference wants:
//!   per-call overhead dominates, so amortising it over a batch is the
//!   dominant throughput lever (`docs/MODEL.md` §7.1).
//! * single-leaf — `FnMut(&Game, &mut R) -> LeafEval` ([`search`]), kept for
//!   the rollout/heuristic evaluators and any caller that has nothing to gain
//!   from batching. It is a thin wrapper over the batched engine.
//!
//! Both are wrappers over [`Search`], a *pull-based coroutine*: you drive it
//! with [`Search::next_batch`] / [`Search::submit`] and it never calls you
//! back. That matters for the browser, where `onnxruntime-web`'s `run()` is
//! async and WASM cannot await — the whole search (selection, virtual loss,
//! backup, node arena) stays in Rust while inference happens in the JS
//! runtime (`docs/ARCHITECTURE.md` §2.5).
//!
//! Batteries-included evaluators:
//!
//!   * [`random_rollout`] — simulate to terminal with uniform-random moves.
//!   * [`heuristic`] — hand-crafted positional eval (see [`mod@eval`]).
//!     **Retired from the training path**; benchmark opponent only.
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
//! the first traversal after expansion. Unvisited children take a
//! **first-play-urgency** value of `Q(parent) - fpu_reduction` rather than
//! `0`, which is optimistic when losing and pessimistic when winning.
//!
//! # Batching and virtual loss
//!
//! [`Search::next_batch`] descends repeatedly until it has collected
//! `cfg.batch_size` *distinct* non-terminal leaves. After each descent it
//! applies a virtual loss along the path — `n_visits += 1` and
//! `total_value += virtual_loss` at every node on it. The value term is
//! added in the node's *own* POV, i.e. as a virtual **win** for the player
//! to move there, which is exactly a virtual loss for the parent that chose
//! it (`score = -Q(child) + U`), so the next descent avoids the path.
//! [`Search::submit`] reverts the value term and turns the virtual visit into
//! the real one, so the tree ends up bit-identical to what a sequential
//! search would have produced *given the same descent order*. With
//! `batch_size = 1` the virtual loss is applied and reverted before any other
//! descent can see it, so the search is exactly the sequential one whatever
//! `virtual_loss` is set to.
//!
//! **Terminal leaves never enter a batch.** They are resolved from
//! `game.state()` and backed up immediately — they need no network. Under
//! score-based adjudication at the ply cap (`docs/MODEL.md` §3) a large
//! fraction of leaves are terminal, so this matters.
//!
//! **Duplicates are deduplicated by node.** A descent can land on a leaf that
//! is already pending in the current batch (it is unexpanded, so it is still
//! a leaf; virtual loss discourages this but does not forbid it). Such a
//! simulation does *not* add a second copy of the position to the batch: it
//! shares the batch slot, and on [`Search::submit`] the same evaluation is
//! backed up once per simulation, each with its own virtual-loss revert. The
//! batch therefore holds distinct positions while the simulation count is
//! still exact.
//!
//! # Node storage
//!
//! A flat arena. Children of a node occupy a contiguous id range
//! `child_start .. child_start + child_len` in one `Vec<Node>` — no per-node
//! `Vec`. Positions are never stored: a node knows only the move that reaches
//! it from its parent, and the leaf state is materialised lazily by applying
//! moves along the descent path (the parent's position is in hand anyway).
//! After the initial reserve the hot path allocates nothing.

pub mod eval;

use abalone_game::{Dir, Game, GameState, Move, Side};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Gamma};

type NodeId = u32;

/// Nodes reserved per simulation up front. One expansion per simulation, each
/// allocating one node per legal move; ~60–80 is typical for Abalone.
const NODES_PER_SIM: usize = 64;
/// Cap on the up-front reserve, so a huge `simulations` does not reserve a
/// gigabyte. Beyond this the arena grows by doubling.
const MAX_RESERVE_NODES: usize = 1 << 18;
/// Give up collecting a full batch after this many descents per
/// `batch_size`. Only bites when duplicates pile up (e.g. `virtual_loss = 0`
/// in a narrow tree); bounds the work of one `next_batch` call.
const MAX_DESCENT_FACTOR: usize = 4;

/// What a leaf evaluator returns. Value is in `[-1, 1]` from the leaf's
/// to_move POV. Priors, if present, must be parallel to the move list
/// produced by `game.legal_moves()` at the leaf and sum to 1 (the
/// caller is responsible for normalisation; we don't re-softmax).
///
/// The network's value head is 3-way (win/draw/loss); the ONNX consumer
/// collapses it to `E[value] = P(win) - P(loss)` before building a
/// `LeafEval`. Search never sees the 3-way form.
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

#[derive(Clone, Copy, Debug)]
struct Node {
    /// Move that reaches this node from its parent. Meaningless at the root.
    mv: Move,
    n_visits: u32,
    total_value: f32,
    /// PUCT prior `P(this | parent)`. Set when the parent expands. The
    /// root has `prior = 0.0` (never used in selection).
    prior: f32,
    /// Children occupy `child_start .. child_start + child_len` in the arena.
    child_start: u32,
    child_len: u32,
    expanded: bool,
}

impl Node {
    fn new(mv: Move, prior: f32) -> Self {
        Self {
            mv,
            n_visits: 0,
            total_value: 0.0,
            prior,
            child_start: 0,
            child_len: 0,
            expanded: false,
        }
    }

    /// Placeholder move for the root, which has no parent edge.
    fn root() -> Self {
        Self::new(
            Move::Inline {
                anchor: 0,
                dir: Dir::E,
                size: 1,
            },
            0.0,
        )
    }
}

#[derive(Clone, Debug)]
pub struct SearchConfig {
    /// Simulations (leaf visits) per search. Tree reuse counts toward it:
    /// after [`Search::reroot`] only the shortfall is run.
    pub simulations: u32,
    pub c_puct: f32,
    /// Distinct leaves collected per evaluator call. `1` reproduces the
    /// sequential search exactly.
    pub batch_size: usize,
    /// Value added (in the node's own POV) to every node on a pending path,
    /// reverted on `submit`. Only has an effect when `batch_size > 1`.
    pub virtual_loss: f32,
    /// First-play urgency: unvisited children are valued at
    /// `Q(parent) - fpu_reduction` instead of `0`.
    pub fpu_reduction: f32,
    /// Dirichlet concentration for root exploration noise. `α ≈ 10/branching`.
    pub dirichlet_alpha: f32,
    /// Weight of the Dirichlet noise at the root. `0` disables it (the
    /// default: only self-play wants noise).
    pub dirichlet_eps: f32,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            simulations: 800,
            c_puct: 1.4,
            batch_size: 1,
            virtual_loss: 1.0,
            fpu_reduction: 0.25,
            dirichlet_alpha: 0.2,
            dirichlet_eps: 0.0,
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

/// One in-flight simulation: the path it took and the batch slot holding the
/// position it is waiting on.
#[derive(Clone, Copy, Debug)]
struct PendingSim {
    path_start: u32,
    path_len: u32,
    slot: u32,
}

/// A search in progress, driven from the outside as a coroutine:
///
/// ```no_run
/// # use abalone_game::Game;
/// # use abalone_mcts::{LeafEval, Search, SearchConfig};
/// let mut s = Search::begin(&Game::new_standard(), &SearchConfig::default(), 0);
/// loop {
///     let batch = s.next_batch();
///     if batch.is_empty() {
///         break;
///     }
///     let evals: Vec<LeafEval> = batch.iter().map(|_| LeafEval::from_value(0.0)).collect();
///     s.submit(&evals);
/// }
/// let result = s.result();
/// ```
pub struct Search {
    cfg: SearchConfig,
    rng: SmallRng,
    root_state: Game,
    nodes: Vec<Node>,
    /// Flat storage for pending descent paths; `PendingSim` indexes into it.
    path_buf: Vec<NodeId>,
    sims: Vec<PendingSim>,
    /// Distinct positions awaiting evaluation, parallel to `batch_leaf`.
    batch: Vec<Game>,
    batch_leaf: Vec<NodeId>,
    noise_buf: Vec<f32>,
    sims_started: u32,
    sims_completed: u32,
    sims_target: u32,
    done: bool,
}

impl Search {
    /// Start a search from `game`. Nothing is evaluated yet: the first
    /// [`next_batch`](Self::next_batch) returns the root position alone (the
    /// root expansion needs its priors before any descent can happen), and
    /// that call does **not** consume a simulation.
    pub fn begin(game: &Game, cfg: &SearchConfig, seed: u64) -> Self {
        let batch_size = cfg.batch_size.max(1);
        let mut nodes = Vec::with_capacity(node_reserve(cfg.simulations));
        nodes.push(Node::root());
        let mut s = Self {
            cfg: cfg.clone(),
            rng: SmallRng::seed_from_u64(seed),
            root_state: *game,
            nodes,
            path_buf: Vec::with_capacity(batch_size * 32),
            sims: Vec::with_capacity(batch_size),
            batch: Vec::with_capacity(batch_size),
            batch_leaf: Vec::with_capacity(batch_size),
            noise_buf: Vec::new(),
            sims_started: 0,
            sims_completed: 0,
            sims_target: cfg.simulations,
            done: false,
        };
        if game.is_terminal() || game.legal_moves().is_empty() {
            s.done = true;
        }
        s
    }

    /// Positions needing evaluation, parallel to the `evals` slice that
    /// [`submit`](Self::submit) expects. Empty means the search is finished.
    ///
    /// Calling this twice without an intervening `submit` discards the
    /// un-submitted batch (reverting its virtual loss) and descends again.
    pub fn next_batch(&mut self) -> &[Game] {
        self.abandon_pending();
        self.batch.clear();
        self.batch_leaf.clear();
        if self.done {
            return &self.batch;
        }
        if !self.nodes[0].expanded {
            // Root expansion: a batch of one, not counted as a simulation, so
            // that root child visits sum to exactly `simulations`.
            self.batch.push(self.root_state);
            self.batch_leaf.push(0);
            return &self.batch;
        }
        let batch_size = self.cfg.batch_size.max(1);
        let max_descents = batch_size.saturating_mul(MAX_DESCENT_FACTOR);
        let mut descents = 0usize;
        while self.sims_started < self.sims_target
            && self.batch.len() < batch_size
        {
            // The descent cap bounds the cost of one call when duplicates
            // pile up, but never returns an empty batch while simulations
            // remain: `next_batch()` is empty if and only if the search is
            // done. Descents that resolve terminally need no evaluation, so
            // running many of them in one call costs nothing to the caller.
            if descents >= max_descents && !self.batch.is_empty() {
                break;
            }
            self.descend_one();
            descents += 1;
        }
        if self.batch.is_empty() {
            // Every remaining simulation resolved terminally.
            self.done = self.sims_completed >= self.sims_target;
        }
        &self.batch
    }

    /// Supply evaluations for the positions returned by the last
    /// [`next_batch`](Self::next_batch). Expands each leaf, reverts virtual
    /// loss, and backs the values up.
    ///
    /// # Panics
    /// If `evals.len()` differs from the last batch length.
    pub fn submit(&mut self, evals: &[LeafEval]) {
        assert_eq!(
            evals.len(),
            self.batch.len(),
            "submit(): expected {} evaluations, got {}",
            self.batch.len(),
            evals.len()
        );
        if self.batch.is_empty() {
            return;
        }
        if !self.nodes[0].expanded {
            let state = self.batch[0];
            self.expand(0, &state, evals[0].priors.as_deref());
            self.mix_root_dirichlet();
            self.batch.clear();
            self.batch_leaf.clear();
            if self.sims_completed >= self.sims_target {
                self.done = true;
            }
            return;
        }

        for (slot, e) in evals.iter().enumerate() {
            let leaf = self.batch_leaf[slot];
            let state = self.batch[slot];
            self.expand(leaf, &state, e.priors.as_deref());
        }
        for i in 0..self.sims.len() {
            let sim = self.sims[i];
            let v = evals[sim.slot as usize].value;
            self.backup_pending(
                sim.path_start as usize,
                sim.path_len as usize,
                v,
            );
            self.sims_completed += 1;
        }
        self.sims.clear();
        self.path_buf.clear();
        self.batch.clear();
        self.batch_leaf.clear();
        if self.sims_completed >= self.sims_target {
            self.done = true;
        }
    }

    pub fn is_done(&self) -> bool {
        self.done
    }

    /// `None` until the root has been expanded (i.e. before the first
    /// `submit`), and for a terminal or move-less root.
    pub fn result(&self) -> Option<SearchResult> {
        let root = self.nodes[0];
        if !root.expanded || root.child_len == 0 {
            return None;
        }
        let range = root.child_start as usize
            ..(root.child_start + root.child_len) as usize;
        let visits: Vec<(Move, u32)> = self.nodes[range.clone()]
            .iter()
            .map(|c| (c.mv, c.n_visits))
            .collect();
        let q_parent_pov: Vec<f32> = self.nodes[range]
            .iter()
            .map(|c| {
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
            .expect("non-empty children");
        Some(SearchResult {
            best,
            visits,
            q_parent_pov,
        })
    }

    /// Position at the root of the tree.
    pub fn root_state(&self) -> &Game {
        &self.root_state
    }

    /// Simulations backed up into the root so far (including any inherited
    /// through [`reroot`](Self::reroot)).
    pub fn root_visits(&self) -> u32 {
        self.nodes[0].n_visits
    }

    /// Nodes currently in the arena. Diagnostics only.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Re-root on `played`, keeping that child's subtree and discarding the
    /// rest. The reachable nodes are copied into a fresh arena — cheaper than
    /// re-searching by a wide margin.
    ///
    /// The budget is *total*: the retained visits count toward
    /// `cfg.simulations`, so the follow-up search runs only the shortfall and
    /// the recovered work is a saving rather than a bonus. Root Dirichlet
    /// noise (if enabled) is re-applied to the new root's priors.
    ///
    /// If `played` is not a child of the current root (root never expanded),
    /// the result is a fresh search on the resulting position.
    pub fn reroot(mut self, played: Move) -> Search {
        // Any un-submitted batch would otherwise leave virtual loss baked in.
        self.abandon_pending();

        let mut root_state = self.root_state;
        root_state.apply(played);

        let root = self.nodes[0];
        let old_root = (root.expanded)
            .then(|| {
                (root.child_start..root.child_start + root.child_len)
                    .find(|&cid| self.nodes[cid as usize].mv == played)
            })
            .flatten();

        let mut nodes: Vec<Node> =
            Vec::with_capacity(node_reserve(self.cfg.simulations));
        match old_root {
            Some(cid) => {
                let mut n = self.nodes[cid as usize];
                n.prior = 0.0;
                n.child_start = 0;
                n.child_len = 0;
                nodes.push(n);
                // BFS: siblings stay contiguous, so `(start, len)` ranges hold.
                let mut queue: Vec<(NodeId, NodeId)> = vec![(cid, 0)];
                let mut head = 0usize;
                while head < queue.len() {
                    let (old, new) = queue[head];
                    head += 1;
                    let o = self.nodes[old as usize];
                    if !o.expanded || o.child_len == 0 {
                        continue;
                    }
                    let start = nodes.len() as u32;
                    for k in 0..o.child_len {
                        let mut c = self.nodes[(o.child_start + k) as usize];
                        c.child_start = 0;
                        c.child_len = 0;
                        nodes.push(c);
                        if self.nodes[(o.child_start + k) as usize].expanded {
                            queue.push((o.child_start + k, start + k));
                        }
                    }
                    nodes[new as usize].child_start = start;
                    nodes[new as usize].child_len = o.child_len;
                }
            }
            None => nodes.push(Node::root()),
        }

        let retained = nodes[0].n_visits;
        let batch_size = self.cfg.batch_size.max(1);
        let mut s = Search {
            cfg: self.cfg,
            rng: self.rng,
            root_state,
            nodes,
            path_buf: self.path_buf,
            sims: self.sims,
            batch: self.batch,
            batch_leaf: self.batch_leaf,
            noise_buf: self.noise_buf,
            sims_started: 0,
            sims_completed: 0,
            sims_target: 0,
            done: false,
        };
        s.path_buf.clear();
        s.sims.clear();
        s.batch.clear();
        s.batch_leaf.clear();
        s.path_buf.reserve(batch_size * 32);
        s.sims_target = s.cfg.simulations.saturating_sub(retained);
        if root_state.is_terminal()
            || root_state.legal_moves().is_empty()
            || s.sims_target == 0
        {
            s.done = true;
        }
        if s.nodes[0].expanded {
            s.mix_root_dirichlet();
        }
        s
    }

    // ----- internals -----

    /// Descend from the root with PUCT until an unexpanded (or terminal)
    /// node. Terminal leaves are resolved and backed up here; others join the
    /// pending batch under virtual loss.
    fn descend_one(&mut self) {
        let path_start = self.path_buf.len();
        let mut cur: NodeId = 0;
        let mut state = self.root_state;
        self.path_buf.push(cur);
        loop {
            let node = self.nodes[cur as usize];
            if !node.expanded || node.child_len == 0 {
                break;
            }
            let child = self.select_child(cur);
            state.apply(self.nodes[child as usize].mv);
            self.path_buf.push(child);
            cur = child;
        }
        self.sims_started += 1;
        let path_len = self.path_buf.len() - path_start;

        if state.is_terminal() {
            // No network needed: the outcome is known exactly.
            let v = outcome_from_pov(state.turn, &state);
            self.backup(path_start, path_len, v);
            self.path_buf.truncate(path_start);
            self.sims_completed += 1;
            return;
        }

        let slot = match self.batch_leaf.iter().position(|&l| l == cur) {
            Some(s) => s,
            None => {
                self.batch.push(state);
                self.batch_leaf.push(cur);
                self.batch.len() - 1
            }
        };
        self.apply_virtual_loss(path_start, path_len);
        self.sims.push(PendingSim {
            path_start: path_start as u32,
            path_len: path_len as u32,
            slot: slot as u32,
        });
    }

    fn select_child(&self, parent: NodeId) -> NodeId {
        let p = self.nodes[parent as usize];
        let sqrt_n_parent = (p.n_visits.max(1) as f32).sqrt();
        // First-play urgency: unvisited children inherit the parent's own
        // value, reduced. `Q = 0` is optimistic when losing, pessimistic when
        // winning.
        let parent_q = if p.n_visits == 0 {
            0.0
        } else {
            p.total_value / p.n_visits as f32
        };
        let fpu = parent_q - self.cfg.fpu_reduction;

        let mut best = p.child_start;
        let mut best_score = f32::NEG_INFINITY;
        for cid in p.child_start..p.child_start + p.child_len {
            let c = self.nodes[cid as usize];
            let q_parent_pov = if c.n_visits == 0 {
                fpu
            } else {
                -(c.total_value / c.n_visits as f32)
            };
            let u = self.cfg.c_puct * c.prior * sqrt_n_parent
                / (1.0 + c.n_visits as f32);
            let score = q_parent_pov + u;
            if score > best_score {
                best_score = score;
                best = cid;
            }
        }
        best
    }

    fn expand(&mut self, node: NodeId, state: &Game, priors: Option<&[f32]>) {
        let moves = state.legal_moves();
        let n = moves.len();
        let start = self.nodes.len() as u32;
        let uniform = if n == 0 { 0.0 } else { 1.0 / n as f32 };
        // Priors of the wrong length are ignored rather than trusted.
        let priors = priors.filter(|p| p.len() == n);
        for (i, &mv) in moves.iter().enumerate() {
            let prior = priors.map_or(uniform, |p| p[i]);
            self.nodes.push(Node::new(mv, prior));
        }
        let p = &mut self.nodes[node as usize];
        p.child_start = start;
        p.child_len = n as u32;
        p.expanded = true;
    }

    /// Mix `Dir(alpha)` noise into the root's priors. No-op when
    /// `dirichlet_eps == 0`, which is also the only case in which the
    /// internal RNG is left untouched.
    fn mix_root_dirichlet(&mut self) {
        let eps = self.cfg.dirichlet_eps;
        let root = self.nodes[0];
        let (start, len) = (root.child_start as usize, root.child_len as usize);
        if eps <= 0.0 || len == 0 {
            return;
        }
        let gamma = match Gamma::new(
            f64::from(self.cfg.dirichlet_alpha.max(1e-4)),
            1.0,
        ) {
            Ok(g) => g,
            Err(_) => return,
        };
        self.noise_buf.clear();
        let mut sum = 0.0f32;
        for _ in 0..len {
            let x = gamma.sample(&mut self.rng) as f32;
            self.noise_buf.push(x);
            sum += x;
        }
        if sum <= 0.0 || !sum.is_finite() {
            return;
        }
        for k in 0..len {
            let noise = self.noise_buf[k] / sum;
            let n = &mut self.nodes[start + k];
            n.prior = (1.0 - eps) * n.prior + eps * noise;
        }
    }

    /// `n_visits += 1`, `total_value += virtual_loss` in each node's own POV —
    /// a virtual win for the mover there, hence a virtual loss for the parent
    /// that selected it.
    fn apply_virtual_loss(&mut self, start: usize, len: usize) {
        let vl = self.cfg.virtual_loss;
        for i in start..start + len {
            let nid = self.path_buf[i] as usize;
            let n = &mut self.nodes[nid];
            n.n_visits += 1;
            n.total_value += vl;
        }
    }

    fn backup(&mut self, start: usize, len: usize, v: f32) {
        let mut sign = 1.0f32;
        for i in (start..start + len).rev() {
            let nid = self.path_buf[i] as usize;
            let n = &mut self.nodes[nid];
            n.n_visits += 1;
            n.total_value += v * sign;
            sign = -sign;
        }
    }

    /// Revert the virtual loss and back the real value up in one pass. The
    /// virtual visit *becomes* the real visit, so `n_visits` is untouched and
    /// only the value term is corrected: `+v*sign - virtual_loss`.
    fn backup_pending(&mut self, start: usize, len: usize, v: f32) {
        let vl = self.cfg.virtual_loss;
        let mut sign = 1.0f32;
        for i in (start..start + len).rev() {
            let nid = self.path_buf[i] as usize;
            let n = &mut self.nodes[nid];
            n.total_value += v * sign - vl;
            sign = -sign;
        }
    }

    /// Undo an un-submitted batch: revert its virtual loss and un-count the
    /// simulations it started.
    fn abandon_pending(&mut self) {
        if self.sims.is_empty() {
            self.path_buf.clear();
            return;
        }
        let vl = self.cfg.virtual_loss;
        for i in 0..self.sims.len() {
            let sim = self.sims[i];
            for k in sim.path_start..sim.path_start + sim.path_len {
                let nid = self.path_buf[k as usize] as usize;
                let n = &mut self.nodes[nid];
                n.n_visits -= 1;
                n.total_value -= vl;
            }
        }
        self.sims_started -= self.sims.len() as u32;
        self.sims.clear();
        self.path_buf.clear();
    }
}

fn node_reserve(simulations: u32) -> usize {
    (simulations as usize)
        .saturating_mul(NODES_PER_SIM)
        .clamp(64, MAX_RESERVE_NODES)
}

/// Run MCTS from `game` with a **batched** evaluator: `evaluator` is called
/// once per batch of up to `cfg.batch_size` distinct leaf positions and must
/// return one [`LeafEval`] per position, in order. Returns `None` if `game`
/// is terminal.
pub fn search_batched<F>(
    game: &Game,
    cfg: &SearchConfig,
    seed: u64,
    mut evaluator: F,
) -> Option<SearchResult>
where
    F: FnMut(&[Game]) -> Vec<LeafEval>,
{
    let mut s = Search::begin(game, cfg, seed);
    loop {
        let batch = s.next_batch();
        if batch.is_empty() {
            break;
        }
        let evals = evaluator(batch);
        s.submit(&evals);
    }
    s.result()
}

/// Run MCTS from `game` for `cfg.simulations` iterations using `eval_fn`
/// to evaluate non-terminal leaves. Returns `None` if `game` is terminal.
///
/// This is [`search_batched`] with the evaluator called once per position;
/// with the default `batch_size = 1` it is exactly the sequential search.
/// `rng` is passed through to `eval_fn` (and one `u64` is drawn from it to
/// seed the search's own RNG, which only Dirichlet noise uses).
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
    let seed: u64 = rng.gen();
    let mut s = Search::begin(game, cfg, seed);
    let mut evals: Vec<LeafEval> = Vec::with_capacity(cfg.batch_size.max(1));
    loop {
        let batch = s.next_batch();
        if batch.is_empty() {
            break;
        }
        evals.clear();
        for state in batch {
            evals.push(eval_fn(state, rng));
        }
        s.submit(&evals);
    }
    s.result()
}

/// Random-rollout evaluator: simulate to terminal with uniform-random
/// moves and return the outcome from `game.turn`'s POV. Uniform priors.
pub fn random_rollout<R: Rng + ?Sized>(game: &Game, rng: &mut R) -> LeafEval {
    LeafEval::from_value(simulate_random(*game, rng))
}

/// Heuristic evaluator with the default weights. Uniform priors. For
/// custom weights pass a closure that calls [`eval::evaluate`] directly.
///
/// Benchmark opponent and test fixture only — see [`mod@eval`].
pub fn heuristic<R: Rng + ?Sized>(game: &Game, _rng: &mut R) -> LeafEval {
    LeafEval::from_value(eval::evaluate(
        &game.board,
        game.turn,
        &eval::Weights::default(),
    ))
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
    use abalone_game::game::{DEFAULT_MAX_PLIES, NO_PROGRESS_DISABLED};
    use abalone_game::Opening;
    use rand::rngs::SmallRng;
    use rand::SeedableRng;
    use std::cell::Cell as StdCell;
    use std::collections::HashSet;

    fn cfg(simulations: u32) -> SearchConfig {
        SearchConfig {
            simulations,
            ..Default::default()
        }
    }

    /// Deterministic stand-in for a network: value and priors are a pure
    /// function of the position, so any two search implementations fed the
    /// same positions see the same numbers.
    fn fake_net(g: &Game) -> LeafEval {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        let mut mix = |x: u64| {
            h ^= x;
            h = h.wrapping_mul(0x1000_0000_01b3);
            h ^= h >> 29;
        };
        mix(g.board.marbles[0] as u64);
        mix((g.board.marbles[0] >> 64) as u64);
        mix(g.board.marbles[1] as u64);
        mix((g.board.marbles[1] >> 64) as u64);
        mix(u64::from(g.ply));
        let value = ((h >> 40) as f32 / 16_777_216.0) * 2.0 - 1.0;

        let n = g.legal_moves().len();
        let mut priors: Vec<f32> = Vec::with_capacity(n);
        let mut sum = 0.0f32;
        for i in 0..n {
            let mut hh = h ^ ((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            hh ^= hh >> 31;
            let p = 0.1 + (hh >> 40) as f32 / 16_777_216.0;
            priors.push(p);
            sum += p;
        }
        for p in priors.iter_mut() {
            *p /= sum;
        }
        LeafEval {
            value,
            priors: Some(priors),
        }
    }

    fn zero_value_net(g: &Game) -> LeafEval {
        LeafEval {
            value: 0.0,
            priors: fake_net(g).priors,
        }
    }

    /// Straightforward sequential MCTS implementing exactly the rules the
    /// arena engine implements (PUCT + parent-Q FPU, no batching, no virtual
    /// loss, no noise). The reference the batched engine is checked against.
    fn reference_search<F: Fn(&Game) -> LeafEval>(
        game: &Game,
        cfg: &SearchConfig,
        eval: F,
    ) -> Option<SearchResult> {
        struct RNode {
            state: Game,
            n: u32,
            w: f32,
            prior: f32,
            children: Vec<(Move, usize)>,
            expanded: bool,
        }
        fn expand<F: Fn(&Game) -> LeafEval>(
            nodes: &mut Vec<RNode>,
            id: usize,
            eval: &F,
        ) -> f32 {
            let state = nodes[id].state;
            let e = eval(&state);
            let moves = state.legal_moves();
            let n = moves.len();
            let uniform = 1.0 / n as f32;
            let priors = e.priors.as_deref().filter(|p| p.len() == n);
            let mut kids = Vec::with_capacity(n);
            for (i, &mv) in moves.iter().enumerate() {
                let mut cs = state;
                cs.apply(mv);
                let cid = nodes.len();
                nodes.push(RNode {
                    state: cs,
                    n: 0,
                    w: 0.0,
                    prior: priors.map_or(uniform, |p| p[i]),
                    children: Vec::new(),
                    expanded: false,
                });
                kids.push((mv, cid));
            }
            nodes[id].children = kids;
            nodes[id].expanded = true;
            e.value
        }

        if game.is_terminal() || game.legal_moves().is_empty() {
            return None;
        }
        let mut nodes = vec![RNode {
            state: *game,
            n: 0,
            w: 0.0,
            prior: 0.0,
            children: Vec::new(),
            expanded: false,
        }];
        expand(&mut nodes, 0, &eval);

        for _ in 0..cfg.simulations {
            let mut path = vec![0usize];
            let mut cur = 0usize;
            while nodes[cur].expanded && !nodes[cur].children.is_empty() {
                let p = &nodes[cur];
                let sqrt_n = (p.n.max(1) as f32).sqrt();
                let parent_q = if p.n == 0 { 0.0 } else { p.w / p.n as f32 };
                let fpu = parent_q - cfg.fpu_reduction;
                let mut best = p.children[0].1;
                let mut best_score = f32::NEG_INFINITY;
                for &(_, cid) in &p.children {
                    let c = &nodes[cid];
                    let q = if c.n == 0 { fpu } else { -(c.w / c.n as f32) };
                    let u = cfg.c_puct * c.prior * sqrt_n / (1.0 + c.n as f32);
                    let s = q + u;
                    if s > best_score {
                        best_score = s;
                        best = cid;
                    }
                }
                path.push(best);
                cur = best;
            }
            let state = nodes[cur].state;
            let v = if state.is_terminal() {
                outcome_from_pov(state.turn, &state)
            } else {
                expand(&mut nodes, cur, &eval)
            };
            let mut sign = 1.0f32;
            for &nid in path.iter().rev() {
                nodes[nid].n += 1;
                nodes[nid].w += v * sign;
                sign = -sign;
            }
        }

        let visits: Vec<(Move, u32)> = nodes[0]
            .children
            .iter()
            .map(|&(mv, cid)| (mv, nodes[cid].n))
            .collect();
        let q_parent_pov: Vec<f32> = nodes[0]
            .children
            .iter()
            .map(|&(_, cid)| {
                let c = &nodes[cid];
                if c.n == 0 {
                    0.0
                } else {
                    -(c.w / c.n as f32)
                }
            })
            .collect();
        let best = visits.iter().max_by_key(|&&(_, n)| n).map(|&(m, _)| m)?;
        Some(SearchResult {
            best,
            visits,
            q_parent_pov,
        })
    }

    /// The pre-batching implementation, verbatim: `Q = 0` for unvisited
    /// children, one eval per simulation.
    fn legacy_search<F: Fn(&Game) -> LeafEval>(
        game: &Game,
        cfg: &SearchConfig,
        eval: F,
    ) -> Vec<(Move, u32)> {
        struct RNode {
            state: Game,
            n: u32,
            w: f32,
            prior: f32,
            children: Vec<(Move, usize)>,
            expanded: bool,
        }
        fn expand<F: Fn(&Game) -> LeafEval>(
            nodes: &mut Vec<RNode>,
            id: usize,
            eval: &F,
        ) -> f32 {
            let state = nodes[id].state;
            let e = eval(&state);
            let moves = state.legal_moves();
            let n = moves.len();
            let uniform = 1.0 / n as f32;
            let priors = e.priors.as_deref().filter(|p| p.len() == n);
            let mut kids = Vec::with_capacity(n);
            for (i, &mv) in moves.iter().enumerate() {
                let mut cs = state;
                cs.apply(mv);
                let cid = nodes.len();
                nodes.push(RNode {
                    state: cs,
                    n: 0,
                    w: 0.0,
                    prior: priors.map_or(uniform, |p| p[i]),
                    children: Vec::new(),
                    expanded: false,
                });
                kids.push((mv, cid));
            }
            nodes[id].children = kids;
            nodes[id].expanded = true;
            e.value
        }
        let mut nodes = vec![RNode {
            state: *game,
            n: 0,
            w: 0.0,
            prior: 0.0,
            children: Vec::new(),
            expanded: false,
        }];
        expand(&mut nodes, 0, &eval);
        for _ in 0..cfg.simulations {
            let mut path = vec![0usize];
            let mut cur = 0usize;
            while nodes[cur].expanded && !nodes[cur].state.is_terminal() {
                let p = &nodes[cur];
                let sqrt_n = (p.n.max(1) as f32).sqrt();
                let mut best = p.children[0].1;
                let mut best_score = f32::NEG_INFINITY;
                for &(_, cid) in &p.children {
                    let c = &nodes[cid];
                    let q = if c.n == 0 { 0.0 } else { -(c.w / c.n as f32) };
                    let u = cfg.c_puct * c.prior * sqrt_n / (1.0 + c.n as f32);
                    let s = q + u;
                    if s > best_score {
                        best_score = s;
                        best = cid;
                    }
                }
                path.push(best);
                cur = best;
            }
            let state = nodes[cur].state;
            let v = if state.is_terminal() {
                outcome_from_pov(state.turn, &state)
            } else {
                expand(&mut nodes, cur, &eval)
            };
            let mut sign = 1.0f32;
            for &nid in path.iter().rev() {
                nodes[nid].n += 1;
                nodes[nid].w += v * sign;
                sign = -sign;
            }
        }
        nodes[0]
            .children
            .iter()
            .map(|&(mv, cid)| (mv, nodes[cid].n))
            .collect()
    }

    fn batched<F: Fn(&Game) -> LeafEval>(
        game: &Game,
        cfg: &SearchConfig,
        eval: F,
    ) -> Option<SearchResult> {
        search_batched(game, cfg, 0, |batch| batch.iter().map(&eval).collect())
    }

    // ---------- basic invariants ----------

    #[test]
    fn terminal_returns_none() {
        let mut g = Game::new_standard();
        g.ply = DEFAULT_MAX_PLIES;
        // 0-0 captures at the cap adjudicates to a Draw, hence terminal.
        assert!(g.is_terminal());
        let mut rng = SmallRng::seed_from_u64(0);
        assert!(search(&g, &cfg(800), &mut rng, random_rollout).is_none());
        assert!(batched(&g, &cfg(800), fake_net).is_none());
    }

    #[test]
    fn search_returns_a_legal_move() {
        let g = Game::new_standard();
        let c = cfg(30);
        let mut rng = SmallRng::seed_from_u64(7);
        let r = search(&g, &c, &mut rng, random_rollout).unwrap();
        let legal: Vec<Move> = g.legal_moves().iter().copied().collect();
        assert!(legal.contains(&r.best));
        let total: u32 = r.visits.iter().map(|&(_, n)| n).sum();
        assert_eq!(total, c.simulations);
    }

    #[test]
    fn root_visits_sum_to_simulations() {
        let g = Game::new_standard();
        let c = cfg(40);
        let mut rng = SmallRng::seed_from_u64(99);
        let r = search(&g, &c, &mut rng, random_rollout).unwrap();
        let sum: u32 = r.visits.iter().map(|&(_, n)| n).sum();
        assert_eq!(sum, c.simulations);
    }

    #[test]
    fn root_visits_sum_to_simulations_batched() {
        let g = Game::new_standard();
        for &bs in &[1usize, 2, 8, 32, 64, 128] {
            let c = SearchConfig {
                simulations: 200,
                batch_size: bs,
                ..Default::default()
            };
            let r = batched(&g, &c, fake_net).unwrap();
            let sum: u32 = r.visits.iter().map(|&(_, n)| n).sum();
            assert_eq!(sum, c.simulations, "batch_size={bs}");
            // Root itself is visited once per simulation too.
            let s = {
                let mut s = Search::begin(&g, &c, 0);
                while !s.next_batch().is_empty() {
                    let evals: Vec<LeafEval> =
                        s.batch.iter().map(fake_net).collect();
                    s.submit(&evals);
                }
                s
            };
            assert_eq!(s.root_visits(), c.simulations, "batch_size={bs}");
        }
    }

    #[test]
    fn heuristic_search_runs() {
        let g = Game::new_standard();
        let mut rng = SmallRng::seed_from_u64(0);
        let r = search(&g, &cfg(30), &mut rng, heuristic).unwrap();
        let legal: Vec<Move> = g.legal_moves().iter().copied().collect();
        assert!(legal.contains(&r.best));
    }

    #[test]
    fn nonuniform_priors_are_used() {
        let g = Game::new_standard();
        let n_legal = g.legal_moves().len();
        let preferred = 0;
        let other_mass = (1.0 - 0.9) / (n_legal as f32 - 1.0);
        let priors: Vec<f32> = (0..n_legal)
            .map(|i| if i == preferred { 0.9 } else { other_mass })
            .collect();
        let c = cfg(200);
        let mut rng = SmallRng::seed_from_u64(0);
        let r = search(&g, &c, &mut rng, |_g, _r| LeafEval {
            value: 0.0,
            priors: Some(priors.clone()),
        })
        .unwrap();
        let best_other = r
            .visits
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != preferred)
            .map(|(_, &(_, n))| n)
            .max()
            .unwrap();
        assert!(r.visits[preferred].1 > best_other);
    }

    // ---------- determinism ----------

    #[test]
    fn seeded_search_is_deterministic() {
        let g = Game::new_standard();
        let c = cfg(60);
        let mut rng1 = SmallRng::seed_from_u64(42);
        let mut rng2 = SmallRng::seed_from_u64(42);
        let r1 = search(&g, &c, &mut rng1, random_rollout).unwrap();
        let r2 = search(&g, &c, &mut rng2, random_rollout).unwrap();
        assert_eq!(r1.best, r2.best);
        assert_eq!(r1.visits, r2.visits);
    }

    #[test]
    fn batched_search_is_deterministic_for_a_fixed_seed() {
        let g = Game::new_standard();
        let c = SearchConfig {
            simulations: 150,
            batch_size: 16,
            dirichlet_eps: 0.25,
            ..Default::default()
        };
        let run = |seed| {
            search_batched(&g, &c, seed, |b| b.iter().map(fake_net).collect())
                .unwrap()
                .visits
        };
        assert_eq!(run(7), run(7));
        assert_ne!(run(7), run(8), "different seeds must diverge via noise");
    }

    // ---------- batching equivalence ----------

    #[test]
    fn batch_one_matches_the_sequential_reference() {
        let g = Game::new_standard();
        for &fpu in &[0.0f32, 0.25, 0.5] {
            let c = SearchConfig {
                simulations: 120,
                fpu_reduction: fpu,
                batch_size: 1,
                ..Default::default()
            };
            let a = batched(&g, &c, fake_net).unwrap();
            let b = reference_search(&g, &c, fake_net).unwrap();
            assert_eq!(a.visits, b.visits, "fpu={fpu}");
            assert_eq!(a.best, b.best);
            for (x, y) in a.q_parent_pov.iter().zip(b.q_parent_pov.iter()) {
                assert!((x - y).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn batch_one_reproduces_the_pre_batching_engine() {
        // With `fpu_reduction = 0` and an evaluator whose value is always 0,
        // parent-Q FPU degenerates to the old `Q = 0` rule, so the new engine
        // must reproduce the old one move for move.
        let g = Game::new_standard();
        let c = SearchConfig {
            simulations: 150,
            fpu_reduction: 0.0,
            batch_size: 1,
            ..Default::default()
        };
        let new = batched(&g, &c, zero_value_net).unwrap().visits;
        let old = legacy_search(&g, &c, zero_value_net);
        assert_eq!(new, old);
    }

    #[test]
    fn virtual_loss_is_fully_reverted() {
        // At batch_size 1 the virtual loss is applied and reverted before any
        // other descent observes it, so the tree must be identical to a
        // search with no virtual loss at all — and to the reference.
        let g = Game::new_standard();
        let mut base = None;
        for &vl in &[0.0f32, 1.0, 3.0] {
            let c = SearchConfig {
                simulations: 120,
                virtual_loss: vl,
                batch_size: 1,
                ..Default::default()
            };
            let r = batched(&g, &c, fake_net).unwrap();
            let reference = reference_search(&g, &c, fake_net).unwrap();
            assert_eq!(r.visits, reference.visits, "vl={vl}");
            for (x, y) in
                r.q_parent_pov.iter().zip(reference.q_parent_pov.iter())
            {
                assert!((x - y).abs() < 1e-6, "vl={vl}: {x} vs {y}");
            }
            match &base {
                None => base = Some(r.visits.clone()),
                Some(b) => assert_eq!(&r.visits, b, "vl={vl}"),
            }
        }
    }

    #[test]
    fn virtual_loss_leaves_no_residue_in_totals() {
        // Deeper check on the arena itself: after a batched search finishes,
        // every node's total_value must be the plain sum of backed-up values,
        // which we verify by re-deriving the root's Q from its children.
        let g = Game::new_standard();
        let c = SearchConfig {
            simulations: 200,
            batch_size: 16,
            virtual_loss: 2.0,
            ..Default::default()
        };
        let mut s = Search::begin(&g, &c, 3);
        while !s.next_batch().is_empty() {
            let evals: Vec<LeafEval> = s.batch.iter().map(fake_net).collect();
            s.submit(&evals);
        }
        let root = s.nodes[0];
        assert_eq!(root.n_visits, c.simulations);
        // Root value = -(sum of child values) + (values of simulations that
        // stopped at the root, of which there are none after expansion).
        let child_sum: f32 = (root.child_start
            ..root.child_start + root.child_len)
            .map(|cid| s.nodes[cid as usize].total_value)
            .sum();
        assert!(
            (root.total_value + child_sum).abs() < 1e-2,
            "root {} vs children {}",
            root.total_value,
            child_sum
        );
        // Visit accounting: every simulation that reaches an expanded node
        // continues into exactly one child, so a node's visits are its
        // children's plus the ones that stopped there while it was still a
        // leaf (exactly zero for the root, which is expanded before any
        // simulation runs; at least one for everything else).
        for id in 0..s.nodes.len() {
            let n = s.nodes[id];
            if !n.expanded || n.child_len == 0 {
                continue;
            }
            let kids: u32 = (n.child_start..n.child_start + n.child_len)
                .map(|cid| s.nodes[cid as usize].n_visits)
                .sum();
            if id == 0 {
                assert_eq!(kids, n.n_visits, "root");
            } else {
                assert!(
                    kids < n.n_visits,
                    "node {id}: {kids} vs {}",
                    n.n_visits
                );
            }
        }
    }

    #[test]
    fn virtual_loss_diversifies_a_batch() {
        let g = Game::new_standard();
        let c = SearchConfig {
            simulations: 256,
            batch_size: 32,
            virtual_loss: 1.0,
            ..Default::default()
        };
        let mut s = Search::begin(&g, &c, 0);
        // First batch is the root expansion.
        assert_eq!(s.next_batch().len(), 1);
        let e: Vec<LeafEval> = s.batch.iter().map(fake_net).collect();
        s.submit(&e);
        // Second batch must contain 32 distinct leaves.
        let batch: Vec<Game> = s.next_batch().to_vec();
        assert_eq!(batch.len(), 32);
        let distinct: HashSet<_> =
            batch.iter().map(|g| (g.board.marbles, g.ply)).collect();
        assert_eq!(distinct.len(), 32, "positions in a batch must be distinct");
        assert_eq!(s.sims.len(), 32, "and one simulation each, no duplicates");
    }

    #[test]
    fn duplicate_leaves_are_shared_not_duplicated() {
        // With virtual_loss = 0 nothing discourages re-descending the same
        // path, so descents collapse onto one leaf. The documented behaviour:
        // the batch holds that position once, several simulations share the
        // slot, and the visit accounting still comes out exact.
        let g = Game::new_standard();
        let c = SearchConfig {
            simulations: 64,
            batch_size: 8,
            virtual_loss: 0.0,
            fpu_reduction: 0.5,
            ..Default::default()
        };
        let mut s = Search::begin(&g, &c, 0);
        assert_eq!(s.next_batch().len(), 1);
        let e: Vec<LeafEval> = s.batch.iter().map(fake_net).collect();
        s.submit(&e);
        let batch_len = s.next_batch().len();
        assert!(
            batch_len < 8,
            "expected duplicates, got {batch_len} distinct"
        );
        assert!(s.sims.len() > batch_len, "duplicates must share a slot");
        let evals: Vec<LeafEval> = s.batch.iter().map(fake_net).collect();
        s.submit(&evals);
        while !s.next_batch().is_empty() {
            let evals: Vec<LeafEval> = s.batch.iter().map(fake_net).collect();
            s.submit(&evals);
        }
        let r = s.result().unwrap();
        let sum: u32 = r.visits.iter().map(|&(_, n)| n).sum();
        assert_eq!(sum, c.simulations);
    }

    #[test]
    fn batching_reduces_evaluator_calls() {
        let g = Game::new_standard();
        let mut calls = 0u32;
        let c = SearchConfig {
            simulations: 256,
            batch_size: 32,
            ..Default::default()
        };
        search_batched(&g, &c, 0, |b| {
            calls += 1;
            b.iter().map(fake_net).collect()
        })
        .unwrap();
        // 1 root call + ceil(256/32) = 9 in the ideal case; allow slack for
        // terminal leaves shrinking batches.
        assert!(calls <= 16, "expected ~9 evaluator calls, got {calls}");
    }

    // ---------- terminal leaves ----------

    #[test]
    fn terminal_leaves_are_never_evaluated() {
        // One ply before the cap: every child of the root is terminal
        // (adjudicated), so the evaluator must be called exactly once — for
        // the root itself — and never with a terminal position.
        let mut g = Game::new_standard();
        g.ply = DEFAULT_MAX_PLIES - 1;
        assert!(!g.is_terminal());
        let calls = StdCell::new(0u32);
        let c = SearchConfig {
            simulations: 300,
            batch_size: 16,
            ..Default::default()
        };
        let r = search_batched(&g, &c, 0, |batch| {
            calls.set(calls.get() + 1);
            batch
                .iter()
                .map(|state| {
                    assert!(
                        !state.is_terminal(),
                        "evaluator called on a terminal position"
                    );
                    fake_net(state)
                })
                .collect()
        })
        .unwrap();
        assert_eq!(calls.get(), 1, "only the root expansion needs the network");
        let sum: u32 = r.visits.iter().map(|&(_, n)| n).sum();
        assert_eq!(sum, c.simulations);
    }

    #[test]
    fn terminal_leaves_are_never_evaluated_mid_tree() {
        // Same guarantee in a tree deep enough to mix terminal and live
        // leaves: 5-5 captures means the next capture ends the game.
        let mut rng = SmallRng::seed_from_u64(4);
        let g = Game::with_handicap(
            Opening::BelgianDaisy,
            DEFAULT_MAX_PLIES,
            NO_PROGRESS_DISABLED,
            5,
            5,
            &mut rng,
        );
        let c = SearchConfig {
            simulations: 400,
            batch_size: 24,
            ..Default::default()
        };
        let r = search_batched(&g, &c, 0, |batch| {
            batch
                .iter()
                .map(|state| {
                    assert!(!state.is_terminal());
                    fake_net(state)
                })
                .collect()
        })
        .unwrap();
        let sum: u32 = r.visits.iter().map(|&(_, n)| n).sum();
        assert_eq!(sum, c.simulations);
    }

    // ---------- first-play urgency ----------

    #[test]
    fn fpu_does_not_prefer_the_unknown_when_losing() {
        // Root is losing badly (Q = -0.9). One child has been visited and is
        // the least bad option (Q from the root's POV = -0.5). Under the old
        // `Q = 0` rule the unvisited children would be preferred purely
        // because their default is 0 > -0.5.
        let g = Game::new_standard();
        let c = SearchConfig {
            simulations: 1,
            fpu_reduction: 0.25,
            ..Default::default()
        };
        let mut s = Search::begin(&g, &c, 0);
        s.next_batch();
        let e: Vec<LeafEval> = s
            .batch
            .iter()
            .map(|st| LeafEval::from_value(fake_net(st).value))
            .collect();
        s.submit(&e);

        let root = s.nodes[0];
        s.nodes[0].n_visits = 100;
        s.nodes[0].total_value = -90.0; // parent Q = -0.9
        let good = root.child_start;
        s.nodes[good as usize].n_visits = 10;
        s.nodes[good as usize].total_value = 5.0; // child Q = +0.5 => root sees -0.5

        assert_eq!(
            s.select_child(0),
            good,
            "FPU should keep the visited, least-bad child"
        );

        // The old rule (unvisited => 0) would have picked something else.
        let p = s.nodes[0];
        let sqrt_n = (p.n_visits.max(1) as f32).sqrt();
        let mut best_old = p.child_start;
        let mut best_score = f32::NEG_INFINITY;
        for cid in p.child_start..p.child_start + p.child_len {
            let ch = s.nodes[cid as usize];
            let q = if ch.n_visits == 0 {
                0.0
            } else {
                -(ch.total_value / ch.n_visits as f32)
            };
            let u = c.c_puct * ch.prior * sqrt_n / (1.0 + ch.n_visits as f32);
            if q + u > best_score {
                best_score = q + u;
                best_old = cid;
            }
        }
        assert_ne!(best_old, good, "the old rule should have explored instead");
    }

    #[test]
    fn fpu_still_explores_when_winning() {
        // Mirror image: a winning parent should not funnel everything into
        // the one visited child.
        let g = Game::new_standard();
        let c = cfg(1);
        let mut s = Search::begin(&g, &c, 0);
        s.next_batch();
        let e: Vec<LeafEval> = s.batch.iter().map(fake_net).collect();
        s.submit(&e);
        let root = s.nodes[0];
        s.nodes[0].n_visits = 100;
        s.nodes[0].total_value = 90.0; // parent Q = +0.9
        let visited = root.child_start;
        s.nodes[visited as usize].n_visits = 10;
        s.nodes[visited as usize].total_value = -1.0; // root sees +0.1
        assert_ne!(
            s.select_child(0),
            visited,
            "a winning parent should try the unvisited children"
        );
    }

    // ---------- Dirichlet noise ----------

    fn root_priors(s: &Search) -> Vec<f32> {
        let r = s.nodes[0];
        (r.child_start..r.child_start + r.child_len)
            .map(|cid| s.nodes[cid as usize].prior)
            .collect()
    }

    fn expanded_root(g: &Game, c: &SearchConfig, seed: u64) -> Search {
        let mut s = Search::begin(g, c, seed);
        s.next_batch();
        let e: Vec<LeafEval> = s.batch.iter().map(fake_net).collect();
        s.submit(&e);
        s
    }

    #[test]
    fn dirichlet_perturbs_root_priors_only() {
        let g = Game::new_standard();
        let plain = SearchConfig {
            simulations: 1,
            dirichlet_eps: 0.0,
            ..Default::default()
        };
        let noisy = SearchConfig {
            dirichlet_eps: 0.25,
            dirichlet_alpha: 0.2,
            ..plain.clone()
        };
        let a = expanded_root(&g, &plain, 5);
        let b = expanded_root(&g, &noisy, 5);
        let pa = root_priors(&a);
        let pb = root_priors(&b);
        assert_eq!(pa.len(), pb.len());
        assert!(pa != pb, "noise must change the root priors");
        assert!(
            pb.iter().all(|&p| p >= 0.0),
            "priors must stay non-negative"
        );
        let sum: f32 = pb.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-3,
            "priors must still sum to 1: {sum}"
        );
    }

    #[test]
    fn dirichlet_eps_zero_is_a_noop() {
        let g = Game::new_standard();
        let c = SearchConfig {
            simulations: 1,
            dirichlet_eps: 0.0,
            ..Default::default()
        };
        let a = expanded_root(&g, &c, 1);
        let b = expanded_root(&g, &c, 999);
        assert_eq!(
            root_priors(&a),
            root_priors(&b),
            "no noise, no seed effect"
        );
        // ...and matches the evaluator's priors verbatim.
        let want = fake_net(&g).priors.unwrap();
        assert_eq!(root_priors(&a), want);
    }

    #[test]
    fn dirichlet_leaves_deeper_priors_untouched() {
        let g = Game::new_standard();
        let c = SearchConfig {
            simulations: 120,
            batch_size: 8,
            dirichlet_eps: 0.5,
            ..Default::default()
        };
        let mut s = Search::begin(&g, &c, 11);
        while !s.next_batch().is_empty() {
            let evals: Vec<LeafEval> = s.batch.iter().map(fake_net).collect();
            s.submit(&evals);
        }
        // Find an expanded non-root node and check its children's priors are
        // exactly what the evaluator produced for that position.
        let root = s.nodes[0];
        let mut checked = 0;
        for cid in root.child_start..root.child_start + root.child_len {
            let child = s.nodes[cid as usize];
            if !child.expanded || child.child_len == 0 {
                continue;
            }
            let mut state = g;
            state.apply(child.mv);
            let want = fake_net(&state).priors.unwrap();
            let got: Vec<f32> = (child.child_start
                ..child.child_start + child.child_len)
                .map(|k| s.nodes[k as usize].prior)
                .collect();
            assert_eq!(got, want, "leaf priors must be unmodified");
            checked += 1;
            if checked == 3 {
                break;
            }
        }
        assert!(checked > 0, "expected at least one expanded child");
    }

    // ---------- tree reuse ----------

    #[test]
    fn reroot_keeps_the_played_subtree() {
        let g = Game::new_standard();
        let c = SearchConfig {
            simulations: 400,
            batch_size: 8,
            ..Default::default()
        };
        let mut s = Search::begin(&g, &c, 2);
        while !s.next_batch().is_empty() {
            let evals: Vec<LeafEval> = s.batch.iter().map(fake_net).collect();
            s.submit(&evals);
        }
        let r = s.result().unwrap();
        let played = r.best;
        let retained_expected = r
            .visits
            .iter()
            .find(|&&(mv, _)| mv == played)
            .map(|&(_, n)| n)
            .unwrap();
        assert!(retained_expected > 1);
        let grandchildren_before: u32 = {
            let root = s.nodes[0];
            let cid = (root.child_start..root.child_start + root.child_len)
                .find(|&cid| s.nodes[cid as usize].mv == played)
                .unwrap();
            let child = s.nodes[cid as usize];
            (child.child_start..child.child_start + child.child_len)
                .map(|k| s.nodes[k as usize].n_visits)
                .sum()
        };

        let mut s = s.reroot(played);
        assert_eq!(s.root_visits(), retained_expected, "visits must survive");
        let mut expected_state = g;
        expected_state.apply(played);
        assert_eq!(*s.root_state(), expected_state);
        let after = s.result().unwrap();
        let sum_after: u32 = after.visits.iter().map(|&(_, n)| n).sum();
        assert_eq!(sum_after, grandchildren_before, "subtree stats survive");
        assert!(sum_after > 0);

        // Continuing accumulates rather than restarting.
        while !s.next_batch().is_empty() {
            let evals: Vec<LeafEval> = s.batch.iter().map(fake_net).collect();
            s.submit(&evals);
        }
        assert_eq!(
            s.root_visits(),
            c.simulations,
            "budget is total: retained + new = simulations"
        );
        let r2 = s.result().unwrap();
        let sum2: u32 = r2.visits.iter().map(|&(_, n)| n).sum();
        assert!(sum2 > sum_after, "search after reroot must accumulate");
        assert_eq!(sum2, sum_after + (c.simulations - retained_expected));
        let legal: Vec<Move> =
            expected_state.legal_moves().iter().copied().collect();
        assert!(legal.contains(&r2.best), "reroot must yield a legal move");
        for &(mv, _) in &r2.visits {
            assert!(legal.contains(&mv));
        }
    }

    #[test]
    fn reroot_on_an_unsearched_root_starts_fresh() {
        let g = Game::new_standard();
        let c = cfg(50);
        let s = Search::begin(&g, &c, 0);
        let mv = g.legal_moves()[3];
        let mut s = s.reroot(mv);
        assert_eq!(s.root_visits(), 0);
        while !s.next_batch().is_empty() {
            let evals: Vec<LeafEval> = s.batch.iter().map(fake_net).collect();
            s.submit(&evals);
        }
        let r = s.result().unwrap();
        let sum: u32 = r.visits.iter().map(|&(_, n)| n).sum();
        assert_eq!(sum, c.simulations);
    }

    #[test]
    fn reroot_across_a_whole_game_is_consistent() {
        let mut rng = SmallRng::seed_from_u64(21);
        let mut g = Game::with_handicap(
            Opening::BelgianDaisy,
            40,
            NO_PROGRESS_DISABLED,
            4,
            4,
            &mut rng,
        );
        let c = SearchConfig {
            simulations: 64,
            batch_size: 8,
            dirichlet_eps: 0.25,
            ..Default::default()
        };
        let mut s = Search::begin(&g, &c, 5);
        while !g.is_terminal() {
            while !s.next_batch().is_empty() {
                let evals: Vec<LeafEval> =
                    s.batch.iter().map(fake_net).collect();
                s.submit(&evals);
            }
            let r = s.result().expect("non-terminal root has a result");
            let legal: Vec<Move> = g.legal_moves().iter().copied().collect();
            assert!(legal.contains(&r.best));
            g.apply(r.best);
            s = s.reroot(r.best);
            assert_eq!(*s.root_state(), g);
        }
        assert!(g.is_terminal());
    }

    // ---------- robustness ----------

    #[test]
    fn plays_several_moves_without_panicking() {
        let mut g = Game::new_standard();
        let c = cfg(25);
        let mut rng = SmallRng::seed_from_u64(1);
        for _ in 0..6 {
            if g.is_terminal() {
                break;
            }
            let r = search(&g, &c, &mut rng, random_rollout).unwrap();
            g.apply(r.best);
        }
        assert_eq!(g.ply, 6);
    }

    #[test]
    fn full_handicapped_games_never_panic() {
        let mut rng = SmallRng::seed_from_u64(0xfeed);
        for game_i in 0..6u64 {
            let black = (game_i % 6) as u8;
            let white = ((game_i * 3) % 6) as u8;
            let mut g = Game::with_handicap(
                Opening::BelgianDaisy,
                60,
                NO_PROGRESS_DISABLED,
                black.min(5),
                white.min(5),
                &mut rng,
            );
            let c = SearchConfig {
                simulations: 48,
                batch_size: 12,
                dirichlet_eps: 0.25,
                ..Default::default()
            };
            let mut plies = 0;
            while !g.is_terminal() {
                let r = search_batched(&g, &c, game_i, |b| {
                    b.iter().map(fake_net).collect()
                })
                .expect("non-terminal => Some");
                let legal: Vec<Move> =
                    g.legal_moves().iter().copied().collect();
                assert!(legal.contains(&r.best));
                let sum: u32 = r.visits.iter().map(|&(_, n)| n).sum();
                assert_eq!(sum, c.simulations);
                g.apply(r.best);
                plies += 1;
                assert!(plies <= 60);
            }
        }
    }

    #[test]
    fn next_batch_without_submit_is_recoverable() {
        let g = Game::new_standard();
        let c = SearchConfig {
            simulations: 64,
            batch_size: 8,
            ..Default::default()
        };
        let mut s = Search::begin(&g, &c, 0);
        s.next_batch();
        let e: Vec<LeafEval> = s.batch.iter().map(fake_net).collect();
        s.submit(&e);
        // Abandon a batch: the virtual loss it applied must be undone.
        let first: Vec<Game> = s.next_batch().to_vec();
        let again: Vec<Game> = s.next_batch().to_vec();
        assert_eq!(first, again, "re-descending a clean tree is reproducible");
        while !s.next_batch().is_empty() {
            let evals: Vec<LeafEval> = s.batch.iter().map(fake_net).collect();
            s.submit(&evals);
        }
        let r = s.result().unwrap();
        let sum: u32 = r.visits.iter().map(|&(_, n)| n).sum();
        assert_eq!(sum, c.simulations);
    }

    #[test]
    fn zero_simulations_is_harmless() {
        let g = Game::new_standard();
        let c = cfg(0);
        let r = batched(&g, &c, fake_net).unwrap();
        assert_eq!(r.visits.len(), g.legal_moves().len());
        assert!(r.visits.iter().all(|&(_, n)| n == 0));
    }
}
