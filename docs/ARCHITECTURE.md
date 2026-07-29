# ARCHITECTURE

> **Status:** target architecture. Describes the system we are building toward.
> Where today's code differs materially, §10 says so explicitly.
>
> Companion documents: [MODEL.md](MODEL.md) (what the network is and how it is
> trained), [2026-07-27-architecture-review.md](2026-07-27-architecture-review.md)
> (current-state review and roadmap).

---

## 1. System overview

Three subsystems, three languages, each chosen for one reason.

```
┌──────────────────────────────────────────────────────────────────────────┐
│  RUST  — rules, search, self-play, inference                             │
│                                                                          │
│   abalone-game ──► abalone-mcts ──► abalone-selfplay ──► shards (parquet)│
│        │                 │            ▲     ▲                            │
│        │        abalone-encoder ──────┴─────┼───┐                        │
│        └─────────────────┴──► abalone-wasm ◄┼───┘                        │
│                                    │        │ ONNX                       │
│                                    │        │                            │
└────────────────────────────────────┼────────┼────────────────────────────┘
                                     │        │
┌────────────────────────────────────┼────────┼────────────────────────────┐
│  PYTHON — training, orchestration  │        │                            │
│                                    │        │                            │
│   replay buffer ◄── shards         │   model/ (PyTorch) ──► ONNX export ─┘
│        │                           │        ▲                            │
│        └────────► SGD ─────────────┼────────┘                            │
│                                    │                                     │
│   train_loop.py orchestrates the whole generation cycle                   │
└────────────────────────────────────┼─────────────────────────────────────┘
                                     │
┌────────────────────────────────────┼─────────────────────────────────────┐
│  TYPESCRIPT — review & play UI     ▼                                     │
│                                                                          │
│   Next.js app ──► Web Worker ──► wasm (rules + search)                   │
│                        └───────► onnxruntime-web (network)               │
│                                                                          │
│   reads: exported games (JSON), exported network (ONNX)                  │
└──────────────────────────────────────────────────────────────────────────┘
```

**Why these boundaries.** Rust owns everything on the search hot path — the
rules and MCTS run millions of times per generation and need zero-allocation bit
manipulation. Python owns training because that is where the ecosystem is. The
browser gets the same Rust engine compiled to WASM, so the UI and the trainer
provably share one implementation of the rules.

---

## 2. Rust — `crates/`

### 2.1 `abalone-game` — rules

Pure game logic. No I/O, no allocation in hot paths, no dependencies beyond
`arrayvec`.

| Module | Responsibility |
| --- | --- |
| `cell` | Axial coordinates, six hex directions, cell naming and parsing |
| `bitboard` | `u128` board representation, direction shifts, iteration |
| `board` | Two bitboards + capture counters; starting positions |
| `moves` | `Move` enum, legal move generation, `apply` |
| `move_index` | Flat 0..2562 move encoding ↔ `Move` |
| `game` | Position + turn + ply + termination rules |
| `notation` | Move display and parsing |

**Representation.** The board is a 9×9 axial grid packed into a `u128`, bit
`r·9 + q`. 61 of 81 slots are valid (`|q − r| ≤ 4`); the rest are permanently
masked by `VALID_MASK`. Direction shifts are uniform integer offsets
(`E=+1, NE=+10, NW=+9`, and negations), so a direction step is one shift and one
mask on the whole board at once. Row wraparound is handled automatically because
any wrapped bit lands outside `VALID_MASK`.

**Move canonicalisation.** Broadside moves anchor on the three "positive"
directions (E, NE, NW) so a group has exactly one representation rather than the
`(start, end)` / `(end, start)` ambiguity of ordinary notation. Inline moves
anchor at the group's rear relative to travel. This is what makes the flat index
space a bijection.

**Termination** is the one place in this crate where policy lives, and per
[MODEL.md §3](MODEL.md#3-game-termination-and-the-outcome-signal) it is
score-based: six captures, or adjudication by capture differential at the ply
cap. Configurable cap, no positional tiebreak.

### 2.2 `abalone-mcts` — search

PUCT search with a **pluggable leaf evaluator**. The evaluator interface is the
crate's central abstraction:

```rust
LeafEval { value: f32, priors: Option<Vec<f32>> }
```

Keeping the evaluator behind a closure means random-rollout, ONNX and
browser-side evaluators all drive the identical search with no branching in the
hot loop.

**Batched evaluation** is the defining feature of the target design. Rather than
one network call per simulation, search descends repeatedly with virtual loss to
collect 16–64 leaves, evaluates them in one call, and backs them all up. This is
worth 5–15× and everything in [MODEL.md](MODEL.md) that assumes 800+ simulations
depends on it.

**Node storage** is a flat arena. Children are `(start, len)` index ranges into a
shared child array rather than a `Vec` per node, and child states are
materialised lazily on first descent. No per-node heap allocation on the hot
path.

**Retired:** the hand-written evaluator (`eval.rs`) leaves the training path
entirely. It remains only as a fixed benchmark opponent for the Elo ladder and as
a test fixture — never as a teacher.

**Outcome statistics** (`track_outcome_stats`, default off) are an analysis
readout bolted to the side of the search, and the design is deliberately
defensive about staying that way.

Search backs up one scalar. `collapse_value` reduces the three-way value head to
`P(win) − P(loss)` before anything is stored, and the score head is not consumed
at all — so a tree holds no draw probability and no marble margin, and neither
can be recovered from it afterwards. That is why the browser once resorted to
re-reading the network per candidate move, and why the numbers it showed were an
unsearched first impression wearing a searched label.

With tracking on, the full distribution and the expected margin are accumulated
alongside the scalar: identical paths, identical visit weighting, and the same
point-of-view flip at each ply — swap win and loss, negate the margin, draw is
invariant. Four things keep it from touching the AlphaZero search self-play runs:

- **Selection never reads it.** PUCT reads `Node::total_value` and nothing else,
  so the accumulators are write-only from the search's perspective. This is
  structural, not a convention.
- **`Node` is unchanged.** The accumulators live in side vectors parallel to the
  arena, allocated only when tracking is on, so with it off the node layout and
  memory footprint are identical rather than merely equivalent.
- **`LeafEval` is unchanged.** The extra distributions arrive through a separate
  `submit_with_stats`; `submit` is byte-for-byte the path the trainer takes.
- **The virtual loss has a matching term.** A pending visit contributes
  `[(1+vl)/2, 0, (1−vl)/2]`, whose collapse is exactly `virtual_loss`, so the
  two accumulators stay consistent *during* a batch and not merely after it.

The invariant `P(win) − P(loss) == eval` holds at every node by construction and
is asserted across batch sizes in the tests. It is the only cheap check on the
flips: apply one at the wrong parity and the distribution still looks like a
distribution — it just describes the position backwards.

### 2.3 `abalone-selfplay` — trajectory generation and inference

The bridge between search and training data.

| Module | Responsibility |
| --- | --- |
| `encoder` | Position → input planes (hot path; mirrors the training encoder) |
| `ort_eval` | ONNX Runtime session, batched forward, legal-mask + softmax |
| `shard` | Parquet writer for trajectories |
| `bin/selfplay-batch` | Multi-threaded game generation |
| `bin/eval-match` | Head-to-head matches, JSON summary |
| `bin/dump-golden` | Conformance fixtures for the Python side (§5.5) |
| `bin/review-probe` | Sweeps games at the browser's exact search config, to measure what the review panel measures (§7.6) |

**Threading.** One worker thread per core minus one; each owns its own ORT
session and its own shard writer. Threads claim games from a shared atomic
counter. No shared mutable state, no locks on the hot path. ORT is configured
with `intra_threads(1)` so its internal pool does not fight the worker pool.

**Shard writes are atomic** — write to `<name>.parquet.tmp`, rename on `finish()`
— so the trainer, which polls the directory, never reads a file without a footer.

### 2.4 `abalone-encoder` — the plane encoding

Position → `(14, 9, 9)` planes, and nothing else. It is its own crate rather
than a module of `abalone-selfplay` because the browser needs the identical
encoding and the `wasm32` target cannot link `ort` or `parquet`. Re-exported as
`abalone_selfplay::encoder`, so the trainer's paths are unchanged.

The point is that trainer and browser cannot drift by construction: there is one
Rust implementation, compiled twice. The Python twin in `model/encoder.py`
remains, pinned by the golden-fixture conformance test (§5.6).

### 2.5 `abalone-wasm` — browser boundary

A deliberately narrow `wasm-bindgen` surface:

- `Game` is opaque to JS; a handle wrapping the Rust struct.
- Moves cross as `u16` indices, never as structs. No serde, no JSON.
- Cells cross as `u8` indices `0..81`; JS derives `(q, r)` arithmetically.

**Search in the browser needs a pull-based API**, because `onnxruntime-web`'s
`run()` is async and WASM cannot await. The search is therefore driven as a
coroutine from the JS side:

```
begin(position, config)
loop {
    planes = next_batch()        // wasm: descend, collect leaves, encode
    if planes.is_empty() break
    (values, policies) = await ort.run(planes)   // JS: async inference
    submit(values, policies)     // wasm: back up, continue
}
result()
```

This keeps the entire search — selection, virtual loss, backup, node arena — in
Rust, shared with self-play, while inference stays in the JS runtime where the
async ONNX API lives.

---

## 3. Python — `model/`

| Module | Responsibility |
| --- | --- |
| `abalone_net.py` | Network definition (trunk, policy/value/auxiliary heads) |
| `encoder.py` | Plane encoding, move-index maths, D6 symmetry tables |
| `replay_buffer.py` | Shard ingest, rolling window, sampling, augmentation |
| `train_step.py` | One optimizer step; loss composition |
| `train_loop.py` | Generation cycle orchestration |
| `validate.py` | Held-out evaluation (policy top-1, value CE, calibration) |
| `eval.py` | Subprocess wrappers around the Rust binaries |
| `export_onnx.py` | Torch → ONNX with the fixed I/O contract |
| `export_game.py` | Shard → reviewable game JSON |
| `config.py` | Typed YAML config; unknown keys rejected |
| `state.py` | Resumable run state, atomic writes |

**Python orchestrates but does not compute on the hot path.** It spawns Rust
subprocesses for self-play and evaluation, ingests their output, and runs SGD.
Everything performance-critical is either in Rust or in PyTorch kernels.

**The data pipeline must be vectorised.** Sampling is a batch-level numpy
operation — no per-example Python loops, no per-example permutation
reconstruction. Positions are held as bitboards and decoded to planes at sample
time; symmetry permutation tables are precomputed once at import. Sampling runs
on a background thread so it overlaps the optimizer step.

---

## 4. The generation cycle

One generation, end to end:

```
  1. self-play        spawn selfplay-batch with the current ONNX
                      N games, deep MCTS, handicap-seeded fraction
                      writes shards incrementally

  2. train            (overlaps 1) poll for new shards, ingest,
                      sample, SGD; evict generations outside the window

  3. export           checkpoint .pt  +  ONNX (EMA weights)

  4. validate         two holdouts: a bounded slice of an early generation
                      (a fixed ruler, hence a drift indicator) and this
                      generation's own withheld slice (current distribution,
                      never trained on — the one to gate on).  MODEL §8.1
                      data health: decisive rate, plies, target entropy
                      curriculum: ratchet handicap_rate on the natural
                      termination rate of unseeded games (MODEL §4.1)

  5. anchor ladder    every N gens: floor anchors (random, heuristic@100)
                      plus frozen and trailing checkpoints of this run
                      → Elo per rung, each with its 95% interval

  6. commit           append metrics.jsonl, TensorBoard scalars,
                      apply retention, atomically advance state.json
```

**Metrics are namespaced and logged generically.** Each measurement module
returns a flat dict under its own prefix — `selfplay/` (`model/export_game.py`),
`buffer/` (`model/replay_buffer.py`), `val_frozen/` and `val_rolling/`
(`model/validate.py`), and `train/`, `curriculum/`, `ladder/`, `perf/` (the
loop) — and the loop iterates whatever it is handed. There is no whitelist, so
a metric added downstream reaches `metrics.jsonl` and TensorBoard without the
loop being edited. `uv run python -m model.report --run latest` reads it back.

**Self-play and training overlap by design.** Self-play is CPU-bound across all
cores; training is GPU/MPS-bound. Running them concurrently within a generation
is most of the reason a generation is affordable at all.

**No gating.** Self-play always uses the latest network. Progress is measured by
the anchor ladder, not by promotion.

**Resume granularity is one generation.** `state.json` records the phase and is
written atomically at each transition. On restart, partial shards for an
in-progress generation are discarded and that generation restarts from self-play.

---

## 5. Cross-language contracts

**This is the highest-risk surface in the system.** Four specifications are
consumed by more than one language, and a silent divergence in any of them
corrupts training invisibly. One such divergence — swapped capture planes between
the Rust and Python encoders — was live in the codebase for three generations and
survived a 43-test Python suite that exercised the encoder heavily.

### 5.1 Move index space

`MOVE_SPACE = 2562`, laid out anchor-major:

```
inline     [0, 1098)     idx = anchor_compact · 18 + dir · 3         + (size − 1)
broadside  [1098, 2562)  idx = anchor_compact · 24 + gi · 8 + mi · 2 + (size − 2) + 1098
```

`anchor_compact` is the anchor's index into the 61 valid cells. The space
includes structurally impossible moves; they never come out of `legal_moves()`
and are masked before softmax.

Consumed by: `abalone-game`, `abalone-wasm`, `model/encoder.py`, the policy head
gather table.

### 5.2 Input planes

`(14, 9, 9)` float32, layout per [MODEL.md §5](MODEL.md#5-input-representation).
Side-to-move relative. The ply normaliser is tied to the configured ply cap and
**must** move in lockstep with it across both implementations.

Consumed by: `abalone-encoder` (and through it `abalone-selfplay` and
`abalone-wasm`), `model/encoder.py`, the browser
evaluator.

### 5.3 ONNX signature

```
input   planes         (B, 14, 9, 9)  float32
output  policy_logits  (B, 2562)      float32   unmasked, unnormalised
        value          (B, 3)         float32   logits over (win, draw, loss)
        score          (B, 13)        float32   logits over d ∈ [−6, +6]
```

Batch is dynamic. Softmax and legal-masking are applied by the consumer, not
baked into the graph, so Rust and the browser keep control of masking.

Consumed by: `ort_eval.rs`, `export_onnx.py`, `onnxruntime-web`.

### 5.4 Shard schema

One row per trajectory position. Columns hold the position POV-relative, the
outcome labels, and the full search result:

```
game_id          u32        source game within (run, generation)
seed             u64        RNG seed — the game is reproducible from this
opening          u8         0 = Standard, 1 = BelgianDaisy
handicap_black   u8         marbles Black conceded at curriculum seeding
handicap_white   u8         marbles White conceded at curriculum seeding

own_bb_lo/hi     u64        side-to-move relative bitboards
opp_bb_lo/hi     u64
black_losses     u8         marbles BLACK has lost
white_losses     u8         marbles WHITE has lost
turn             u8         0 = Black, 1 = White
ply              u16
max_plies        u16        this game's cap — the ply plane's denominator

move_played      u16        flat move index applied next
is_full_search   bool       true iff this position ran the FULL simulation
                            count; only these carry a policy target (§7.2)

z                i8         outcome from this POV: +1 win, 0 draw, −1 loss
score_diff       i8         final capture differential from this POV, [−6, 6]
q                f32        MCTS root value from this POV (diagnostics / UI)

child_move_idxs  list<u16>  the search result — the policy target
child_visits     list<u32>  parallel to child_move_idxs

cap_map_idx      list<u16>  sparse capture map: channel*81 + cell, 0..162
cap_map_val      list<f32>  parallel discounted weights, clamped to [0, 1]
```

**Capture-map target.** For a position at ply `t` with side-to-move `S`, every
future capture at ply `t' ≥ t` that removed a marble of side `X` from cell `c`
contributes `γ^(t'−t)` (γ ≈ 0.98) to channel `0` if `X == S` else channel `1`, at
cell `c`. Cells are absolute board indices — only the *channel* is POV-relative,
matching the own/opp convention of the input planes. Weights are clamped to
`[0, 1]` and only non-zero entries are stored.

Column names state whose *losses* they are. The previous naming
(`pushed_off_black`) meant "pushed off **by** black" and read naturally as the
opposite; that ambiguity is what produced the plane-swap bug.

`game_id` and `seed` make shards self-describing: games can be reconstructed and
replayed without inferring boundaries from `ply` resets.

### 5.5 Training batch

The contract between `replay_buffer.py` and `train_step.py`:

```
planes        (B, 14, 9, 9) float32
policy        (B, 2562)     float32   normalised over legal moves; zeros if no target
legal_mask    (B, 2562)     float32   1.0 on legal moves
policy_weight (B,)          float32   1.0 if is_full_search else 0.0
value         (B,)          int64     class index: 0 = win, 1 = draw, 2 = loss
score         (B,)          int64     class index: score_diff + 6, in [0, 12]
capture_map   (B, 2, 9, 9)  float32   targets in [0, 1]
q             (B,)          float32   carried for diagnostics, not a loss term
```

**The value head trains on the game outcome alone — there is no z/q blend.** The
blend existed to work around an all-draws data distribution; the curriculum in
[MODEL.md §4](MODEL.md#4-curriculum-capture-handicap-seeding) removed that
problem (measured decisive rate 82.5%, mean |d| 2.06), so the blend would now be
complexity in service of a solved problem. `q` stays in the shard for diagnostics
and the review UI. If value learning proves noisy at low game counts, the
principled fix is to have MCTS back up a 3-way distribution rather than a scalar
— not to reintroduce a soft-label hack.

`policy_weight` implements playout cap randomisation: positions searched at the
fast simulation count still supply value, score and capture-map targets, but must
not contribute to the policy loss.

### 5.6 Enforcement

**Interim — golden-file conformance test.** `dump-golden` emits ~200
`(position, planes, legal move indices, encoded planes)` fixtures as JSON; a
pytest asserts the Python implementation reproduces them byte-for-byte. Cheap,
and it makes this bug class impossible to reintroduce silently.

**Target — eliminate the duplication.** Expose the Rust encoder to Python via
PyO3, taking a batch of bitboards and returning an `(N, 14, 9, 9)` array in one
crossing. Then training and self-play do not merely *agree* on the encoding, they
*are* the same code. This is strictly better than testing for agreement, and the
batch-oriented signature is also what the vectorised replay buffer wants.

**Also required:** record the git SHA in `state.json` and warn on mismatch at
resume. `config_hash` covers YAML only — changing the ply cap or the plane layout
silently invalidates every shard in the buffer with no detection today.

---

## 6. Storage layout

```
runs/<run-id>/
├── config.yaml              frozen resolved config
├── state.json               atomic; phase, generation, git SHA, annealed
│                            handicap_rate, history
├── metrics.jsonl            one line per generation
├── tb/                      TensorBoard events
├── checkpoints/
│   ├── gen_NNN.pt           model + optimizer + EMA
│   └── gen_NNN.onnx         exported network (EMA weights)
├── shards/gen_NNN/
│   └── shard_tNN_NNNN.parquet
├── eval/gen_NNN_*.json      match results
├── games/gen_NNN/*.json     exported reviewable games
└── logs/gen_NNN_*.log       subprocess output
```

Run IDs are `<adjective>-<noun>-<YYYYMMDD>-<HHMM>` — memorable and chronologically
sortable.

**Retention** keeps the last K checkpoints, ONNX exports and shard generations.
The checkpoint required for resume and any file referenced by `state.json` are
never collected.

---

## 7. Web — `web/`

Next.js (App Router) + React. Two distinct modes sharing one board renderer.

### 7.1 Play mode

Interactive board with click-to-select and drag-with-snap, legal-move
highlighting, and sumito preview. The engine is the WASM build of the same Rust
crates the trainer uses, so browser rules and training rules cannot drift.

Two tabs, because the two jobs want opposite things from the same engine:

- **Play vs engine** — pick a side and a difficulty (a named simulation budget,
  from `Beginner` = raw policy with no search up to `Maximum` = 1600). The board
  rotates 180° when you take Black so your own marbles are always the near side,
  and the pointer delta is mirrored with it. No win bar and no move list: the
  network searches on its own turn only, so it is neither computing nor
  displaying an opinion while you think.
- **Analysis** — both sides played by hand, with the win bar (`WinBar.tsx`), the
  ranked move list, hover preview and click-to-play. Board orientation is a
  control here rather than a consequence of the colour you picked (`F`), and
  `← →` step back and forward through the line you have entered.

There is no evaluator picker. Analysis briefly offered a choice between the
network and the hand-written heuristic, which stopped being a question worth
asking once the network was unambiguously the stronger of the two; the
`WasmGame::analyze` and `eval_white_pov` bindings that existed to serve it are
gone with it. The heuristic itself remains in `abalone-mcts` where the trainer
and the benchmarks use it.

**The search budget is three named settings** — Quick (120), Standard (500),
Deep (2000) — not a slider. Strength goes with the log of the budget, so a
50-step slider over 50–2000 offered forty settings nobody could distinguish;
worse, it searched on every step of a drag, so crossing the range spawned and
abandoned dozens of searches.

**The panel refines rather than blanks.** Progress messages carry the whole
tree, not a counter (§7.4), so the rows on screen are always this position's,
updating as visits accumulate. The earlier design showed nothing at all until
the budget was spent — and, because "no rows" and "no legal moves" were the
same state, spent every search telling you the position was over.

**The panel speaks in probabilities, and they are searched.** Each ranked move
shows the chance the game is won, drawn or lost from there, plus the expected
final capture differential in marbles. Both come out of the tree — see the
outcome statistics in §2.2 — not from a forward pass on the resulting position.

That distinction is the whole design. An earlier version read the network's
heads directly for each candidate move, which put a searched eval and an
unsearched first impression side by side in the same table with nothing saying
which was which. On the opening position the two disagree by eight points:
the raw network says White 49% / draw 7% / Black 44%, and 500 simulations say
White 41% / Black 52%. Displaying the first while ranking by the second is not
a presentation choice, it is a wrong number.

**One scale, not three.** `eval` and win/draw/loss are the same axis in
different units, so only the probabilities are shown; the marble margin is a
genuinely different axis and survives alongside them, because probabilities
saturate — at 92% they stop discriminating between grinding out one marble and
taking four, which is exactly where a human still wants to know.

The signed eval appears **nowhere in the UI**: not in the table, not in the move
detail, not on the win bar. It remains the tree's internal scalar and the thing
the `P(win) − P(loss)` invariant is asserted against, but that check belongs in
the Rust tests where it fails loudly, not on screen where it asks a reader to
notice. Visit counts went the same way — the row's background bar shows search
concentration without a unit, which is the only part of "412 visits" that meant
anything without also knowing the budget.

What is left follows one rule: numbers are White-positive, and colour names the
*side* (the marble ramp) rather than good-versus-bad, which would invert every
ply. `web/lib/outcomeFormat.ts` owns it.

**Each ranked move carries the line search explored under it** — the
most-visited path from that root child, walked off the tree the search already
built (`Search::principal_variation`), so it costs no extra inference. Hovering
a move in that line replays the board to it, which is the cheapest way to make
"why is this move ranked first" answerable without playing it.

### 7.2 Game review — your own games

A finished game (or one in progress) can be reviewed from the result banner or
the toolbar. Entering review snapshots the move list, so starting a new game
does not pull the record out from under it, and then sweeps the whole game:
every position it passed through is searched in play order at review depth.

That sweep is the point. During play the engine only ever searched its *own*
turns, so the interesting half of the record — yours — has never been looked at.

Review depth is **800 simulations**, four times what the panel first shipped
with. `review-probe` (§7.6) swept 909 positions across four games at 200, 800
and 3200: at 200 the most-visited root move held a median 12% of the visits and
agreed with a 3200-simulation search on 43% of positions, while labelling 60% of
moves "best" where the deep search allowed 24%. Everything on the panel keys off
*which* move is best — the label, the "engine wants" line, the hover preview,
the yardstick a cost is measured against — so the depth is the feature's
premise, not a nicety. At 800 those become 19%, 63% and 36%.

Each move is then graded by **what it cost against the best move available from
the same position**, in points of expected score, into best / good / inaccuracy
/ blunder. Two properties earn that definition:

- **Both numbers come off one tree.** The obvious alternative — the root eval
  before the move against the root eval after it — subtracts two independent
  searches, and their disagreement is dominated by the search revising itself.
  Measured that way a move the engine *itself picked* is charged +2.06 ± 0.35
  points at 200 simulations, +1.49 at 800, +1.07 at 3200: shrinking with depth,
  never reaching zero, and never the player's doing. The eval a search reports is
  the Q of its own best child, so in a converged search playing that move
  preserves the number exactly. Across-move swing survives only as a fallback.
- **It is zero when you played the engine's move**, by construction, so a row can
  no longer read `BEST` and a penalty at once.

Expected score (`P(win) + P(draw)/2`) rather than `P(win)`, because draw mass
climbs over a game — 11% before ply 10, 31% after ply 40 — and every point
leaving both players' win column would otherwise be charged to whoever moved
last. It is also exactly the axis the graph plots, `rootEval = 2·score − 1`, so
`Δscore = Δeval / 2` holds outright and the 4-point and 10-point bands are the
old 0.08 and 0.2 restated. Validated against the 3200 sweep as ground truth: at
800 the review flags 0 of 71 moves the deep search calls best and catches 21 of
26 it charges 4 points or more. The bands are deliberately wide and the grades
few: the underlying estimate is a 3M-parameter network, and a scale nobody
trusts is worse than a coarse one they do.

The screen is a board with a ply scrubber (slider, transport buttons, ← →),
a graph over the game, and the move list, which follows the scrubber so arrowing
forward never hides the move being read. Hovering a flagged move previews what
the engine wanted instead, using the same overlay the analysis panel uses.

The graph's curve is `rootEval`, which is *identically* `P(win) − P(loss)` — so
its shape is already a probability difference, and only the readout needed
changing: the tooltip gives the full win/draw/loss triple rather than a signed
decimal. It plots on a square root scale — real games sit inside ±0.2, which a
linear [-1, 1] axis renders as a flat line. One consequence to keep in mind
reading it: a modest change near equality is drawn as a cliff, and a point or two
of its per-ply jaggedness is the alternating-search bias above rather than
anything that happened on the board.

Beneath it, on the same x axis and inside the same `<svg>` so one cursor crosses
both, is the **marble lead** — stepped and linear, deliberately unlike the curve
above it, because captures are discrete events at a known ply and interpolating
would draw a marble leaving the board over four moves. It comes off the game
record rather than the sweep, so it is complete before the analysis finishes and
owes the network nothing. The engine's *expected* margin stays in the text
readout: it correlates with the eval curve, and a third view of one thing costs
clutter for no information.

**One rule the review made explicit:** never hold a wasm handle across renders.
The position for a ply is created, read, and freed inside a single memo, and
only plain data escapes. The earlier pattern — memoise the handle, free the
stale one from an effect — dies under React's strict-mode remount, which runs
the unmount cleanup and then reuses the handle it just freed
(`null pointer passed to rust`).

### 7.3 Self-play review — the primary goal

```
┌──────────────────────────────────────────────────────────────┐
│  game selector    gen 034 · game 117 · handicap (2,1)        │
├────────────────────────────────┬─────────────────────────────┤
│                                │  eval bar   W ▓▓▓▓▓░░░ B    │
│                                │  win 0.62 draw 0.21 loss …  │
│          board                 │  expected score  +1.4       │
│                                ├─────────────────────────────┤
│                                │  TRAINING-TIME SEARCH       │
│                                │   1. C3-C5:NE   412 visits  │
│                                │   2. D4E:3      201         │
│                                ├─────────────────────────────┤
│                                │  CURRENT NETWORK            │
│                                │   1. D4E:3      688  +0.31  │
├────────────────────────────────┴─────────────────────────────┤
│  ◀ ▶  ply 84/163   ●────────────●──────────────────          │
└──────────────────────────────────────────────────────────────┘
```

The distinguishing feature: **every self-play position already carries the full
MCTS visit distribution and root value from the moment it was generated.** The
review tool shows what search believed at training time alongside what the
current network believes now. That is a training-pipeline debugger as much as a
product feature — it makes "the policy target is uniform" something you can
*see*, not something you have to infer from a loss curve.

### 7.4 Browser inference

`onnxruntime-web` running inside a Web Worker, driven by the pull-based search
API from §2.5. WebGPU is preferred and the threaded WASM backend is the
fallback; the provider that took is reported in the analysis panel. The worker
keeps the UI thread responsive during search. The network is the same ONNX
artifact the trainer exports, served from `web/public/models/`.

Four things are load-bearing and easy to get wrong:

- **One `run()` per session, ever.** `onmessage` is `async`, so two searches
  posted in quick succession — a position change, a depth change, React's
  strict-mode double mount — will both be inside `session.run` at once unless
  something stops them. `onnxruntime-web` permits exactly one run per session:
  the second throws `Session mismatch` and the engine wedges for the rest of
  the page's life. The worker therefore chains `runSearch` calls through a
  promise queue. Bumping `generation` first is what keeps the wait short — the
  superseded search returns as soon as its current forward pass resolves,
  rather than spending its whole budget on a position nobody is looking at.
- **Masking and collapsing happen in Rust**, not JS. The worker hands
  `policy_logits` and `value` back to `WasmSearch::submit` exactly as the graph
  emitted them; the legal-move gather, the softmax over them and the
  `P(win) − P(loss)` collapse are the same lines `ort_eval.rs` runs. JS only
  moves `Float32Array`s.
- **Cross-origin isolation is required for threads.** `SharedArrayBuffer` needs
  COOP/COEP, set in `next.config.mjs`; without them ORT is capped at one thread.
- **`ort.env.wasm.wasmPaths` must point somewhere real.** Left unset it resolves
  against the hashed webpack chunk URL and the first session 404s, so
  `scripts/copy-ort-assets.mjs` stages the binaries into `public/ort/`.

The `score` head rides along with `policy_logits` and `value` into
`WasmSearch::submit`, which softmaxes both and hands the distributions to
`submit_with_stats`. Everything the panel displays therefore comes off the tree
in one read, and there is no second forward pass anywhere on this path — an
earlier design ran one per progress tick to annotate the ranked moves, which
cost about 8% of throughput to produce numbers that were not searched.

Progress messages carry a full `SearchSnapshot` — ranked moves, visit counts,
win/draw/loss, margins, principal variations — and not just a visit count, which
is what lets the panel refine in place. It stays cheap because it is bounded by
wall-clock (120 ms) rather than by batch count, and because notation and the PV
walk are done only for the five rows that will be displayed rather than for all
~50 legal moves.

The Q values are a different matter and `allMoves` carries every one of them,
beside those five rows. The cap was always about presentation work, never about
the numbers — and review needs the numbers for the move a *player* chose, which
is frequently not in the top five by visits. Grading it against the best move in
the same search is impossible without it (§7.2).

### 7.5 Game format and notation

Games are JSON, emitted by `export_game.py` directly from shards:

```jsonc
{
  "run_id": "...", "gen": 34, "game_id": 117,
  "opening": "belgian_daisy", "handicap": [2, 1], "seed": 8891,
  "result": { "outcome": "black_wins", "score_diff": 3, "plies": 163 },
  "moves": [ { "idx": 1204, "notation": "C3-C5:NE",
               "visits": [[1204, 412], [881, 201]], "q": 0.18 } ]
}
```

Standard Abalone notation is added for display and parsing alongside the existing
engine form; the engine form remains canonical on the wire.

**Position permalinks** encode board state, capture counters and ply in the URL,
so "a particular spot" is shareable — the second half of the tool's stated
purpose.

### 7.6 Measuring the review — `review-probe`

The review makes claims: this move was best, that one cost you six points. Those
are checkable, and checking them needs the browser's numbers rather than a
plausible reconstruction of them.

`crates/selfplay/src/bin/review-probe` plays games and then sweeps every position
they passed through with the configuration `WasmGame::begin_search` builds —
`c_puct` 1.4, batch 16, `dirichlet_eps` 0, `track_outcome_stats` on, and the
worker's own per-ply seed — emitting one JSON line per position with every root
child's Q and visit count. `--review-sims` repeats, so one game record can be
swept at several depths and the same verdict compared across them; taking the
deepest sweep as ground truth is what turns "does this metric work" into a false
positive rate. `ABALONE_USE_COREML=1` makes a 3200-simulation sweep minutes
rather than most of an hour.

```text
review-probe --model web/public/models/best.onnx \
             --games 4 --review-sims 200 --review-sims 800 --review-sims 3200
```

It is a measurement tool, not part of any loop: nothing depends on it and it
writes nothing but its own output. The one thing it needed from the library was
`OrtEvaluator::evaluate_batch_with_wdl`, which returns the distribution
`evaluate_batch` was already computing and discarding, so `submit_with_stats` can
be fed exactly what the browser feeds it.

What it found on first use is in NOTEBOOK.md (2026-07-29) and drives §7.2: the
across-move measure the panel shipped with charged players ~2 points a move for
the search's failure to converge, and a review at 200 simulations agreed with a
3200-simulation search about which move was best on 43% of positions.

---

## 8. Concurrency and process model

```
train_loop.py  (parent)
  ├── selfplay-batch      (subprocess, T worker threads, T ORT sessions)
  ├── eval-match          (subprocess, periodic)
  ├── tensorboard         (subprocess, optional)
  └── SGD                 (in-process, MPS/CUDA)
        └── sampler       (background thread, CPU)
```

- Rust workers share nothing but an atomic game counter.
- The parent communicates with self-play only through the filesystem: shards
  appear atomically, progress is parsed from the log. There is no IPC channel to
  get out of sync.
- Signal handlers terminate tracked subprocesses so an interrupted run does not
  leave orphans.

---

## 9. Invariants

| Invariant | Enforced by |
| --- | --- |
| Rules are identical in trainer and browser | one Rust crate, compiled twice |
| Plane encoding identical in training and inference | conformance test → PyO3 shared implementation |
| Move index space is a bijection over legal moves | round-trip tests over all 2562 indices |
| The trainer never reads a partial shard | write-to-`.tmp` + atomic rename |
| A crash never corrupts run state | atomic fsync'd `state.json` at each phase |
| The resume checkpoint is never garbage-collected | retention skips referenced files |
| Illegal moves receive zero probability | mask before softmax, both sides |
| Symmetry augmentation is outcome-preserving | D6 group axiom tests on cell and move permutations |
| A move's cost is measured inside one search, never across two | `lossVersusBest`; zero when played move is the engine's, by construction (§7.2) |

---

## 10. Implementation status

### 10.1 Landed (2026-07-27)

The training pipeline has been rebuilt against this document.

| Area | Was | Now | Ref |
| --- | --- | --- | --- |
| Termination | 400-ply → draw | 200-ply → adjudicate by capture differential | [MODEL §3](MODEL.md#3-game-termination-and-the-outcome-signal) |
| Cold start | heuristic evaluator, gens 1–2 | capture-handicap seeding, no heuristic anywhere | [MODEL §4](MODEL.md#4-curriculum-capture-handicap-seeding) |
| Opening | standard, fixed | configurable + randomised plies | [MODEL §4](MODEL.md#4-curriculum-capture-handicap-seeding) |
| Search batching | 1 leaf per NN call | pull-based coroutine, 16–64 with virtual loss | [MODEL §7.1](MODEL.md#71-batched-evaluation-is-the-enabling-change) |
| Simulations | 100 (1.6× branching) | 200 fast / 800 full + playout cap randomisation | [MODEL §7](MODEL.md#7-search) |
| Policy head | dense, 3,321,282 params | conv `(42,9,9)` gather, 48,426 | [MODEL §6.2](MODEL.md#62-policy-head--convolutional-not-dense) |
| Value head | tanh scalar | 3-way win/draw/loss | [MODEL §6.3](MODEL.md#63-value-head--3-way-not-tanh) |
| Auxiliary heads | none | score (13-way) + capture-map `(2,9,9)` | [MODEL §6.4](MODEL.md#64-score-head--auxiliary-from-the-trajectory) |
| Trunk | 4 × 64 (~300 k) | 10 × 128 (2,954,240 = 97.4% of the model) | [MODEL §6.1](MODEL.md#61-trunk) |
| Model selection | per-gen gate, 21 games | anchor ladder → Elo | [MODEL §8.1](MODEL.md#81-measurement--the-part-that-was-missing) |
| Held-out eval | none | `validate.py` every generation | [MODEL §8.1](MODEL.md#81-measurement--the-part-that-was-missing) |
| Encoder sharing | duplicated, untested across languages | 832-fixture golden conformance test | §5.6 |
| Replay buffer | per-example loop, 4,536 B/position | vectorised, 32 B/position bitboards | §3 |
| Game export | none | `export_game.py` → reviewable JSON | §7.4 |
| Web engine | heuristic MCTS in WASM | trained network via `onnxruntime-web` in a worker | §7.4 |
| WASM search | single-leaf `search()` | pull-based coroutine driven from JS | §2.5 |
| Move grading | eval before vs eval after, in `P(win)` | played move vs the same search's best, in expected score | §7.2 |
| Review depth | 200 simulations, unmeasured | 800, chosen against a 3200-simulation ground truth | §7.6 |

### 10.2 Remaining

| Area | Current | Target | Ref |
| --- | --- | --- | --- |
| Encoder sharing | one Rust crate (`abalone-encoder`) + conformance-tested Python twin | single implementation via PyO3 | §5.6 |
| Web mode | play vs engine, analysis, review of your own games | + review of exported self-play games | §7.3 |
| Notation | engine-internal only | standard Abalone notation for display/parse | §7.4 |
| Position sharing | none | permalinks encoding board + counters + ply | §7.4 |
| Repo entry point | no root README | README pointing at `docs/` | — |
| Stale artifacts | `runs/` holds 6-plane, 2-output checkpoints | purge; they cannot load under the current contract | §5.3 |

### 10.3 Known environment hazard

Release builds link against cached `ort-sys` / `zstd-sys` build-script output. If
the Xcode Command Line Tools are upgraded, that cache keeps pointing at the old
clang runtime directory and `cargo build --release` fails with
`ld: library 'clang_rt.osx' not found`. **Debug builds do not exhibit this**,
which makes it easy to misdiagnose — and training runs release binaries. Fix:

```sh
cargo clean --release -p ort-sys -p zstd-sys
```

---

## 11. Extension points

Where to add things, so they land in the right layer:

- **A new starting position or curriculum** → `abalone-game::board` plus a
  seeding option in `abalone-selfplay`. Never in the network.
- **A new input plane** → `model/encoder.py` and `selfplay/encoder.rs` together,
  bump the plane count in the ONNX contract, regenerate golden fixtures. (Under
  the PyO3 target, one place.)
- **A new auxiliary head** → `abalone_net.py` for the head, `shard.rs` for the
  label column, `train_step.py` for the loss term.
- **A new evaluator or opponent** → implement the `LeafEval` closure; register it
  in `eval-match`. This is how the frozen heuristic remains available as a
  benchmark without re-entering the training loop.
- **A new UI analysis view** → consumes exported game JSON; requires no engine
  change.
