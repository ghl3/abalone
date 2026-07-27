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
│        │                 │                  ▲                            │
│        └─────────────────┴──► abalone-wasm  │ ONNX                       │
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

**Threading.** One worker thread per core minus one; each owns its own ORT
session and its own shard writer. Threads claim games from a shared atomic
counter. No shared mutable state, no locks on the hot path. ORT is configured
with `intra_threads(1)` so its internal pool does not fight the worker pool.

**Shard writes are atomic** — write to `<name>.parquet.tmp`, rename on `finish()`
— so the trainer, which polls the directory, never reads a file without a footer.

### 2.4 `abalone-wasm` — browser boundary

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

  4. validate         held-out policy top-1, value CE, calibration
                      data health: decisive rate, plies, target entropy
                      curriculum: ratchet handicap_rate on the natural
                      termination rate of unseeded games (MODEL §4.1)

  5. anchor ladder    every 5 gens: vs random / heuristic@100 /
                      heuristic@800 / frozen earlier checkpoints → Elo

  6. commit           append metrics.jsonl, TensorBoard scalars,
                      apply retention, atomically advance state.json
```

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

Consumed by: `abalone-selfplay::encoder`, `model/encoder.py`, the browser
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

### 7.2 Review mode — the primary goal

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

### 7.3 Browser inference

`onnxruntime-web` (WASM backend, WebGPU where available) running inside a Web
Worker, driven by the pull-based search API from §2.4. The worker keeps the UI
thread responsive during analysis. The network is the same ONNX artifact the
trainer exports, served from `web/public/models/`.

### 7.4 Game format and notation

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

### 10.2 Remaining

| Area | Current | Target | Ref |
| --- | --- | --- | --- |
| Encoder sharing | conformance-tested duplication | single implementation via PyO3 | §5.6 |
| Web engine | heuristic MCTS in WASM | trained network via `onnxruntime-web` | §7.3 |
| WASM search | old single-leaf `search()` | pull-based coroutine (already exists in `abalone-mcts`) | §2.4 |
| Web mode | play only | play + review | §7.2 |
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
