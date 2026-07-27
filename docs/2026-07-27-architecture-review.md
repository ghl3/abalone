# Abalone AlphaZero — architecture, code and modeling review

**Date:** 2026-07-27
**Reviewer:** Claude (Opus 5)
**Scope:** full codebase at `diagonal-version` @ `87efbf6`, plus run artifacts from
`runs/joyful-panther-20260510-2305` (last training run, ~2.5 months old)
**Test status at time of review:** 43 Python tests pass, 16 Rust tests pass

---

## 1. Executive summary

The engineering scaffolding is genuinely good — better than most hobby AlphaZero
projects. The rules engine is correct and fast, the crate layering is sensible,
the training harness has a resumable phase state machine with atomic writes,
config-hash drift detection, retention policy and TensorBoard integration.

**The learning loop, however, is provably not learning, and the run artifacts say
so precisely.** Three generations produced a model that outputs a uniform policy
and a constant-zero value, and it converged there rather than drifting there.

There are also **two silent correctness bugs** that would sabotage training even
after the loop is unblocked — one of which inverts the network's material
awareness between training and play. Both should be fixed before spending another
GPU-hour.

Priority ordering used throughout this document:

1. **Correctness** — bugs that make results meaningless (§3)
2. **Signal** — the loop cannot learn from the data it currently generates (§2)
3. **Observability** — the collapse should have been visible in minutes, not days (§7)
4. **Throughput** — the search is ~8× too slow for the simulation counts it needs (§5)
5. **Capacity** — 92% of parameters are in the wrong layer (§6)
6. **Product** — the web tool is a playground, not a review tool (§9)

---

## 2. The blocking problem: the training signal is empty

### 2.1 The evidence

From [`runs/joyful-panther-20260510-2305/metrics.jsonl`](../runs/joyful-panther-20260510-2305/metrics.jsonl):

| metric | gen 1 | gen 2 | gen 3 | interpretation |
| --- | --- | --- | --- | --- |
| `plies_per_game_avg` | 399.2 | 400.0 | 400.0 | **every game hits the 400-ply draw cap** |
| `train_loss_value` | 0.0020 | 0.0038 | 0.0027 | value head collapsed to constant 0 |
| `train_loss_policy` | 4.1319 | 4.1403 | 4.1435 | `ln(62) = 4.127` — target *is* uniform |
| `train_grad_norm` | 0.051 | 0.046 | 0.039 | converging to that fixed point, not escaping it |
| `gate_winrate` | 0.0 | 0.0 | 0.0 | 21 draws every time |

Cross-entropy cannot go below the entropy of its target. A policy loss pinned at
`ln(branching factor)` means the MCTS visit distributions being trained on carry
**zero bits of information**. `value_loss ≈ 0` on all-draw data means the value
head correctly learned "always output 0". The network sits at a perfect, useless
optimum, and the shrinking gradient norm confirms it is settling there rather
than escaping.

### 2.2 Cause A — nobody ever wins

Counting outcomes in the gen-1 self-play log (which used the *heuristic*
evaluator, i.e. the cold-start fix from commit `87efbf6`):

```
196 final=Draw
  3 final=Wins(Black)
  1 final=Wins(White)
```

**98% draws, with the hand-written evaluator driving search.** Independently
corroborated by `cargo run --bin movegen-bench`:

```
random self-play: 3.23K/s playouts/sec (~399 ply each, ~1.29M/s positions/sec)
```

Abalone from the standard opening is enormously drawish for weak players.
Removing 6 marbles requires sustained coordinated pressure that neither random
play nor 100-simulation heuristic MCTS can produce. So `z = 0` for ~98% of
training positions, permanently.

Contributing factors in the code:

- [`crates/selfplay/src/lib.rs:146`](../crates/selfplay/src/lib.rs#L146) hardcodes
  `Game::new_standard()`. `Board::belgian_daisy()` is implemented and tested at
  [`crates/game/src/board.rs:59`](../crates/game/src/board.rs#L59) but **never
  wired up**. Belgian Daisy exists as a tournament opening precisely because the
  standard layout is too passive — it starts with the clusters already in contact.
- `MAX_PLIES = 400` at [`crates/game/src/game.rs:17`](../crates/game/src/game.rs#L17)
  produces a *draw*, not an adjudication. Real games run 60–100 moves (120–200
  plies), so the cap is both too generous and wastes half the compute on a
  meaningless tail.
- No repetition or no-progress detection at all
  ([`game.rs:67`](../crates/game/src/game.rs#L67)). Two similar players shuffle
  marbles until the cap with nothing to stop them.
- No opening randomization, so all 200 games per generation explore the same
  narrow region.

### 2.3 Cause B — fewer simulations than legal moves

`simulations_per_move: 100` against a measured branching factor of 44 (standard
opening), 52 (Belgian Daisy), and ~62 in the midgame (derived: `exp(4.132) = 62.3`,
matching the observed policy loss exactly).

Every root child receives one or two visits. The visit histogram is sampling
noise, not search output.

For comparison: AlphaZero used 800 simulations at chess's ~35 branching (≈23×);
AlphaGo Zero used 1600 at Go's ~250 (≈6×). **This project is at 1.6×.** There is
no usable policy target below roughly 10× branching.

### 2.4 Why the heuristic bootstrap did not rescue it

The `evaluator_schedule: {2: heuristic}` cold-start fix
([`config/medium.yaml:29`](../config/medium.yaml#L29)) was the right instinct, but
it addressed the wrong link in the chain. The heuristic evaluator produces better
*moves*, but from the standard opening at 100 simulations it still cannot force
6 captures inside 400 plies — so it produces the same all-draw data. The
diagnostic that would have caught this (`plies_per_game_avg` = 399.2 in gen 1) was
being logged correctly and simply wasn't acted on.

---

## 3. Silent correctness bugs

### 3.1 🔴 Planes 2 and 3 are swapped between training and inference

**This is the most serious defect in the codebase.**

The Rust encoder that feeds MCTS during self-play and evaluation sets plane 2 =
**own marbles lost**:

```rust
// crates/selfplay/src/encoder.rs:47
let own_lost = game.board.lost(game.turn) as f32 / 6.0;   // -> plane 2
let opp_lost = game.board.lost(game.turn.other()) as f32 / 6.0; // -> plane 3
```

The Python decoder that builds training data sets plane 2 = **own captures made**:

```python
# model/replay_buffer.py:87-95
# `pushed_off_black` = number of black marbles pushed off the
# board (= white's captures).                              <-- this comment is wrong
if int(turn[i]) == 0:  # Black to move
    own_po = int(pushed_off_black[i])
    opp_po = int(pushed_off_white[i])
```

The root cause is the shard column name. `pushed_off_black` holds
`Board::pushed_off[Black]`, which per
[`crates/game/src/board.rs:16`](../crates/game/src/board.rs#L16) is *"number of
white marbles black has pushed"* — i.e. **Black's captures**, not Black's losses.
The name reads naturally as the opposite.

**Verified against real shard data.** Row 254 of
`shards/gen_001/shard_t00_0000.parquet`: Black to move, 13 black marbles on board
(so Black has lost 1):

```
pushed_off_black_col = 0     pushed_off_white_col = 1
replay_buffer assigns:   own_po = 0,  opp_po = 1        <-- inverted
Rust encoder would feed:  own_lost = 1, opp_lost = 0
```

**Impact.** The network is trained believing plane 2 means "captures I have made"
and then played with plane 2 meaning "captures against me". Every material
judgment it learns is sign-flipped at play time. This has not yet visibly bitten
only because all games are draws with ≤1 capture — it becomes the dominant bug
the moment §2 is fixed.

**Fix.** Swap the branches in
[`model/replay_buffer.py:90-95`](../model/replay_buffer.py#L90-L95) *and* rename
the shard columns to `black_losses` / `white_losses` in
[`crates/selfplay/src/shard.rs`](../crates/selfplay/src/shard.rs) so the
ambiguity cannot recur. Then add the cross-language golden test from §4.3.

### 3.2 🔴 The 21-game gate is a 2-game experiment

`eval-match` consumes its RNG only inside the leaf evaluator. Both
`Player::Model` and `Player::Heuristic` ignore it
([`eval_match.rs:142-148`](../crates/selfplay/src/bin/eval_match.rs#L142-L148)),
and every game starts from the same fixed position — so MCTS is fully
deterministic and all games sharing a colour assignment are **byte-identical**.

Proof, from `logs/gen_002_heuristic.log` — all 21 games:

```
a_is_black=true   -> Wins(White)   (11 games, all identical)
a_is_black=false  -> Draw          (10 games, all identical)
```

Two distinct games, replayed 21 times. Every `21 draws` gate result in the run
has the same explanation.

**Fix.** Temperature-sample the first ~10 plies from visit counts (or apply small
root Dirichlet noise in eval), plus randomized openings. Then 21 games is 21
samples.

### 3.3 🟠 The gate metric cannot fire, and would not matter if it did

- `winrate_a = wins_a / games`
  ([`eval_match.rs:263`](../crates/selfplay/src/bin/eval_match.rs#L263)) scores a
  **draw as a loss**. In a drawish game the `gate_threshold: 0.55` is
  mathematically unreachable. Use the standard score `(wins + 0.5·draws) / games`.
- Self-play always loads `state.current_onnx`, which
  [`train_loop.py:1223`](../model/train_loop.py#L1223) sets to the newest
  generation regardless of promotion. So `best.onnx` gates only the web export.
  That is a defensible choice (AlphaZero-2017 behaviour vs. AlphaGo Zero's) but
  the code currently pays ~170 s/generation for gate matches that influence
  nothing.

**Recommendation:** drop per-generation gating entirely. Always self-play with
the latest network, and replace the gate with a fixed **anchor ladder** run every
5 generations (see §7.2). Simpler, cheaper, and a strictly better measurement.

---

## 4. Architecture and boundaries

### 4.1 What is right

The layering is sound and the language boundaries are drawn in the right places:

```
crates/game      pure rules, bitboards, no I/O, no allocation in hot paths
   |-> crates/mcts      PUCT search with a pluggable leaf evaluator
   |     |-> crates/selfplay   trajectory generation, ONNX inference, parquet
   |     |-> crates/wasm       narrow JS boundary (u16 move indices, no serde)
   \-> model/            PyTorch training, ONNX export, run orchestration
```

Specific good calls worth preserving:

- **`LeafEval { value, priors }`** as the evaluator interface. Lets random
  rollout / heuristic / ONNX all plug into the same search with no branching in
  the hot loop.
- **Parquet over a hand-rolled binary format.** The rationale in
  [`shard.rs:5-16`](../crates/selfplay/src/shard.rs#L5-L16) is correct and the
  atomic `.tmp` + rename means the trainer never reads a partial file.
- **Per-thread ORT sessions** instead of `Arc<Mutex<Session>>`, with
  `intra_threads(1)` to stop pool over-subscription. The benchmark table in
  [`ort_eval.rs:55-63`](../crates/selfplay/src/ort_eval.rs#L55-L63) is exactly
  the kind of decision record that should exist.
- **The u16-index WASM boundary.** Opaque `Game` handle, moves as integers, no
  serde, no JSON. Clean and fast.
- **Resumable phase state machine** with atomic fsync'd `state.json`, config-hash
  drift refusal, and partial-shard cleanup on crash recovery.

### 4.2 The structural weakness: duplicated cross-language specs

| Specification | Rust | Python |
| --- | --- | --- |
| Position → planes | [`selfplay/src/encoder.rs`](../crates/selfplay/src/encoder.rs) | [`model/encoder.py:272`](../model/encoder.py#L272) |
| Move index space (2562) | [`game/src/move_index.rs`](../crates/game/src/move_index.rs) | [`model/encoder.py:159`](../model/encoder.py#L159) |
| `MOVE_SPACE`, plane count, ply normalizer | constants | constants |

Each side is independently tested; **neither is tested against the other**. Bug
§3.1 is exactly the failure this predicts, and it survived a full 43-test Python
suite that exercises the encoder heavily.

### 4.3 Recommended fix: a golden-file conformance test

Add a small Rust binary that dumps ~200 `(position, planes, legal move indices)`
triples to JSON, and a pytest that asserts Python reproduces them byte-for-byte.
Cheap to build, makes this entire bug class impossible, and doubles as
documentation of the wire format.

### 4.4 Other structural notes

- **Shards have no `game_id` column.** Rows can only be split back into games by
  detecting `ply` resets. That happens to work today (one thread writes
  sequentially) but is fragile, and it blocks the game-review tool (§9). Add
  `game_id: u32` and `seed: u64` — cheap, and it makes shards self-describing.
- **`config_hash` does not cover code.** Changing `MAX_PLIES` or the plane layout
  silently invalidates every shard in the replay buffer with no detection.
  `_git_short_sha()` is already computed for the banner
  ([`train_loop.py:104`](../model/train_loop.py#L104)) — record it in
  `state.json` and warn on mismatch at resume.
- **Stale doc comments.** The `eval_match.rs` module doc still describes the
  abandoned `Arc<Mutex<Session>>` design and is contradicted by an inline comment
  20 lines later. `moves.rs`'s "anchored at the low end of a group" is wrong — the
  anchor is the group's *rear* relative to the direction of travel.
- **Dead code.** [`encoder.py:330-335`](../model/encoder.py#L330-L335) has a loop
  whose body does nothing. `Phase` in `state.py` declares `heuristic_eval` /
  `random_eval` values that are never assigned.

---

## 5. MCTS engine

### 5.1 The throughput ceiling

Derived from the run: `200 games × 399 plies × 100 sims / 1453 s` ≈
**5,490 NN evaluations/second** across 9 threads.

Budget arithmetic for what §2 requires:

| scenario | evals/gen | time/gen at 5.5k/s | time/gen at 6× |
| --- | --- | --- | --- |
| current (200 games × 399 ply × 100 sims) | 8.0 M | 24 min | — |
| after adjudication (200 × ~150 ply × 400 sims) | 12.0 M | 36 min | 6 min |
| validation scale (60 × ~150 ply × 400 sims) | 3.6 M | 11 min | 2 min |

So a short validation run is affordable *today* at reduced games/generation, but
a real 50-generation run is not without the speedups below.

### 5.2 Batch the leaf evaluations — the single biggest lever

[`mcts/src/lib.rs:151`](../crates/mcts/src/lib.rs#L151) calls `session.run` once
per simulation at **batch size 1**. This is the largest inefficiency in the
project.

The standard fix is to collect 16–64 leaves per NN call using virtual loss and
evaluate them as one batch. Typically 5–15× on CPU, considerably more on
CoreML/GPU. Note this would also **invert the CoreML benchmark** in
[`ort_eval.rs:55-63`](../crates/selfplay/src/ort_eval.rs#L55-L63): ANE loses
today purely on per-call overhead, which batching amortizes away. The per-thread
session design is a good workaround for a problem that batching mostly dissolves.

### 5.3 Other search improvements

- **Subtree reuse.** Every `play_game` move discards the whole tree
  ([`selfplay/src/lib.rs:172`](../crates/selfplay/src/lib.rs#L172)). Re-rooting on
  the played child is nearly free and recovers a meaningful share of visits.
- **Node representation.** Each expansion eagerly allocates ~60–80 `Node`s, each
  holding a full `Game` (48 B) *and* its own `children: Vec`
  ([`mcts/src/lib.rs:225`](../crates/mcts/src/lib.rs#L225)). At 100 simulations
  that is ~6–8k nodes and ~100 heap allocations per move. A flat arena with
  `(child_start, child_len)` index ranges and lazily-materialised states removes
  allocation from the hot path entirely.
- **First-play urgency.** Unvisited children get `Q = 0`
  ([`mcts/src/lib.rs:249`](../crates/mcts/src/lib.rs#L249)) — optimistic when
  losing, pessimistic when winning. "Parent Q minus a small reduction" is a
  well-tested improvement and is roughly five lines.
- **Dirichlet plumbing is fragile.** Root noise is applied by intercepting "the
  first `eval_fn` call"
  ([`selfplay/src/lib.rs:160-171`](../crates/selfplay/src/lib.rs#L160-L171)),
  which silently depends on `search`'s internal call ordering. Move it into
  `SearchConfig` as an explicit root-noise parameter.
- **`MAX_LEGAL = 256`** is an `ArrayVec` that panics on overflow. Observed max is
  ~52–80 so there is headroom, but `try_push` with a hard error is cheaper than
  a production panic.

---

## 6. Network architecture and modeling

### 6.1 🟠 92% of parameters are in one dense layer

`policy_fc` is `16·81 → 2562` = **3.32 M parameters**. The entire 4×64 residual
tower is ~300 k. This is a large linear readout bolted onto a tiny representation
— almost exactly backwards for an AlphaZero network.

**The move encoding already has the right shape to fix this.** Both halves of the
index space are anchor-major:

```
inline     idx = anchor_compact · 18 + dir · 3     + (size − 1)
broadside  idx = anchor_compact · 24 + gi · 8 + mi · 2 + (size − 2) + 1098
```

So the whole 2562-space is exactly a **(42, 9, 9) spatial tensor** — 18 inline
planes plus 24 broadside planes — gathered through a fixed index table
(`plane · 81 + COMPACT_TO_CELL[anchor]`). 42 × 61 = 2562 exactly.

Replace `policy_fc` with a `1×1 conv 64 → 42`:

| | current | proposed |
| --- | --- | --- |
| policy head params | 3,321,282 | 2,730 |
| equivariance | none | translation-equivariant |
| budget freed for the tower | — | ~3.3 M |

That budget supports an 8–10 block × 96–128 channel tower. This is the same trick
AlphaZero-chess uses with its 73×8×8 policy head, it composes properly with the
D6 augmentation, and I expect it to be the largest *modeling* win available.
(A 3×3 conv instead of 1×1 gives each anchor a little more local context for
~9× the head params — still negligible.)

### 6.2 Encoding — mostly right

- **The 3×3 conv on axial coordinates is correct.** All six hex neighbours fall
  inside the 3×3 kernel: `(0,±1)` = E/W, `(±1,±1)` = NE/SW, `(±1,0)` = NW/SE. The
  two extra corners are distance-2 non-neighbours the network can learn to ignore.
  Good choice — worth a comment so nobody "fixes" it later.
- **POV-relative own/opp planes plus D6 augmentation is the right call.** Since
  D6 includes the 180° rotation, the network never has to learn both board
  orientations separately. No side-to-move embedding needed.
- **Off-board cells leak.** 20 of 81 cells are invalid; plane 5 identifies them
  but nothing stops activations bleeding through. Multiplying by the mask after
  each residual block is cheap insurance. (It also means ~25% of conv FLOPs are
  wasted on dead cells — acceptable for the simplicity.)
- **The ply plane is coupled to `MAX_PLIES` in two places.**
  `ply.min(400)/400` appears in both
  [`selfplay/src/encoder.rs:49`](../crates/selfplay/src/encoder.rs#L49) and
  [`model/encoder.py:279`](../model/encoder.py#L279). Changing the cap (§2.2)
  requires changing both, and invalidates every existing shard. Another argument
  for §4.3.
- **Value head bottleneck** of 64 → 1 channel is very tight. 2–4 channels costs
  almost nothing.

### 6.3 Loss and optimizer

- The z/q blended value target
  ([`train_step.py:35`](../model/train_step.py#L35)) is a good KataGo-style
  choice, and the linear ramp is sensible.
- `torch.optim.Adam(weight_decay=…)` is L2-in-Adam, not decoupled — use `AdamW`.
- **No learning-rate schedule at all.** Constant `1e-3` for the whole run. A step
  decay at fixed generation milestones is standard and matters.
- `grad_clip=1.0` is aggressive for AlphaZero but harmless.

---

## 7. Training harness

### 7.1 The data pipeline will be the next bottleneck

`ReplayBuffer.sample` ([`replay_buffer.py:210`](../model/replay_buffer.py#L210))
is a pure-Python loop over the batch, and inside it:

- `apply_sym_to_planes` rebuilds an 81-element inverse permutation **per example**
  ([`encoder.py:299`](../model/encoder.py#L299)) — should be precomputed once at
  import as `CELL_PERM_INV`;
- two dense 2562-float vectors are allocated per example;
- three scatter operations run per example.

At batch 256 that is roughly 20 k Python-level operations per SGD step,
serialized with the step itself. Vectorizing the gather over the batch axis and
moving sampling to a background thread should be 10×+.

**Memory.** Chunks store dense `float32` planes at 1,944 B/position. The run held
240 k positions ≈ **470 MB**, and `replay_buffer_gens: 12–20` is designed to grow
that further. Store raw bitboards and decode on sample with vectorized
`np.unpackbits` — roughly 30× less memory.

**Quadratic ingest.** `_concat` re-copies the entire generation chunk on every
shard ingest ([`replay_buffer.py:156`](../model/replay_buffer.py#L156)). With
`shard_games_per_file: 1` that is 200 growing copies per generation.

### 7.2 🟠 No held-out evaluation

Every metric logged is training loss on the replay buffer. There is no validation
set, no policy top-1 agreement, no value calibration, no Elo.

A frozen validation set of a few thousand positions reporting **policy top-1
agreement with the MCTS choice** and **value MSE** would have surfaced this
collapse in minutes instead of after three 24-minute generations. This is the
cheapest possible instrumentation and it is the single highest-value addition to
the harness.

Pair it with an **anchor ladder** — `random`, `heuristic@100 sims`,
`heuristic@800 sims` — run every 5 generations, converted to an Elo estimate.
Fixed opponents give a monotone progress curve that self-play gating cannot.

### 7.3 Smaller harness notes

- **Wasted wall-clock.** When training hits `steps_per_gen_max` and self-play is
  still running, the loop just polls
  ([`train_loop.py:635-641`](../model/train_loop.py#L635-L641)). The trainer also
  competes for CPU with a self-play process already using `cores − 1` threads.
- **Retention vs. `best.onnx` on non-symlink filesystems.** `_link_or_copy` falls
  back to a plain copy; in that case `best.onnx.resolve()` no longer protects the
  generation file it was copied from, and `_retain_checkpoints` can delete a file
  `state.best_onnx` still points at.
- `_retain_checkpoints` takes an unused `current_gen`; retention configs are
  passed as `asdict(...)` dicts, discarding the dataclass typing they were
  defined with.

---

## 8. Rules engine

Reviewed in detail; **no correctness defects found.**

- Bitboard direction shifts handle row wraparound correctly — the `VALID_MASK`
  and `|q − r| ≤ 4` predicate mean E/W wraps always land off-board.
- Push legality is right: strict outnumbering, 1-vs-1 and 2-vs-2 correctly
  blocked, 3-vs-2 with off-board rear correctly drops one marble.
- `apply_inline`'s two-opponent case is correct (front opponent leaves first,
  rear advances into its slot).
- Move canonicalization via `POSITIVE_DIRS` anchoring genuinely removes the
  `(start, end)` / `(end, start)` broadside ambiguity; `no_duplicate_moves` and
  the encode/decode round-trip over all 2562 indices back this up.
- `belgian_daisy()` is a correct daisy (verified: each cluster is a centre cell
  plus its six neighbours).

Performance is healthy: 1.2–1.6 M `legal_moves`/s, 176 M `apply`/s.

The one gap is **notation**. [`notation.rs`](../crates/game/src/notation.rs)
emits engine-internal forms (`A1E:2`, `A1-A3:NE`) and the module doc says as
much. A game-review tool needs standard Abalone notation plus a PGN-equivalent
container — see §9.

---

## 9. The web tool

Against the stated goal — *"a web tool for reviewing games and particular
spots"* — what exists today is a **playground**, not a review tool.

What is good: the WASM boundary is clean, the drag-with-snap UX and sumito
preview are thoughtful, and the eval bar plus top-5 analysis panel work.

The gaps:

| Need | Status |
| --- | --- |
| Plays the trained model | ❌ `analyze()` runs *heuristic* MCTS ([`wasm/src/lib.rs:235`](../crates/wasm/src/lib.rs#L235)); `web_export_path` points at `web/public/models/best.onnx` and `web/public/` does not exist |
| Load a game | ❌ no loader, no format |
| Ply navigation | ❌ no move list, no back/forward, no jump-to-ply |
| Serialization | ❌ engine-internal notation only |
| Surface training data | ❌ nothing reads the shards |

**The most valuable thing this UI could do is also the easiest.** Every self-play
position already carries its full MCTS visit distribution and root Q in the
parquet shards. A "load self-play game *N*, scrub through plies, see what search
actually thought at each one" view is mostly plumbing — and it is a **debugging
tool for the training pipeline**, not just a product feature. Given that the
pipeline silently produced three generations of garbage, that has immediate
diagnostic value. This is why the roadmap below promotes it ahead of the
full-featured review tool.

---

# 10. Roadmap

Re-reviewed against the original priority ordering. Four changes from a naive
"fix bugs → speed → model → product" sequence:

1. **A de-risking experiment comes first.** The Belgian Daisy + adjudication
   change is the largest single change proposed here and rests on a hypothesis.
   It costs ~30 minutes to test the hypothesis directly first.
2. **Observability moves ahead of throughput.** The collapse was detectable in
   minutes with a validation set. Building that before making the loop faster
   means the faster loop is also legible.
3. **The shard viewer moves up.** It is a pipeline debugger, not just product
   work.
4. **Gating gets deleted rather than fixed.** Fixing it costs more than replacing
   it with an anchor ladder that measures the thing actually wanted.

Effort estimates assume focused solo work.

---

### Phase 0 — De-risk the hypothesis · ~half a day

Before committing to the largest change, measure it.

| # | Task | File(s) |
| --- | --- | --- |
| 0.1 | Add `--opening {standard,belgian}` and `--max-plies N` to `selfplay-batch`; thread through to `play_game` | [`selfplay/src/bin/selfplay_batch.rs`](../crates/selfplay/src/bin/selfplay_batch.rs), [`selfplay/src/lib.rs:146`](../crates/selfplay/src/lib.rs#L146) |
| 0.2 | Run a 2×2 matrix — {standard, belgian} × {400-ply, 200-ply} — 50 heuristic games each at 400 sims | — |
| 0.3 | Record decisive rate, mean plies, mean capture differential at termination | — |

**Acceptance:** Belgian Daisy at 200 plies yields **> 40% decisive games**.

**If it fails:** the drawishness is deeper than the opening, and the fix is
adjudication-by-score at the cap (Phase 1.4) rather than a better opening. Either
way this half-day tells you which lever matters, and the numbers become the
baseline for every later comparison.

---

### Phase 1 — Correctness · 1–2 days

Nothing downstream is trustworthy until these land.

| # | Task | File(s) |
| --- | --- | --- |
| 1.1 | Swap the `pushed_off_*` branches so plane 2 = own losses | [`replay_buffer.py:90-95`](../model/replay_buffer.py#L90-L95) |
| 1.2 | Rename shard columns `pushed_off_black`/`_white` → `black_losses`/`white_losses` | [`shard.rs`](../crates/selfplay/src/shard.rs), [`replay_buffer.py`](../model/replay_buffer.py) |
| 1.3 | Add `game_id: u32` and `seed: u64` columns to the shard schema | [`shard.rs`](../crates/selfplay/src/shard.rs) |
| 1.4 | Golden-file conformance test: Rust binary dumps 200 `(position, planes, legal indices)` triples; pytest asserts byte-equality | new `crates/selfplay/src/bin/dump_golden.rs`, new `tests/test_conformance.py` |
| 1.5 | Score draws as 0.5 in `winrate_a` | [`eval_match.rs:263`](../crates/selfplay/src/bin/eval_match.rs#L263) |
| 1.6 | Record git SHA in `state.json`; warn on mismatch at resume | [`state.py`](../model/state.py), [`train_loop.py`](../model/train_loop.py) |

**Acceptance:** the golden test passes; it **fails** if you deliberately
re-introduce the plane swap. All existing tests still green.

**Note:** all existing shards become invalid (column rename + semantics change).
Start a fresh run; do not attempt to resume `joyful-panther`.

---

### Phase 2 — Restore the training signal · 2–3 days

| # | Task | File(s) |
| --- | --- | --- |
| 2.1 | Default self-play to the Phase-0-winning opening; expose as config | [`config.py`](../model/config.py), [`selfplay/src/lib.rs`](../crates/selfplay/src/lib.rs) |
| 2.2 | Adjudicate at the ply cap by capture differential, tie-broken by centrality; keep true `Draw` only for exact ties | [`game.rs:67`](../crates/game/src/game.rs#L67) |
| 2.3 | Lower `MAX_PLIES` to 200; make it a config value, not a constant | [`game.rs:17`](../crates/game/src/game.rs#L17) |
| 2.4 | Update the ply-plane normalizer in **both** encoders to track the new cap | [`selfplay/src/encoder.rs:49`](../crates/selfplay/src/encoder.rs#L49), [`encoder.py:279`](../model/encoder.py#L279) |
| 2.5 | No-progress rule: adjudicate after N plies without a capture | [`game.rs`](../crates/game/src/game.rs) |
| 2.6 | Randomize openings: 1–2 random plies, or sample a small opening book | [`selfplay/src/lib.rs`](../crates/selfplay/src/lib.rs) |
| 2.7 | Raise `simulations_per_move` to 400 in a new `config/validation.yaml` (60 games/gen, 6 gens) | new `config/validation.yaml` |

**Acceptance** — run `config/validation.yaml` for 6 generations (~1 hour) and
require all four:

- `plies_per_game_avg` < 180
- decisive rate > 40%
- `train_loss_value` > 0.05 and moving
- `train_loss_policy` < `ln(mean branching) − 0.3`

**If any fail, stop and diagnose. Do not scale up.** This is the gate that the
previous run lacked.

---

### Phase 3 — Observability · 2 days

| # | Task | File(s) |
| --- | --- | --- |
| 3.1 | Frozen validation set (~5k positions from a held-out generation), evaluated every N steps | new `model/validate.py` |
| 3.2 | Log val policy top-1 agreement, val value MSE, and value-prediction histogram to TensorBoard | [`train_loop.py`](../model/train_loop.py) |
| 3.3 | Delete per-generation gating; always self-play with the latest network | [`train_loop.py:1199-1221`](../model/train_loop.py#L1199-L1221) |
| 3.4 | Anchor ladder every 5 gens: `random`, `heuristic@100`, `heuristic@800`; convert to Elo | [`eval.py`](../model/eval.py), [`train_loop.py`](../model/train_loop.py) |
| 3.5 | Add opening/temperature variation to `eval-match` so N games are N samples | [`eval_match.rs`](../crates/selfplay/src/bin/eval_match.rs) |
| 3.6 | Shard → JSON exporter (one game: positions, visit distributions, Q, outcome) | new `model/export_game.py` |

**Acceptance:** a training run produces a monotone Elo-vs-generation curve, and
`export_game.py` round-trips a self-play game to JSON that replays correctly
through the WASM engine.

---

### Phase 4 — Throughput · 3–5 days

Required before any run longer than ~10 generations.

| # | Task | Expected gain |
| --- | --- | --- |
| 4.1 | Batched leaf evaluation with virtual loss (16–64 leaves/NN call) | **5–15×** |
| 4.2 | Vectorize `ReplayBuffer.sample`; precompute `CELL_PERM_INV`; background sampling thread | 10× on the data path |
| 4.3 | Store bitboards not dense planes in the buffer; decode via `np.unpackbits` | ~30× memory |
| 4.4 | Subtree reuse between moves in `play_game` | 10–30% effective sims |
| 4.5 | Flat node arena with `(child_start, child_len)` ranges | removes hot-path allocation |
| 4.6 | Re-benchmark CoreML at batch 32+ — batching may invert the current CPU/ANE result | possibly large |

**Acceptance:** ≥ 30k NN evals/second aggregate (from 5,490 today), and a
generation at 200 games × 800 sims completing in under 15 minutes.

---

### Phase 5 — Model capacity · 2–3 days

| # | Task | File(s) |
| --- | --- | --- |
| 5.1 | Convolutional policy head: `1×1 conv 64 → 42`, gathered to 2562 via a fixed index table | [`abalone_net.py`](../model/abalone_net.py), [`encoder.py`](../model/encoder.py) |
| 5.2 | Update the ONNX contract and Rust-side extraction for the new head shape | [`export_onnx.py`](../model/export_onnx.py), [`ort_eval.rs`](../crates/selfplay/src/ort_eval.rs) |
| 5.3 | Grow the tower to 8–10 blocks × 96–128 channels with the freed budget | [`abalone_net.py`](../model/abalone_net.py) |
| 5.4 | Mask off-board cells after each residual block | [`abalone_net.py`](../model/abalone_net.py) |
| 5.5 | `AdamW` + step LR decay at generation milestones | [`train_loop.py`](../model/train_loop.py), [`config.py`](../model/config.py) |
| 5.6 | Widen the value head bottleneck to 2–4 channels | [`abalone_net.py`](../model/abalone_net.py) |

**Acceptance:** at matched parameter count and matched wall-clock, the new
architecture beats the old one on the Phase 3 anchor ladder.

---

### Phase 6 — Game review tool · ~1 week

| # | Task |
| --- | --- |
| 6.1 | Standard Abalone move notation (display + parse), alongside the existing engine form |
| 6.2 | A game file format (JSON: opening, move list, per-ply eval/visits, result) emitted directly by self-play |
| 6.3 | `onnxruntime-web` in the browser so the UI plays the *trained* network, not the heuristic; create `web/public/models/` |
| 6.4 | Game loader + move list + ply scrubbing + keyboard navigation |
| 6.5 | Per-position analysis view sourced from shard data: what search thought at the time vs. what the current network thinks now |
| 6.6 | Position permalinks (encode board state in the URL) for sharing "particular spots" |

**Acceptance:** load any self-play game from any generation, scrub it, and
compare the network's opinion at training time against the latest network's.

---

## 11. Decisions needed

Points where the roadmap assumes an answer that is really yours to give:

| # | Decision | Recommendation |
| --- | --- | --- |
| D1 | Opening: Belgian Daisy only, or mixed with standard? | Belgian for training; keep standard playable in the UI |
| D2 | Ply cap value | 200, pending Phase 0 data |
| D3 | Adjudication rule at the cap | Capture differential, tie-broken by centrality |
| D4 | Keep gating, or switch to the anchor ladder? | Switch — cheaper and measures the right thing |
| D5 | Target strength | "Beats `heuristic@800 sims` at equal simulation count" is a concrete, honest milestone |

---

## 12. Expectations

Full AlphaZero scale is not reachable on a single M1 Pro, and it is worth setting
the bar accordingly. With Phases 0–5 complete, a realistic target is a network
that clearly beats the hand-written heuristic at equal simulation counts and
plays recognisably purposeful Abalone — coherent group formation, sumito threats,
edge avoidance. That is an achievable and genuinely satisfying result.

Reaching it requires the loop to actually close. Right now it does not, and the
instrumentation to notice that was one validation set away.
