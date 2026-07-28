# MODEL — target design

> **Status:** aspirational design doc. Describes what we are building toward, not
> what exists today. For the current state and the path from here to there, see
> [2026-07-27-architecture-review.md](2026-07-27-architecture-review.md).

---

## 1. Design principles

1. **No hand-written positional knowledge.** No centrality term, no cohesion
   term, no tuned weights. Everything the network knows about *how to play*
   Abalone must come from self-play outcomes.
2. **The game's own score is not a heuristic.** Abalone's win condition is a
   counter: push off six. Using that counter to resolve a truncated game is
   reading the rulebook, not injecting strategy. (§3.1 defends this line.)
3. **Auxiliary supervision must be derivable from the trajectory.** Extra
   training targets are allowed if they are computed *from the game record
   itself*. They densify the gradient without encoding anyone's opinion.
4. **Solve cold-start with the curriculum, not with a teacher.** Where AlphaGo
   used human games and the current code uses a hand-written evaluator, we
   instead move the *starting position* closer to a decision.
5. **Scale down before scaling up.** Every design choice must have a small,
   fast configuration for iteration and a large one for the real run.

The hand-written evaluator (`crates/mcts/src/eval.rs`) is **retired from the
training loop entirely**. It survives in exactly one role: a fixed benchmark
opponent on the Elo ladder. A frozen yardstick is not a teacher.

---

## 2. The cold-start problem, stated precisely

This is the problem the whole design turns on.

A randomly initialised network gives MCTS a uniform prior and a meaningless
value. Under such a policy, Abalone games from the standard opening essentially
never reach six captures — measured at **98% draws over 200 games**, even with
search driving move choice. So `z = 0` everywhere, the value head learns the
constant zero, and there is nothing for the policy to distil.

### 2.1 Does deeper MCTS fix this on its own?

Partly — and it is worth being precise about which half.

**Deeper search does fix the policy signal.** With branching ~60 and 100
simulations, every root child gets one or two visits and the visit histogram is
noise. Search needs roughly **10× the branching factor** before visit counts
concentrate into a usable target. That alone is a decisive argument for 800+
simulations and it costs nothing conceptually.

**Deeper search does not fix the value signal from the opening.** MCTS can only
discover what lies inside its horizon. At 800 simulations with branching 60, the
tree is a few ply wide-deep plus maybe 10–15 ply along the principal variation.
Under near-random play a capture occurs roughly once per 100+ plies and six are
needed to terminate. Terminal states are simply not in the horizon, so there is
nothing for search to find and back up. Multiplying simulations by 10 moves the
horizon by a couple of ply against a gap of hundreds.

**The conclusion that shapes this design:** deep search converts *reachable*
outcomes into learning signal extremely well. Our job is therefore not to search
harder from a hopeless position — it is to **make outcomes reachable**, then
search deeply. Two mechanisms do that, and neither is a heuristic:

- **§3** — treat Abalone as a scored game, so truncation yields an outcome.
- **§4** — seed self-play near the capture threshold, so terminals are inside
  the search horizon from generation one.

---

## 3. Game termination and the outcome signal

### 3.1 Abalone is a scored game

Current rule: first to six captures wins; at the ply cap, **draw**. That draw is
our invention — it is not in the rules, it is what we do when we run out of
patience. It discards the one quantity the game actually tracks.

Target rule:

| Condition | Result |
| --- | --- |
| `captures[side] ≥ 6` | `side` wins, score difference `d = +6` |
| ply cap reached, `d ≠ 0` | winner is `sign(d)`, score difference `d` |
| ply cap reached, `d = 0` | genuine draw |
| no capture in `K` plies | adjudicate as above (optional; see §3.3) |

where `d = captures[side] − captures[other]`.

**Why this is not a heuristic.** The distinction that matters is *objective*
versus *strategy*. Capture count is the game's own scoring quantity — the thing
the win condition counts. Resolving a truncated game by score is precisely what
Go does with area scoring and what every scored game does. It says nothing about
*how* to get captures: nothing about the centre, nothing about group cohesion,
nothing about edge danger. Contrast the evaluator we are deleting, whose
`Weights { w_capture, w_center, w_cohesion }` encodes three hand-picked beliefs
about what makes a position good. That is the line, and score-based adjudication
sits firmly on the safe side of it.

**It also self-anneals.** As play improves, more games end naturally at six
captures and adjudication fires less often. Its influence decays to zero on its
own, with no unlearning phase and no schedule to tune.

### 3.2 Shorter games

Cap at **200 plies**, not 400. Real Abalone runs 60–100 moves. The current cap
spends half of every game's compute on a tail that under weak play is pure noise.
Halving it roughly doubles throughput and makes the capture differential a
tighter, less-diluted signal.

### 3.3 No-progress rule

Optionally adjudicate after `K` plies without a capture (K ≈ 80). Same class of
rule as the ply cap — a horizon device, not strategy. With a 200-ply cap it is
close to redundant; keep it configurable and off by default.

### 3.4 Graded value targets

Do not throw away the magnitude. A 4–1 game is more informative than a 1–0 game,
and the network should see the difference. See the value and score heads in §6.

---

## 4. Curriculum: capture-handicap seeding

**The mechanism that replaces the heuristic bootstrap.**

The win condition is a counter, which means we can start a game *near* the
threshold without knowing anything about good play. For a fraction of self-play
games, sample a starting handicap:

```
a, b  ~  chosen from 0..=5              (independently, per game)
set   captures[Black] = a, captures[White] = b
remove b marbles from Black at random, a from White at random
```

The position stays perfectly consistent — a side that has conceded 5 has 9
marbles on the board — and **which** marbles are removed is uniformly random, so
no positional judgment enters. At `(5, 5)` the very next capture ends the game.

Why this works where a heuristic teacher does not:

- **Terminals land inside the search horizon immediately.** At a 1-capture
  distance, an 800-simulation search genuinely finds forced wins. That is real
  signal in generation one, from search rather than from an author's opinion.
- **It teaches the endgame first, then propagates backward.** Classic AlphaZero
  value bootstrapping — the tail is learned, and each generation extends
  competence a horizon further toward the opening. Handicap seeding just gives
  that process a place to start.
- **It is a pure data-distribution intervention.** No term in the loss, no term
  in the evaluator, nothing to unlearn. It changes which positions we visit, not
  what we believe about them.
- **It fixes an exploration gap that never closes on its own.** Positions at 5–4
  captures are strategically critical and, under self-play from the standard
  start, essentially never visited.

**Opening diversity.** Independently of handicap: play from Belgian Daisy
(clusters start in contact, far livelier than the standard layout) and randomise
the first 1–2 plies. Both are knowledge-free and both decorrelate the 200 games
per generation that currently explore one narrow corridor.

### 4.1 Annealing — retiring the crutch

Start with ~70% of games seeded, uniformly over handicap levels, and ratchet
that down toward a permanent floor as the network learns to finish games on its
own. Left static, a 0.7 rate would still be spending 70% of generation 40's
compute on artificially-seeded positions instead of the ones real play produces.

**The control signal is the natural termination rate of unseeded games**: of the
games recorded with `handicap == (0, 0)`, the fraction that reach six captures
**before** the ply cap.

**Decisive rate was considered and rejected.** It is the intuitive choice and it
is wrong. Measured over 200 uniformly-random playouts, Belgian Daisy with
adjudication and no handicap is already **63% decisive**, with a mean margin of
0.89 captures — because adjudication at the ply cap resolves nearly any game on
one lucky push. Any defensible threshold on decisive rate therefore fires in
generation one, against a *random* network, and pulls the crutch before it has
done anything. Over those same playouts the natural termination rate was
**0.0%** — mean plies came out at exactly the cap (400.0 / 200.0 / 200.0), so not
one game in 200 ended on its own. That number rises only when the network can
genuinely close a game out, which is precisely the condition under which the
endgame curriculum has stopped earning its keep.

**Detecting it.** With the no-progress rule off (the default),
`Game::state()` has exactly two exits: six captures, or `ply >= max_plies`
adjudicated on capture differential. So a game whose recorded ply count is short
of its own `max_plies` ended on captures — no flag needed, and none exists in
the shard schema. With the no-progress rule *on*, a third exit stops the game
early without six captures and the shards cannot tell the two apart after the
fact (`black_losses`/`white_losses` are recorded before each move, so the
capture that ends the game is in no row). In that case the controller reports
the signal as unmeasurable and holds the rate for the whole run.

**The rule** (`model/curriculum.py`, evaluated once per generation after
self-play, applying from the next generation on):

```
mode: off          rate unchanged
mode: schedule     rate = most recent {gen: rate} entry with key <= gen
mode: controller   unseeded_games < min_unseeded_games  → hold (too noisy)
                   natural_termination_rate >= target   → rate = max(floor, rate − Δ)
                   otherwise                            → hold

    where  Δ = step × clamp(natural_termination_rate / target, 1, max_step_multiple)
```

Two invariants hold in every mode: the rate never increases, and it never lands
below `floor`. A curriculum does not go backwards — if the network regresses,
holding is right and re-teaching endgames it already knows is not — and a
monotone rule cannot oscillate, which is what makes a controller this crude safe
to leave running for 50 generations. One *decision* per generation: each step
changes the distribution the next measurement is taken from.

**The step is proportional to how far above target the signal sits.** A fixed
step ignores the magnitude of its own error signal, and the six-generation
validation run showed what that costs: `natural_termination_rate` reached 1.00
at generation 4 — *four times* the 0.25 target, completely saturated — and the
controller still moved 0.05 a generation, going 0.70 → 0.50 over six
generations with eight more needed to reach the floor. The whole run was spent
retiring a crutch the network had visibly outgrown. At
`max_step_multiple: 4.0` the same signal retires it in three generations. The
clamp is what keeps this a ratchet rather than a controller with gain: the step
never shrinks below the configured one, and never exceeds K of them, so one
lucky generation cannot dump the entire curriculum.

`floor: 0.10` is deliberately non-zero. Positions at 5–4 captures are
strategically critical and essentially never reached by self-play from a fresh
start, so a permanent trickle of seeded games is worth keeping purely as endgame
coverage.

```yaml
self_play:
  handicap_rate: 0.7          # initial value only; also the fixed value when mode is off
  handicap_max: 5
  handicap_anneal:
    mode: controller          # controller | schedule | off
    target_natural_termination: 0.25
    step: 0.05                # at a signal sitting exactly on target
    max_step_multiple: 4.0    # step × clamp(signal/target, 1, K)
    floor: 0.10
    min_unseeded_games: 20
    schedule: {}              # {gen: rate}, used only when mode == schedule
```

**Where the live rate lives.** In `state.json`, not in the config. The config's
`handicap_rate` is an initial condition and is inside `config_hash`; the
annealed value is persisted per-run in `RunState.handicap_rate` and restored on
resume, so a resumed run continues the curriculum instead of silently
restarting it at 0.7. Nothing writes the annealed value back into the config —
that would make every resume after the first step refuse on a hash mismatch.

**Sample size.** `min_unseeded_games: 20` is what stops the controller acting on
noise: 19 unseeded games at 100% natural termination is one plausible run of
luck, not evidence. It is checked *before* the target, so a small sample holds
however good it looks. At 200 games a generation and rate 0.7 there are ~60
unseeded games, comfortably clear; the 60-game predecessor to
`config/validation.yaml` had ~18 and correctly held every generation.

**What a run logs.** Every generation, one line naming the current rate, the
unseeded game count, the natural termination rate, the target, and whether the
controller stepped or why it held. `handicap_rate` (and the next rate, the
sample size and the signal) go to `metrics.jsonl` and TensorBoard; the
seeded-vs-unseeded split of decisive rate and mean |score| is on the data-health
lines beside it.

---

## 5. Input representation

`(B, 14, 9, 9)` float32. The 9×9 axial grid holds 61 valid cells; 20 slots are
off-board.

| Plane | Contents |
| --- | --- |
| 0 | own marbles (binary) |
| 1 | opponent marbles (binary) |
| 2–6 | own losses, thermometer: `own_losses ≥ 1 … ≥ 5` |
| 7–11 | opponent losses, thermometer: `opp_losses ≥ 1 … ≥ 5` |
| 12 | `ply / max_plies` |
| 13 | valid-cell mask |

**Side-to-move relative.** Planes 0/1 are own/opponent, never black/white, so one
network plays both colours with no side-to-move embedding.

**Thermometer, not scalar, for the counters.** Losses are an *ordinal* quantity
with a hard threshold at 6. `≥1, ≥2, …` lets a single linear layer read off both
the magnitude and "one away from losing", which a `count/6` scalar makes the
network work for. Five planes each is cheap at 9×9.

**No history planes.** Abalone is Markov given board, capture counters and ply —
there is no repetition rule and no castling/en-passant analogue. This is a real
simplification over chess and Go and we should take it. (Two previous frames as
tactical context is a legitimate experiment, but it must earn its place.)

**Hex geometry is already handled.** In axial coordinates all six hex neighbours
fall inside a standard 3×3 kernel: `(0,±1)` = E/W, `(±1,±1)` = NE/SW, `(±1,0)` =
NW/SE. The two extra kernel corners are distance-2 non-neighbours the network
learns to ignore. Plain `Conv2d` is correct here — this should be commented in
the code so nobody "fixes" it.

**Off-board cells are masked after every block**, not merely flagged on input, so
activations cannot bleed through the 20 dead slots.

---

## 6. Network

```
                    (B, 14, 9, 9)
                          |
                     stem 3×3 conv -> C channels, BN, ReLU
                          |
                    N × ResidualBlock(C)      [+ optional squeeze-excite]
                          |
        +-----------+-----+------+-------------+
        |           |            |             |
   policy head  value head  score head   capture-map head
   (42,9,9)      3-way        13-way       (2,9,9)
        |
   gather -> 2562 logits
```

### 6.0 A note on "value" vs "score"

Two heads model outcome-ish quantities and the names are easy to confuse. They
are distinct and both are kept:

| Head | Quantity | Range | Role |
| --- | --- | --- | --- |
| `value` | **Who wins** — the game-theoretic outcome | `softmax` over (win, draw, loss) | The RL value function. This is what MCTS backs up; `E[value] = P(win) − P(loss)`. |
| `score` | **By how much** — final capture differential | `softmax` over `d ∈ [−6, +6]` | Auxiliary target. Denser gradient than a 3-way outcome, and the natural number to show a human. |

Put another way: `value` is *score* in the reinforcement-learning sense (expected
return), while `score` is *score* in the Abalone sense (the marble count the rules
track). Only `value` participates in search. `score` exists because the difference
between winning 6–0 and 6–5 is real information that a 3-way outcome discards,
and because "expected +1.4 marbles" is far more legible in the review UI than
"+0.31".

These names are load-bearing: they appear in the ONNX output signature
([ARCHITECTURE §5.3](ARCHITECTURE.md#53-onnx-signature)), the shard schema, and
the loss terms in §6.6.

### 6.1 Trunk

`N` blocks × `C` channels, pre-activation residual blocks, stride 1 throughout.

| Configuration | N × C | ~params | Use |
| --- | --- | --- | --- |
| `small` | 6 × 96 | ~1.0 M | fast iteration, CI, smoke runs |
| `base` | 10 × 128 | ~3.0 M | default training runs |
| `large` | 14 × 192 | ~9.3 M | final run, if throughput allows |

**Squeeze-and-excitation is a natural fit here** and worth evaluating early.
Abalone's most important state is partly *global* — the capture counters, overall
material, whether either side is one push from losing. SE blocks let global
pooled context modulate channels directly rather than forcing that information to
propagate spatially through the trunk.

### 6.2 Policy head — convolutional, not dense

The current dense head holds **92% of all parameters** in a single
`16·81 → 2562` layer. That is backwards. The move encoding already has the shape
to fix it, because both halves of the index space are anchor-major:

```
inline     idx = anchor_compact · 18 + dir · 3            + (size − 1)
broadside  idx = anchor_compact · 24 + gi · 8 + mi · 2    + (size − 2)  + 1098
```

So the entire 2562-move space **is** a `(42, 9, 9)` tensor — 18 inline planes
(6 directions × 3 sizes) plus 24 broadside planes (3 group directions × 4 move
directions × 2 sizes) — read out through a fixed index table
`plane · 81 + COMPACT_TO_CELL[anchor]`. And `42 × 61 = 2562` exactly.

| | dense (current) | convolutional (target) |
| --- | --- | --- |
| params | 3,321,282 | ~48 k (3×3) / ~2.7 k (1×1) |
| equivariance | none | translational |

Use a **3×3** conv so each anchor sees local context. This is the same
construction AlphaZero-chess uses for its 73×8×8 head, it composes correctly with
D6 augmentation, and it frees the entire parameter budget for the trunk — which
is where representation quality actually comes from.

Illegal moves are masked before the softmax at both training and inference time.

### 6.3 Value head — 3-way, not tanh

Output `softmax` over **(win, draw, loss)** rather than a `tanh` scalar.

Abalone under our termination rules is genuinely drawish, and a scalar collapses
"50/50 sharp" and "certainly drawn" onto the same number. A three-way
distribution keeps them distinct, is better calibrated, trains against a clean
cross-entropy, and gives the review UI an honest draw probability to display.
`E[value] = P(win) − P(loss)` recovers the scalar for MCTS backup.

### 6.4 Score head — auxiliary, from the trajectory

Predict the **final capture differential** `d ∈ [−6, +6]` as a 13-way
distribution.

This is the densest legitimate signal available. It is computed directly from the
game record, it is knowledge-free, and it carries far more gradient than a 3-way
outcome — the difference between winning 6–0 and 6–5 is real information the
outcome head throws away. It also gives the UI something genuinely useful:
"expected +1.4 marbles" is more legible to a human than "+0.31".

### 6.5 Capture-map head — auxiliary, from the trajectory

A `(2, 9, 9)` sigmoid map: for each cell, the time-discounted probability that a
marble is pushed off **from that cell** during the remainder of the game — one
channel for own losses, one for opponent losses.

Computed purely by replaying the trajectory and recording where captures
originated. This is the Abalone analogue of KataGo's ownership head and it
targets the central skill of the game — recognising which marbles are vulnerable
— **without ever telling the network that edges are dangerous**. It has to
discover that from where captures actually happen. Dozens of labels per position
instead of one.

### 6.6 Loss

```
L = L_policy
  + w_v · L_value          (3-way cross-entropy)
  + w_s · L_score          (13-way cross-entropy)
  + w_c · L_capture_map    (masked BCE)
  + weight decay (AdamW)
```

Start at `w_v = 1.0`, `w_s = 0.15`, `w_c = 0.15`. Auxiliary weights should be
small — their job is to shape the representation, not to dominate it. Ablate both
against a no-auxiliary baseline on the Elo ladder; if they do not pay, drop them.

---

## 7. Search

MCTS with PUCT. Design targets:

| Parameter | Target | Rationale |
| --- | --- | --- |
| simulations (full) | 800–1600 | ≥ 10× branching (~60) for a usable policy target |
| simulations (fast) | 100–200 | playout cap randomisation, §7.2 |
| batch size per NN call | 16–64 | virtual loss; the dominant throughput lever |
| `c_puct` | 1.25–2.0, tuned | |
| root Dirichlet | `α ≈ 10/branching ≈ 0.15–0.3`, `ε = 0.25` | explicit `SearchConfig` field, not an `eval_fn` side effect |
| FPU | parent-Q minus reduction | `Q = 0` for unvisited is optimistic when losing, pessimistic when winning |
| tree reuse | re-root on the played move | recovers a meaningful share of visits for free |

### 7.1 Batched evaluation is the enabling change

One `session.run` per simulation at batch size 1 is the single largest
inefficiency in the system. Collecting 16–64 leaves per call with virtual loss is
worth **5–15× on CPU** and considerably more on GPU/ANE. Everything in this
document that depends on 800+ simulations depends on this landing first.

It also likely inverts the recorded CoreML benchmark, which currently loses
purely on per-call overhead — exactly what batching amortises.

### 7.2 Playout cap randomisation

Most self-play moves run at the *fast* simulation count; a randomly chosen ~25%
run at the *full* count, and **only those positions produce policy targets**.
Roughly 2–3× more games per unit compute at equal target quality. Purely an
efficiency device — no domain knowledge, no bias.

---

## 8. Training loop

```
  ┌─ self-play (N games, deep MCTS, handicap-seeded fraction)
  │        ↓ parquet shards
  │   replay buffer (rolling W generations, D6 augmentation)
  │        ↓ uniform sample
  │   SGD (AdamW, LR schedule, EMA weights)
  │        ↓ export ONNX
  └────────┘
        every 5 gens ─→ anchor ladder ─→ Elo per rung, with CIs
```

**Self-play always uses the latest network.** No gating, no promotion, no
`best.onnx` in the loop — AlphaZero-2017 behaviour. Gating on 21 games is a
noisy, expensive measurement of something a fixed anchor ladder measures better.

**Value target.** Blend the game outcome with the MCTS root value, ramping toward
pure outcome as generations progress. Q is denser and less noisy early; z is
unbiased and should win out.

**Augmentation.** Full D6 (12 elements) via precomputed cell and move-index
permutations. Free 12× data multiplier — this is the one place the hex symmetry
of the board pays real dividends.

**EMA.** Maintain an exponential moving average of the weights and export *that*
for self-play. Cheap variance reduction; standard practice.

**Optimizer.** AdamW with decoupled weight decay, plus a step LR schedule keyed
to generation milestones. Constant LR for a whole run leaves strength on the
table.

### 8.1 Measurement — the part that was missing

Every metric below is logged under a namespace (`selfplay/`, `buffer/`,
`train/`, `val_frozen/`, `val_rolling/`, `curriculum/`, `ladder/`, `perf/`) to
`metrics.jsonl` and TensorBoard. The training loop iterates whatever the
measurement layer returns — there is no whitelist, so a metric added to
`model/validate.py` appears without anything in the loop being edited. Read a
run back with `uv run python -m model.report --run latest`.

Every generation:

- **Two held-out sets, measuring different things.**
  - `val_frozen/` — `validation.holdout_positions` positions (whole games) of
    an early generation, frozen and never trained on again, scored with a fixed
    seed so the same rows come back every generation. That fixity is what makes
    the curve comparable and also what limits it: by generation 30 it is scoring
    the network on positions a thirty-generation-weaker network produced, so a
    rising loss there is as likely to be *progress* as regression. **It is a
    drift indicator. Do not gate on it.**

    Which is why it is a *bounded sample* of that generation and not the whole
    of it. Freezing the generation wholesale cost a live run 57,699 training
    positions — generation 1 is the largest of any run, because games are
    longest under random play — and, by leaving nothing in the pool outside the
    current generation, it also silently skipped the rolling holdout below. The
    missing rolling metrics then produced a false "search is producing no
    information" alarm read off the frozen set's constant-by-construction
    `data_*` values. The remainder of the generation is ordinary training data
    and ages out of the replay window normally.
  - `val_rolling/` — 10% of each generation's own games, withheld by whole
    game from that generation's training and never sampled again. Same
    distribution as the training data, provably never seen. `val_rolling`
    total loss minus mean training loss is memorisation of the replay buffer
    and nothing else; that is the one to gate on, and the loop warns when the
    gap exceeds `validation.overfit_warn_delta`.

  **They are reported side by side and must never be collapsed into one
  number.** They disagree exactly when the curriculum is working. At generation
  5 of run `ruby-panther`, rolling value accuracy fell 0.667 → 0.554 while
  frozen accuracy rose 0.644 → 0.649: the controller had cut `handicap_rate`
  from 0.33 to 0.13, so the newest generation held far fewer near-terminal
  seeded positions where the winner is obvious. The rolling holdout's *task*
  got harder; nothing regressed. Read the frozen column for the trend and the
  rolling column for the alarm.
- **The generalisation gap, decomposed by head** — because "more games" and
  "fewer steps" are opposite actions and the total loss cannot tell you which
  one you need.

  `value` and `score` are labelled **per game**: every position in a game
  shares one outcome `z` and one final capture differential. Their effective
  sample size is therefore the number of *games* in the replay buffer —
  hundreds — not the number of positions, tens of thousands, and each label is
  seen once per position per epoch, of the order of a hundred times a
  generation. `policy` and `capture_map` are labelled per position and have no
  such ceiling.

  Generation 5 of `ruby-panther`, `train → val_rolling`:

  | head | train | rolling | gap | × w | share | labels |
  |---|---|---|---|---|---|---|
  | value | 0.5824 | 0.8143 | +0.232 | 1.00 | 118% | per-game |
  | score | 1.7344 | 1.9781 | +0.244 | 0.15 | 19% | per-game |
  | policy | 3.7899 | 3.7158 | **−0.074** | 1.00 | −38% | per-position |
  | capture_map | 0.0768 | 0.0884 | +0.012 | 0.15 | 1% | per-position |
  | **total** | | | **+0.196** | | | |

  The two per-game heads carry more than the whole gap; the per-position heads
  generalise at or better than they train. Acting on the total alone would mean
  cutting the step budget, which starves the policy head to treat a problem it
  does not have. The lever for a per-game head is `self_play.games_per_gen` and
  `train.replay_buffer_gens`.
- **Data health:** decisive rate, mean plies, mean `|d|` at termination,
  `captures_per_100_plies`, and the policy-target entropy gap. If target
  entropy sits at `ln(branching)` — a gap of zero — search is not producing
  information and nothing downstream matters.
- **Training intensity:** `buffer/epochs_this_gen = steps × batch / buffer`.
  A generation taking twenty raw passes over the same positions is overfitting
  by construction, and reading that off a loss curve after the fact is exactly
  the ambiguity this exists to remove. `train.target_epochs_per_gen` sets the
  step budget *in these units*, so a generation whose games came out short gets
  a proportionally smaller budget instead of silently multiplying its epochs.

Every `anchor_ladder.every_gens` generations:

- **Anchor ladder → Elo, per rung, with intervals.** Three kinds of rung, and
  they are not interchangeable:
  - **floor** — `random` and one cheap `heuristic@N`. A sanity check: losing to
    these is a bug, beating them is not a result. Few games; the answer is
    binary.
  - **frozen** — checkpoints at absolute generations, kept for the whole run.
    Fixed references, so their Elo curve is comparable end to end. The headline
    number is the mean over floor and frozen rungs.
  - **trailing** — `gen − k`. A *moving* reference that improves as the network
    does, so it never saturates and it is the rung with resolution left at
    generation 45. Excluded from the headline mean precisely because it moves.

    **`k` must be small — 1 or 2.** The trailing rung's entire value is that it
    is *near*; a distant offset is a fixed reference wearing a moving rung's
    clothes and belongs in `frozen_gens`. This was not obvious enough to leave
    as advice: `every_gens: 4` paired with `trailing_gens: [4]` resolves to
    generation 0 at the first ladder, so the rung is dropped, and reaches back
    past the previous ladder at every one after — an opponent already shown to
    be beaten. Two complete runs measured nothing for this reason. The config
    now rejects it.
- **Per-rung results are always reported, never only the mean**, and every Elo
  carries its 95% interval. `ladder/clamped_fraction` says how many rungs were
  swept to 0 or 1 and are therefore reporting a sample-size bound rather than a
  measurement; at 1.0 the ladder measured nothing and the loop says so loudly.
  Generation 6 of the validation run swept all four rungs 12-0-0 and reported
  "+545" four times over — the bound for a 12-game match, not a strength.
- Eval matches must **randomise openings and temperature-sample the early
  plies** — with a deterministic evaluator and a fixed start, N games is
  otherwise 1 game replayed N times.
- The ladder is the most expensive phase of a generation by a wide margin
  (`eval-match` sustains ~2.1k NN evaluations/s aggregate against
  `selfplay-batch`'s ~16k on the same machine), which is why the cadence is a
  knob and why the floor rungs get fewer games than the checkpoint rungs.

### 8.2 Success criteria, in order

`heuristic@800` used to sit at position 5 as *the* milestone. It has been
retired to a generation-1 sanity floor, for three reasons. Its evaluator is
`tanh(6.0·capture_diff + …)` and `tanh(6.0) = 0.99999`, so it cannot tell being
one marble up from four — one capture ahead it believes it has won, one behind
it believes it has lost. Its weights were tuned under the old 400-ply
draw-at-cap rules and were never retuned. And a 1.1M-parameter network on a
six-generation validation run cleared it at **generation 3, after 180 total
games of self-play**. A criterion a nearly-untrained network satisfies is a
miscalibrated yardstick, not an achievement.

1. **Sanity floor, generation 1–2.** Beats `random` and `heuristic@100`
   decisively; policy-target entropy gap clearly above zero; `train_loss_value`
   above 0.05 and moving. Failing any of these means the pipeline is broken,
   not that the network is weak. Stop and diagnose.
2. **The training signal is real.** Policy-target entropy well below
   `ln(branching)` and the gap widening generation over generation; decisive
   rate > 40%; value cross-entropy falling on `val_rolling`, calibration curve
   near diagonal, and `val_rolling` loss *not* pulling away from the training
   loss.
3. **Sustained strength gain — the primary milestone.** ≥ **+100 Elo per 10
   generations** against the frozen-checkpoint rungs, with 95% intervals that
   exclude zero, sustained across at least two consecutive ladders. Elo against
   a *frozen* opponent is the only strength number in the system that means the
   same thing at generation 10 and generation 40. Two disqualifiers, both of
   which the old criterion would have missed:
   - a ladder where every rung is clamped measures nothing, whatever number it
     prints (`ladder/clamped_fraction == 1.0`);
   - a gain measured only against `random` or `heuristic` is a gain against a
     ceiling that stopped moving at generation 3.
4. **Games lengthen while still finishing — the secondary milestone, and the
   genuine skill signal.** Unseeded games trend *toward* 60–100 moves
   (120–200 plies) while `natural_termination_rate` stays high, with
   `captures_per_100_plies` falling.

   This is the criterion the validation run failed while appearing to succeed.
   Natural termination went 0% → 100% and every headline number improved — but
   mean plies went 148 → 71, and games reaching six captures in ~30 moves is
   roughly one capture every five moves. Both sides were conceding marbles
   almost on contact. Competent defence looks like the opposite: games get
   *longer* because captures get *harder*, and they still finish because the
   network can convert an advantage once it has one. Short-and-decisive is a
   brawl; long-and-decisive is a game. If plies fall while natural termination
   rises, the network is learning to attack an opponent that cannot defend —
   and it is playing itself.
5. **Held-out generalisation holds up at scale.** `val_rolling` value CE and
   policy top-1 improving while `buffer/epochs_this_gen` stays in single
   figures. Improving one by spending the other is not progress.
6. **Plays recognisably purposeful Abalone:** coherent groups, sumito threats,
   edge avoidance — all learned, none of it told. Judged from the exported
   games in the review UI, not from a number.

---

## 9. Compute budget

Measured today: **~5,490 NN evaluations/second** aggregate on an M1 Pro
(9 self-play threads, batch size 1).

| Configuration | evals/gen | @ 5.5k/s | @ 35k/s (batched) |
| --- | --- | --- | --- |
| current (200 games × 399 ply × 100 sims) | 8.0 M | 24 min | — |
| target (200 × ~150 ply × 800 sims) | 24 M | 73 min | 11 min |
| target + playout cap randomisation | ~10 M | 30 min | **5 min** |
| validation scale (60 games) | ~3 M | 9 min | 1.5 min |

A 50-generation run at 5 min/generation is roughly 4 hours. That is the loop we
are building toward, and §7.1 plus §7.2 are what make it reachable.

---

## 10. Explicit non-goals

- **No hand-written positional evaluation** anywhere in the training loop.
- **No supervised pre-training** on human games or engine games.
- **No opening book** beyond uniform randomisation of the first plies.
- **No AlphaZero-scale ambitions.** Single-machine training. The target is a
  network that gains strength steadily against its own frozen past selves and
  plays purposefully (§8.2) — not a superhuman engine. Note that the target is
  no longer "beats the hand-written heuristic at equal search": that bar was
  cleared at generation 3 by a 1.1M network after 180 games, which says more
  about the heuristic than about the network.
- **No premature distributed training.** Single-node until throughput is a proven
  wall.

---

## 11. Open questions

| # | Question | How to settle it |
| --- | --- | --- |
| Q1 | Does handicap seeding actually produce learnable endgame value in gen 1? | Seed at `(5,5)`, train 1 gen, check value CE on held-out near-terminal positions |
| Q2 | Are the auxiliary heads worth their weight? | Ablate score and capture-map heads against the Elo ladder |
| Q3 | Does squeeze-excite pay on a 9×9 board? | A/B at `small` scale |
| Q4 | 3-way value vs. scalar tanh under our draw rate? | A/B on validation calibration |
| Q5 | Belgian Daisy only, or mixed with standard? | Compare decisive rate and Elo; standard stays playable in the UI regardless |
| Q6 | Optimal handicap anneal schedule? | Tune once Q1 is answered |
| Q7 | Do 2 history frames help despite the position being Markov? | A/B at `small` scale; default off |
