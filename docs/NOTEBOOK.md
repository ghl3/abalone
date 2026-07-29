# Notebook

A running log of work sessions. One dated entry per session: what it set out to
do, what came back, the numbers needed to read that later, and what it left for
the next person.

This is the informal record. It is not a design doc and not a review — when a
session produces a result that deserves an argument rather than a log line, it
gets its own dated document under `docs/` and the entry links to it.

## Conventions

- **Newest entry first.** The most common read is "what happened last time".
- **Heading format:** `## YYYY-MM-DD — <short title>`.
- **Sections:** goal, what ran, results, context, next steps. Skip any that
  would be empty; do not pad.
- **Numbers over adjectives.** An entry that says a run "went well" is worth
  nothing in a month. Record the measurement and its interval.
- **Record what was refuted too.** A hypothesis checked and killed is the most
  perishable thing a session produces, and the most expensive to re-derive.
- Entries are point-in-time. If a conclusion later turns out wrong, write it in
  the newer entry and leave the old one standing.

---

## 2026-07-29 — the analysis view, reviewed and rebuilt

### Goal

Review the analysis view as a piece of UI and act on what the review found. No
model work; this session touched `web/`, plus the two crates it reads through.

### What the review found

Driving the built app rather than reading it was what produced the list — four
of these are invisible in source and obvious within seconds of using it.

- **"No legal moves." on screen during every search.** `topMoves` is empty for
  the whole duration of a search, and the empty state said the position was
  over. Also shown on first load while the 12 MB `best.onnx` downloads. The
  `busy` prop's own doc comment described rows persisting from the previous
  position; nothing implemented it.
- **The heuristic evaluator displayed the network's numbers.** Selecting it
  kept the `Network read · no search` block on screen, directly above a footer
  reading "Hand-written evaluator, no network."
- **Eval colour inverted its meaning every ply.** Evals are White-POV under a
  heading naming the side to move, so at ply 4 Black's strongest move — 1280 of
  2000 visits — rendered `−0.14` in red, the most alarming number in the list.
- **Layout broke below ~700 px** (measured `scrollWidth 705` vs `clientWidth
  585` at a 600 px viewport): eval bar and plate labels off the left edge.
- **The move list was invisible to keyboard and screen readers** — `div`s with
  `onClick`, no role, no tabIndex. The a11y tree contained no move entries.
- **The eval bar stacked two estimators on one gauge**: bands from the raw
  network read, number from search. They contradict each other whenever search
  changes its mind.

### What changed

Numbers first, because the rest follows from them. One convention —
White-positive, as in chess, never restated per mover — which forces colour to
encode *side* rather than sentiment; `web/lib/outcomeFormat.ts` owns it.
`Network read · no search` is gone.

Then the panel stopped speaking in evals at all — see *Searched probabilities*
below, which was the largest piece of the day and started from a mistake of
mine.

Then the engine surface. `Search::principal_variation` walks the most-visited
path off the arena the search already built, exposed through `WasmSearch`, so
every ranked move carries its line at no inference cost — hovering a move in
the line replays the board to it. Progress messages now carry a full
`SearchSnapshot` instead of a visit count, so rows refine in place (measured
120 → 256 → 832 → 1888 visits across one Deep search, with the top move
changing at 256) and the blank-state bug has nowhere to live. The 50–2000
slider became Quick/Standard/Deep; visit share is drawn as a bar behind each
row, which is the one thing four bare integers never communicated.

The heuristic evaluator is gone, along with the `WasmGame::analyze` and
`eval_white_pov` bindings that existed only to serve it. Analysis gained a flip
control, `← →` stepping with a redo stack, and `F`; the row list became real
buttons with labels; the layout wraps instead of overflowing (`585/585` at
600 px, was `705/585`).

### Searched probabilities — and the wrong turn on the way there

The panel's units went through three states, and the middle one was wrong in a
way worth recording because it looked entirely reasonable.

Asked for win/draw/loss and a marble margin per move, I reached for the
network: encode the position after each ranked move, one batched forward pass,
read the `value` and `score` heads. It worked, it was cheap, and it was junk —
a 1-ply first impression displayed in a table whose rows were *ranked* by a
searched eval, with nothing on screen saying the two came from different
places. On the opening position the disagreement is not subtle: the raw network
says White 49% / draw 7% / Black 44%, and 500 simulations say White 41% /
Black 52%. Eight points, and the version I shipped first would have shown the
49%.

The reason I reached for the network at all is that the tree genuinely does not
hold these numbers. `collapse_value` reduces the three-way head to
`P(win) − P(loss)` *before* anything is backed up, and the score head is never
fed to search — so a node stores one scalar and the draw share and the margin
are gone. The right fix was not a better source for the missing numbers; it was
to stop destroying them.

`abalone-mcts` now accumulates the full distribution and the expected margin
alongside the scalar, under `track_outcome_stats` (default off). The algebra is
exact rather than approximate: since `Q = P(win) − P(loss)` and the three sum
to one, a draw share is the only missing degree of freedom, and backing it up
recovers the rest identically.

Four things keep this from touching training, which was the explicit
constraint:

- Selection reads `Node::total_value` and nothing else, so PUCT has no path to
  the new data. Structural, not conventional.
- The accumulators live in side vectors parallel to the arena, not in `Node` —
  so with tracking off the node layout and footprint are *identical*, not
  merely equivalent.
- `LeafEval` is untouched; the distributions arrive via a separate
  `submit_with_stats`. Not one training file changed except `search_config`,
  which now states `track_outcome_stats: false` explicitly rather than
  inheriting it.
- The virtual loss got a matching term, `[(1+vl)/2, 0, (1−vl)/2]`, whose
  collapse is exactly `virtual_loss` — so the two accumulators stay consistent
  *during* a batch, not only after it resolves.

**The invariant is the load-bearing test.** `P(win) − P(loss)` must reproduce
the eval at every node. A point-of-view flip applied at the wrong parity still
produces a valid-looking distribution — it just describes the position
backwards — and nothing in the UI would show it. Asserted across batch sizes 1,
8 and 32, so virtual loss is covered, plus a test that turning tracking on
leaves `visits` and `q_parent_pov` bit-identical.

It earned its keep immediately: the first run failed at `batch=1 child=4`,
`-0.4199` against `-0.4901`. The backup was fine — my *test evaluator* fed a
fixed 0.2 draw mass, which clamps the collapse for any leaf past |0.8| and made
the distribution disagree with the scalar it was supposed to decompose. Shrink
the mass as |v| → 1 and it passes. A test that catches a flaw in its own
fixture before it catches one in the code is doing its job.

The move table is now `move · White% · draw% · Black% · marbles`, all searched.
No eval column, no visits column, and — after a round of "I still see eval
here" — no signed eval anywhere in the UI at all. Both were engine units for
things already shown better: eval is the same axis as the percentages, and
visit share is the row's background bar. The invariant is still checked, in
the Rust tests where it fails loudly rather than on screen where it asks a
reader to notice. `EvalBar` became `WinBar` and `evalFormat.ts` became
`outcomeFormat.ts`, because a component that deliberately shows no eval should
not be named for one.

Removing the per-move forward pass also gave back what it cost:
**795 → ~1020 sims/s at Deep**.

Review followed, so the two screens share a vocabulary. Move grading is now the
winning chance given up in percentage points (`−6%`) rather than eval given up
(`−0.12`); the bands moved from 0.08/0.2 to 4/10 points, which is the same
measurement restated since `Δwin ≈ Δeval / 2`. The "engine wants" line reports
the triple and the margin. The graph's *geometry* did not change and did not
need to: `rootEval` is identically `P(win) − P(loss)`, so the curve was already
plotting a probability difference — only the tooltip was speaking the wrong
language.

### The bug the work uncovered

Adding an explicit cancel on search supersession surfaced `Session mismatch`
from ORT, which wedged the engine permanently — play mode stuck at "thinking"
forever. Cause: `onmessage` is `async`, so two `runSearch` calls can both be
inside `session.run`, and `onnxruntime-web` permits exactly one run per
session. **This was a latent race before this session, not a new one** —
React's strict-mode double mount posts two searches on every mount, and the
cancel only widened the window. Fixed by chaining `runSearch` through a promise
queue in the worker, with `generation` still bumped first so the superseded
search bails at its next batch boundary. Documented in ARCHITECTURE §7.4, where
it now leads the load-bearing list.

Also fixed while in there: a transient engine error stayed pinned under the
board for the rest of the session, describing an engine that had recovered.

### The move arrow, in three wrong shapes

Small, but a good example of a thing that cannot be reviewed from source. The
hover preview draws one arrow for the group:

1. **Centroid to centroid.** For a 3-marble inline push both centroids land on
   squares occupied *before and after*, so the arrow began mid-group and said
   nothing about where the line came from. Looked fine for single marbles and
   broadside slides, which is why it survived the first check.
2. **Rearmost origin to frontmost destination.** Fixed the tail, but spanned
   the whole line — 131 px of arrow for a move that travels 52. Overstated the
   motion as badly as (1) understated the origin.
3. **Rearmost origin, plus one step.** The centroid difference *is* the shift
   vector, and it is one cell for inline and broadside alike, so adding it to
   the tail gives a one-step arrow with no need to know which kind of move it
   is. A broadside's cells all tie on the travel axis, so the tail resolves to
   the group's centre; a single marble resolves to itself. All five ranked
   moves now measure 27 px whatever their size.

### Verified

179 Rust tests pass, clippy clean, `next build` clean, `tsc --noEmit` clean.

Driven in the browser: live row refinement, depth switching under rapid
alternation, line hover and board stepping, flip (including that the preview
arrow rotates with the board while rim labels stay upright), play mode with the
engine to move, and a full review sweep — the protocol change touches
`sweepGame`, so review was re-checked end to end.

The searched-probability invariant was checked *in the browser too*, not only in
Rust: every displayed row satisfies `White% − Black% == eval` and the three
percentages sum to 100.

Two PV tests are worth keeping honest: the line is replayed against a real
`Game` and asserted legal at every step, because a PV that cannot be played is
worse than no PV.

### Next steps

- Analysis has no move list, and that is a **decision, not a gap**: `← →` with
  the redo stack covers stepping through a line, and a list changes the layout
  shape enough to want its own pass. Revisit only if entering long lines starts
  to feel lossy.
- The PV is capped at 6 plies and only the top 5 moves are ranked (`TOP_N` in
  the worker). Neither is a considered limit, just a default.
- `EvalGraph` in review still colours by "up is the reviewer" rather than by
  side. It shares the palette constants now but not the convention.
- `track_outcome_stats` costs one `[f32; 3]` and one `f32` per node when on.
  Unmeasured, because self-play never turns it on — but if a future tool wants
  it at training scale, measure before assuming it is free.

---

## 2026-07-29 — ruby-panther extended to 24 generations

### Goal

Extend the best model by ten generations. `ruby-panther-20260727-2159` was the
only full-scale run in `runs/` (`base` preset, 400 games/gen, 200/800 sims);
everything else is a smoke test at `small`. It had stopped at generation 14.
Target: generations 15–24, with the question inherited from
[2026-07-28-first-strength-measurement.md](2026-07-28-first-strength-measurement.md)
— does the 75–90 Elo/generation rate measured over generations 1–12 survive?

### What ran

```bash
uv run python -m model.train_loop \
  --config <copy of the run's archived config, gens: 24> \
  --resume ruby-panther-20260727-2159
```

Two deltas against the archived config, both in `HASH_EXCLUDED` so neither
blocked the resume (`semantic_diff` was empty):

| field | was | now | why |
|---|---|---|---|
| `gens` | 14 | 24 | the extension |
| `retention.keep_last_onnx` | 15 | 30 | at 15, generations 0–8 would have been collected during the extension. The first-strength measurement was only recoverable because checkpoints had been retained; 12 MB each is not worth losing that again. |

Preflight, both worth repeating on any resume:

- **Rebuilt `selfplay-batch` and `eval-match`.** HEAD had moved seven commits
  past the SHA in `state.json` (016f363 → cc436bd) and `eval-match` in
  `target/release` predated `c23183e`, which moved the plane encoder into its
  own crate. Checked the diff before trusting the buffer: it is a file move
  plus a re-export, no encoding change, so the 14 generations of shards stayed
  valid. `train_loop` warns on SHA drift but cannot check this for you.
- **Confirmed generation 1 shards were still present.** They are the frozen
  holdout and outlive the rolling window by design;
  `_apply_retention` protects them explicitly.

8.4 hours wall clock, 41–50 min/generation. TensorBoard was already serving
`runs/` on :6006 and was left alone rather than restarted.

### Results

**The headline: generation 22 beats generation 14 — the model the extension
started from — 30–0–2, score 0.938, +470 Elo [+306, +720].**

Per generation (`—` = rung not yet reachable, `bound` = 32–0–0 sweep reporting
the sample-size bound rather than a measurement):

| gen | plies | nat term | cap/100 | train | val_roll | −1 | −2 | −4 | −8 | min |
|---|---|---|---|---|---|---|---|---|---|---|
| 14 | 118.2 | 0.91 | 7.44 | 3.9164 | 3.686 | −22 | +89 | +221 | bound | 43 |
| 15 | 112.8 | 0.91 | 7.43 | 3.8243 | 3.619 | +22 | +124 | +221 | bound | 41 |
| 16 | 118.1 | 0.89 | 7.11 | 3.7421 | 3.442 | +55 | +77 | +137 | +597 | 44 |
| 17 | 116.8 | 0.90 | 7.35 | 3.6732 | 3.671 | +77 | +137 | +429 | +597 | 45 |
| 18 | 126.1 | 0.86 | 6.73 | 3.6149 | 3.500 | +66 | +33 | +150 | +597 | 48 |
| 19 | 121.9 | 0.91 | 7.06 | 3.5630 | 3.555 | +11 | +137 | +221 | +470 | 47 |
| 20 | 126.9 | 0.86 | 6.75 | 3.5296 | 3.539 | +124 | −44 | +137 | +364 | 49 |
| 21 | 132.5 | 0.78 | 6.42 | 3.5036 | 3.340 | −55 | −77 | +124 | +293 | 50 |
| 22 | 131.8 | 0.84 | 6.47 | 3.4843 | 3.643 | −66 | +11 | +33 | +470 | 50 |
| 23 | 130.2 | 0.84 | 6.56 | 3.4717 | 3.516 | +89 | +44 | +112 | +163 | 50 |
| 24 | 130.4 | 0.83 | 6.51 | 3.4610 | 3.829 | +33 | +66 | −100 | +191 | 82 |

**The rate decelerated, and most of the deceleration is in the last four
generations.** Both trailing offsets agree, which is what lifts this above
single-rung noise:

| | gen−4 rung | gen−8 rung |
|---|---|---|
| gens 13–20 | +215 mean → **54 Elo/gen** | +525 mean → **66 Elo/gen** |
| gens 21–24 | +42 mean → **11 Elo/gen** | +279 mean → **35 Elo/gen** |

against 75–90 Elo/gen over generations 1–12. Overall for the extension,
~45–60 Elo/gen depending on which offset you weight.

**A single 32-game rung is worth almost nothing on its own.** The gen−4 series
across the extension reads +221, +137, +429, +150, +221, +137, +124, +33, +112,
−100 — sd ≈ 100 Elo. The gen−8 rung produced +470 and +163 on *consecutive*
generations. Three separate readings during this session looked like a trend
break and were not; only the pooled means and the two-offset agreement survived.
Design ladders and read them accordingly.

**The value head began memorising, and that is the binding constraint.**
Generation 24 tripped a warning that had not fired since generation 3:
`val_rolling` 3.8286 against a training loss of 3.4610, gap +0.368, of which 88%
is the per-game heads. The value head's train→rolling gap by generation:

```
gen  14     15     16     17     18     19     20     21     22     23     24
   -0.002 +0.026 -0.192 -0.007 -0.020 +0.030 +0.125 -0.068 +0.128 +0.002 +0.308
```

Negative — holding out *better* than it trained — through generation 21, then
positive and climbing. The mechanism is in the report's own diagnosis: value and
score labels are per-game, so the effective sample size is 400 games per
generation, not the ~51,000 positions. Games have grown from 77 plies at
generation 4 to 130 now, so each per-game label is shared by ~70% more positions
than when the game budget was set. Same games, more redundancy.

This is a data-quantity problem, not a capacity problem, and it arrived at the
same time as the sharp slowdown in generations 21–24. The two are plausibly the
same event.

**Refuted during the session:** that rising draws were compressing the near-rung
Elo toward zero. Self-play draw rate did nearly double (0.048 → 0.085 at
generation 21), and draw-heavy play does compress rating differences — but the
ladder games contain 0–3 draws in 32 across every recent rung. Generation 21 vs
20 was 13–1–18. The near-rung losses are losses, not diluted scores.

**Physical signature of the extension**, all consistent with defence improving:
plies 118 → 130, captures per 100 plies 7.44 → 6.51 (run low), natural
termination 0.91 → 0.83, mean absolute score difference falling. Near-peer
ladder pairings ran 140–157 plies against 47–65 for the floor anchors. Roughly
17% of games now end at the 200-ply cap and are scored by margin.

**Two traps in the final report:**

- **`best.onnx` → generation 24 is weakly supported.** Its ladder is 11–20–1
  against generation 20, −100 Elo [−245, **+16**]. It was promoted under "no
  *resolved* regression" and the interval includes zero by 16 points. The web
  app is now serving a checkpoint with no evidence it beats generation 20, and
  mild evidence against.
- **The "mean plies fell 146 → 130" warning is an artifact.** It compares
  against generation 1, which ran at 0.70 handicap with 2% natural termination —
  long games because neither side could finish. From generation 4's trough of 77
  plies the trend is monotonically upward.

### Context

- Run directory: `runs/ruby-panther-20260727-2159/`. The config that produced
  generations 15–24 is archived in it as `config.yaml` (rewritten at resume);
  per-generation metrics in `metrics.jsonl`; per-rung W–D–L with CIs and
  per-game transcripts in `eval/gen_0NN_ladder_*.json`; TensorBoard events in
  `tb/`.
- `uv run python -m model.report --run ruby-panther-20260727-2159` reproduces
  the trajectory, ladder, per-head generalisation table and warnings.
- Final state: `current_gen: 24`, `best_gen: 24`, handicap parked at its 0.10
  floor since generation 6. Every generation 15–24 sweeps `random` and
  `heuristic@100` 16–0–0.
- All 25 ONNX checkpoints (gen 000–024) retained, so any pairing can be
  re-measured post hoc.

### Next steps

1. **Before generation 25, fix the label supply.** Raise
   `self_play.games_per_gen` (400 → 800) or widen `train.replay_buffer_gens`
   (8 → 12–16). Cutting the step budget will not help: the gap is per-game
   labels, not over-training on positions.
2. **Settle generation 20 vs 24 with a real sample.** `posthoc_ladder` at 200+
   games. At 32 games the pairing cannot resolve, and it decides what the web
   app should serve.
3. **The anchor ladder needs more games or deeper offsets.** One generation is
   now worth ~50 Elo and 32 games cannot resolve that — roughly 200 games would.
   The gen−1 and gen−2 rungs spent this whole extension reporting noise.
4. **Watch the ply cap.** Natural termination 0.94 → 0.83 while plies rose
   112 → 130. Not a problem at this level, but if games keep lengthening, a
   growing share of value targets come from margin-at-cap rather than an actual
   six-marble win.
5. **Unexplained:** generation 24's `heuristic@100` rung sat inside a 16.6-minute
   wall-clock window while `eval-match` reported 0.4 minutes of match time; the
   generation took 82 minutes against ~50 for its neighbours. Not diagnosed.
6. Still unestablished, unchanged from the first strength measurement: the
   engine has never been measured against anything but itself, `random` and
   `heuristic@100`. "Improving against its own history" is not "playing Abalone
   well".

### Addendum, same day — next steps 3 and 4 applied to the configs

`medium.yaml` and `standard.yaml` only; `dry_run.yaml` and `validation.yaml` are
instruments for checking the loop runs at all and were left alone.

| knob | was | now | why |
|---|---|---|---|
| `anchor_ladder.games` | 32 | 64 | a 32-game rung resolves ±120 Elo; the run was gaining ~35/gen |
| `anchor_ladder.trailing_gens` | `[1,2,4,8]` | `[2,4,8,16]` (`[2,4,8]` in medium) | the `1` rung returned ten readings from −66 to +124, every interval spanning zero. `16` is the rung that will still resolve at generation 50. Medium stops at 8 — a 12-generation run never reaches a `gen − 16`. |
| `self_play.random_opening_plies` | 2 | 5 | distinct *games* are what the per-game heads learn from, and this buys them free |
| `self_play.max_plies` | 200 | 300 | fewer value labels decided by the sign of a one-marble margin |
| `train.steps_per_gen_max` (standard) | 5000 | 8000 | consequence of the above: ~150-ply games over a 20-generation window put 1.5 epochs at ~7000 steps, so the cap would have silently bound at 1.07 epochs and retuned the training regime |

Ladder and self-play `max_plies` / `random_opening_plies` were moved together —
a rung capped at 200 while self-play runs to 300 would adjudicate a class of
position the training distribution resolves properly.

**Cost, stated plainly.** Self-play holds 22.2 pos/s (steady to three figures
over generations 20–23), so 400 games × ~150 plies is ~46 min, and the wider
ladder is ~26 min against ~12 before. `standard.yaml` goes from ~40 h to ~60 h
for its 50 generations. The ladder overhead roughly triples in absolute terms,
from ~12% of a generation to ~30%. That was bought deliberately: a measurement
that cannot resolve the effect it is measuring is not cheap, it is worthless.

**Verified**: all four configs validate, `pytest tests/ -q` is 626 passed, and
an 8-game `selfplay-batch` at the new parameters runs clean. The binary prints
`note: ply cap 300 differs from the default 200; the ply input plane is
normalised by it` — the plumbing for a non-default cap already existed.

**One caution the smoke test surfaced.** At 32/64 simulations, 3 of 8 games ran
to 295 plies. Play that weak is not representative, but it is a reminder that
raising the cap *moves* adjudication rather than removing it. Whether the 17%
of generation-24 games that capped at 200 now finish naturally, or simply run to
300, is a measurement for the next run — `selfplay/natural_termination_rate` is
where it will show.

Note also that these edits put `self_play` outside `ruby-panther`'s
`config_hash`, so neither config can resume it any more. Intended: this is the
configuration for the *next* run, and next step 1 — more games per generation —
is deliberately not applied here, because it is the one change worth its own
run rather than a config edit.

### Proposals for further training

Written at the end of the session, from the generation 13–24 evidence. Nothing
here is run yet.

#### The diagnosis these rest on

Split the generalisation gap by head and two different stories separate cleanly.

**The policy head is healthy and is not capacity-limited.** Its cross-entropy
minus its target's entropy is the KL to its own teacher — how far the network is
from reproducing the search that trained it:

| gen | KL on train | KL on unseen | unseen − train |
|---|---|---|---|
| 8 | 0.870 | 0.542 | −0.328 |
| 16 | 0.465 | 0.366 | −0.099 |
| 24 | 0.349 | 0.321 | −0.028 |

It generalises *better than it fits* at every generation, and is now within
~0.32 nats of reproducing 200/800-simulation search on positions it has never
seen. A head that does not overfit, whose loss is still falling, is not short of
parameters.

**The value head is the entire problem** — +0.308 of the +0.368 total gap at
generation 24, having been negative through generation 21. Its effective sample
size is `games_per_gen × replay_buffer_gens` = 400 × 8 = **3,200 distinct
labels**, and games grew 77 → 130 plies, so each label is smeared over ~70% more
positions than when that budget was set.

By the run's own instrument this is nowhere near converged: the GCP proposal §7
calls `score_vs_gen_minus_8 ≤ 0.55` converged and **> 0.65 data-limited**, and
the last four ladders read 0.844, 0.938, 0.719, 0.750.

#### Ranked interventions

1. **More distinct games.** `games_per_gen` 400 → 1200 and/or
   `replay_buffer_gens` 8 → 20. Both multiply the same quantity. This is the
   diagnosed problem; everything below is secondary.
2. **Fix the instrument.** *Applied this session* — see the addendum above.
3. **Widen the openings.** *Applied.*
4. **Raise the ply cap.** *Applied.*
5. **More search** (`sims_full` 800 → 1200) — later. The policy head is close to
   its teacher, so the teacher eventually becomes the ceiling; but this competes
   for exactly the compute item 1 needs, and item 1 is the measured problem.
6. **`net_preset: large` — not indicated, and this contradicts the existing
   plan.** GCP proposal §11 sets the rule: *"if `val_rolling` pulls away from
   training loss while the gauntlet flattens, the constraint is capacity, and
   `large` is the right call."* Both conditions are now literally true, so the
   rule as written says spend on parameters. **That would be wrong.** The rule
   predates the per-head generalisation table; the pull-away is 88% per-game
   heads, which is a data signature. A capacity limit would show as the *policy*
   head plateauing with a positive train→holdout gap, and it is doing the
   opposite on both counts. Re-read §11 before acting on it.

#### How to test item 1 without losing a week

**First, free, and it may make the rest unnecessary.** The *diagnosis* can be
tested offline from shards already on disk — generations 13–24, ~600k positions
across ~4,800 games. Train from one fixed initialisation twice at **equal
position count** but different distinct-game count (e.g. 240k positions drawn
from 1,900 games vs from 4,800) and compare the value head's train→rolling gap.
If the gap tracks game count rather than position count, the diagnosis holds and
the run below is justified. If it doesn't, the whole ranking above is wrong.
Tens of minutes, no self-play, no money.

**Then the run — and resume, do not restart.** A fresh run at 1200 games spends
its first ~15 generations re-deriving what `ruby-panther` already knows; the
memorisation only appeared at generation 22. Resuming from generation 24 puts
the extra data exactly where the problem is: the 8-generation window would hold
~9,600 distinct games instead of 3,200. Seven or eight generations yields a
`gen 32 vs gen 24` −8 rung, four or five gen−4 readings, and eight
per-generation readings of the value-head gap.

**The blocker is a bug in the exclusion set.** `self_play.games_per_gen` is not
in `HASH_EXCLUDED`, so changing it refuses the resume — but the set's own
criterion is *"run identity, an outer-loop bound, or infrastructure — none of it
changes the distribution of the data"*, and games-per-generation changes how
many samples are drawn, not the distribution each is drawn from. It cannot
invalidate a single existing shard. This is the same argument commit `54e037a`
accepted for `anchor_ladder`, whose absence had locked this very run out of
being extended. One line plus a test.

Useful side effect: a resume keeps `max_plies: 200` and
`random_opening_plies: 2`, since those *are* hash-covered. So the experiment
moves one variable, and the changes applied above land in the next fresh run.

#### On renting a machine — the GPU is not the expensive part

The proposal picks `g2-standard-32` for its **32:1 vCPU-to-GPU ratio**, not for
the L4. This is a CPU-bound workload with a GPU-shaped inner loop: the forward
pass is 0.489 GFLOP while tree search costs ~4–5 pos/s per vCPU. You rent cores;
the L4 comes with them.

Which creates an arbitrage for a *short* run. At `base` fp32 the L4 binds at
~50–70 pos/s against a CPU ceiling of ~110–135, so the extra 16 vCPUs of the
32-core shape do nothing until the fp16 export exists. **A `g2-standard-16` at
fp32 gets ~50–60 pos/s for roughly half the hourly rate**, and lets item 4 of
the port (fp16, 2–4 h) be skipped entirely.

| option | wall clock | $ | engineering | what it answers |
|---|---|---|---|---|
| Offline games-vs-positions ablation | ~30 min, local | 0 | none | is the diagnosis right? |
| `g2-standard-16` fp32, resume from gen 24 @ 1200 games | 8 h | ~$5 | ~6 h port + 1-line hash fix | does the fix restore the Elo rate? |
| `g2-standard-32` + fp16 | 8 h | ~$8 | +2–4 h | same, ~2× the generations |
| Fresh 30-generation run, laptop | 2.5 days | 0 | none | same, slowly |

Dollar figures extrapolate from the proposal's measured ~$0.90–1.00/hr spot for
the 32-core shape and **must be verified with `gcloud` before committing**.
Ruled out: CPU-only cloud (§2 sizes a 128-vCPU C3 under 40 pos/s at ~2× the
price) and the cheap GPU marketplaces, which sell thin vCPU allocations — the
wrong ratio for this workload.

**Implementation caution for the port.** Item 3 of §2 proposes replacing
`use_coreml` with `inference_backend`. `RunConfig` rejects unknown keys and
`ruby-panther`'s archived `config.yaml` contains `use_coreml: true`, so a
straight rename breaks resuming the very run this plan depends on. Keep
`use_coreml` as an accepted alias.
