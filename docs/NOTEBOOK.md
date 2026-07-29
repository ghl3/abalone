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
