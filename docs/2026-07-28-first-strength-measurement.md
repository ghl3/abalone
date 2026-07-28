# First strength measurement — run `ruby-panther`, 12 generations

*2026-07-28*

The engine measurably plays Abalone better than it did twelve generations ago,
and this is the first time that sentence has been supported by a number. Every
prior ladder in the project's history reported a sample-size bound rather than a
measurement. This one resolves.

## Result

Generation 12 against every earlier checkpoint, 32 games per rung, 200
simulations, Belgian Daisy opening with randomised first plies and
temperature-sampled early moves:

| opponent | W–D–L | score | mean plies | Elo | 95% CI |
|---|---|---|---|---|---|
| gen 11 | 18–1–13 | 0.578 | 97.9 | +55 | [−63, +188] |
| gen 10 | 23–1–8 | 0.734 | 98.7 | +177 | [+59, +353] |
| gen 9 | 22–0–10 | 0.688 | 97.2 | +137 | [+19, +299] |
| gen 8 | 28–1–3 | 0.891 | 104.0 | +364 | [+227, +720] |
| gen 6 | 30–0–2 | 0.938 | 88.8 | +470 | [+306, +720] |
| gen 4 | 31–0–1 | 0.969 | 64.8 | +597 | [+399, +720] |
| gen 2 | 32–0–0 | 1.000 | 59.2 | +720 | bound |
| gen 1 | 32–0–0 | 1.000 | 51.7 | +720 | bound |

Generation 8 against the same anchors, as a second point on the curve:

| opponent | W–D–L | score | mean plies | Elo | 95% CI |
|---|---|---|---|---|---|
| gen 6 | 27–0–5 | 0.844 | 77.2 | +293 | [+162, +601] |
| gen 4 | 32–0–0 | 1.000 | 64.5 | +720 | bound |
| gen 2 | 32–0–0 | 1.000 | 54.0 | +720 | bound |
| gen 1 | 32–0–0 | 1.000 | 46.2 | +720 | bound |

Six of eight rungs resolved. The curve is monotone in the expected direction
throughout, with one exception — generation 12 scored slightly better against
generation 10 than against generation 9 — whose intervals overlap almost
entirely and which is noise at this sample size.

**Roughly 75–90 Elo per generation.** Taking the resolved rungs at face value:
+364 over four generations (91/gen), +470 over six (78/gen), +597 over eight
(75/gen). MODEL.md §8.2 sets the bar at ≥100 Elo per *ten* generations, so the
run is running about eight times ahead of that target. The estimate should be
read as a slope through overlapping measurements, not eight independent
observations.

An earlier reading of this run put the figure near 120 Elo/generation. That came
from the in-run generation-12 ladder, whose `gen_008` rung measured +470 where
this one measures +364 for the same pairing under a different seed. Both sit
inside each other's intervals; 75–90 is the better-supported number.

## The two numbers worth more than the Elo

**Mean plies rises with opponent strength**: 51.7 against generation 1, 64.8
against generation 4, 88.8 against generation 6, and 97–104 against generations
8 through 11. Nobody designed this metric to measure that. The network dispatches
weak opponents quickly and needs twice as long against near-peers, which is what
a real strength gradient looks like from the inside.

**Generation 12 versus generation 11 is +55 [−63, +188]** — consistent with zero.
A single generation of improvement is now below what a 32-game match can resolve.
That is not a plateau; it is the measuring instrument running out of precision,
and it says what future ladders need: anchors two or more generations back, or
substantially more games.

## Where saturation begins

Generations 1 and 2 are swept 32–0–0 by both probes, and generation 4 is swept by
generation 8. So a gap of roughly eight or more generations saturates a 32-game
match at this stage of training. Anchors within about six generations still carry
information. That is the empirical basis for `anchor_ladder.trailing_gens: [1, 2]`
plus a spread of frozen anchors, rather than any single offset.

## The run itself

| gen | plies | nat. term | entropy gap | handicap | train loss |
|---|---|---|---|---|---|
| 1 | 146 | 0.02 | 0.16 | 0.70 | — |
| 4 | 77 | 0.99 | 1.04 | 0.33 | 4.6809 |
| 8 | 102 | 0.95 | 1.28 | 0.10 | 4.4863 |
| 12 | 107 | 0.92 | 1.42 | 0.10 | 4.1483 |

Games collapsed from 146 plies to 77 as the network learned to force captures,
then climbed back to 113 by generation 11 and held near 107 — the shape MODEL.md
§8.2 predicts, where the initial shortening is competence at winning and the
subsequent lengthening is competence at not losing. Captures per 100 plies peaked
at 10.13 in generation 4 and fell to 7.94 by generation 12: it now takes more
moves to win a marble.

The curriculum retired itself from 0.70 to its 0.10 floor in five generations on
evidence alone, and natural termination on unseeded games went from 2% to 92%.
No hand-written evaluator was involved at any point.

By generation 12 every head held out *better* than it trained (total −0.379),
so the run ended with no measurable memorisation of the replay buffer.

## What this does not establish

The engine has only ever been measured against itself and against `random` and
`heuristic@100`, both of which it sweeps. There is no external reference point,
no human game, and no published engine in the comparison. "Improving rapidly
against its own history" and "playing Abalone well" are different claims, and
only the first is supported.

Twelve generations is also short. Whether the 75–90 Elo/generation rate survives
to generation 50 is the question `config/standard.yaml` exists to answer.

## Method note

The run's own ladders were misconfigured — `trailing_gens: [4]` against
`every_gens: 4` resolves to generation 0 at the first ladder and to
already-swept checkpoints afterwards — and its config was frozen in memory at
startup, so the fix could not apply mid-run. The measurements above are
post-hoc, replayed from retained checkpoints with `model/posthoc_ladder.py`.
That the measurement was recoverable at all is an argument for retaining every
checkpoint and for keeping `eval-match` a standalone binary.

See also: [MODEL.md](MODEL.md) §8 for the measurement design,
[2026-07-27-architecture-review.md](2026-07-27-architecture-review.md) for the
state this run started from.
