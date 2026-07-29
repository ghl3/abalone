# Published models

`web/public/models/` is gitignored — these files exist only on disk and in the
originating run's `checkpoints/`. A training run overwrites `best.onnx` at its
first ladder, and that network is *weaker* than a finished run's, so archive
anything worth keeping under a named copy before starting a new run.

| file | run | generation | strength |
|---|---|---|---|
| `best.onnx` | `ruby-panther-20260727-2159` | 24 | **latest**, and what the web app loads. Beats gen 16 23-7-2 (+191) and gen 14 decisively; see the caveat below. |
| `ruby-panther-gen013.onnx` | `ruby-panther-20260727-2159` | 13 | +206 Elo [+87, +400] over gen 9; beats gen 12 head-to-head 17-14-1 |
| `ruby-panther-gen012.onnx` | `ruby-panther-20260727-2159` | 12 | +364 Elo [+227, +720] over gen 8; superseded by gen 13 |

`best.onnx` is whatever ran last — check `runs/<run>/state.json` → `best_gen`
before trusting the row above. It currently matches `checkpoints/gen_024.onnx`
of `ruby-panther-20260727-2159` byte for byte.

**"Latest" is not established as "strongest."** Generation 24 was promoted under
the no-*resolved*-regression rule, and its own ladder has it losing to
generation 20 by 11-20-1, −100 Elo [−245, **+16**] — an interval that includes
zero by 16 points. One generation is now worth ~50 Elo and a 32-game rung cannot
resolve that. If it matters which network the browser serves, settle gen 20 vs
gen 24 with `posthoc_ladder` at 200+ games first. Both ONNX files are retained
in the run's `checkpoints/`. See docs/NOTEBOOK.md, 2026-07-29.

3.03M parameters, 10x128 trunk, four heads (policy / value / score / capture
map), 14 input planes. See docs/MODEL.md §5-6 for the contract; checkpoints
predating 2026-07-27 use an incompatible 6-plane input and will not load.

## What the browser does with it

`web/` loads `best.onnx` through `onnxruntime-web` and drives the same
`abalone_mcts::Search` self-play uses, so a move it plays in the browser is the
move it would play in an eval match at the same simulation count. Masking,
softmax and the value collapse all happen in Rust — see ARCHITECTURE §7.3.

A note on what you will see: at the standard opening this network reports the
**side to move** as losing (E[value] ≈ −0.42, expected score ≈ −1.4 marbles),
while a 400-simulation search of the same position comes back at +0.12 for the
mover. Both numbers are shown in Analysis mode, and both are reproducible
outside the browser — the encoder and heads were cross-checked against
`model/encoder.py` + `onnxruntime` and agree to three decimals.
