# Abalone

An AlphaZero-style engine for [Abalone](https://en.wikipedia.org/wiki/Abalone_(board_game)),
and a web tool for reviewing games and individual positions.

**No hand-written heuristics anywhere in the training loop.** The network
starts from random weights and learns from self-play alone. A heuristic
evaluator survives in `crates/mcts/src/eval.rs`, used only as a benchmark
opponent — never as a teacher, never as a bootstrap.

## Start here

| document | what it is |
|---|---|
| [docs/MODEL.md](docs/MODEL.md) | The target design: representation, heads, search, curriculum, and how success is measured. Aspirational, and the authority when code and doc disagree. |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | How the pieces fit — the Rust crates, the Python trainer, the shard schema, the web app. |
| [docs/2026-07-27-architecture-review.md](docs/2026-07-27-architecture-review.md) | Point-in-time review that motivated the current design, with the evidence chain. |
| [docs/README.md](docs/README.md) | Documentation conventions: which files are living and which are dated records. |

## Layout

```
crates/game/       rules, board, move generation, termination
crates/mcts/       PUCT search with a pull-based batched API
crates/encoder/    position -> (14,9,9) planes; shared by trainer and browser
crates/selfplay/   ORT inference, selfplay-batch, eval-match
crates/wasm/       browser bindings for the search
model/             PyTorch training loop, replay buffer, evaluation, reporting
web/               Next.js app: play against the trained network, or analyse
config/            run configs: dry_run, validation, medium, standard
docs/              design docs and dated records
```

## Running things

```bash
# Rust tests, then the Python suite
cargo test --release
uv run pytest tests/ -q

# A three-generation smoke test of the whole loop (minutes, not hours)
uv run python -m model.train_loop --config config/dry_run.yaml

# A real run
uv run python -m model.train_loop --config config/medium.yaml

# Read a run back — trajectory, ladder, per-head generalisation, warnings
uv run python -m model.report --run latest

# Play the trained network in a browser (loads web/public/models/best.onnx)
cd web && npm install && npm run dev

# Re-measure a finished run against anchors of your choosing
ABALONE_USE_COREML=1 uv run python -m model.posthoc_ladder --run <run-id>
```

## Things that will cost you an afternoon

- **`ABALONE_USE_COREML=1` is not set for you.** `train_loop` sets it for the
  subprocesses it spawns; a hand-run `eval-match` or `selfplay-batch` silently
  takes the CPU path instead. A 12× throughput fix once measured as no change
  because of this.
- **Release builds break after an Xcode CLT upgrade** with
  `ld: library 'clang_rt.osx' not found`. Fix with
  `cargo clean --release -p ort-sys -p zstd-sys`. Debug builds hide it.
- **Exclude `runs/` from Spotlight.** A run writes ~200 parquet shards per
  generation; indexing them cost about 1.6 cores continuously and grew as the
  run went on. There is a `.metadata_never_index` marker in `runs/`, but
  System Settings → Spotlight → Search Privacy is the reliable version.
- **Metrics prefixed `data_` describe the held-out *dataset*, not the model.**
  They are constant by construction on the frozen holdout. Never alarm on them.
- **The two validation holdouts are not interchangeable.** `val_frozen` is a
  fixed position set and the only value number comparable across generations;
  `val_rolling` is a slice of the newest generation and detects memorisation.
  They disagree exactly when the curriculum is working. See MODEL.md §8.1.
