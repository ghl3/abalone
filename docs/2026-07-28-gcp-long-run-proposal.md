# Proposal: a long self-play run on rented GCP hardware

*2026-07-28*

**Verdict: worth doing, and money is not the constraint.** A run long enough to
approach convergence costs **under $200 of spot compute**. The constraint is
that the inference path is CoreML-only, so roughly **one to two days of
engineering** stands between here and a machine that can be rented at all.

This document sizes the machine from measured throughput, names the code that
has to change, and defines what "nearly converged" means in terms the repo
already computes.

---

## 1. What we actually know

Everything in this table is measured in this repo, not estimated.

| quantity | value | source |
| --- | --- | --- |
| self-play throughput, M1 Pro, `base`, CoreML | **22 pos/s** | `ruby-panther` gens 2, 3, 5, 13, 14 |
| … same, while Spotlight indexed `runs/` | 10–16 pos/s | gens 6–12, before `.metadata_never_index` landed |
| average simulations per position | 350 | `0.75·200 + 0.25·800`, playout-cap randomisation |
| → aggregate network evaluations | **~7,700 evals/s** | derived from the two rows above |
| FLOPs per evaluation, `base` (10×128, 3.03 M) | **0.489 GFLOP** | counted through the graph today |
| FLOPs per evaluation, `large` (14×192, 9.41 M) | **1.522 GFLOP** | same; 3.11× `base` |
| → effective accelerator throughput (ANE) | **~3.8 TFLOPS** | 7,700 × 0.489 |
| ORT **CPU** path | **139 evals/s per M1 P-core** | `ort_eval.rs` provider table |
| MCTS-side CPU cost | ~2.3 cores for 22 pos/s → **~9.6 pos/s per M1 P-core** | `selfplay_batch.rs` ("~23% of a 10-core machine") |
| training, MPS, batch 256, uncontended | 7–8.5 steps/s | `perf/train_active_seconds` ÷ `train/steps`, gens 1–5, 13–14 |
| training, same, contended by self-play | 0.7–1.7 steps/s | gens 7–12 |
| replay buffer resident cost | ~450 B/position | 55 mean legal moves × 6 B ragged + ~48 B fixed |
| shards on disk | ~7 MB/gen at 400 games | `du runs/ruby-panther/shards` |
| anchor ladder, 6 rungs × 32 games | ~490 s | gens 13–14, post-padding-fix |

Two facts drive every decision below.

**The network is tiny and the tree is not.** At 0.489 GFLOP per evaluation the
forward pass is nothing; the MCTS descent, backup, move generation and plane
encoding around it cost ~9.6 pos/s per fast core. This is a **CPU-bound
workload with a GPU-shaped inner loop**, and it wants the machine with the most
cores per accelerator, not the biggest accelerator.

**The CPU inference path is a dead end.** 139 evals/s per core is 68 GFLOPS.
A 128-vCPU C3 would land somewhere under 40 pos/s — under 2× the laptop, at
roughly 2× the hourly price of the GPU box recommended below. Rent an
accelerator or don't rent anything.

---

## 2. The blocker: inference is CoreML-only

[crates/selfplay/Cargo.toml:25](crates/selfplay/Cargo.toml#L25) builds `ort`
with the `coreml` feature and nothing else, and
[ort_eval.rs:136](crates/selfplay/src/ort_eval.rs#L136) is the only place an
execution provider is ever attached. On a Linux GPU box today, `selfplay-batch`
runs the **ORT CPU provider** — the 139 evals/s path — and the run would be
slower than the laptop it left.

The good news is that everything else is portable:

- `ort` 2.0.0-rc.10 has a `cuda` feature, and `ort-sys`'s `dist.txt` carries a
  prebuilt `cu12` distribution for `x86_64-unknown-linux-gnu` (ONNX Runtime
  1.22.0). `download-binaries` fetches it — no building ORT from source.
- `_device()` in [train_loop.py:752](model/train_loop.py#L752) already falls
  through MPS → CUDA → CPU, so the trainer needs no change.
- `use_coreml` is already in `HASH_EXCLUDED`
  ([config.py:765](model/config.py#L765)), so generalising it into a backend
  selector cannot break resume on existing runs.
- `eval-match` sets fixed-batch padding unconditionally
  ([eval_match.rs:433](crates/selfplay/src/bin/eval_match.rs#L433)); only
  `selfplay-batch` gates it on `use_coreml()`.
- The Rust binaries are plain subprocesses located under `target/release`
  ([eval.py:73](model/eval.py#L73)). Nothing is macOS-specific in the plumbing.

### The work

| # | change | where | effort |
| --- | --- | --- | --- |
| 1 | Add `cuda` to the `ort` features; select `CUDAExecutionProvider` at session build | `crates/selfplay/Cargo.toml`, `ort_eval.rs::from_onnx` | 2–3 h |
| 2 | Replace `use_coreml()` with a three-way backend (`cpu`/`coreml`/`cuda`) read from one env var; keep padding on for CoreML **and** CUDA | `ort_eval.rs`, `selfplay_batch.rs:306`, `lib.rs:906` | 2 h |
| 3 | Config: `use_coreml: bool` → `inference_backend: str`, plumbed to the subprocess env | `model/config.py`, `train_loop.py:1597` | 1 h |
| 4 | fp16 ONNX export behind a flag | `model/export_onnx.py` | 2–4 h |
| 5 | Startup script + systemd unit that resumes across spot preemption | new, `deploy/` | 2 h |
| 6 | Periodic `gsutil rsync` of `runs/<id>/` to GCS | new, cron or systemd timer | 1 h |

Items 1–3 are mandatory. Item 4 is the single largest GPU lever (see §3) and
should be treated as mandatory for `large`. Items 5–6 are what make a
multi-day spot run survivable.

Two smaller gaps worth knowing about:

- `--resume latest` **raises** if no run exists yet
  ([train_loop.py:1518](model/train_loop.py#L1518)), so the systemd unit needs a
  three-line wrapper: resume if any `state.json` exists, else start fresh.
- `_start_tensorboard` passes `--bind_all`
  ([train_loop.py:1490](model/train_loop.py#L1490)). That is fine behind an SSH
  tunnel and a hole in the world with a public IP. Do not open port 6006.

---

## 3. Machine

### Recommendation: `g2-standard-32` — 32 vCPU, 128 GB, 1× NVIDIA L4, Spot

Region `us-central1-a` (check L4 capacity; `us-east4`, `europe-west4` are
fallbacks). 200 GB `pd-balanced`. Deep Learning VM image from the
`deeplearning-platform-release` family on a **CUDA 12.4+ / cuDNN 9** build —
ORT 1.22 requires cuDNN 9, so verify with `ldconfig -p | grep libcudnn.so.9`
before anything else.

**Why this shape.** G2 is the only GCP family that sells a good vCPU:GPU ratio,
and `g2-standard-32` is the best in it at **32:1**. Everything larger drops to
12:1 by bundling more L4s (`g2-standard-48` is 48 vCPU / 4 L4). The A100 and
H100 shapes go the wrong way entirely: `a2-highgpu-1g` is 12 vCPU to one A100,
which would leave the accelerator idle while twelve threads did tree search.

### Projected throughput

These are estimates, and they carry the widest error bars in this document —
which is exactly why Stage 0 in §5 exists. A GCP vCPU is a hyperthread; assume
0.4–0.5× an M1 performance core on this scalar-heavy work.

| limit | `base` fp32 | `base` fp16 | `large` fp16 |
| --- | --- | --- | --- |
| CPU ceiling (28 workers × ~4–5 pos/s) | ~110–135 pos/s | ~110–135 pos/s | ~110–135 pos/s |
| GPU ceiling (L4, 30–40% of peak on 9×9 convs) | ~50–70 pos/s | ~140–200 pos/s | ~60–90 pos/s |
| **binding constraint** | **GPU** | **CPU, ~110 pos/s** | **GPU, ~60–90 pos/s** |

Two readings:

- **fp32 wastes the machine.** The L4's fp32 peak is ~30 TFLOPS against
  ~121 TFLOPS of fp16 tensor throughput. At fp32 the GPU binds at roughly half
  the CPU ceiling. Item 4 in §2 is worth a factor of two to three.
- **The speedup depends on the preset, and not in the obvious direction.**
  Compared like for like — the same config on both machines:

  | preset | M1 Pro | `g2-standard-32`, fp16 | speedup |
  | --- | --- | --- | --- |
  | `base` (0.489 GFLOP/eval) | 22 pos/s *(measured)* | ~110 pos/s, CPU-bound | **~5×** |
  | `large` (1.522 GFLOP/eval) | ~7 pos/s *(estimated: ANE-bound, so 22 ÷ 3.11)* | ~60–90 pos/s, GPU-bound | **~9–13×** |

  The gap *widens* with model size, because the M1 is accelerator-bound at both
  presets while the L4 still has CPU headroom at `base`. The laptop pays the
  full 3.11× FLOPs penalty for `large`; the L4 pays roughly half of it.
  **`large` is the case that justifies renting**, and `base` is the case where
  the laptop is closer than it looks.

### The upgrade path, if Stage 0 says contention hurts

Training and self-play share the one L4, exactly as they share the one M1 today
— and the M1 numbers show what that costs: steps/s fell from 8.5 to 0.7–1.7
under contention. If the same shows up on the L4, move to `g2-standard-48`
(48 vCPU, 4× L4): pin the trainer to device 3 via `CUDA_VISIBLE_DEVICES`, and
round-robin self-play workers across devices 0–2 with
`CUDAExecutionProvider::default().with_device_id(i % 3)`. That is a
five-line change on top of item 1, roughly doubles the hourly rate, and roughly
halves the wall clock — near-neutral on total cost.

Start on `g2-standard-32`. It is the cheapest thing that can work, and the run
resumes onto a bigger box without losing a generation.

---

## 4. Cost

Approximate `us-central1` rates — **verify before committing**, with
`gcloud compute machine-types describe g2-standard-32 --zone us-central1-a` and
the pricing calculator.

| item | on-demand | spot |
| --- | --- | --- |
| `g2-standard-32` | ~$2.2–2.4 /hr | ~$0.90–1.00 /hr |
| 200 GB pd-balanced | ~$20 /month | same |
| egress (checkpoints + tb to GCS) | negligible | same |

| run | wall clock | spot cost |
| --- | --- | --- |
| Stage 0 shakedown (§5) | ~6 h | **~$6** |
| `base`, 150 gens × 1500 games | ~3.5 days | **~$80** |
| `large`, 150 gens × 1500 games | ~4.5 days | **~$105** |
| `large`, 200 gens × 2000 games | ~7.5 days | **~$175** |

The largest option is under $200. **Pick the configuration most likely to reach
convergence, not the cheapest one** — the difference between them is a rounding
error against a day of engineering time.

---

### What the laptop can still do

The same throughput model, run backwards, says where the laptop's ceiling
actually is. Self-play plus a ~8 min ladder, `base` unless noted:

| config | per generation | 50 gens | 200 gens |
| --- | --- | --- | --- |
| 400 games (`standard.yaml`) | ~41 min | **~34 h** | ~5.7 days |
| 1500 games | ~2.2 h | ~4.6 days | ~18.5 days |
| 1500 games, `large` | ~6.6 h | ~14 days | ~54 days |

So `standard.yaml` is a **weekend**, not a rental. The boundary is sharp:
anything up to ~50 generations at `base`/400 games is laptop territory;
anything involving `large` or >1000 games per generation is not. §11 works
through which side of that line the goal falls on.

---

## 5. Plan

### Stage 0 — shakedown, ~6 hours, ~$6

Do not size a week-long run on the estimates in §3.

1. Provision, build `--release` with the CUDA feature, verify cuDNN 9.
2. `cargo test --release && uv run pytest tests/ -q` — the golden-transcript
   tests are the check that the CUDA path produces the same search as CoreML.
3. `cargo test --release -p abalone-selfplay -- --ignored --nocapture
   measure_selfplay_throughput` with `ABALONE_BENCH_MODEL` set. This harness
   already exists ([lib.rs:895](crates/selfplay/src/lib.rs#L895)) and gives
   pos/s and evals/s directly. Run it at both presets, fp32 and fp16, and at
   `worker_threads` of 16, 24, 28, 32 to find where the CPU ceiling bites.
4. `config/dry_run.yaml`, then ~6 generations of `config/medium.yaml` at
   `net_preset: large`. **`large` has never been trained.** Six generations
   costs an hour and tells us whether it learns at all before we bet a week
   on it.
5. Rewrite §6's `gens` and `games_per_gen` from the measured pos/s.

Stage 0 gates Stage 1. If measured throughput lands under ~50 pos/s at `large`,
fall back to `base` and spend the compute on generations instead of parameters.

### Stage 1 — the run

Launch under systemd with the resume wrapper, `--tensorboard`, and the GCS
sync timer. Check in daily with `model/report.py`. Expected 4–8 days.

---

## 6. Run configuration — `config/cloud.yaml`

Deltas from `config/standard.yaml`, with the reasoning. Everything unlisted
stays as `standard.yaml` has it.

```yaml
gens: 200                    # a ceiling, not a target — see the stop rule below
net_preset: large            # 14×192, 9.41M params — contingent on Stage 0
inference_backend: cuda      # replaces use_coreml

self_play:
  games_per_gen: 1500        # 165k positions/gen
  worker_threads: 28         # 32 vCPU less the trainer and the OS
  shard_games_per_file: 8    # 188 shards/gen rather than 375

train:
  steps_per_gen_max: 25000   # 1.5 epochs over a 3.3M window is ~19,300 steps;
                             # standard.yaml's 5000 cap would bind at 0.39
  replay_buffer_gens: 20     # 20 × 165k = 3.3M positions ≈ 1.5 GB resident
  lr_schedule:               # standard.yaml's {20, 35} is sized for 50 gens
    50:  5.0e-4
    100: 2.0e-4
    150: 7.0e-5

anchor_ladder:
  trailing_gens: [1, 2, 4, 8, 16, 32]
                             # standard.yaml already argues the informative rung
                             # *moves* as per-generation gains shrink. Over 200
                             # generations it moves a long way; 32 is the rung
                             # that will still resolve at the end, and 1 is the
                             # one that stops resolving first.

retention:
  keep_last_onnx: 200        # keep every one
  keep_last_checkpoints: 10
  keep_last_shard_gens: 25   # the buffer window plus margin
```

**On `keep_last_onnx: 200`.** 200 × ~38 MB is 7.6 GB — noise against a 200 GB
disk. The first strength measurement was only recoverable *at all* because
checkpoints had been retained past a misconfigured ladder
([2026-07-28-first-strength-measurement.md](2026-07-28-first-strength-measurement.md),
method note). On a 200-generation run, post-hoc re-measurement is not a
contingency, it is the plan. `RunConfig.validate` also requires
`keep_last_onnx` to cover the deepest trailing offset, which 32 now is.

**On `games_per_gen: 1500`.** 3.75× `standard.yaml`. Distinct positions are the
scarce resource — the training step budget is divided by them — and the
per-generation compute is already dominated by self-play, so more games per
generation is close to free in overhead terms. 200 × 1500 = **300,000 games,
~33M positions**, against `standard.yaml`'s 20,000 games / 2.2M positions.

**On `large`.** MODEL.md calls it "the final run, if throughput allows", and on
an L4 it does. 3.03 M parameters is thin for a 2,562-move space; a 200-
generation run at `base` risks plateauing on capacity rather than on data, and
we would not be able to tell which from the outside. Gated on Stage 0 step 4.

---

## 7. What "nearly converged" means, concretely

The repo already computes the right number. `ladder/score_vs_gen_minus_k` is a
trailing gauntlet — the current network against its own self *k* generations
ago — and unlike a fixed anchor it never saturates. `standard.yaml` puts it
well: a flat 0.75 against `gen − 4` means the learning rate is holding; a drift
toward 0.50 means it has stalled.

**Stop when all three hold for five consecutive generations:**

1. `ladder/score_vs_gen_minus_8 ≤ 0.55` — eight generations of training no
   longer buys a measurable win. (32 games per rung resolves ~±120 Elo; 0.55 is
   the edge of what it can see, so this is "below the instrument", not "zero".)
2. `selfplay/mean_plies` inside 120–200 with
   `selfplay/natural_termination_rate ≥ 0.85` — MODEL.md §8.2 criterion 4, the
   one the validation run failed while every headline number improved.
3. `val_rolling` value CE flat, and `buffer/epochs_this_gen` still in single
   figures — not converged by memorisation.

**If `score_vs_gen_minus_8` is still above 0.65 at generation 200**, the run is
data-limited, not converged. The answer then is more games per generation, not
more generations, and that is a second run rather than an extension of this one.

Note what this cannot establish. The engine has still only ever been measured
against itself, `random`, and `heuristic@100`. "Converged against its own
history" and "plays Abalone well" remain different claims, and only the first
will be supported.

---

## 8. Monitoring

### TensorBoard — SSH tunnel over IAP

Event files are always written to `runs/<id>/tb/`; `--tensorboard` serves them.
The run serves `runs/` as a whole, so every generation of every run compares
side by side.

```bash
gcloud compute ssh abalone-run --zone us-central1-a --tunnel-through-iap \
  -- -NL 6006:localhost:6006
# then http://localhost:6006
```

No external IP, no firewall rule, no unauthenticated TensorBoard on the public
internet. The VM should have **no** external address; IAP handles both SSH and
the tunnel.

### The offline fallback

A systemd timer running `gsutil -m rsync -r runs/<id>/tb gs://<bucket>/<id>/tb`
every few minutes, plus `metrics.jsonl`, `state.json` and `checkpoints/`. Then
`gsutil -m rsync` down to a local directory and run TensorBoard against it.
This works when the VM is preempted, it works from a phone, and it is the
backup that means a lost VM is not a lost run. (Direct `gs://` log dirs need
extra packages that plain `tensorboard` does not ship — rsync down instead of
fighting that.)

### The human-readable view

`model/report.py` is the thing to actually read daily — trajectory, ladder,
per-head generalisation, warnings:

```bash
gcloud compute ssh abalone-run --zone us-central1-a --tunnel-through-iap \
  --command 'cd ~/abalone && uv run python -m model.report --run latest'
```

Worth adding to the timer as a file written into the GCS bucket, so a day's
summary is one `gsutil cat` away.

---

## 9. Risks

| risk | severity | mitigation |
| --- | --- | --- |
| CUDA path diverges from CoreML — different search, silently | **high** | Golden-transcript tests in Stage 0 step 2 before anything long runs |
| 28 ORT CUDA sessions exhaust 24 GB of L4 memory (each builds its own arena) | medium | Cap the arena, or drop to 16 workers × batch 64; worst case, item 1's follow-on is a shared batching evaluator fed by `Search::next_batch`, which the pull-based API already supports |
| GPU contention between trainer and self-play | medium | Measured in Stage 0; escape hatch is `g2-standard-48` (§3) |
| `large` fails to train, or trains worse than `base` | medium | Six generations at `medium.yaml` scale in Stage 0, before committing |
| Spot preemption | low | State is per-generation; a preemption costs ≤ one generation (~35 min). systemd `Restart=always` + resume wrapper |
| L4 capacity unavailable in-region | low | Three fallback regions; the run is region-agnostic |
| fp16 export changes network behaviour | low | Compare fp16 and fp32 ONNX head-to-head over a 32-game `eval-match` in Stage 0 |

---

## 10. Recommendation in one paragraph

Spend a day or two adding a CUDA execution provider and an fp16 export, then
rent a `g2-standard-32` spot instance and run `large` at 1500 games per
generation until the trailing-8 gauntlet drops below 0.55 — expect four to
eight days and under $200. Do not skip Stage 0: the throughput estimates in §3
are the weakest numbers here, `large` has never been trained, and six hours and
six dollars buys certainty on both. Monitor with TensorBoard over an IAP tunnel
and a GCS rsync, and read `model/report.py` daily.

---

## 11. …but run `standard.yaml` on the laptop first

The two are not alternatives, and the ordering is not a compromise.

**A 10-hour overnight run settles nothing.** At 41 min/generation it reaches
generation ~15, which is where `ruby-panther` already stopped. It would
re-measure a number we have.

**`standard.yaml` — 50 generations, ~34 hours — settles a lot, for free.** It
answers the question its own header poses: whether 75–90 Elo/generation survives
to generation 50. Every parameter guessed in §6 is currently extrapolated from
14 generations:

- If `score_vs_gen_minus_8` is still ≥ 0.65 at generation 50, the cloud run is
  justified on evidence rather than on prediction, and `games_per_gen` and the
  LR schedule get sized from 50 generations of trajectory instead of 14.
- If `val_rolling` pulls away from training loss while the gauntlet flattens,
  the constraint is **capacity**, and `large` is the right call.
- If generalisation holds and the gauntlet flattens anyway, the constraint is
  **data**, and the money belongs in `games_per_gen`, not parameters.
- If it converges outright, that is $200 and a week of engineering saved.

**Critically, it costs nothing to run it in parallel with the port.** Items 1–4
of §2 are Rust and config work that cannot be *tested* on this machine anyway —
there is no CUDA device here, so the test loop is Stage 0 on rented hardware
regardless. And self-play is ANE-bound at ~23% of ten cores, so there is ample
CPU left for `cargo build` while it runs.

**The order, then:** launch `standard.yaml` tonight and give it the ~34 hours it
needs; write the CUDA backend while it runs; read the generation-50 gauntlet;
size and launch the cloud run from that. The laptop run is a prerequisite for
the cloud run, not a substitute for it.

The honest prediction is that `standard.yaml` will *not* converge — 2.2M
positions is small, and at generation 14 `ruby-panther` showed no sign of
slowing, with mean plies still climbing (113 → 118) and the curriculum only just
retired to its floor. But that is a prediction, and it costs one weekend of
otherwise-idle laptop to test.

See also: [MODEL.md](MODEL.md) §8.2 for the success criteria and §9 for the
compute budget this supersedes,
[2026-07-28-first-strength-measurement.md](2026-07-28-first-strength-measurement.md)
for the 75–90 Elo/generation figure the sizing assumes,
[config/standard.yaml](../config/standard.yaml) for the single-machine run this
scales up from.
