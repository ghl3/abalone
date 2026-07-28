/** The network-guided engine, off the UI thread.
 *
 * Search lives in Rust (the same `abalone_mcts::Search` self-play drives) and
 * inference lives in `onnxruntime-web`, so the two have to interleave. WASM
 * cannot await a JS promise, so the search is driven as a coroutine from here:
 * `next_batch()` hands out the leaves it wants evaluated, `run()` evaluates
 * them, `submit()` backs the values up. See docs/ARCHITECTURE.md §2.5.
 *
 * A worker rather than the UI thread because the loop is CPU-bound in bursts:
 * on the wasm execution provider a forward pass blocks whichever thread it
 * runs on, and dragging a marble through that is unpleasant.
 */

import * as ort from "onnxruntime-web";
import type {
  ProgressMsg,
  RootNetRead,
  ScoredMove,
  SearchRequest,
  WorkerRequest,
  WorkerResponse,
} from "./protocol";

type WasmModule = typeof import("abalone-wasm");

/** `self` is typed as a `Window` here because the project compiles with the
 *  DOM lib; pulling in the webworker lib instead would break every other file.
 *  Narrowing it to the two members this file uses is cheaper than maintaining
 *  a second tsconfig for one module. */
const ctx = self as unknown as {
  postMessage(msg: WorkerResponse): void;
  onmessage: ((e: MessageEvent<WorkerRequest>) => void) | null;
};

const TOP_N = 5;
/** Progress messages are for a spinner, not a readout; keep them cheap. */
const PROGRESS_INTERVAL_MS = 120;

let wasm: WasmModule | null = null;
/** Resolves once the ONNX session exists. A `search` posted immediately after
 *  `init` — which is exactly what the hook does — arrives while the `init`
 *  handler is still awaiting `InferenceSession.create`, so search must wait on
 *  the promise rather than read a variable that is not set yet. */
let sessionPromise: Promise<{
  session: ort.InferenceSession;
  provider: string;
}> | null = null;
/** Bumped by `cancel` and by each new request: the search loop checks it
 *  between batches and abandons a superseded search rather than finishing it. */
let generation = 0;

function post(msg: WorkerResponse) {
  ctx.postMessage(msg);
}

async function loadWasm(): Promise<WasmModule> {
  if (!wasm) wasm = await import("abalone-wasm");
  return wasm;
}

/** Create the ORT session, preferring WebGPU and falling back to the wasm
 *  CPU backend. Returns the session and the provider that actually took. */
async function createSession(
  modelPath: string
): Promise<{ session: ort.InferenceSession; provider: string }> {
  ort.env.wasm.wasmPaths = "/ort/";
  // Multi-threading needs SharedArrayBuffer, which needs cross-origin
  // isolation (the COOP/COEP headers set in next.config.mjs). Without it ORT
  // throws rather than silently degrading, so ask for what we can have.
  ort.env.wasm.numThreads = crossOriginIsolated
    ? Math.min(navigator.hardwareConcurrency || 4, 8)
    : 1;

  const providers: string[] = [];
  if ((navigator as Navigator & { gpu?: unknown }).gpu) providers.push("webgpu");
  providers.push("wasm");

  let lastErr: unknown = null;
  for (const provider of providers) {
    try {
      const session = await ort.InferenceSession.create(modelPath, {
        executionProviders: [provider],
        graphOptimizationLevel: "all",
      });
      return { session, provider };
    } catch (e) {
      lastErr = e;
    }
  }
  throw lastErr ?? new Error("no execution provider available");
}

function softmax(logits: Float32Array | number[]): number[] {
  let max = -Infinity;
  for (const x of logits) if (x > max) max = x;
  const exps = Array.from(logits, (x) => Math.exp(x - max));
  const sum = exps.reduce((s, x) => s + x, 0);
  return sum > 0 ? exps.map((x) => x / sum) : exps;
}

/** Class order of the 3-way value head, matching `model/batch.py`. */
const VALUE_WIN = 0;
const VALUE_DRAW = 1;
const VALUE_LOSS = 2;
/** The score head is a softmax over a capture differential of −6..+6. */
const SCORE_OFFSET = 6;

/** Pull the value and score heads out of the root's forward pass and restate
 *  them from White's POV. Both heads are side-to-move relative (the encoder's
 *  planes 0/1 are own/opponent), so playing Black means reading them backwards. */
function readRootHeads(
  out: ort.InferenceSession.OnnxValueMapType,
  blackToMove: boolean
): RootNetRead {
  const value = out.value.data as Float32Array;
  const p = softmax(value.subarray(0, 3));
  const win = p[VALUE_WIN];
  const draw = p[VALUE_DRAW];
  const loss = p[VALUE_LOSS];
  const wdlWhite: [number, number, number] = blackToMove
    ? [loss, draw, win]
    : [win, draw, loss];

  let expectedScoreWhite: number | null = null;
  const scoreOut = out.score;
  if (scoreOut) {
    const logits = scoreOut.data as Float32Array;
    const q = softmax(logits.subarray(0, 13));
    // E[d] over the side to move, then signed for White.
    const expected = q.reduce((s, pi, i) => s + pi * (i - SCORE_OFFSET), 0);
    expectedScoreWhite = blackToMove ? -expected : expected;
  }
  return { wdlWhite, expectedScoreWhite };
}

/** Rebuild the position by replaying `moves` from the opening. Cheaper than it
 *  looks (a few hundred microseconds for a full game) and it keeps the wire
 *  format free of any board representation. */
function positionFrom(mod: WasmModule, req: SearchRequest) {
  const game =
    req.opening === "belgian" ? mod.WasmGame.belgian_daisy() : new mod.WasmGame();
  for (const idx of req.moves) game.apply_index(idx);
  return game;
}

async function runSearch(req: SearchRequest) {
  if (!sessionPromise) throw new Error("engine not initialised");
  const [mod, { session }] = await Promise.all([loadWasm(), sessionPromise]);

  const myGeneration = generation;
  const game = positionFrom(mod, req);
  const blackToMove = game.turn() === mod.WasmSide.Black;
  const started = performance.now();
  let batches = 0;
  let lastProgress = started;
  let rootNet: RootNetRead | null = null;

  const search = game.begin_search(
    req.simulations,
    req.batchSize,
    1.4,
    // Deterministic per position: the same board searched twice gives the
    // same move, which makes a surprising engine move reproducible.
    (req.moves.length * 2654435761 + req.simulations) >>> 0
  );

  try {
    for (;;) {
      const planes = search.next_batch();
      if (planes.length === 0) break;
      const n = search.batch_len();

      const input = new ort.Tensor("float32", planes, [n, 14, 9, 9]);
      const out = await session.run({ planes: input });

      // A newer request (or a cancel) landed while inference was in flight.
      if (generation !== myGeneration) return;

      // The first batch is always the root alone — that is how `Search` opens,
      // because the root needs its priors before any descent can happen — so
      // this reads the root's own heads with no extra inference.
      if (batches === 0) rootNet = readRootHeads(out, blackToMove);

      const policy = out.policy_logits.data as Float32Array;
      const value = out.value.data as Float32Array;
      search.submit(policy, value);
      batches++;

      const now = performance.now();
      if (now - lastProgress > PROGRESS_INTERVAL_MS) {
        lastProgress = now;
        const msg: ProgressMsg = {
          kind: "progress",
          id: req.id,
          visits: search.root_visits(),
          simulations: req.simulations,
        };
        post(msg);
      }
    }

    const result = search.result();
    const topMoves: ScoredMove[] = [];
    let totalVisits = 0;
    let rootEval = 0;
    if (result) {
      const indices = Array.from(result.indices());
      const evals = Array.from(result.evals());
      const visits = Array.from(result.visits());
      rootEval = result.root_eval();
      result.free();
      totalVisits = visits.reduce((s, v) => s + v, 0);
      const all = indices.map((idx, i) => ({
        idx,
        notation: mod.move_notation(idx),
        evalWhite: evals[i],
        visits: visits[i],
      }));
      all.sort((a, b) => b.visits - a.visits);
      topMoves.push(...all.slice(0, TOP_N));
    }

    post({
      kind: "result",
      id: req.id,
      bestIdx: search.best_index(),
      rootEval,
      rootNet,
      topMoves,
      totalVisits,
      elapsedMs: performance.now() - started,
      batches,
    });
  } finally {
    search.free();
    game.free();
  }
}

ctx.onmessage = async (e: MessageEvent<WorkerRequest>) => {
  const msg = e.data;
  try {
    if (msg.kind === "init") {
      const started = performance.now();
      sessionPromise = createSession(msg.modelPath);
      const [{ provider }] = await Promise.all([sessionPromise, loadWasm()]);
      post({
        kind: "ready",
        provider,
        threads: ort.env.wasm.numThreads ?? 1,
        modelPath: msg.modelPath,
        loadMs: performance.now() - started,
      });
    } else if (msg.kind === "search") {
      generation++;
      await runSearch(msg);
    } else if (msg.kind === "cancel") {
      generation++;
    }
  } catch (err) {
    post({
      kind: "error",
      id: "id" in msg ? msg.id : undefined,
      message: err instanceof Error ? err.message : String(err),
    });
  }
};
