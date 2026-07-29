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
  ScoredMove,
  SearchRequest,
  SearchSnapshot,
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
/** How far to follow each move's line. Long enough to show the idea, short
 *  enough that the tail — where visit counts thin out to single digits and the
 *  line stops meaning much — stays off the screen. */
const PV_LENGTH = 6;
/** Progress ticks carry the whole tree, not just a counter, so the panel can
 *  refine its rows in place. Still cheap: reading the root's children and six
 *  PV nodes is arena indexing, and it is bounded by wall-clock, not by batch
 *  count, so a fast provider does not flood the main thread. */
const PROGRESS_INTERVAL_MS = 120;
/** Stands in for a `score` head the model does not have. */
const EMPTY = new Float32Array(0);

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
/** Serialises `runSearch` calls. See the `search` branch of `onmessage`. */
let searchQueue: Promise<void> = Promise.resolve();

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

/** `WasmSearch` has a private constructor — it only ever comes out of
 *  `begin_search` — so its type is named through that return, not `InstanceType`. */
type WasmSearchHandle = ReturnType<
  InstanceType<WasmModule["WasmGame"]>["begin_search"]
>;

/** Read the tree as it currently stands. `result()` is valid mid-search — it
 *  reports the arena as-is — so the same function serves a progress tick and
 *  the final answer, and there is exactly one definition of "what the search
 *  thinks" for the UI to render.
 *
 *  Everything here comes out of the tree, including the win/draw/loss split and
 *  the marble margin: `Search` accumulates those alongside the scalar it backs
 *  up (`track_outcome_stats`, on for the browser only). There is no forward
 *  pass in this function and no network read anywhere on this path. */
function readSnapshot(
  mod: WasmModule,
  search: WasmSearchHandle
): SearchSnapshot | null {
  const result = search.result();
  if (!result) return null;

  const indices = Array.from(result.indices());
  const evals = Array.from(result.evals());
  const visits = Array.from(result.visits());
  const wdl = result.wdl();
  const margins = result.margins();
  const rootEval = result.root_eval();
  const rootWdlRaw = result.root_wdl();
  const rootMargin = result.root_margin() ?? null;
  result.free();

  const triple = (src: Float32Array, i: number) =>
    src.length >= (i + 1) * 3
      ? ([src[i * 3], src[i * 3 + 1], src[i * 3 + 2]] as [number, number, number])
      : null;

  const all = indices.map((idx, i) => ({
    idx,
    evalWhite: evals[i],
    visits: visits[i],
    wdlWhite: triple(wdl, i),
    marginWhite: i < margins.length ? margins[i] : null,
  }));
  all.sort((a, b) => b.visits - a.visits);

  // Notation and the PV walk are only done for the rows that will be shown.
  // Doing it for all ~50 legal moves on every 120 ms tick is the difference
  // between a progress message and a stall.
  const topMoves: ScoredMove[] = all.slice(0, TOP_N).map((m) => {
    const pv = Array.from(search.principal_variation(m.idx, PV_LENGTH));
    return {
      ...m,
      notation: mod.move_notation(m.idx),
      pv,
      pvNotation: pv.map((idx) => mod.move_notation(idx)),
    };
  });

  return {
    bestIdx: search.best_index(),
    rootEval,
    rootWdlWhite: triple(rootWdlRaw, 0),
    rootMarginWhite: rootMargin,
    topMoves,
    totalVisits: visits.reduce((s, v) => s + v, 0),
  };
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
  const started = performance.now();
  let batches = 0;
  let lastProgress = started;

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

      const policy = out.policy_logits.data as Float32Array;
      const value = out.value.data as Float32Array;
      // The score head rides along so the margin can be backed up with
      // everything else. A model exported without one sends an empty array,
      // and the margin column simply has nothing to report.
      const score = (out.score?.data as Float32Array) ?? EMPTY;
      search.submit(policy, value, score);
      batches++;

      const now = performance.now();
      if (now - lastProgress > PROGRESS_INTERVAL_MS) {
        lastProgress = now;
        const msg: ProgressMsg = {
          kind: "progress",
          id: req.id,
          visits: search.root_visits(),
          simulations: req.simulations,
          snapshot: readSnapshot(mod, search),
        };
        post(msg);
      }
    }

    post({
      kind: "result",
      id: req.id,
      snapshot: readSnapshot(mod, search) ?? {
        bestIdx: -1,
        rootEval: 0,
        rootWdlWhite: null,
        rootMarginWhite: null,
        topMoves: [],
        totalVisits: 0,
      },
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
      // Bump first, so a search already in flight abandons at its next batch
      // boundary rather than spending its whole budget on a stale position.
      generation++;
      // Then queue behind it. `onmessage` is async, so two searches posted in
      // quick succession — a position change, a depth change, React's
      // strict-mode double mount — would otherwise both be inside
      // `session.run` at once, and `onnxruntime-web` permits exactly one run
      // per session: the second throws "Session mismatch" and the engine
      // wedges. The wait is short by construction, because the search being
      // superseded returns as soon as its current forward pass resolves.
      searchQueue = searchQueue
        .catch(() => {})
        .then(() => runSearch(msg));
      await searchQueue;
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
