/** Messages between the UI thread and the engine worker.
 *
 * The worker owns the ONNX session and the WASM search tree; the UI owns the
 * game. They stay in sync by *replaying* rather than by shipping positions:
 * a request carries the opening and the move indices played from it, and the
 * worker rebuilds the position with the same Rust rules the UI used. There is
 * no board encoding on the wire, so there is no second place for the rules to
 * drift (docs/ARCHITECTURE.md §2.5).
 */

export type Opening = "standard" | "belgian";

/** Which side, if any, the network plays. */
export type AiSide = "none" | "black" | "white";

export interface SearchRequest {
  /** Echoed back on every response, so stale results can be dropped. */
  id: number;
  opening: Opening;
  /** Move indices applied in order from the opening position. */
  moves: number[];
  simulations: number;
  /** Leaves per forward pass. Larger amortises inference over more of the tree. */
  batchSize: number;
}

/** The network's opinion of the root position, straight from the heads — no
 *  search. Free to collect: the search's first batch is the root expansion, so
 *  this is the forward pass that was happening anyway.
 *
 *  Kept separate from `rootEval` on purpose. `rootEval` is the *searched* Q,
 *  a backed-up scalar that cannot be decomposed into outcome probabilities;
 *  these are the raw distributions. They can legitimately disagree, and the
 *  gap is itself informative — it is what search found. */
export interface RootNetRead {
  /** `[P(win), P(draw), P(loss)]` for **White**, summing to 1. */
  wdlWhite: [number, number, number];
  /** Expected final capture differential, signed for White (MODEL.md §6.0:
   *  "by how much", the number that is legible to a human). Null if the model
   *  has no `score` head. */
  expectedScoreWhite: number | null;
}

export interface ScoredMove {
  idx: number;
  notation: string;
  /** Q from White's POV: positive favours White. */
  evalWhite: number;
  visits: number;
}

export interface SearchResultMsg {
  kind: "result";
  id: number;
  /** Most-visited root move, or -1 for a terminal position. */
  bestIdx: number;
  /** Searched Q of the most-visited root child, from White's POV. */
  rootEval: number;
  /** Raw network read of the root. Null for a terminal position. */
  rootNet: RootNetRead | null;
  topMoves: ScoredMove[];
  totalVisits: number;
  /** Wall-clock milliseconds for the whole search. */
  elapsedMs: number;
  /** Network forward passes it took. */
  batches: number;
}

export interface ProgressMsg {
  kind: "progress";
  id: number;
  visits: number;
  simulations: number;
}

export interface ReadyMsg {
  kind: "ready";
  /** Execution provider ORT actually settled on. */
  provider: string;
  /** Threads the wasm backend was given. */
  threads: number;
  modelPath: string;
  loadMs: number;
}

export interface ErrorMsg {
  kind: "error";
  id?: number;
  message: string;
}

export type WorkerRequest =
  | { kind: "init"; modelPath: string }
  | ({ kind: "search" } & SearchRequest)
  | { kind: "cancel" };

export type WorkerResponse = ReadyMsg | SearchResultMsg | ProgressMsg | ErrorMsg;
