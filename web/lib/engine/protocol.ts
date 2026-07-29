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

export interface ScoredMove {
  idx: number;
  notation: string;
  /** Q from White's POV: positive favours White. **Searched** — this is the
   *  value backed up through the tree, and it is the number the move list
   *  ranks and the eval bar shows. */
  evalWhite: number;
  visits: number;
  /** `[P(win), P(draw), P(loss)]` for **White** after this move. Searched:
   *  accumulated through the tree alongside `evalWhite` and visit-weighted the
   *  same way, so `wdlWhite[0] - wdlWhite[2]` reproduces `evalWhite`. */
  wdlWhite: [number, number, number] | null;
  /** Expected final capture differential after this move, signed for White.
   *  Also searched — the score head is backed up like the value head. Null if
   *  the model has no `score` head. */
  marginWhite: number | null;
  /** The line search explored under this move — move indices along the
   *  most-visited path, starting with this move itself. Read off the tree the
   *  search already built, so it costs no extra inference. */
  pv: number[];
  /** `pv` rendered as notation, so the UI never has to call into wasm to
   *  print a line it is only going to display. */
  pvNotation: string[];
}

/** What the search currently believes. Identical in shape whether it comes
 *  from a finished search or a progress tick mid-flight — the panel renders
 *  one thing and never has to decide whether it is looking at a real answer
 *  or a placeholder. */
export interface SearchSnapshot {
  /** Most-visited root move, or -1 before the root has been expanded. */
  bestIdx: number;
  /** Searched Q of the most-visited root child, from White's POV. */
  rootEval: number;
  /** The position's win/draw/loss for White, taken from the most-visited child
   *  — the line the engine intends — rather than averaged over moves it has
   *  already rejected. Same provenance as `rootEval`, and consistent with it:
   *  `[0] - [2] === rootEval`. */
  rootWdlWhite: [number, number, number] | null;
  /** The position's expected capture differential, signed for White. */
  rootMarginWhite: number | null;
  topMoves: ScoredMove[];
  /** Every legal move with its searched Q and visit count, ranked with
   *  `topMoves` and including it — no notation, no principal variation.
   *
   *  `topMoves` is capped at five because generating notation and walking a PV
   *  for ~50 moves on every 120 ms tick is a stall. The Q values themselves are
   *  free: the search returns all of them and the cap was only ever about the
   *  presentation work. Review needs them, because grading a move against the
   *  best one in the same search requires the move the player actually chose to
   *  be *in* the data — and a human's move is often not in the top five. */
  allMoves: { idx: number; evalWhite: number; visits: number }[];
  /** Visits across *all* root children, not just the ones in `topMoves`. */
  totalVisits: number;
}

export interface SearchResultMsg {
  kind: "result";
  id: number;
  snapshot: SearchSnapshot;
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
  /** The tree so far. Null only until the root expands, which is the first
   *  forward pass — after that every tick carries usable rows. */
  snapshot: SearchSnapshot | null;
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
