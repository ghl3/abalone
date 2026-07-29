/** Post-game review: search every position the game passed through, then read
 *  the move actually played against what the search wanted.
 *
 *  The whole feature rests on one asymmetry — during the game the engine only
 *  ever searched its own turns, so the interesting half of the record (yours)
 *  has never been looked at. A sweep afterwards is cheap: on WebGPU a position
 *  at review depth costs well under a tenth of a second, so a full game is
 *  seconds, not minutes.
 */

import type { Opening, ScoredMove, SearchResultMsg } from "./protocol";

/** What the engine thinks of one position in the game. */
export interface PlyRead {
  /** Number of moves played before this position; 0 is the opening. */
  ply: number;
  /** Searched Q of the best root move, White's POV. Identically
   *  `P(win) − P(loss)`, which is why the graph can plot it as "who is ahead"
   *  while every number on screen is a probability. */
  rootEval: number;
  /** Searched `[P(win), P(draw), P(loss)]` for White. */
  wdlWhite: [number, number, number] | null;
  expectedScoreWhite: number | null;
  bestIdx: number;
  topMoves: ScoredMove[];
}

/** The mover's chance of winning, from a read. `wdlWhite` is White's, so Black
 *  reads it from the other end. Falls back to splitting the eval when a read
 *  predates the searched distributions. */
export function winChanceFor(side: 0 | 1, read: PlyRead): number {
  if (read.wdlWhite) return side === 1 ? read.wdlWhite[0] : read.wdlWhite[2];
  const q = side === 1 ? read.rootEval : -read.rootEval;
  return (q + 1) / 2;
}

/** Severity of a played move. Deliberately three grades, not five: at review
 *  depth the eval is not precise enough to defend finer distinctions, and a
 *  scale nobody trusts is worse than a coarse one they do. */
export type MoveQuality = "best" | "good" | "inaccuracy" | "blunder";

export interface ReviewedMove {
  /** Position index the move was played from. */
  ply: number;
  idx: number;
  notation: string;
  /** 0 = Black, 1 = White. Black moves from even plies. */
  side: 0 | 1;
  evalBefore: number;
  evalAfter: number | null;
  /** Percentage points of winning chance the mover gave up. Positive is worse.
   *
   *  Points rather than eval units because "this cost you 6 points of winning
   *  chances" is a sentence, and "−0.12" is a unit you have to be taught. The
   *  two are the same measurement — `Δwin ≈ Δeval / 2` at a fixed draw share —
   *  so the bands below are the old ones restated, not loosened. */
  loss: number;
  quality: MoveQuality;
  bestIdx: number;
  bestNotation: string | null;
}

/** Winning chance given up, in percentage points. The bands are wide because
 *  the underlying estimate is a 3M-parameter network at a few hundred
 *  simulations, not a tablebase. */
const INACCURACY = 4;
const BLUNDER = 10;

export function classify(loss: number, played: number, best: number): MoveQuality {
  if (played === best) return "best";
  if (loss < INACCURACY) return "good";
  if (loss < BLUNDER) return "inaccuracy";
  return "blunder";
}

export const QUALITY_LABEL: Record<MoveQuality, string> = {
  best: "Best",
  good: "",
  inaccuracy: "Inaccuracy",
  blunder: "Blunder",
};

/** Status colours, validated for CVD separation and contrast against the
 *  panel surface. Every use is paired with the text label above — colour is
 *  never the only carrier. */
export const QUALITY_COLOR: Record<MoveQuality, string> = {
  best: "#5dadec",
  good: "transparent",
  inaccuracy: "#f6c343",
  blunder: "#e2645c",
};

export interface SweepOptions {
  opening: Opening;
  moves: number[];
  simulations: number;
  batchSize: number;
  search: (req: {
    opening: Opening;
    moves: number[];
    simulations: number;
    batchSize: number;
  }) => Promise<SearchResultMsg | null>;
  /** Called after each position so the UI can show progress. */
  onProgress?: (done: number, total: number) => void;
  /** Checked between positions; abandons the sweep when it returns true. */
  isCancelled?: () => boolean;
}

/** Search every position from the opening through to the final one, in order.
 *
 *  Sequential rather than parallel on purpose: the worker holds a single ORT
 *  session and one search tree, so overlapping requests would only supersede
 *  each other. Going in play order also means the eval graph fills in left to
 *  right, which makes the progress legible rather than abstract.
 */
export async function sweepGame(opts: SweepOptions): Promise<PlyRead[]> {
  const { opening, moves, simulations, batchSize, search } = opts;
  const total = moves.length + 1;
  const reads: PlyRead[] = [];

  for (let ply = 0; ply < total; ply++) {
    if (opts.isCancelled?.()) break;
    const res = await search({
      opening,
      moves: moves.slice(0, ply),
      simulations,
      batchSize,
    });
    // A null result means the search was superseded, and a terminal position
    // yields no root moves. Both end the sweep: there is nothing after a
    // finished game, and a superseded search means the user has moved on.
    if (!res) break;
    reads.push({
      ply,
      rootEval: res.snapshot.rootEval,
      wdlWhite: res.snapshot.rootWdlWhite,
      expectedScoreWhite: res.snapshot.rootMarginWhite,
      bestIdx: res.snapshot.bestIdx,
      topMoves: res.snapshot.topMoves,
    });
    opts.onProgress?.(ply + 1, total);
    if (res.snapshot.bestIdx < 0) break; // terminal
  }
  return reads;
}

/** Join the move record to the sweep. A move is judged by what the eval did
 *  across it, which needs the read on *both* sides — so the last move can only
 *  be graded once the final position has been searched. */
export function reviewMoves(
  moves: number[],
  notations: string[],
  reads: PlyRead[]
): ReviewedMove[] {
  const out: ReviewedMove[] = [];
  for (let i = 0; i < moves.length; i++) {
    const before = reads[i];
    if (!before) break;
    const after = reads[i + 1] ?? null;
    const side: 0 | 1 = i % 2 === 0 ? 0 : 1;
    // Both reads are taken from the mover's side, so "loss" always means the
    // player to move made things worse for themselves — and it survives the
    // change of turn, because `wdlWhite` names its sides rather than relying
    // on whose move it is.
    const loss =
      after === null
        ? 0
        : (winChanceFor(side, before) - winChanceFor(side, after)) * 100;
    const bestIdx = before.bestIdx;
    out.push({
      ply: i,
      idx: moves[i],
      notation: notations[i],
      side,
      evalBefore: before.rootEval,
      evalAfter: after?.rootEval ?? null,
      loss,
      quality:
        after === null ? "good" : classify(loss, moves[i], bestIdx),
      bestIdx,
      bestNotation: null,
    });
  }
  return out;
}
