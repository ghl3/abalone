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
  /** Searched Q of the best root move, White's POV. */
  rootEval: number;
  wdlWhite: [number, number, number] | null;
  expectedScoreWhite: number | null;
  bestIdx: number;
  topMoves: ScoredMove[];
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
  /** Eval given up by this move, from the mover's POV. Positive is worse. */
  loss: number;
  quality: MoveQuality;
  bestIdx: number;
  bestNotation: string | null;
}

/** Eval given up, in [0, 2] units of the searched Q. The bands are wide
 *  because the underlying eval is a 3M-parameter network at a few hundred
 *  simulations, not a tablebase. */
const INACCURACY = 0.08;
const BLUNDER = 0.2;

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
      rootEval: res.rootEval,
      wdlWhite: res.rootNet?.wdlWhite ?? null,
      expectedScoreWhite: res.rootNet?.expectedScoreWhite ?? null,
      bestIdx: res.bestIdx,
      topMoves: res.topMoves,
    });
    opts.onProgress?.(ply + 1, total);
    if (res.bestIdx < 0) break; // terminal
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
    // Q is White's POV throughout; flip it for Black so "loss" always means
    // the player to move made things worse for themselves.
    const sign = side === 1 ? 1 : -1;
    const loss =
      after === null ? 0 : sign * before.rootEval - sign * after.rootEval;
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
