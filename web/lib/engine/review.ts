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
  /** Searched Q of *every* legal move, White's POV. What makes grading a move
   *  against its alternatives possible for any move, rather than only for the
   *  five the panel happens to display. */
  allMoves: { idx: number; evalWhite: number; visits: number }[];
}

/** The mover's expected score: `P(win) + P(draw)/2`, on 0…1.
 *
 *  This, and not `P(win)`, is what a move's cost has to be measured in. Draw
 *  mass accumulates as a game runs on, and every point that leaves *both*
 *  players' win column for the draw column reads as a loss for whoever just
 *  moved — and reads as one again for their opponent next ply. Measured in
 *  `P(win)` alone, every move in a quiet game looks like a small mistake, with
 *  no move responsible for any of it. Expected score is neutral to that shift.
 *
 *  It is also exactly what the eval graph plots, since `rootEval` is
 *  `P(win) − P(loss)` and so `score = (rootEval + 1) / 2` — the curve and the
 *  cost column now describe one axis instead of two. `rootWdlWhite` agrees with
 *  `rootEval` by construction (`[0] − [2] === rootEval`, protocol.ts), so there
 *  is no second path to keep in step. */
export function scoreFor(side: 0 | 1, read: PlyRead): number {
  const q = side === 1 ? read.rootEval : -read.rootEval;
  return (q + 1) / 2;
}

/** Same, for one root move's searched Q. */
function scoreOf(side: 0 | 1, evalWhite: number): number {
  return ((side === 1 ? evalWhite : -evalWhite) + 1) / 2;
}

/** What the move cost against the best move available *from the same
 *  position*, both numbers read off the same tree.
 *
 *  Preferred over comparing the reads either side of the move, because that
 *  subtracts two independent searches: their disagreement includes every point
 *  the deeper look revised, which is the engine changing its mind rather than
 *  the player doing anything. Within one search the comparison is like for
 *  like, and — the part that shows — it is exactly zero when the move played
 *  *was* the engine's pick, so "best" and a cost can no longer contradict each
 *  other on the same row.
 *
 *  Reads `allMoves`, which covers every legal move, and falls back to the five
 *  display rows only for a read taken before that field existed. Null after
 *  that, which is the one case still needing the across-move measure. */
function lossVersusBest(
  side: 0 | 1,
  read: PlyRead,
  played: number
): number | null {
  if (read.bestIdx < 0) return null;
  if (played === read.bestIdx) return 0;
  const rows = read.allMoves.length > 0 ? read.allMoves : read.topMoves;
  const playedRow = rows.find((m) => m.idx === played);
  const bestRow = rows.find((m) => m.idx === read.bestIdx);
  if (!playedRow || !bestRow) return null;
  return Math.max(
    0,
    (scoreOf(side, bestRow.evalWhite) - scoreOf(side, playedRow.evalWhite)) * 100
  );
}

/** Severity of a played move. Deliberately three grades, not five: at review
 *  depth the eval is not precise enough to defend finer distinctions, and a
 *  scale nobody trusts is worse than a coarse one they do. */
export type MoveQuality = "best" | "good" | "inaccuracy" | "blunder";

/** Where a move's `loss` came from. `alternatives` is the honest one — the
 *  played move against the best move in the same search. `swing` is the
 *  fallback for a move the search never ranked, and carries the engine's own
 *  revision along with the player's mistake; the UI says so on hover rather
 *  than presenting the two as interchangeable. */
export type LossBasis = "alternatives" | "swing" | "unknown";

export interface ReviewedMove {
  /** Position index the move was played from. */
  ply: number;
  idx: number;
  notation: string;
  /** 0 = Black, 1 = White. Black moves from even plies. */
  side: 0 | 1;
  evalBefore: number;
  evalAfter: number | null;
  /** Points of expected score the mover gave up. Positive is worse, and zero
   *  whenever the move played was the engine's own choice.
   *
   *  Points rather than eval units because "this cost you 6 points" is a
   *  sentence and "−0.12" is a unit you have to be taught. The two are one
   *  measurement: `Δscore = Δeval / 2` exactly, no assumption about the draw
   *  share required, so the bands below are the old ones restated. */
  loss: number;
  lossBasis: LossBasis;
  quality: MoveQuality;
  bestIdx: number;
  bestNotation: string | null;
}

/** Expected score given up, in points. The bands are wide because the
 *  underlying estimate is a 3M-parameter network at a few hundred simulations,
 *  not a tablebase. */
/** Exported because the move list colours the number at the same threshold it
 *  is labelled at; two copies of "4" would drift apart. */
export const INACCURACY_POINTS = 4;
const BLUNDER_POINTS = 10;

export function classify(loss: number, played: number, best: number): MoveQuality {
  if (played === best) return "best";
  if (loss < INACCURACY_POINTS) return "good";
  if (loss < BLUNDER_POINTS) return "inaccuracy";
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
      allMoves: res.snapshot.allMoves,
    });
    opts.onProgress?.(ply + 1, total);
    if (res.snapshot.bestIdx < 0) break; // terminal
  }
  return reads;
}

/** Join the move record to the sweep. A move is judged against the
 *  alternatives the search saw beside it, so a move is gradeable as soon as the
 *  position it was played from has been read — the final move included. Only
 *  the fallback needs the read on the far side. */
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

    const versusBest = lossVersusBest(side, before, moves[i]);
    // Both scores are taken from the mover's side, so a positive swing always
    // means the player to move made things worse for themselves — and it
    // survives the change of turn, because the eval names its sides rather
    // than relying on whose move it is. Clamped: a position that improved is
    // not a move that cost something.
    const swing =
      after === null
        ? null
        : Math.max(0, (scoreFor(side, before) - scoreFor(side, after)) * 100);

    const loss = versusBest ?? swing ?? 0;
    const lossBasis: LossBasis =
      versusBest !== null ? "alternatives" : swing !== null ? "swing" : "unknown";

    const bestIdx = before.bestIdx;
    out.push({
      ply: i,
      idx: moves[i],
      notation: notations[i],
      side,
      evalBefore: before.rootEval,
      evalAfter: after?.rootEval ?? null,
      loss,
      lossBasis,
      quality: classify(loss, moves[i], bestIdx),
      bestIdx,
      bestNotation: null,
    });
  }
  return out;
}
