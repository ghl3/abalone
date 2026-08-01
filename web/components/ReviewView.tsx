"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import HexBoard, { DIR_SHIFTS, type LastMove, type MovingState } from "./HexBoard";
import PlayerPlate from "./PlayerPlate";
import EvalGraph from "./EvalGraph";
import MoveTable, { type MoveRowData } from "./MoveTable";
import { buildHoverPreview } from "@/lib/boardPreview";
import { useEngine } from "@/lib/engine/useEngine";
import { formatMargin, tintFor } from "@/lib/outcomeFormat";
import type { Opening } from "@/lib/engine/protocol";
import { loadReview, reviewKey, saveReview } from "@/lib/engine/reviewCache";
import {
  INACCURACY_POINTS,
  QUALITY_COLOR,
  QUALITY_LABEL,
  reviewMoves,
  sweepGame,
  type PlyRead,
} from "@/lib/engine/review";

type WasmModule = typeof import("abalone-wasm");

/** The game being reviewed, snapshotted when review is entered so that
 *  starting a new game does not pull the record out from under it. */
export interface ReviewGame {
  opening: Opening;
  moves: number[];
  /** Colour the human had, or null for a game played by hand on both sides. */
  playerSide: 0 | 1 | null;
  difficulty: string | null;
}

interface Props {
  wasm: WasmModule;
  game: ReviewGame;
  onExit: () => void;
}

/** Review depth. Higher than the engine plays at casually, because a review is
 *  a judgement about moves already made: being slow is fine, being wrong is not.
 *
 *  Was 200, which measurement did not support. `review-probe` swept 909
 *  positions across four games at 200/800/3200 simulations
 *  (docs/NOTEBOOK.md, 2026-07-29). At 200 the most-visited root move held a
 *  median 12% of the visits and agreed with a 3200-simulation search on 43% of
 *  positions; it labelled 60% of moves "best" where the deep search allowed
 *  24%. At 800 that is 19%, 63% and 36%, and the share of real mistakes caught
 *  rises from 69% to 81% with false positives at 0 of 71. The whole panel
 *  depends on *which* move is best — the label, the "engine wants" line, the
 *  hover preview, and the yardstick the cost is measured against — so a
 *  4× sweep buys the feature its premise.
 *
 *  Not free: the sweep is one search per position, so a 60-move game is 4× the
 *  work it was. The progress bar was already there; this is what it is for —
 *  and a finished sweep is cached (reviewCache.ts), so it is paid once per
 *  game per model rather than once per page load. */
const REVIEW_SIMS = 800;
const NN_BATCH_SIZE = 16;
const SIDE_NAME = ["Black", "White"] as const;

export default function ReviewView({ wasm, game, onExit }: Props) {
  const engine = useEngine();
  const [reads, setReads] = useState<PlyRead[]>([]);
  const [done, setDone] = useState(0);
  const [ply, setPly] = useState(0);
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);
  /** A move within a shown line, being previewed on the board: the line's move
   *  indices and how far along it to replay. Same shape as analysis. */
  const [linePreview, setLinePreview] = useState<{
    moves: number[];
    step: number;
  } | null>(null);

  const total = game.moves.length + 1;
  const upSide: 0 | 1 = game.playerSide ?? 1;
  const flipped = upSide === 0;

  const { search } = engine;
  useEffect(() => {
    let cancelled = false;
    setReads([]);
    setDone(0);
    (async () => {
      // The cache is consulted under the model's identity, which needs one
      // HEAD request the first time; after that this is synchronous in
      // practice. A hit skips the sweep entirely — same reads, no work.
      const key = await reviewKey({
        opening: game.opening,
        moves: game.moves,
        sims: REVIEW_SIMS,
      });
      if (cancelled) return;
      const cached = loadReview(key);
      if (cached) {
        setReads(cached);
        setDone(total);
        return;
      }
      const r = await sweepGame({
        opening: game.opening,
        moves: game.moves,
        simulations: REVIEW_SIMS,
        batchSize: NN_BATCH_SIZE,
        search,
        isCancelled: () => cancelled,
        onProgress: (d) => {
          if (!cancelled) setDone(d);
        },
      });
      if (cancelled) return;
      setReads(r);
      // Only a finished sweep is worth remembering: it either read every
      // position or stopped at a terminal one. A sweep abandoned mid-game
      // would otherwise be served forever as if it were the whole answer.
      const complete =
        r.length === total || (r.length > 0 && r[r.length - 1].bestIdx < 0);
      if (complete) saveReview(key, r);
    })();
    return () => {
      cancelled = true;
    };
  }, [game, search, total]);

  // Everything the board needs, derived in one pass that creates the position,
  // reads it, and frees it before returning. Holding a wasm handle across
  // renders and freeing it from an effect is what produced "null pointer
  // passed to rust": React's strict-mode remount runs the unmount cleanup and
  // then reuses the memoised handle it just freed. Plain data cannot dangle.
  //
  // A hovered line replays here too — the game to `ply`, then the line as far
  // as the pointer has walked it — so the board, the plates and the turn light
  // all describe the previewed position for free.
  const view = useMemo(() => {
    const g =
      game.opening === "belgian"
        ? wasm.WasmGame.belgian_daisy()
        : new wasm.WasmGame();
    try {
      for (let i = 0; i < ply; i++) g.apply_index(game.moves[i]);
      const lineMoves = linePreview
        ? linePreview.moves.slice(0, linePreview.step + 1)
        : [];
      for (const idx of lineMoves) g.apply_index(idx);

      const cells = new Int8Array(81);
      for (let c = 0; c < 81; c++) cells[c] = g.cell(c);
      const turn = g.turn() as 0 | 1;

      const lastIdx =
        lineMoves.length > 0
          ? lineMoves[lineMoves.length - 1]
          : ply > 0
            ? game.moves[ply - 1]
            : null;
      let lastMove: LastMove | null = null;
      if (lastIdx != null) {
        const from = Array.from(g.move_source_cells(lastIdx)).filter(
          (c) => c !== 0xff
        );
        const shift = DIR_SHIFTS[wasm.move_motion_dir(lastIdx)];
        lastMove = { fromCells: from, toCells: from.map((c) => c + shift) };
      }

      // The hover preview needs a live position too, so it is built here
      // rather than in its own memo. Replaying a game costs microseconds; a
      // second lifetime to reason about does not.
      const suggestion: MovingState | null =
        lineMoves.length > 0 || hoveredIdx == null
          ? null
          : buildHoverPreview(g, wasm, cells, turn, hoveredIdx);

      return {
        cells,
        turn,
        lostBlack: g.lost(wasm.WasmSide.Black),
        lostWhite: g.lost(wasm.WasmSide.White),
        lastMove,
        suggestion,
        lineDepth: lineMoves.length,
      };
    } finally {
      g.free();
    }
  }, [wasm, game, ply, hoveredIdx, linePreview]);

  const notations = useMemo(
    () => game.moves.map((m) => wasm.move_notation(m)),
    [game.moves, wasm]
  );

  const reviewed = useMemo(
    () => reviewMoves(game.moves, notations, reads),
    [game.moves, notations, reads]
  );

  const seek = useCallback(
    (p: number) => {
      setPly(Math.max(0, Math.min(total - 1, p)));
      // A previewed move or line belongs to the position it was read from.
      setHoveredIdx(null);
      setLinePreview(null);
    },
    [total]
  );

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "ArrowLeft") seek(ply - 1);
      else if (e.key === "ArrowRight") seek(ply + 1);
      else if (e.key === "Home") seek(0);
      else if (e.key === "End") seek(total - 1);
      else return;
      e.preventDefault();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [ply, seek, total]);

  const read = reads.find((r) => r.ply === ply) ?? null;
  const sweeping = done < total && reads.length === 0;
  const topSide: 0 | 1 = flipped ? 1 : 0;
  const bottomSide: 0 | 1 = flipped ? 0 : 1;
  const name = (s: 0 | 1) =>
    game.playerSide === null
      ? SIDE_NAME[s]
      : s === game.playerSide
        ? "You"
        : "Network";

  const plate = (s: 0 | 1) => (
    <PlayerPlate
      side={s}
      name={name(s)}
      // Only when it adds something. A game played by hand on both sides has
      // no "you", so `name` is already the colour and the detail line was
      // rendering "Black" under "Black".
      detail={name(s) === SIDE_NAME[s] ? undefined : SIDE_NAME[s]}
      captures={s === 1 ? view.lostBlack : view.lostWhite}
      isTurn={view.turn === s}
    />
  );

  // Keep the row for the current ply in view while arrowing through the game.
  // Scrolls the list by its own `scrollTop` rather than calling
  // `scrollIntoView`, which walks up to every scrollable ancestor and takes the
  // page along with it. Only moves when the row has actually left the box, so
  // reading down a visible stretch of moves does not re-centre under you.
  const listRef = useRef<HTMLDivElement | null>(null);
  const currentRowRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    const list = listRef.current;
    const row = currentRowRef.current;
    if (!list || !row) return;
    const lb = list.getBoundingClientRect();
    const rb = row.getBoundingClientRect();
    const margin = 4;
    if (rb.top < lb.top) list.scrollTop -= lb.top - rb.top + margin;
    else if (rb.bottom > lb.bottom) list.scrollTop += rb.bottom - lb.bottom + margin;
  }, [ply, reviewed.length]);

  const blunders = reviewed.filter((m) => m.quality === "blunder");
  const yourBlunders = blunders.filter(
    (m) => game.playerSide === null || m.side === game.playerSide
  );

  // Who moves from the position on screen — by parity off the record, not
  // from `view`, which follows a hovered line: the panel describes the
  // position the game was at, whatever the board is momentarily showing.
  const baseTurn: 0 | 1 = ply % 2 === 0 ? 0 : 1;
  const terminal = read !== null && read.bestIdx < 0;

  // What the panel on the right shows: the searched candidates for this
  // position, with the move actually played tagged and, when it fell outside
  // the displayed five, appended at its true rank. `allMoves` covers every
  // legal move precisely so that a human move — often nowhere near the top —
  // still has a row to point at.
  const panelRows = useMemo<MoveRowData[]>(() => {
    if (!read || read.bestIdx < 0) return [];
    const rows: MoveRowData[] = read.topMoves.map((m, i) => ({
      move: m,
      rank: i + 1,
    }));
    if (ply < game.moves.length) {
      const playedIdx = game.moves[ply];
      const quality = reviewed[ply]?.quality;
      const tag = {
        label: "played",
        // The tag doubles as the verdict, in the same colours as the move
        // list. "Good" has no colour of its own there (its label is empty),
        // so it falls back to muted rather than to transparent text.
        color:
          quality && QUALITY_COLOR[quality] !== "transparent"
            ? QUALITY_COLOR[quality]
            : "var(--muted)",
      };
      const inTop = rows.find((r) => r.move.idx === playedIdx);
      if (inTop) {
        inTop.tag = tag;
      } else {
        const rank = read.allMoves.findIndex((m) => m.idx === playedIdx);
        if (rank >= 0) {
          const am = read.allMoves[rank];
          rows.push({
            move: {
              idx: playedIdx,
              notation: notations[ply],
              evalWhite: am.evalWhite,
              visits: am.visits,
              // The sweep keeps full outcome rows only for the top five; the
              // table prints "·" for what was not retained.
              wdlWhite: null,
              marginWhite: null,
              pv: [],
              pvNotation: [],
            },
            rank: rank + 1,
            tag,
          });
        }
      }
    }
    return rows;
  }, [read, ply, game.moves, notations, reviewed]);

  const handleHover = useCallback((idx: number | null) => {
    setHoveredIdx(idx);
    if (idx === null) setLinePreview(null);
  }, []);

  const handleHoverLine = useCallback(
    (rootIdx: number, step: number | null) => {
      if (step === null) {
        setLinePreview(null);
        return;
      }
      const row = panelRows.find((r) => r.move.idx === rootIdx);
      if (row && row.move.pv.length > 0) {
        setLinePreview({ moves: row.move.pv, step });
      }
    },
    [panelRows]
  );

  return (
    <div
      style={{
        display: "flex",
        gap: 18,
        alignItems: "flex-start",
        justifyContent: "center",
        flexWrap: "wrap",
      }}
    >
      {/* The game, top to bottom in the order you read it: position (board),
          trajectory (graph), record (move list). All three share an x — click
          anywhere in any of them and the other two follow. */}
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          gap: 10,
          width: "fit-content",
        }}
      >
        {plate(topSide)}
        <HexBoard
          cells={view.cells}
          selection={[]}
          moving={view.suggestion}
          lastMove={view.lastMove}
          onCellPointerDown={() => {}}
          flipped={flipped}
          idle
        />
        {plate(bottomSide)}

        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <button className="btn" onClick={() => seek(0)} aria-label="First move">
            ⏮
          </button>
          <button className="btn" onClick={() => seek(ply - 1)} aria-label="Previous move">
            ◀
          </button>
          <input
            type="range"
            min={0}
            max={total - 1}
            value={ply}
            onChange={(e) => seek(parseInt(e.target.value, 10))}
            style={{ flex: 1, accentColor: "var(--accent)" }}
            aria-label="Ply"
          />
          <button className="btn" onClick={() => seek(ply + 1)} aria-label="Next move">
            ▶
          </button>
          <button className="btn" onClick={() => seek(total - 1)} aria-label="Last move">
            ⏭
          </button>
          <span
            style={{
              fontFamily: "var(--mono)",
              fontSize: 12,
              color: "var(--muted)",
              minWidth: 54,
              textAlign: "right",
            }}
          >
            {ply}/{total - 1}
          </span>
        </div>

        {/* Fixed height whether or not a line is being walked, so the panel
            below does not jump when a hover starts. */}
        <div
          style={{
            height: 14,
            fontSize: 11,
            color: "var(--highlight)",
            padding: "0 2px",
          }}
        >
          {view.lineDepth > 0 &&
            `showing the line, ${view.lineDepth} move${
              view.lineDepth === 1 ? "" : "s"
            } ahead`}
        </div>

        <div
          className="panel"
          style={{
            alignSelf: "stretch",
            padding: "12px 14px",
            display: "flex",
            flexDirection: "column",
            gap: 8,
          }}
        >
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "baseline",
            }}
          >
            <div style={{ fontSize: 13, fontWeight: 600 }}>Game review</div>
            <button
              className="btn"
              style={{ padding: "3px 8px", fontSize: 11 }}
              onClick={onExit}
            >
              Close
            </button>
          </div>

          {sweeping ? (
            <div style={{ fontSize: 12, color: "var(--muted)" }}>
              Analysing {done}/{total} positions…
              <div
                style={{
                  marginTop: 6,
                  height: 3,
                  borderRadius: 2,
                  background: "var(--well)",
                  overflow: "hidden",
                }}
              >
                <div
                  style={{
                    width: `${(done / total) * 100}%`,
                    height: "100%",
                    background: "var(--accent)",
                    transition: "width 200ms",
                  }}
                />
              </div>
            </div>
          ) : (
            <>
              <EvalGraph
                reads={reads}
                moves={reviewed}
                currentPly={ply}
                onSeek={seek}
                upSide={upSide}
                totalPlies={total}
              />
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  fontSize: 10,
                  color: "var(--faint)",
                }}
              >
                <span>▲ {name(upSide)} ahead</span>
                <span>▼ {name(upSide === 0 ? 1 : 0)} ahead</span>
              </div>
            </>
          )}

        </div>
      </div>

      {/* The position and the record share the right column: what the options
          were, then where the game went. The list used to sit under the graph,
          which pushed it below the fold — the record is consulted constantly
          while scrubbing, and it belongs beside the board, not beneath it. */}
      <div
        style={{
          width: 360,
          display: "flex",
          flexDirection: "column",
          gap: 18,
        }}
      >
        <div
          className="panel"
          style={{
            padding: 14,
            display: "flex",
            flexDirection: "column",
            gap: 10,
          }}
        >
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "baseline",
            }}
          >
            <div style={{ fontSize: 13, fontWeight: 600 }}>
              {terminal ? "Game over" : `${SIDE_NAME[baseTurn]} to move`}
            </div>
            {read?.expectedScoreWhite != null && (
              <div
                style={{
                  fontSize: 11,
                  color: "var(--muted)",
                  fontFamily: "var(--mono)",
                }}
                title="Expected final capture differential for this position, White-positive. Searched, from the engine's intended line."
              >
                <span style={{ color: tintFor(read.expectedScoreWhite) }}>
                  {formatMargin(read.expectedScoreWhite)}
                </span>{" "}
                marbles
              </div>
            )}
          </div>

          <MoveTable
            rows={panelRows}
            hoveredIdx={hoveredIdx}
            onHover={handleHover}
            onHoverLine={handleHoverLine}
            emptyText={
              sweeping
                ? "Analysing…"
                : terminal
                  ? "Game over — no moves to search."
                  : "This position was not analysed."
            }
          />

          <div
            style={{
              paddingTop: 8,
              borderTop: "1px solid var(--border)",
              color: "var(--faint)",
              fontSize: 10,
              lineHeight: 1.5,
            }}
          >
            Chances of each result, and the margin in marbles, after each move —
            searched at review depth, {REVIEW_SIMS} simulations per position.
            Hover a move to see it on the board; hover along its line to walk
            forward.
          </div>
        </div>

        <div
          className="panel"
          style={{
            padding: "10px 8px",
            display: "flex",
            flexDirection: "column",
            gap: 6,
          }}
        >
          <div style={{ fontSize: 11, color: "var(--muted)", padding: "0 6px" }}>
            {sweeping
              ? "Moves grade as the sweep reaches them."
              : yourBlunders.length === 0
                ? "No blunders found."
                : `${yourBlunders.length} blunder${
                    yourBlunders.length === 1 ? "" : "s"
                  } — click one to jump there.`}
          </div>
          <div
            ref={listRef}
            style={{
              display: "flex",
              flexDirection: "column",
              gap: 1,
              overflowY: "auto",
              maxHeight: 330,
              borderTop: "1px solid var(--border)",
              paddingTop: 6,
            }}
          >
            {reviewed.map((m) => {
              // Clicking a move lands *before* it — the position it was
              // played from, with the panel showing the choice that was
              // open. The highlighted row is therefore the move about to be
              // made, which is also the row the panel tags "played": the
              // record and the candidates agree on what "here" means.
              const isCurrent = m.ply === ply;
              const label = QUALITY_LABEL[m.quality];
              return (
                <div
                  key={m.ply}
                  ref={isCurrent ? currentRowRef : undefined}
                  onClick={() => seek(m.ply)}
                  style={{
                    display: "grid",
                    gridTemplateColumns: "26px 14px 1fr auto auto",
                    gap: 6,
                    alignItems: "center",
                    padding: "4px 6px",
                    borderRadius: 4,
                    cursor: "pointer",
                    fontSize: 12,
                    fontFamily: "var(--mono)",
                    background: isCurrent ? "var(--accent-soft)" : "transparent",
                  }}
                >
                  <span style={{ color: "var(--faint)" }}>{m.ply + 1}.</span>
                  <span
                    aria-hidden
                    style={{
                      width: 9,
                      height: 9,
                      borderRadius: "50%",
                      background:
                        m.side === 0 ? "var(--black)" : "var(--white)",
                      border: "1px solid var(--border-strong)",
                    }}
                  />
                  <span>{m.notation}</span>
                  {label && (
                    <span
                      style={{
                        color: QUALITY_COLOR[m.quality],
                        fontSize: 10,
                        textTransform: "uppercase",
                        letterSpacing: "0.04em",
                      }}
                    >
                      {label}
                    </span>
                  )}
                  {/* Expected score handed over, in points. Below half a point
                      there is nothing to report: at review depth that is noise,
                      and a column of "-0.3" on every move would bury the two
                      moves that mattered. A move the engine itself picked reads
                      zero and so prints nothing — the label has already said the
                      only thing there is to say about it. */}
                  <span
                    style={{
                      color: m.loss >= INACCURACY_POINTS ? "var(--illegal)" : "var(--faint)",
                      minWidth: 40,
                      textAlign: "right",
                    }}
                    title={
                      m.lossBasis === "swing"
                        ? "How much worse the position got across this move — measured across two searches, so it carries the engine's own second thoughts as well as yours"
                        : "Expected score given up against the engine's best move, both read from the same search"
                    }
                  >
                    {m.loss >= 0.5 ? `-${Math.round(m.loss)}%` : ""}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
