"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import HexBoard, { DIR_SHIFTS, type LastMove, type MovingState } from "./HexBoard";
import PlayerPlate from "./PlayerPlate";
import EvalGraph from "./EvalGraph";
import { buildHoverPreview } from "@/lib/boardPreview";
import { useEngine } from "@/lib/engine/useEngine";
import { SIDE_TINT, formatMargin, percent, tintFor } from "@/lib/outcomeFormat";
import type { Opening } from "@/lib/engine/protocol";
import {
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
 *  a judgement about moves already made — being slow is fine, being wrong is
 *  not — but still low enough that a 60-move game sweeps in a few seconds. */
const REVIEW_SIMS = 200;
const NN_BATCH_SIZE = 16;
const SIDE_NAME = ["Black", "White"] as const;

export default function ReviewView({ wasm, game, onExit }: Props) {
  const engine = useEngine();
  const [reads, setReads] = useState<PlyRead[]>([]);
  const [done, setDone] = useState(0);
  const [ply, setPly] = useState(0);
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);

  const total = game.moves.length + 1;
  const upSide: 0 | 1 = game.playerSide ?? 1;
  const flipped = upSide === 0;

  const { search } = engine;
  useEffect(() => {
    let cancelled = false;
    setReads([]);
    setDone(0);
    sweepGame({
      opening: game.opening,
      moves: game.moves,
      simulations: REVIEW_SIMS,
      batchSize: NN_BATCH_SIZE,
      search,
      isCancelled: () => cancelled,
      onProgress: (d) => {
        if (!cancelled) setDone(d);
      },
    }).then((r) => {
      if (!cancelled) setReads(r);
    });
    return () => {
      cancelled = true;
    };
  }, [game, search]);

  // Everything the board needs, derived in one pass that creates the position,
  // reads it, and frees it before returning. Holding a wasm handle across
  // renders and freeing it from an effect is what produced "null pointer
  // passed to rust": React's strict-mode remount runs the unmount cleanup and
  // then reuses the memoised handle it just freed. Plain data cannot dangle.
  const view = useMemo(() => {
    const g =
      game.opening === "belgian"
        ? wasm.WasmGame.belgian_daisy()
        : new wasm.WasmGame();
    try {
      for (let i = 0; i < ply; i++) g.apply_index(game.moves[i]);

      const cells = new Int8Array(81);
      for (let c = 0; c < 81; c++) cells[c] = g.cell(c);
      const turn = g.turn() as 0 | 1;

      let lastMove: LastMove | null = null;
      if (ply > 0) {
        const idx = game.moves[ply - 1];
        const from = Array.from(g.move_source_cells(idx)).filter(
          (c) => c !== 0xff
        );
        const shift = DIR_SHIFTS[wasm.move_motion_dir(idx)];
        lastMove = { fromCells: from, toCells: from.map((c) => c + shift) };
      }

      // The hover preview needs a live position too, so it is built here
      // rather than in its own memo. Replaying a game costs microseconds; a
      // second lifetime to reason about does not.
      const suggestion: MovingState | null =
        hoveredIdx == null
          ? null
          : buildHoverPreview(g, wasm, cells, turn, hoveredIdx);

      return {
        cells,
        turn,
        lostBlack: g.lost(wasm.WasmSide.Black),
        lostWhite: g.lost(wasm.WasmSide.White),
        lastMove,
        suggestion,
      };
    } finally {
      g.free();
    }
  }, [wasm, game, ply, hoveredIdx]);

  const snapshot = view;
  const lastMove = view.lastMove;
  const suggestion = view.suggestion;

  const notations = useMemo(
    () => game.moves.map((m) => wasm.move_notation(m)),
    [game.moves, wasm]
  );
  const reviewed = useMemo(
    () => reviewMoves(game.moves, notations, reads),
    [game.moves, notations, reads]
  );

  const seek = useCallback(
    (p: number) => setPly(Math.max(0, Math.min(total - 1, p))),
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
      captures={s === 1 ? snapshot.lostBlack : snapshot.lostWhite}
      isTurn={snapshot.turn === s}
    />
  );

  const blunders = reviewed.filter((m) => m.quality === "blunder");
  const yourBlunders = blunders.filter(
    (m) => game.playerSide === null || m.side === game.playerSide
  );

  return (
    <div style={{ display: "flex", gap: 18, alignItems: "flex-start" }}>
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
          cells={snapshot.cells}
          selection={[]}
          moving={suggestion}
          lastMove={lastMove}
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
      </div>

      <div
        className="panel"
        style={{
          width: 340,
          alignSelf: "stretch",
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
            <div style={{ fontSize: 11, color: "var(--muted)" }}>
              {yourBlunders.length === 0
                ? "No blunders found."
                : `${yourBlunders.length} blunder${
                    yourBlunders.length === 1 ? "" : "s"
                  } — click one to jump there.`}
            </div>
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
                marginTop: -6,
              }}
            >
              <span>▲ {name(upSide)} ahead</span>
              <span>▼ {name(upSide === 0 ? 1 : 0)} ahead</span>
            </div>
          </>
        )}

        {read && (
          <div
            style={{
              fontFamily: "var(--mono)",
              fontSize: 11,
              color: "var(--muted)",
              borderTop: "1px solid var(--border)",
              paddingTop: 8,
              display: "flex",
              flexDirection: "column",
              gap: 3,
            }}
          >
            <div>
              engine wants{" "}
              <span
                style={{ color: "var(--accent)", cursor: "pointer" }}
                onMouseEnter={() => setHoveredIdx(read.bestIdx)}
                onMouseLeave={() => setHoveredIdx(null)}
              >
                {read.bestIdx >= 0 ? wasm.move_notation(read.bestIdx) : "—"}
              </span>
            </div>
            <div style={{ fontSize: 10, display: "flex", gap: 8 }}>
              {read.wdlWhite && (
                <>
                  <span>
                    White{" "}
                    <span style={{ color: SIDE_TINT.white }}>
                      {percent(read.wdlWhite[0])}
                    </span>
                  </span>
                  <span>draw {percent(read.wdlWhite[1])}</span>
                  <span>
                    Black{" "}
                    <span style={{ color: SIDE_TINT.black }}>
                      {percent(read.wdlWhite[2])}
                    </span>
                  </span>
                </>
              )}
              {read.expectedScoreWhite != null && (
                <span>
                  <span style={{ color: tintFor(read.expectedScoreWhite) }}>
                    {formatMargin(read.expectedScoreWhite)}
                  </span>{" "}
                  marbles
                </span>
              )}
            </div>
          </div>
        )}

        <div
          style={{
            display: "flex",
            flexDirection: "column",
            gap: 1,
            overflowY: "auto",
            maxHeight: 340,
            marginTop: 2,
          }}
        >
          {reviewed.map((m) => {
            const isCurrent = m.ply === ply - 1;
            const label = QUALITY_LABEL[m.quality];
            return (
              <div
                key={m.ply}
                onClick={() => seek(m.ply + 1)}
                onMouseEnter={() =>
                  m.quality !== "best" && m.bestIdx >= 0
                    ? setHoveredIdx(m.bestIdx)
                    : undefined
                }
                onMouseLeave={() => setHoveredIdx(null)}
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
                {/* Winning chance handed over, in percentage points. Below
                    half a point there is nothing to report: at review depth
                    that is noise, and a column of "-0.3" on every move would
                    bury the two moves that mattered. */}
                <span
                  style={{
                    color: m.loss >= 4 ? "var(--illegal)" : "var(--faint)",
                    minWidth: 40,
                    textAlign: "right",
                  }}
                  title="Winning chance given up by this move"
                >
                  {m.loss >= 0.5 ? `-${Math.round(m.loss)}%` : ""}
                </span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
