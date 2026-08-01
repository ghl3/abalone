"use client";

import { formatMargin, tintFor } from "@/lib/outcomeFormat";
import type { ScoredMove } from "@/lib/engine/protocol";
import MoveTable from "./MoveTable";

/** Named search budgets. Three, not a slider: the old 50–2000 range in steps
 *  of 50 offered forty settings nobody could tell apart, and dragging it fired
 *  a fresh search at every step on the way past. Strength goes with the log of
 *  the budget, so these are spaced geometrically — each one is a visibly
 *  different engine, and picking one is a single click that cancels once. */
export const DEPTHS = [
  { key: "quick", label: "Quick", sims: 120 },
  { key: "standard", label: "Standard", sims: 500 },
  { key: "deep", label: "Deep", sims: 2000 },
] as const;
export type DepthKey = (typeof DEPTHS)[number]["key"];

interface Props {
  topMoves: ScoredMove[];
  totalVisits: number;
  /** 0 = Black, 1 = White. */
  turnSide: 0 | 1;
  hoveredIdx: number | null;
  onHover: (idx: number | null) => void;
  /** Hovering a move *within* a line: `step` indexes into that move's `pv`,
   *  and `null` means the pointer has left the line and the board should go
   *  back to showing the move itself. */
  onHoverLine: (rootIdx: number, step: number | null) => void;
  onApply: (idx: number) => void;
  depth: DepthKey;
  onDepthChange: (d: DepthKey) => void;
  /** Search in flight. Rows keep updating underneath it — they are this
   *  position's, not the last one's. */
  busy: boolean;
  busyVisits: number;
  busyTarget: number;
  /** One-line provenance: provider, timing, load state. */
  footer?: string;
  /** Expected final capture differential at the root, signed for White. */
  marginWhite?: number | null;
  /** Nothing to search — the game is over. */
  terminal?: boolean;
}

const SIDE_NAME = ["Black", "White"] as const;

export default function AnalysisPanel({
  topMoves,
  totalVisits,
  turnSide,
  hoveredIdx,
  onHover,
  onHoverLine,
  onApply,
  depth,
  onDepthChange,
  busy,
  busyVisits,
  busyTarget,
  footer,
  marginWhite,
  terminal = false,
}: Props) {
  const progress = busyTarget > 0 ? Math.min(1, busyVisits / busyTarget) : 0;

  return (
    <div
      className="panel"
      style={{
        width: 360,
        // Sized to its contents rather than stretched to the board. Stretching
        // left a fixed amount of content floating above a few hundred pixels of
        // nothing, and reserving space for the move detail (below) is only
        // worth doing at the size the detail actually needs.
        alignSelf: "flex-start",
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
          {SIDE_NAME[turnSide]} to move
        </div>
        {marginWhite != null && (
          <div
            style={{
              fontSize: 11,
              color: "var(--muted)",
              fontFamily: "var(--mono)",
            }}
            title="Expected final capture differential for this position, White-positive. Searched, from the engine's intended line."
          >
            <span style={{ color: tintFor(marginWhite) }}>
              {formatMargin(marginWhite)}
            </span>{" "}
            marbles
          </div>
        )}
      </div>

      {/* Search state, always occupying the same height whether or not a
          search is running. The bar used to appear and vanish with the rows
          it describes, which moved everything below it on every move. */}
      <div
        style={{
          height: 3,
          borderRadius: 2,
          background: "var(--well)",
          overflow: "hidden",
        }}
        role="progressbar"
        aria-valuemin={0}
        aria-valuemax={busyTarget}
        aria-valuenow={busy ? busyVisits : busyTarget}
        aria-label="Search progress"
      >
        <div
          style={{
            width: busy ? `${progress * 100}%` : "100%",
            height: "100%",
            background: busy ? "var(--accent)" : "var(--border-strong)",
            transition: "width 140ms linear, background 200ms",
          }}
        />
      </div>

      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: 8,
        }}
      >
        <div className="segmented" role="group" aria-label="Search depth">
          {DEPTHS.map((d) => (
            <button
              key={d.key}
              className="segment"
              aria-pressed={depth === d.key}
              onClick={() => onDepthChange(d.key)}
              title={`${d.sims} simulations`}
            >
              {d.label}
            </button>
          ))}
        </div>
        <span
          style={{
            fontSize: 11,
            fontFamily: "var(--mono)",
            color: busy ? "var(--accent)" : "var(--faint)",
          }}
        >
          {busy ? `${busyVisits}/${busyTarget}` : `${busyTarget} sims`}
        </span>
      </div>

      <MoveTable
        rows={topMoves.map((m, i) => ({ move: m, rank: i + 1 }))}
        hoveredIdx={hoveredIdx}
        onHover={onHover}
        onHoverLine={onHoverLine}
        onApply={onApply}
        emptyText={
          terminal
            ? "Game over — no moves to search."
            : busy
              ? "Searching…"
              : "No legal moves."
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
        <div>
          Chances of each result, and the margin in marbles, after this move —
          all searched, not a single network read.
        </div>
        {footer && <div style={{ marginTop: 3 }}>{footer}</div>}
      </div>
    </div>
  );
}
