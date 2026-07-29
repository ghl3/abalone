"use client";

import {
  SIDE_TINT,
  formatMargin,
  percent,
  tintFor,
} from "@/lib/outcomeFormat";
import type { ScoredMove } from "@/lib/engine/protocol";

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
const COLUMNS = "16px 1fr 40px 38px 40px 48px";

/** One outcome probability. Draw is left untinted — the two win columns are
 *  the ones that carry a side, and tinting all three would make the row read
 *  as decoration rather than as three related numbers. */
function Pct({ value, tint }: { value?: number; tint?: string }) {
  return (
    <span
      style={{
        position: "relative",
        textAlign: "right",
        fontSize: 12,
        color: value == null ? "var(--faint)" : (tint ?? "var(--muted)"),
      }}
    >
      {value == null ? "·" : percent(value)}
    </span>
  );
}

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
  // Visit share is measured against the most-visited move rather than the
  // total: what the bar is for is "how far ahead of the alternatives is the
  // top move", and normalising by the total makes every bar short in a wide
  // position and tells you nothing about the comparison you care about.
  const maxVisits = Math.max(1, ...topMoves.map((m) => m.visits));
  const progress = busyTarget > 0 ? Math.min(1, busyVisits / busyTarget) : 0;

  // The detail area falls back to the top move so it always has something to
  // show and never changes height. Hovering only ever swaps its contents.
  const detail = topMoves.find((m) => m.idx === hoveredIdx) ?? topMoves[0] ?? null;

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

      <div
        style={{
          display: "grid",
          gridTemplateColumns: COLUMNS,
          gap: 8,
          padding: "0 8px",
          // 9px and no letter-spacing because "OUTCOME" and "MARBLES" set wider
          // than the columns they head and ran into each other.
          fontSize: 9,
          textTransform: "uppercase",
          color: "var(--faint)",
          borderTop: "1px solid var(--border)",
          paddingTop: 10,
          whiteSpace: "nowrap",
        }}
      >
        <span />
        <span>move</span>
        <span style={{ textAlign: "right" }} title="Chance White wins">
          white
        </span>
        <span style={{ textAlign: "right" }} title="Chance the game ends drawn">
          draw
        </span>
        <span style={{ textAlign: "right" }} title="Chance Black wins">
          black
        </span>
        <span
          style={{ textAlign: "right" }}
          title="Expected final capture differential, White-positive"
        >
          marbles
        </span>
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
        {topMoves.length === 0 && (
          <div style={{ color: "var(--muted)", fontSize: 12, padding: "4px 8px" }}>
            {terminal
              ? "Game over — no moves to search."
              : busy
                ? "Searching…"
                : "No legal moves."}
          </div>
        )}
        {topMoves.map((m, i) => {
          const active = hoveredIdx === m.idx;
          const share = m.visits / maxVisits;
          return (
            <div
              key={m.idx}
              role="button"
              tabIndex={0}
              // Announces what the row shows, in the same terms. A label that
              // describes different quantities from the visible columns is a
              // second interface to keep in step, and it will fall behind.
              aria-label={
                m.notation +
                (m.wdlWhite
                  ? `, White ${percent(m.wdlWhite[0])}, draw ${percent(
                      m.wdlWhite[1]
                    )}, Black ${percent(m.wdlWhite[2])}`
                  : "") +
                (m.marginWhite != null
                  ? `, margin ${formatMargin(m.marginWhite)} marbles for White`
                  : "")
              }
              onMouseEnter={() => onHover(m.idx)}
              onMouseLeave={() => onHover(null)}
              onFocus={() => onHover(m.idx)}
              onBlur={() => onHover(null)}
              onClick={() => onApply(m.idx)}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  onApply(m.idx);
                }
              }}
              style={{
                position: "relative",
                display: "grid",
                gridTemplateColumns: COLUMNS,
                gap: 8,
                alignItems: "center",
                padding: "7px 8px",
                borderRadius: 4,
                cursor: "pointer",
                fontSize: 13,
                fontFamily: "var(--mono)",
                background: active ? "var(--accent-soft)" : "transparent",
                transition: "background 60ms",
              }}
            >
              {/* Visit share, drawn behind the row. The count itself is gone
                  from the table — "412 visits" is an engine-internal unit that
                  means nothing without knowing the budget — but *how much of
                  the search went here* is the most informative thing in an
                  MCTS readout, and as a bar it needs no unit at all. The
                  number is still in the detail area for anyone who wants it. */}
              <span
                aria-hidden
                style={{
                  position: "absolute",
                  left: 0,
                  top: 0,
                  bottom: 0,
                  width: `${share * 100}%`,
                  borderRadius: 4,
                  background: "var(--border-strong)",
                  opacity: active ? 0.55 : 0.75,
                  transition: "width 200ms cubic-bezier(0.2,0.8,0.2,1)",
                  pointerEvents: "none",
                }}
              />
              <span style={{ color: "var(--faint)", position: "relative" }}>
                {i + 1}
              </span>
              <span style={{ position: "relative" }}>{m.notation}</span>
              <Pct value={m.wdlWhite?.[0]} tint={SIDE_TINT.white} />
              <Pct value={m.wdlWhite?.[1]} />
              <Pct value={m.wdlWhite?.[2]} tint={SIDE_TINT.black} />
              <span
                style={{
                  position: "relative",
                  textAlign: "right",
                  fontSize: 12,
                  color:
                    m.marginWhite != null ? tintFor(m.marginWhite) : "var(--faint)",
                }}
              >
                {m.marginWhite != null ? formatMargin(m.marginWhite) : "·"}
              </span>
            </div>
          );
        })}
      </div>

      {/* Detail for whichever move is being pointed at. A fixed region below
          the table rather than a row that expands in place: expanding pushed
          every row under it down, so reaching for row 3 moved row 3 out from
          under the pointer. Height is constant and it defaults to the top
          move, so nothing in this panel ever reflows. */}
      <div
        style={{
          borderTop: "1px solid var(--border)",
          paddingTop: 9,
          // Fixed, not content-sized: swapping which move it describes must
          // not change its height, or the legend below it twitches — a smaller
          // version of the reflow that moving this out of the table was meant
          // to end.
          height: 72,
          overflow: "hidden",
          fontSize: 11,
        }}
      >
        {detail ? (
          <Detail move={detail} onHoverLine={onHoverLine} />
        ) : (
          <span style={{ color: "var(--faint)" }}>No move selected.</span>
        )}
      </div>

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

function Detail({
  move,
  onHoverLine,
}: {
  move: ScoredMove;
  onHoverLine: (rootIdx: number, step: number | null) => void;
}) {
  // Neither the raw eval nor the visit count appears here. Both are engine
  // units for things the table already says better: the eval is the same axis
  // as the win/draw/loss percentages, and visit share is the bar behind each
  // row. What is left is the one thing nothing else shows — the line.
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
      <div
        style={{
          fontFamily: "var(--mono)",
          fontSize: 10,
          letterSpacing: "0.04em",
          textTransform: "uppercase",
          color: "var(--faint)",
        }}
      >
        line after {move.notation}
      </div>

      {move.pvNotation.length > 1 ? (
        <div
          onMouseLeave={() => onHoverLine(move.idx, null)}
          style={{
            display: "flex",
            flexWrap: "wrap",
            gap: 3,
            height: 40,
            overflow: "hidden",
            alignContent: "flex-start",
          }}
        >
          {move.pvNotation.map((n, step) => (
            <span
              key={step}
              className="pv-step"
              onMouseEnter={() => onHoverLine(move.idx, step)}
              style={{
                padding: "1px 4px",
                borderRadius: 3,
                cursor: "pointer",
                fontFamily: "var(--mono)",
                fontSize: 11,
                color: step === 0 ? "var(--text)" : "var(--muted)",
                background: step === 0 ? "var(--surface-raised)" : "transparent",
              }}
            >
              {n}
            </span>
          ))}
        </div>
      ) : (
        <div style={{ height: 40, color: "var(--faint)", fontSize: 10 }}>
          Search did not look past this move.
        </div>
      )}
    </div>
  );
}
