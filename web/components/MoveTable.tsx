"use client";

import { SIDE_TINT, formatMargin, percent, tintFor } from "@/lib/outcomeFormat";
import type { ScoredMove } from "@/lib/engine/protocol";

/** The candidate-move table: ranked moves with their searched outcomes, the
 *  visit-share bar behind each row, and the line detail underneath. Shared by
 *  the analysis panel and the review panel so the two read as one instrument —
 *  same columns, same hover behaviour, same vocabulary — and cannot drift.
 */

/** One row: a searched move, the rank the search gave it, and an optional tag
 *  rendered beside the notation ("played", in review). Rank travels with the
 *  row rather than being the array position because review appends the move
 *  actually played at its true rank — often well outside the displayed five,
 *  which is itself the point being made. */
export interface MoveRowData {
  move: ScoredMove;
  rank: number;
  tag?: { label: string; color: string };
}

interface Props {
  rows: MoveRowData[];
  hoveredIdx: number | null;
  onHover: (idx: number | null) => void;
  /** Hovering a move *within* a line: `step` indexes into that move's `pv`,
   *  and `null` means the pointer has left the line and the board should go
   *  back to showing the move itself. */
  onHoverLine: (rootIdx: number, step: number | null) => void;
  /** Clicking a row plays it. Absent in review, where the game is a record:
   *  rows still preview on hover, they just cannot rewrite history. */
  onApply?: (idx: number) => void;
  /** Shown when there are no rows; the caller knows why there are none. */
  emptyText: string;
}

/** Rank column widened from 16px: review ranks reach two digits. The gap is
 *  tighter than it was for the same reason — the widest cell is a broadside
 *  notation plus the "played" tag, and it has to clear the outcome columns
 *  without wrapping. */
const COLUMNS = "20px 1fr 40px 38px 40px 48px";
const COL_GAP = 6;

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

export default function MoveTable({
  rows,
  hoveredIdx,
  onHover,
  onHoverLine,
  onApply,
  emptyText,
}: Props) {
  // Visit share is measured against the most-visited move rather than the
  // total: what the bar is for is "how far ahead of the alternatives is the
  // top move", and normalising by the total makes every bar short in a wide
  // position and tells you nothing about the comparison you care about.
  const maxVisits = Math.max(1, ...rows.map((r) => r.move.visits));

  // The detail area falls back to the top move so it always has something to
  // show and never changes height. Hovering only ever swaps its contents.
  const detail =
    rows.find((r) => r.move.idx === hoveredIdx)?.move ?? rows[0]?.move ?? null;

  return (
    <>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: COLUMNS,
          gap: COL_GAP,
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
        {rows.length === 0 && (
          <div style={{ color: "var(--muted)", fontSize: 12, padding: "4px 8px" }}>
            {emptyText}
          </div>
        )}
        {rows.map(({ move: m, rank, tag }) => {
          const active = hoveredIdx === m.idx;
          const share = m.visits / maxVisits;
          return (
            <div
              key={m.idx}
              role={onApply ? "button" : undefined}
              tabIndex={0}
              // Announces what the row shows, in the same terms. A label that
              // describes different quantities from the visible columns is a
              // second interface to keep in step, and it will fall behind.
              aria-label={
                m.notation +
                (tag ? `, ${tag.label}` : "") +
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
              onClick={onApply ? () => onApply(m.idx) : undefined}
              onKeyDown={
                onApply
                  ? (e) => {
                      if (e.key === "Enter" || e.key === " ") {
                        e.preventDefault();
                        onApply(m.idx);
                      }
                    }
                  : undefined
              }
              style={{
                position: "relative",
                display: "grid",
                gridTemplateColumns: COLUMNS,
                gap: COL_GAP,
                alignItems: "center",
                padding: "7px 8px",
                borderRadius: 4,
                cursor: onApply ? "pointer" : "default",
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
                {rank}
              </span>
              <span
                style={{
                  position: "relative",
                  display: "flex",
                  alignItems: "baseline",
                  gap: 6,
                  // A broadside notation contains a hyphen the line breaker
                  // would happily wrap at, doubling the row's height.
                  whiteSpace: "nowrap",
                }}
              >
                {m.notation}
                {tag && (
                  <span
                    style={{
                      fontSize: 9,
                      textTransform: "uppercase",
                      letterSpacing: "0.04em",
                      color: tag.color,
                    }}
                  >
                    {tag.label}
                  </span>
                )}
              </span>
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
    </>
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
