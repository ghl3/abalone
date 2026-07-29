"use client";

import { percent } from "@/lib/outcomeFormat";

interface Props {
  /** Searched `[P(win), P(draw), P(loss)]` for White, summing to 1. Null until
   *  the first result lands, when the bar sits level rather than guessing.
   *
   *  These come out of the tree, not off a forward pass: the value head's three
   *  classes are backed up and visit-weighted exactly as the scalar eval is.
   *  The bar therefore shows the same estimate the move list ranks by, at the
   *  same search depth — where it once mixed a searched number with an
   *  unsearched split and could visibly contradict itself. */
  wdlWhite: [number, number, number] | null;
  /** Colour occupying the bottom of the board: 0 = Black, 1 = White. The bar
   *  mirrors the board so "your side" is always the near end of both. */
  bottomSide: 0 | 1;
  /** Dim while a search is in flight, so a number being revised does not look
   *  like a number being asserted. */
  busy?: boolean;
}

/** Colour of the winning bands. The bar reads as "how much of the outcome
 *  belongs to each side", so the bands are the marble colours. */
const FILL = [
  "linear-gradient(90deg, var(--black-shade), var(--black) 55%, var(--black-lit))",
  "linear-gradient(90deg, var(--white-shade), var(--white) 55%, var(--white-lit))",
] as const;

const SIDE_NAME = ["Black", "White"] as const;

export default function WinBar({ wdlWhite, bottomSide, busy = false }: Props) {
  const topSide = (bottomSide === 0 ? 1 : 0) as 0 | 1;
  const [whiteWin, draw, blackWin] = wdlWhite ?? [0.5, 0, 0.5];
  const share = (s: 0 | 1) => (s === 1 ? whiteWin : blackWin);

  return (
    <div
      style={{
        alignSelf: "stretch",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: 6,
        opacity: busy ? 0.75 : 1,
        transition: "opacity 200ms",
      }}
    >
      {/* Both ends carry a name and a number. The bar used to say only
          "-0.13" in grey above an unlabelled column: correct, and unreadable
          without already knowing the convention it was written in. */}
      <Label side={topSide} pct={share(topSide)} />

      <div
        style={{
          width: 36,
          flex: 1,
          // Stretching to the board's height is the normal case, but on a
          // narrow window the bar wraps onto a line of its own, where `flex: 1`
          // has nothing to stretch against and it collapses to a sliver. A
          // floor keeps it a gauge instead of a hairline.
          minHeight: 150,
          borderRadius: 999,
          border: "1px solid var(--border)",
          boxShadow: "inset 0 0 8px rgba(0,0,0,0.45)",
          position: "relative",
          overflow: "hidden",
          display: "flex",
          flexDirection: "column",
        }}
        title={
          `${SIDE_NAME[topSide]} ${percent(share(topSide))} · ` +
          `draw ${percent(draw)} · ` +
          `${SIDE_NAME[bottomSide]} ${percent(share(bottomSide))}`
        }
      >
        {/* Stacked top → bottom, so the near player's band grows upward from
            the bottom edge the way the board's near side sits at the bottom. */}
        <div
          style={{
            height: `${share(topSide) * 100}%`,
            background: FILL[topSide],
            transition: "height 220ms cubic-bezier(0.2, 0.8, 0.2, 1)",
          }}
        />
        <div
          style={{
            height: `${draw * 100}%`,
            background: "var(--faint)",
            opacity: 0.5,
            transition: "height 220ms cubic-bezier(0.2, 0.8, 0.2, 1)",
          }}
        />
        <div
          style={{
            height: `${share(bottomSide) * 100}%`,
            background: FILL[bottomSide],
            transition: "height 220ms cubic-bezier(0.2, 0.8, 0.2, 1)",
          }}
        />

        {/* Centerline at 50% — the equal-position reference. */}
        <div
          style={{
            position: "absolute",
            left: 0,
            right: 0,
            top: "50%",
            height: 1,
            background: "rgba(255,255,255,0.18)",
            pointerEvents: "none",
          }}
        />

        {/* The draw share, on the draw band. It used to sit in the caption
            under the bar, which put the number nowhere near the thing it
            measured. Anchored to the band's midpoint as a percentage of the
            bar rather than to a measured pixel height, so it lands correctly
            at any bar height; it floats over the band instead of sitting
            inside it, because a thin band cannot hold text but still has a
            midpoint worth pointing at. */}
        {draw > 0.005 && (
          <span
            style={{
              position: "absolute",
              left: "50%",
              top: `${(share(topSide) + draw / 2) * 100}%`,
              transform: "translate(-50%, -50%)",
              padding: "1px 3px",
              borderRadius: 3,
              background: "rgba(8, 11, 14, 0.72)",
              color: "var(--text)",
              fontFamily: "var(--mono)",
              fontSize: 9,
              lineHeight: 1.2,
              whiteSpace: "nowrap",
              pointerEvents: "none",
            }}
            title="Chance the game ends drawn"
          >
            {percent(draw)}
          </span>
        )}
      </div>

      <Label side={bottomSide} pct={share(bottomSide)} />

      <span
        style={{
          fontSize: 9,
          letterSpacing: "0.06em",
          textTransform: "uppercase",
          color: "var(--faint)",
        }}
        title="Chance of winning from here, as the search sees it"
      >
        win
      </span>
    </div>
  );
}

function Label({ side, pct }: { side: 0 | 1; pct: number }) {
  return (
    <span
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        fontFamily: "var(--mono)",
        fontSize: 11,
        lineHeight: 1.3,
        color: "var(--text)",
      }}
      title={`${SIDE_NAME[side]} wins ${percent(pct)} of the time from here`}
    >
      <span>{percent(pct)}</span>
      <span style={{ fontSize: 9, color: "var(--faint)" }}>
        {SIDE_NAME[side]}
      </span>
    </span>
  );
}
