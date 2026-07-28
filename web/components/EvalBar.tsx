"use client";

interface Props {
  /** Searched eval from White's POV: +1 = White winning, -1 = Black winning. */
  evalWhitePov: number;
  /** `[P(win), P(draw), P(loss)]` for White, from the network's value head.
   *  Absent while the network is still thinking, or on the heuristic engine —
   *  the bar then falls back to splitting on the scalar alone. */
  wdlWhite?: [number, number, number] | null;
  /** Expected final capture differential, signed for White. */
  expectedScoreWhite?: number | null;
  /** Colour occupying the bottom of the board: 0 = Black, 1 = White. The bar
   *  mirrors the board so "your side" is always the near end of both. */
  bottomSide: 0 | 1;
}

function format(v: number): string {
  if (v >= 0.999) return "+M";
  if (v <= -0.999) return "-M";
  return `${v >= 0 ? "+" : ""}${v.toFixed(2)}`;
}

function pct(p: number): string {
  return `${Math.round(p * 100)}%`;
}

/** Marbles, not eval. `format` above saturates to "+M" past 0.999 because an
 *  eval of 1 is a won game; a *score* of 1 is one marble, and there are six to
 *  take. Running the score through the eval formatter printed "+M marbles" for
 *  any lead over a single marble. */
function marbles(v: number): string {
  return `${v >= 0 ? "+" : ""}${v.toFixed(1)}`;
}

/** Colour of the losing/winning bands. The bar reads as "how much of the
 *  outcome belongs to each side", so the bands are the marble colours. */
const FILL = [
  "linear-gradient(90deg, var(--black-shade), var(--black) 55%, var(--black-lit))",
  "linear-gradient(90deg, var(--white-shade), var(--white) 55%, var(--white-lit))",
] as const;

export default function EvalBar({
  evalWhitePov,
  wdlWhite,
  expectedScoreWhite,
  bottomSide,
}: Props) {
  const topSide = (bottomSide === 0 ? 1 : 0) as 0 | 1;

  // Three bands, bottom-anchored to the near player. Without a value head to
  // read, synthesise a two-band split from the scalar so the bar still moves.
  let bottomWin: number;
  let drawShare: number;
  if (wdlWhite) {
    const [whiteWin, draw, whiteLoss] = wdlWhite;
    bottomWin = bottomSide === 1 ? whiteWin : whiteLoss;
    drawShare = draw;
  } else {
    const clamped = Math.max(-1, Math.min(1, evalWhitePov));
    const whiteShare = (clamped + 1) / 2;
    bottomWin = bottomSide === 1 ? whiteShare : 1 - whiteShare;
    drawShare = 0;
  }
  const topWin = Math.max(0, 1 - bottomWin - drawShare);

  const bottomLabel = bottomSide === 0 ? "Black" : "White";
  const topLabel = topSide === 0 ? "Black" : "White";
  const title = wdlWhite
    ? `${bottomLabel} win ${pct(bottomWin)} · draw ${pct(drawShare)} · ` +
      `${topLabel} win ${pct(topWin)}\n` +
      `searched eval (white POV) ${format(evalWhitePov)}` +
      (expectedScoreWhite != null
        ? `\nexpected score ${marbles(expectedScoreWhite)} marbles (white POV)`
        : "")
    : `Eval (white POV): ${format(evalWhitePov)}`;

  return (
    <div
      style={{
        alignSelf: "stretch",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: 8,
      }}
      title={title}
    >
      {/* The number lives above the bar rather than floating inside it: a
          26px column cannot hold "-0.13" without clipping, and widening the
          bar to fit text would make it read as a panel instead of a gauge. */}
      <span
        style={{
          fontFamily: "var(--mono)",
          fontSize: 11,
          color: "var(--muted)",
          letterSpacing: "0.02em",
        }}
      >
        {format(evalWhitePov)}
      </span>
      <div
        style={{
          width: 22,
          flex: 1,
          borderRadius: 999,
          border: "1px solid var(--border)",
          boxShadow: "inset 0 0 8px rgba(0,0,0,0.45)",
          position: "relative",
          overflow: "hidden",
          display: "flex",
          flexDirection: "column",
        }}
      >
      {/* Stacked top → bottom, so the near player's band grows upward from
          the bottom edge the way the board's near side sits at the bottom. */}
      <div
        style={{
          height: `${topWin * 100}%`,
          background: FILL[topSide],
          transition: "height 220ms cubic-bezier(0.2, 0.8, 0.2, 1)",
        }}
      />
      <div
        style={{
          height: `${drawShare * 100}%`,
          background: "var(--faint)",
          opacity: 0.5,
          transition: "height 220ms cubic-bezier(0.2, 0.8, 0.2, 1)",
        }}
      />
      <div
        style={{
          height: `${bottomWin * 100}%`,
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

      </div>
    </div>
  );
}
