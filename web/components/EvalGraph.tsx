"use client";

import { useState } from "react";
import { QUALITY_COLOR, type PlyRead, type ReviewedMove } from "@/lib/engine/review";

/** Bare number — the tooltip is tight and already says which side each is. */
const pct = (p: number) => `${Math.round(p * 100)}`;

/** Marble counts are small integers, so they get a sign rather than a decimal;
 *  `±0` says "level" without claiming a direction it does not have. */
const signedInt = (n: number) => (n > 0 ? `+${n}` : n < 0 ? `${n}` : "±0");

interface Props {
  reads: PlyRead[];
  moves: ReviewedMove[];
  currentPly: number;
  onSeek: (ply: number) => void;
  /** Side whose advantage points up. The reviewer's own colour, so the graph
   *  answers "was I winning?" rather than making you flip signs in your head. */
  upSide: 0 | 1;
  /** Plies expected in total, so a partial sweep still lays out to full width
   *  instead of stretching and re-scaling as results arrive. */
  totalPlies: number;
  /** Capture differential in each position, signed for White, straight off the
   *  game record. Owes nothing to the sweep: it is what happened, not what the
   *  search thinks will happen, which is the whole reason it earns its own
   *  band. */
  captureDiffs: number[];
}

/** Geometry is in viewBox units and the panel renders the box at roughly 0.6,
 *  so a 16-unit label lands at about 10px. Height is left to the aspect ratio
 *  (`height: auto`) rather than pinned: a fixed `height` with a 100% width
 *  letterboxes the drawing inside a taller box and wastes the difference. */
const W = 520;
const EVAL_H = 96;
/** Room for the band's own caption, between the two plots. */
const LABEL_H = 22;
const MARBLE_H = 44;
const H = EVAL_H + LABEL_H + MARBLE_H;
const PAD_Y = 6;

/** Poles lifted from the marble colours until they clear 3:1 against the panel
 *  surface — the board versions are too dark to read as a fill. Validated for
 *  CVD separation (ΔE 22.5 protan) and normal vision (ΔE 23.8). */
const UP_FILL = "#ded6c2";
const DOWN_FILL = "#8090ad";

export default function EvalGraph({
  reads,
  moves,
  currentPly,
  onSeek,
  upSide,
  totalPlies,
  captureDiffs,
}: Props) {
  const [hover, setHover] = useState<number | null>(null);

  if (reads.length === 0) return null;

  const span = Math.max(1, totalPlies - 1);
  const x = (ply: number) => (ply / span) * W;
  // Q is White's POV; flip when the reviewer is Black so up is always theirs.
  const sign = upSide === 1 ? 1 : -1;
  // Plotted on a square-root scale rather than linearly. Real games spend
  // almost all their time inside ±0.2, so a linear [-1, 1] axis renders them
  // as a flat line and hides exactly the swings a review exists to show;
  // sqrt expands the middle while keeping the ends anchored and the sign
  // honest. The tooltip reports the raw number, so nothing is misread.
  const y = (evalWhite: number) => {
    const raw = Math.max(-1, Math.min(1, evalWhite * sign));
    const v = Math.sign(raw) * Math.sqrt(Math.abs(raw));
    return PAD_Y + ((1 - v) / 2) * (EVAL_H - PAD_Y * 2);
  };
  const mid = y(0);

  const pts = reads.map((r) => ({ x: x(r.ply), y: y(r.rootEval), read: r }));
  const line = pts.map((p, i) => `${i === 0 ? "M" : "L"}${p.x},${p.y}`).join(" ");
  const lastX = pts[pts.length - 1].x;
  // Two areas clipped to their own half so the fill reads as "who is ahead"
  // rather than a single blob hanging off one edge.
  const area = `${line} L${lastX},${mid} L${pts[0].x},${mid} Z`;

  // The marble band. Linear and stepped, deliberately unlike the curve above
  // it: captures are discrete events at a known ply, and interpolating between
  // them would draw a marble leaving the board over four moves. Scaled to the
  // largest lead the game reached, floored at two so an early capture is a step
  // rather than a spike to the ceiling — six marbles is the whole game, and a
  // band fixed at ±6 leaves every real lead flat against the axis.
  const diffs = captureDiffs.length > 0 ? captureDiffs : [0];
  const peak = Math.max(2, ...diffs.map((d) => Math.abs(d)));
  const mBase = EVAL_H + LABEL_H + MARBLE_H / 2;
  const my = (d: number) =>
    mBase - (Math.max(-peak, Math.min(peak, d * sign)) / peak) * (MARBLE_H / 2 - 3);
  let mLine = `M0,${my(diffs[0])}`;
  for (let p = 1; p < diffs.length; p++) {
    mLine += ` L${x(p)},${my(diffs[p - 1])} L${x(p)},${my(diffs[p])}`;
  }
  const mLastX = x(diffs.length - 1);
  const mArea = `${mLine} L${mLastX},${mBase} L0,${mBase} Z`;

  const active = hover ?? currentPly;
  const activeRead = reads.find((r) => r.ply === active);
  const activeDiff = captureDiffs[active] ?? 0;

  const seekFromEvent = (e: React.PointerEvent<SVGSVGElement>) => {
    const box = e.currentTarget.getBoundingClientRect();
    const frac = (e.clientX - box.left) / box.width;
    return Math.max(0, Math.min(totalPlies - 1, Math.round(frac * span)));
  };

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      style={{
        display: "block",
        width: "100%",
        height: "auto",
        touchAction: "none",
        cursor: "pointer",
      }}
      onPointerMove={(e) => setHover(seekFromEvent(e))}
      onPointerLeave={() => setHover(null)}
      onPointerDown={(e) => onSeek(seekFromEvent(e))}
      role="img"
      aria-label={`Evaluation and marble lead across ${totalPlies} plies, positive is ${
        upSide === 0 ? "Black" : "White"
      }`}
    >
      <defs>
        <clipPath id="eg-up">
          <rect x="0" y="0" width={W} height={mid} />
        </clipPath>
        <clipPath id="eg-down">
          <rect x="0" y={mid} width={W} height={EVAL_H - mid} />
        </clipPath>
        <clipPath id="eg-m-up">
          <rect x="0" y={EVAL_H + LABEL_H} width={W} height={mBase - EVAL_H - LABEL_H} />
        </clipPath>
        <clipPath id="eg-m-down">
          <rect x="0" y={mBase} width={W} height={H - mBase} />
        </clipPath>
      </defs>

      <path d={area} fill={UP_FILL} opacity={0.34} clipPath="url(#eg-up)" />
      <path d={area} fill={DOWN_FILL} opacity={0.42} clipPath="url(#eg-down)" />

      {/* Equal-position reference. Recessive: it orients, it does not compete. */}
      <line
        x1="0"
        y1={mid}
        x2={W}
        y2={mid}
        stroke="var(--border-strong)"
        strokeWidth={1}
      />

      <path d={line} fill="none" stroke="var(--text)" strokeWidth={2} opacity={0.7} />

      {/* Severity markers only — labelling every ply would bury the two or
          three moments that actually decided the game. */}
      {moves.map((m) =>
        m.quality === "inaccuracy" || m.quality === "blunder" ? (
          <circle
            key={m.ply}
            cx={x(m.ply + 1)}
            cy={y(m.evalAfter ?? m.evalBefore)}
            r={4}
            fill={QUALITY_COLOR[m.quality]}
            stroke="var(--surface)"
            strokeWidth={2}
          />
        ) : null
      )}

      <text
        x={1}
        y={EVAL_H + 15}
        fontSize={15}
        fill="var(--faint)"
        fontFamily="var(--mono)"
      >
        marble lead
      </text>
      <path d={mArea} fill={UP_FILL} opacity={0.34} clipPath="url(#eg-m-up)" />
      <path d={mArea} fill={DOWN_FILL} opacity={0.42} clipPath="url(#eg-m-down)" />
      <line
        x1="0"
        y1={mBase}
        x2={W}
        y2={mBase}
        stroke="var(--border-strong)"
        strokeWidth={1}
      />
      <path d={mLine} fill="none" stroke="var(--text)" strokeWidth={1.5} opacity={0.55} />

      {/* One cursor for both plots: the point of stacking them on a shared x is
          that a capture and the swing that paid for it line up vertically. */}
      <line
        x1={x(active)}
        y1={0}
        x2={x(active)}
        y2={H}
        stroke="var(--accent)"
        strokeWidth={1}
        opacity={0.65}
      />
      {activeRead && (
        <circle
          cx={x(active)}
          cy={y(activeRead.rootEval)}
          r={4.5}
          fill="var(--accent)"
          stroke="var(--surface)"
          strokeWidth={2}
        />
      )}

      {/* The curve is `P(win) − P(loss)`, so its *shape* is already a
          probability difference — but the readout says so in words rather than
          making anyone decode a signed decimal. Sized against the 0.6 render
          scale: the old 10-unit text arrived on screen at six pixels. */}
      {hover !== null && activeRead && (
        <g
          transform={`translate(${Math.min(W - 206, Math.max(2, x(hover) + 10))}, 6)`}
          pointerEvents="none"
        >
          <rect
            width={204}
            height={62}
            rx={5}
            fill="var(--surface-raised)"
            stroke="var(--border)"
          />
          <text
            x={10}
            y={21}
            fontSize={15}
            fill="var(--faint)"
            fontFamily="var(--mono)"
          >
            ply {activeRead.ply}
          </text>
          <text
            x={10}
            y={39}
            fontSize={16}
            fill="var(--text)"
            fontFamily="var(--mono)"
          >
            {activeRead.wdlWhite
              ? `W ${pct(activeRead.wdlWhite[0])} D ${pct(
                  activeRead.wdlWhite[1]
                )} B ${pct(activeRead.wdlWhite[2])}`
              : `${activeRead.rootEval >= 0 ? "+" : ""}${activeRead.rootEval.toFixed(2)}`}
          </text>
          <text
            x={10}
            y={56}
            fontSize={15}
            fill="var(--faint)"
            fontFamily="var(--mono)"
          >
            marbles {signedInt(activeDiff)}
          </text>
        </g>
      )}
    </svg>
  );
}
