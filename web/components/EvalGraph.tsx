"use client";

import { useState } from "react";
import { QUALITY_COLOR, type PlyRead, type ReviewedMove } from "@/lib/engine/review";

/** Bare number — the tooltip is tight and already says which side each is. */
const pct = (p: number) => `${Math.round(p * 100)}`;

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
}

const W = 520;
const H = 96;
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
    return PAD_Y + ((1 - v) / 2) * (H - PAD_Y * 2);
  };
  const mid = y(0);

  const pts = reads.map((r) => ({ x: x(r.ply), y: y(r.rootEval), read: r }));
  const line = pts.map((p, i) => `${i === 0 ? "M" : "L"}${p.x},${p.y}`).join(" ");
  const lastX = pts[pts.length - 1].x;
  // Two areas clipped to their own half so the fill reads as "who is ahead"
  // rather than a single blob hanging off one edge.
  const area = `${line} L${lastX},${mid} L${pts[0].x},${mid} Z`;

  const active = hover ?? currentPly;
  const activeRead = reads.find((r) => r.ply === active);

  const seekFromEvent = (e: React.PointerEvent<SVGSVGElement>) => {
    const box = e.currentTarget.getBoundingClientRect();
    const frac = (e.clientX - box.left) / box.width;
    return Math.max(0, Math.min(totalPlies - 1, Math.round(frac * span)));
  };

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      width="100%"
      height={H}
      style={{ display: "block", touchAction: "none", cursor: "pointer" }}
      onPointerMove={(e) => setHover(seekFromEvent(e))}
      onPointerLeave={() => setHover(null)}
      onPointerDown={(e) => onSeek(seekFromEvent(e))}
      role="img"
      aria-label={`Evaluation across ${totalPlies} plies, positive is ${
        upSide === 0 ? "Black" : "White"
      }`}
    >
      <defs>
        <clipPath id="eg-up">
          <rect x="0" y="0" width={W} height={mid} />
        </clipPath>
        <clipPath id="eg-down">
          <rect x="0" y={mid} width={W} height={H - mid} />
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
          making anyone decode a signed decimal. */}
      {hover !== null && activeRead && (
        <g
          transform={`translate(${Math.min(W - 132, Math.max(2, x(hover) + 8))}, 6)`}
          pointerEvents="none"
        >
          <rect
            width={128}
            height={32}
            rx={4}
            fill="var(--surface-raised)"
            stroke="var(--border)"
          />
          <text
            x={7}
            y={13}
            fontSize={10}
            fill="var(--faint)"
            fontFamily="var(--mono)"
          >
            ply {activeRead.ply}
          </text>
          <text
            x={7}
            y={26}
            fontSize={10}
            fill="var(--text)"
            fontFamily="var(--mono)"
          >
            {activeRead.wdlWhite
              ? `W ${pct(activeRead.wdlWhite[0])} D ${pct(
                  activeRead.wdlWhite[1]
                )} B ${pct(activeRead.wdlWhite[2])}`
              : `${activeRead.rootEval >= 0 ? "+" : ""}${activeRead.rootEval.toFixed(2)}`}
          </text>
        </g>
      )}
    </svg>
  );
}
