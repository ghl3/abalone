"use client";

import { useState } from "react";
import { formatMargin } from "@/lib/outcomeFormat";
import { QUALITY_COLOR, type PlyRead, type ReviewedMove } from "@/lib/engine/review";

/** Bare number — the tooltip is tight and already says which side each is. */
const pct = (p: number) => `${Math.round(p * 100)}`;

/** Parity filter for the plotted curves: `[1, 2, 1] / 4`, edges clamped.
 *
 *  Measured, not assumed (review-probe, docs/NOTEBOOK.md 2026-07-30): each
 *  fresh search revises the position toward the side about to move by
 *  +0.9…+1.8 points of expected score *even when the move played was that
 *  search's own best pick*, and ~90% of all transitions lean the new mover's
 *  way. Every read is optimistic for its mover — the most-visited child's Q
 *  is a max-biased estimate, and the depth barely helps (the bias at 800
 *  simulations is within noise of 200) — so the raw series carries a
 *  per-ply sawtooth that is a property of the estimator, not of the game.
 *  The sqrt axis below then amplifies it ~5× near the centreline, which is
 *  where real games live.
 *
 *  A binomial 3-tap has zero gain at exactly the alternating frequency, so
 *  it removes what the measurement showed to be artefact while leaving
 *  trends and multi-ply swings standing. The tooltip keeps quoting the raw
 *  searched numbers; only the drawn shape is filtered. */
const smooth = (vals: number[]): number[] =>
  vals.map((v, i) => ((vals[i - 1] ?? v) + 2 * v + (vals[i + 1] ?? v)) / 4);

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

/** Geometry is in viewBox units. The graph sits under the board and stretches
 *  to its width, which renders the box at roughly full scale — so an 11-unit
 *  label lands at about 10px, in keeping with the rest of the chrome. Height
 *  is left to the aspect ratio (`height: auto`) rather than pinned: a fixed
 *  `height` with a 100% width letterboxes the drawing inside a taller box and
 *  wastes the difference. */
const W = 520;
const EVAL_H = 96;
/** Room for the band's own caption, between the two plots. */
const LABEL_H = 20;
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

  // Reads arrive in ply order, so index and ply agree; the map is for the
  // markers and cursor below, which look up by ply.
  const smoothedEval = smooth(reads.map((r) => r.rootEval));
  const evalAt = new Map(reads.map((r, i) => [r.ply, smoothedEval[i]]));
  const pts = reads.map((r, i) => ({ x: x(r.ply), y: y(smoothedEval[i]) }));
  const line = pts.map((p, i) => `${i === 0 ? "M" : "L"}${p.x},${p.y}`).join(" ");
  const lastX = pts[pts.length - 1].x;
  // Two areas clipped to their own half so the fill reads as "who is ahead"
  // rather than a single blob hanging off one edge.
  const area = `${line} L${lastX},${mid} L${pts[0].x},${mid} Z`;

  // The marble band: the sweep's expected final capture differential at every
  // position. A prediction, exactly like the curve above it — it was the
  // realised count for a while, which answered "what has happened" under a
  // curve answering "what will happen" and the two disagreed whenever it
  // mattered. Drawn continuous for the same reason: a forecast revises, it
  // does not step. Scaled to the largest predicted lead, floored at two so a
  // quiet game reads as flat rather than as noise stretched to the ceiling.
  const rawMargins = reads.flatMap((r) =>
    r.expectedScoreWhite == null
      ? []
      : [{ x: x(r.ply), v: r.expectedScoreWhite }]
  );
  // Same estimator, same backup, same mover optimism — same filter.
  const smoothedMargins = smooth(rawMargins.map((p) => p.v));
  const marginPts = rawMargins.map((p, i) => ({ x: p.x, v: smoothedMargins[i] }));
  const peak = Math.max(2, ...marginPts.map((p) => Math.abs(p.v)));
  const mBase = EVAL_H + LABEL_H + MARBLE_H / 2;
  const my = (v: number) =>
    mBase - (Math.max(-peak, Math.min(peak, v * sign)) / peak) * (MARBLE_H / 2 - 3);
  const mLine = marginPts
    .map((p, i) => `${i === 0 ? "M" : "L"}${p.x},${my(p.v)}`)
    .join(" ");
  const mArea =
    marginPts.length > 0
      ? `${mLine} L${marginPts[marginPts.length - 1].x},${mBase} L${marginPts[0].x},${mBase} Z`
      : "";

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
      aria-label={`Win probability and predicted marble lead across ${totalPlies} plies, positive is ${
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
            // On the drawn (filtered) curve, not the raw read — a marker
            // floating off its own line reads as a second data series.
            cy={y(evalAt.get(m.ply + 1) ?? m.evalAfter ?? m.evalBefore)}
            r={4}
            fill={QUALITY_COLOR[m.quality]}
            stroke="var(--surface)"
            strokeWidth={2}
          />
        ) : null
      )}

      <text
        x={1}
        y={EVAL_H + 14}
        fontSize={11}
        fill="var(--faint)"
        fontFamily="var(--mono)"
      >
        predicted marble lead
      </text>
      {marginPts.length > 0 && (
        <>
          <path d={mArea} fill={UP_FILL} opacity={0.34} clipPath="url(#eg-m-up)" />
          <path d={mArea} fill={DOWN_FILL} opacity={0.42} clipPath="url(#eg-m-down)" />
        </>
      )}
      <line
        x1="0"
        y1={mBase}
        x2={W}
        y2={mBase}
        stroke="var(--border-strong)"
        strokeWidth={1}
      />
      {marginPts.length > 0 && (
        <path
          d={mLine}
          fill="none"
          stroke="var(--text)"
          strokeWidth={1.5}
          opacity={0.55}
        />
      )}

      {/* One cursor for both plots: the point of stacking them on a shared x is
          that a swing and the marbles it is expected to cost line up
          vertically. */}
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
          cy={y(evalAt.get(active) ?? activeRead.rootEval)}
          r={4.5}
          fill="var(--accent)"
          stroke="var(--surface)"
          strokeWidth={2}
        />
      )}

      {/* The curve is `P(win) − P(loss)`, so its *shape* is already a
          probability difference — but the readout says so in words rather than
          making anyone decode a signed decimal. The margin is formatted by the
          same rule as everywhere else: marbles, signed for White. */}
      {hover !== null && activeRead && (
        <g
          transform={`translate(${Math.min(W - 162, Math.max(2, x(hover) + 10))}, 6)`}
          pointerEvents="none"
        >
          <rect
            width={160}
            height={52}
            rx={5}
            fill="var(--surface-raised)"
            stroke="var(--border)"
          />
          <text
            x={8}
            y={15}
            fontSize={11}
            fill="var(--faint)"
            fontFamily="var(--mono)"
          >
            ply {activeRead.ply}
          </text>
          <text
            x={8}
            y={31}
            fontSize={12}
            fill="var(--text)"
            fontFamily="var(--mono)"
          >
            {activeRead.wdlWhite
              ? `W ${pct(activeRead.wdlWhite[0])} D ${pct(
                  activeRead.wdlWhite[1]
                )} B ${pct(activeRead.wdlWhite[2])}`
              : `${activeRead.rootEval >= 0 ? "+" : ""}${activeRead.rootEval.toFixed(2)}`}
          </text>
          {activeRead.expectedScoreWhite != null && (
            <text
              x={8}
              y={46}
              fontSize={11}
              fill="var(--faint)"
              fontFamily="var(--mono)"
            >
              marbles {formatMargin(activeRead.expectedScoreWhite)}
            </text>
          )}
        </g>
      )}
    </svg>
  );
}
