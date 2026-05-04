"use client";

const HEX_SIZE = 30;
const W = Math.sqrt(3) * HEX_SIZE; // ~51.96 — horizontal step (E direction)
const H = 1.5 * HEX_SIZE; // 45 — vertical step
const PAD = HEX_SIZE + 4;

// Total board: r = 0..8 (9 rows), q = 0..8 (varies per row).
// At row E (r=4), q ∈ 0..=8 spans the widest x-range. Cells offset by indent r/2.
// Convert (q, r) to pixel center; r=8 (I) at top of screen.
function cellCenter(q: number, r: number) {
  // Shift x by +2*W so the row-E leftmost cell sits at x=0.
  const x = W * (q - r / 2 + 2) + PAD;
  const y = H * (8 - r) + PAD;
  return { x, y };
}

function hexPath(cx: number, cy: number, size: number): string {
  // Pointy-top hex: 6 vertices at angles -90°, -30°, 30°, 90°, 150°, 210°.
  const pts: string[] = [];
  for (let i = 0; i < 6; i++) {
    const a = (-Math.PI / 2) + (Math.PI / 3) * i;
    pts.push(`${cx + size * Math.cos(a)},${cy + size * Math.sin(a)}`);
  }
  return pts.join(" ");
}

function isValid(q: number, r: number): boolean {
  return q >= 0 && q < 9 && r >= 0 && r < 9 && Math.abs(q - r) <= 4;
}

const VIEW_W = W * 8 + 4 * W + PAD * 2; // generous; we'll recompute from cells
const VIEW_H = H * 8 + PAD * 2;

interface Props {
  cells: Int8Array;
  highlightedSources: number[];
}

export default function HexBoard({ cells, highlightedSources }: Props) {
  // Compute exact bounds for SVG viewBox.
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  const positions: { q: number; r: number; x: number; y: number; c: number }[] = [];
  for (let r = 0; r < 9; r++) {
    for (let q = 0; q < 9; q++) {
      if (!isValid(q, r)) continue;
      const { x, y } = cellCenter(q, r);
      positions.push({ q, r, x, y, c: r * 9 + q });
      minX = Math.min(minX, x - HEX_SIZE);
      minY = Math.min(minY, y - HEX_SIZE);
      maxX = Math.max(maxX, x + HEX_SIZE);
      maxY = Math.max(maxY, y + HEX_SIZE);
    }
  }
  const width = maxX - minX;
  const height = maxY - minY;
  const highlight = new Set(highlightedSources);

  return (
    <svg
      width={width}
      height={height}
      viewBox={`${minX} ${minY} ${width} ${height}`}
      style={{ background: "var(--panel)", borderRadius: 12 }}
    >
      {positions.map(({ q, r, x, y, c }) => {
        const owner = cells[c]; // -1 empty, 0 black, 1 white
        const isHighlighted = highlight.has(c);
        return (
          <g key={c}>
            <polygon
              points={hexPath(x, y, HEX_SIZE)}
              fill={isHighlighted ? "var(--accent-soft)" : "var(--bg)"}
              stroke={isHighlighted ? "var(--highlight)" : "var(--border)"}
              strokeWidth={isHighlighted ? 2 : 1}
            />
            {owner === 0 && (
              <circle cx={x} cy={y} r={HEX_SIZE * 0.62} fill="var(--black)" stroke="#444" strokeWidth={1} />
            )}
            {owner === 1 && (
              <circle cx={x} cy={y} r={HEX_SIZE * 0.62} fill="var(--white)" stroke="#666" strokeWidth={1} />
            )}
            <text
              x={x}
              y={y + HEX_SIZE * 0.95}
              fontSize={9}
              fill="var(--muted)"
              textAnchor="middle"
              style={{ pointerEvents: "none", userSelect: "none" }}
            >
              {String.fromCharCode(65 + r)}
              {q + 1}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
