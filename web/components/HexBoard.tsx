"use client";

const HEX_SIZE = 30;
const W = Math.sqrt(3) * HEX_SIZE; // horizontal step (E direction)
const H = 1.5 * HEX_SIZE; // vertical step (NW/NE rows)
const PAD = HEX_SIZE + 4;

function cellCenter(q: number, r: number) {
  // r=8 (row I) at top; row E (r=4) at center.
  const x = W * (q - r / 2 + 2) + PAD;
  const y = H * (8 - r) + PAD;
  return { x, y };
}

export function isValid(c: number): boolean {
  if (c < 0 || c >= 81) return false;
  const r = Math.floor(c / 9);
  const q = c % 9;
  return q >= 0 && q < 9 && r < 9 && Math.abs(q - r) <= 4;
}

function hexPath(cx: number, cy: number, size: number): string {
  // Pointy-top hex: vertices at angles -90°, -30°, 30°, 90°, 150°, 210°.
  const pts: string[] = [];
  for (let i = 0; i < 6; i++) {
    const a = (-Math.PI / 2) + (Math.PI / 3) * i;
    pts.push(`${cx + size * Math.cos(a)},${cy + size * Math.sin(a)}`);
  }
  return pts.join(" ");
}

interface Props {
  cells: Int8Array;
  selection: number[];
  ghost: {
    sourceCells: number[];
    destCells: number[];
    legal: boolean;
    movingOwner: 0 | 1;
  } | null;
  onCellPointerDown: (cell: number, clientX: number, clientY: number) => void;
}

export default function HexBoard({
  cells,
  selection,
  ghost,
  onCellPointerDown,
}: Props) {
  // Compute exact bounds for SVG viewBox.
  const positions: { q: number; r: number; x: number; y: number; c: number }[] = [];
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (let r = 0; r < 9; r++) {
    for (let q = 0; q < 9; q++) {
      if (Math.abs(q - r) > 4) continue;
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

  const selSet = new Set(selection);
  const ghostSrc = new Set(ghost?.sourceCells ?? []);
  const ghostDst = new Set(ghost?.destCells ?? []);
  const ghostColor = ghost?.movingOwner === 0 ? "var(--black)" : "var(--white)";
  const ghostStrokeColor = ghost?.movingOwner === 0 ? "#444" : "#666";
  const targetRing = ghost?.legal ? "#62d35e" : "#e25c5c";

  return (
    <svg
      width={width}
      height={height}
      viewBox={`${minX} ${minY} ${width} ${height}`}
      style={{
        background: "var(--panel)",
        borderRadius: 12,
        userSelect: "none",
        touchAction: "none",
      }}
    >
      {positions.map(({ x, y, c }) => {
        const owner = cells[c]; // -1, 0, 1
        const isSelected = selSet.has(c);
        const isGhostSrc = ghostSrc.has(c);
        const isGhostDst = ghostDst.has(c);
        const marbleOpacity = isGhostSrc ? 0.25 : 1;
        return (
          <g key={c}>
            <polygon
              points={hexPath(x, y, HEX_SIZE)}
              fill="var(--bg)"
              stroke="var(--border)"
              strokeWidth={1}
              onPointerDown={(e) => {
                e.preventDefault();
                onCellPointerDown(c, e.clientX, e.clientY);
              }}
              style={{ cursor: owner >= 0 ? "grab" : "default" }}
            />
            {owner === 0 && (
              <circle
                cx={x}
                cy={y}
                r={HEX_SIZE * 0.62}
                fill="var(--black)"
                stroke="#444"
                strokeWidth={1}
                opacity={marbleOpacity}
                pointerEvents="none"
              />
            )}
            {owner === 1 && (
              <circle
                cx={x}
                cy={y}
                r={HEX_SIZE * 0.62}
                fill="var(--white)"
                stroke="#666"
                strokeWidth={1}
                opacity={marbleOpacity}
                pointerEvents="none"
              />
            )}
            {isGhostDst && ghost && (
              <>
                <circle
                  cx={x}
                  cy={y}
                  r={HEX_SIZE * 0.62}
                  fill={ghostColor}
                  stroke={ghostStrokeColor}
                  strokeWidth={1}
                  opacity={0.55}
                  pointerEvents="none"
                />
                <circle
                  cx={x}
                  cy={y}
                  r={HEX_SIZE * 0.78}
                  fill="none"
                  stroke={targetRing}
                  strokeWidth={2}
                  pointerEvents="none"
                />
              </>
            )}
            {isSelected && (
              <circle
                cx={x}
                cy={y}
                r={HEX_SIZE * 0.78}
                fill="none"
                stroke="var(--highlight)"
                strokeWidth={2.5}
                pointerEvents="none"
              />
            )}
            <text
              x={x}
              y={y + HEX_SIZE * 0.95}
              fontSize={9}
              fill="var(--muted)"
              textAnchor="middle"
              style={{ pointerEvents: "none", userSelect: "none" }}
            >
              {String.fromCharCode(65 + Math.floor(c / 9))}
              {(c % 9) + 1}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
