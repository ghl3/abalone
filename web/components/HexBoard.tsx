"use client";

export const HEX_SIZE = 30;
export const W = Math.sqrt(3) * HEX_SIZE; // horizontal step (E direction)
export const H_VERT = 1.5 * HEX_SIZE; // vertical step
const PAD = HEX_SIZE + 4;

export const DIR_SHIFTS = [1, 10, 9, -1, -10, -9]; // E, NE, NW, W, SW, SE
export const DIR_PIXEL = [
  { dx: W, dy: 0 }, // E
  { dx: W / 2, dy: -H_VERT }, // NE
  { dx: -W / 2, dy: -H_VERT }, // NW
  { dx: -W, dy: 0 }, // W
  { dx: -W / 2, dy: H_VERT }, // SW
  { dx: W / 2, dy: H_VERT }, // SE
];

export function cellCenter(c: number) {
  const r = Math.floor(c / 9);
  const q = c % 9;
  return {
    x: W * (q - r / 2 + 2) + PAD,
    y: H_VERT * (8 - r) + PAD,
  };
}

export function isValid(c: number): boolean {
  if (c < 0 || c >= 81) return false;
  const r = Math.floor(c / 9);
  const q = c % 9;
  return q >= 0 && q < 9 && r < 9 && Math.abs(q - r) <= 4;
}

function hexPath(cx: number, cy: number, size: number): string {
  const pts: string[] = [];
  for (let i = 0; i < 6; i++) {
    const a = -Math.PI / 2 + (Math.PI / 3) * i;
    pts.push(`${cx + size * Math.cos(a)},${cy + size * Math.sin(a)}`);
  }
  return pts.join(" ");
}

export interface MovingState {
  ownerColor: 0 | 1;
  oppColor: 0 | 1;
  ownFromCells: number[];
  ownPositions: { x: number; y: number }[];
  snapped: boolean;
  ownToCells: number[];
  oppFromCells: number[];
  oppToPositions: { x: number; y: number; offBoard: boolean }[];
}

interface Props {
  cells: Int8Array;
  selection: number[];
  moving: MovingState | null;
  onCellPointerDown: (cell: number, clientX: number, clientY: number) => void;
}

export default function HexBoard({
  cells,
  selection,
  moving,
  onCellPointerDown,
}: Props) {
  const positions: { x: number; y: number; c: number }[] = [];
  let minX = Infinity,
    minY = Infinity,
    maxX = -Infinity,
    maxY = -Infinity;
  for (let r = 0; r < 9; r++) {
    for (let q = 0; q < 9; q++) {
      if (Math.abs(q - r) > 4) continue;
      const c = r * 9 + q;
      const { x, y } = cellCenter(c);
      positions.push({ x, y, c });
      // Pad enough to fit off-board "pushed off" marbles too.
      const slack = HEX_SIZE * 1.6;
      minX = Math.min(minX, x - slack);
      minY = Math.min(minY, y - slack);
      maxX = Math.max(maxX, x + slack);
      maxY = Math.max(maxY, y + slack);
    }
  }
  const width = maxX - minX;
  const height = maxY - minY;

  const selSet = new Set(selection);
  const ownFromSet = new Set(moving?.ownFromCells ?? []);
  const oppFromSet = new Set(moving?.oppFromCells ?? []);

  const ownFill = (s: 0 | 1) => (s === 0 ? "var(--black)" : "var(--white)");
  const ownStroke = (s: 0 | 1) =>
    s === 0 ? "var(--black-stroke)" : "var(--white-stroke)";

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
      {/* 1. Hex polygons (clickable surfaces) */}
      {positions.map(({ x, y, c }) => (
        <polygon
          key={`hex-${c}`}
          points={hexPath(x, y, HEX_SIZE)}
          fill="var(--bg)"
          stroke="var(--border)"
          strokeWidth={1}
          onPointerDown={(e) => {
            e.preventDefault();
            onCellPointerDown(c, e.clientX, e.clientY);
          }}
          style={{ cursor: cells[c] >= 0 ? "grab" : "default" }}
        />
      ))}

      {/* 2. Static marbles (with source cells faded as outlines, opp-from
          marbles dimmed to indicate they're being pushed) */}
      {positions.map(({ x, y, c }) => {
        const owner = cells[c];
        if (owner === -1) return null;
        const isOwnFrom = ownFromSet.has(c);
        const isOppFrom = oppFromSet.has(c);

        if (isOwnFrom) {
          return (
            <circle
              key={`mb-${c}`}
              cx={x}
              cy={y}
              r={HEX_SIZE * 0.62}
              fill="none"
              stroke={owner === 0 ? "var(--black)" : "var(--white)"}
              strokeWidth={1.5}
              strokeDasharray="3,3"
              opacity={0.4}
              pointerEvents="none"
            />
          );
        }
        if (isOppFrom) {
          return (
            <circle
              key={`mb-${c}`}
              cx={x}
              cy={y}
              r={HEX_SIZE * 0.62}
              fill={owner === 0 ? "var(--black)" : "var(--white)"}
              stroke={owner === 0 ? "var(--black-stroke)" : "var(--white-stroke)"}
              strokeWidth={1}
              opacity={0.25}
              pointerEvents="none"
              style={{ transition: "opacity 100ms" }}
            />
          );
        }
        return (
          <circle
            key={`mb-${c}`}
            cx={x}
            cy={y}
            r={HEX_SIZE * 0.62}
            fill={owner === 0 ? "var(--black)" : "var(--white)"}
            stroke={owner === 0 ? "var(--black-stroke)" : "var(--white-stroke)"}
            strokeWidth={1}
            pointerEvents="none"
          />
        );
      })}

      {/* 3. Cell labels */}
      {positions.map(({ x, y, c }) => (
        <text
          key={`lbl-${c}`}
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
      ))}

      {/* 4. Selection rings (only when no drag in progress) */}
      {!moving &&
        positions.map(({ x, y, c }) => {
          if (!selSet.has(c)) return null;
          return (
            <circle
              key={`sel-${c}`}
              cx={x}
              cy={y}
              r={HEX_SIZE * 0.78}
              fill="none"
              stroke="var(--highlight)"
              strokeWidth={2.5}
              pointerEvents="none"
            />
          );
        })}

      {/* 5. Opp ghost destinations + pushed-off marbles (only when snapped) */}
      {moving?.snapped &&
        moving.oppToPositions.map((p, i) => (
          <circle
            key={`oppto-${i}`}
            cx={p.x}
            cy={p.y}
            r={HEX_SIZE * 0.62}
            fill={ownFill(moving.oppColor)}
            stroke={ownStroke(moving.oppColor)}
            strokeWidth={1}
            opacity={p.offBoard ? 0.32 : 0.7}
            pointerEvents="none"
            style={{ transition: "opacity 110ms ease-out" }}
          />
        ))}

      {/* 6. Snap rings around own destinations (legality color is implicit:
          if `moving.snapped` is true here, find_move already returned a
          legal index, so the ring is always green). */}
      {moving?.snapped &&
        moving.ownToCells.map((c) => {
          const { x, y } = cellCenter(c);
          return (
            <circle
              key={`ring-${c}`}
              cx={x}
              cy={y}
              r={HEX_SIZE * 0.84}
              fill="none"
              stroke="var(--legal)"
              strokeWidth={2.25}
              pointerEvents="none"
              style={{ transition: "opacity 100ms" }}
            />
          );
        })}

      {/* 7. The dragged own marbles -- on top of everything else.
          Snapped state uses a CSS transform transition so the snap-in
          feels springy; free-drag follows the cursor instantly. */}
      {moving &&
        moving.ownFromCells.map((c, i) => {
          const pos = moving.ownPositions[i];
          return (
            <g
              key={`own-${c}`}
              transform={`translate(${pos.x}, ${pos.y})`}
              style={{
                transition: moving.snapped
                  ? "transform 110ms cubic-bezier(0.2,0.8,0.2,1)"
                  : "none",
                pointerEvents: "none",
              }}
            >
              <circle
                r={HEX_SIZE * 0.62}
                fill={ownFill(moving.ownerColor)}
                stroke={ownStroke(moving.ownerColor)}
                strokeWidth={1.5}
                style={{
                  filter: moving.snapped
                    ? "drop-shadow(0 1px 3px rgba(0,0,0,0.45))"
                    : "drop-shadow(0 4px 9px rgba(0,0,0,0.55))",
                }}
              />
            </g>
          );
        })}
    </svg>
  );
}
