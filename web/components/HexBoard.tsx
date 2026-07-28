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

/** Where the previous move came from and went to, for the trace overlay.
 *  Playing an engine without this means watching pieces teleport. */
export interface LastMove {
  fromCells: number[];
  toCells: number[];
}

interface Props {
  cells: Int8Array;
  selection: number[];
  moving: MovingState | null;
  lastMove?: LastMove | null;
  onCellPointerDown: (cell: number, clientX: number, clientY: number) => void;
  /** Rotate the board 180° so the player's own marbles are on the near side.
   *  Everything inside the SVG stays in unflipped coordinates and a single
   *  transform does the work — which means callers must mirror the *pointer*
   *  deltas they feed back in (see `GameView`), because a drag toward the
   *  bottom of the screen is a drag toward the top of the board. */
  flipped?: boolean;
  /** Dim the board and ignore hover affordances — the engine is to move. */
  idle?: boolean;
}

const MARBLE_R = HEX_SIZE * 0.62;

export default function HexBoard({
  cells,
  selection,
  moving,
  lastMove,
  onCellPointerDown,
  flipped = false,
  idle = false,
}: Props) {
  const positions: { x: number; y: number; c: number; r: number; q: number }[] =
    [];
  let minX = Infinity,
    minY = Infinity,
    maxX = -Infinity,
    maxY = -Infinity;
  for (let r = 0; r < 9; r++) {
    for (let q = 0; q < 9; q++) {
      if (Math.abs(q - r) > 4) continue;
      const c = r * 9 + q;
      const { x, y } = cellCenter(c);
      positions.push({ x, y, c, r, q });
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
  const spin = flipped
    ? `rotate(180, ${minX + width / 2}, ${minY + height / 2})`
    : undefined;

  // Coordinates only around the rim. Labelling all 61 cells turns the board
  // into a wall of text you have to read past to see the position; the rank
  // letter beside each row and the file number under each column carry the
  // same information and disappear when you aren't looking for them.
  const rimLabels: { x: number; y: number; text: string }[] = [];
  for (let r = 0; r < 9; r++) {
    const q = Math.max(0, r - 4);
    const { x, y } = cellCenter(r * 9 + q);
    rimLabels.push({
      x: x - W * 0.78,
      y: y + 3.5,
      text: String.fromCharCode(65 + r),
    });
  }
  for (let q = 0; q < 9; q++) {
    const r = Math.max(0, q - 4);
    const { x, y } = cellCenter(r * 9 + q);
    // Files run diagonally, so only files 1-5 bottom out on the bottom row;
    // 6-9 bottom out on the lower-right edge and want to be pushed out along
    // that edge's normal, not straight down into the board.
    const onBottomRow = r === 0;
    rimLabels.push({
      x: onBottomRow ? x : x + W * 0.62,
      y: onBottomRow ? y + H_VERT * 0.92 : y + H_VERT * 0.5,
      text: String(q + 1),
    });
  }

  const selSet = new Set(selection);
  const ownFromSet = new Set(moving?.ownFromCells ?? []);
  const oppFromSet = new Set(moving?.oppFromCells ?? []);
  const lastFromSet = new Set(lastMove?.fromCells ?? []);
  const lastToSet = new Set(lastMove?.toCells ?? []);

  const fillFor = (s: 0 | 1) => (s === 0 ? "url(#mb-black)" : "url(#mb-white)");
  const strokeFor = (s: 0 | 1) =>
    s === 0 ? "var(--black-stroke)" : "var(--white-stroke)";

  const marble = (
    key: string,
    x: number,
    y: number,
    owner: 0 | 1,
    opacity = 1
  ) => (
    // No specular highlight: a glint on 28 pieces at once competes with the
    // position for attention. The soft radial shading below is enough to read
    // as a sphere without drawing the eye.
    <circle
      key={key}
      cx={x}
      cy={y}
      r={MARBLE_R}
      fill={fillFor(owner)}
      stroke={strokeFor(owner)}
      strokeWidth={1}
      opacity={opacity}
      pointerEvents="none"
    />
  );

  return (
    <svg
      width={width}
      height={height}
      viewBox={`${minX} ${minY} ${width} ${height}`}
      style={{
        background: "var(--well)",
        borderRadius: 14,
        border: "1px solid var(--border)",
        boxShadow: "inset 0 1px 24px rgba(0,0,0,0.45)",
        userSelect: "none",
        touchAction: "none",
        opacity: idle ? 0.92 : 1,
        transition: "opacity 200ms",
      }}
    >
      <defs>
        {/* The gradient's lit side is mirrored on a flipped board: the parent
            180° rotation would otherwise carry the shading around with the
            pieces, lighting them from below. */}
        <radialGradient
          id="mb-black"
          cx={flipped ? "66%" : "34%"}
          cy={flipped ? "70%" : "30%"}
          r="75%"
        >
          <stop offset="0%" stopColor="var(--black-soft)" />
          <stop offset="62%" stopColor="var(--black)" />
          <stop offset="100%" stopColor="var(--black-shade)" />
        </radialGradient>
        <radialGradient
          id="mb-white"
          cx={flipped ? "66%" : "34%"}
          cy={flipped ? "70%" : "30%"}
          r="75%"
        >
          <stop offset="0%" stopColor="var(--white-soft)" />
          <stop offset="62%" stopColor="var(--white)" />
          <stop offset="100%" stopColor="var(--white-shade)" />
        </radialGradient>
        <linearGradient
          id="hex-face"
          x1="0"
          y1={flipped ? "1" : "0"}
          x2="0"
          y2={flipped ? "0" : "1"}
        >
          <stop offset="0%" stopColor="#1b222a" />
          <stop offset="100%" stopColor="#141a20" />
        </linearGradient>
      </defs>

      <g transform={spin}>
        {/* 1. Hex cells (the clickable surfaces) */}
        {positions.map(({ x, y, c }) => (
          <polygon
            key={`hex-${c}`}
            data-cell={c}
            points={hexPath(x, y, HEX_SIZE)}
            fill="url(#hex-face)"
            stroke="rgba(255,255,255,0.045)"
            strokeWidth={1}
            onPointerDown={(e) => {
              e.preventDefault();
              onCellPointerDown(c, e.clientX, e.clientY);
            }}
            style={{ cursor: !idle && cells[c] >= 0 ? "grab" : "default" }}
          />
        ))}

        {/* 2. Last-move trace, under the marbles: a filled origin and a ring
            on the landing, so the engine's reply is readable at a glance. */}
        {lastMove &&
          positions.map(({ x, y, c }) => {
            if (!lastFromSet.has(c) && !lastToSet.has(c)) return null;
            const isTo = lastToSet.has(c);
            return (
              <polygon
                key={`last-${c}`}
                points={hexPath(x, y, HEX_SIZE - 1)}
                fill="var(--accent)"
                opacity={isTo ? 0.2 : 0.1}
                stroke={isTo ? "var(--accent)" : "none"}
                strokeWidth={1.5}
                strokeOpacity={0.45}
                pointerEvents="none"
              />
            );
          })}

        {/* 3. Static marbles. Source cells of an in-flight drag become dashed
            outlines; opponents about to be pushed dim. */}
        {positions.map(({ x, y, c }) => {
          const owner = cells[c];
          if (owner === -1) return null;
          const side = owner as 0 | 1;

          if (ownFromSet.has(c)) {
            return (
              <circle
                key={`mb-${c}`}
                cx={x}
                cy={y}
                r={MARBLE_R}
                fill="none"
                stroke={side === 0 ? "var(--black)" : "var(--white)"}
                strokeWidth={1.5}
                strokeDasharray="3,3"
                opacity={0.4}
                pointerEvents="none"
              />
            );
          }
          if (oppFromSet.has(c)) return marble(`mb-${c}`, x, y, side, 0.25);
          return marble(`mb-${c}`, x, y, side);
        })}

        {/* 4. Selection rings (only when no drag is in progress) */}
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
            <g key={`oppto-${i}`} opacity={p.offBoard ? 0.32 : 0.7}>
              {marble(`oppto-m-${i}`, p.x, p.y, moving.oppColor)}
            </g>
          ))}

        {/* 6. Snap rings around own destinations (legality colour is implicit:
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

        {/* 7. The dragged own marbles — above everything else. Snapped state
            uses a transform transition so the snap-in feels springy; free drag
            follows the cursor instantly. */}
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
                  filter: moving.snapped
                    ? "drop-shadow(0 1px 3px rgba(0,0,0,0.45))"
                    : "drop-shadow(0 5px 10px rgba(0,0,0,0.6))",
                }}
              >
                <circle
                  r={MARBLE_R}
                  fill={fillFor(moving.ownerColor)}
                  stroke={strokeFor(moving.ownerColor)}
                  strokeWidth={1.5}
                />
              </g>
            );
          })}

        {/* 8. Rim coordinates, counter-rotated so they read upright when the
            board is turned around. */}
        {rimLabels.map((l, i) => (
          <text
            key={`rim-${i}`}
            x={l.x}
            y={l.y}
            transform={flipped ? `rotate(180, ${l.x}, ${l.y})` : undefined}
            fontSize={10}
            fill="var(--faint)"
            textAnchor="middle"
            style={{ pointerEvents: "none", userSelect: "none" }}
          >
            {l.text}
          </text>
        ))}
      </g>
    </svg>
  );
}
