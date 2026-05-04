"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import HexBoard, {
  cellCenter,
  DIR_PIXEL,
  DIR_SHIFTS,
  HEX_SIZE,
  isValid,
  type MovingState,
} from "./HexBoard";

type WasmModule = typeof import("abalone-wasm");

const DRAG_THRESHOLD = 10; // px below which a release is treated as a click
const SNAP_RADIUS = HEX_SIZE * 0.85; // px from the ideal cell center to engage snap

// Positive directions used to validate a 2/3-marble line group.
const POSITIVE_DIR_SHIFTS = [1, 10, 9];

/** Cells form a contiguous line in one of the 3 hex axes (or a single cell). */
function isValidGroup(cells: number[]): boolean {
  if (cells.length <= 1) return true;
  if (cells.length > 3) return false;
  const sorted = [...cells].sort((a, b) => a - b);
  for (const shift of POSITIVE_DIR_SHIFTS) {
    let ok = true;
    for (let i = 1; i < sorted.length; i++) {
      if (sorted[i] - sorted[i - 1] !== shift) {
        ok = false;
        break;
      }
    }
    if (ok) return true;
  }
  return false;
}

interface SnapResult {
  dirIdx: number;
  moveIdx: number;
  dist: number;
}

/** Pick the closest legal-move snap candidate (by Euclidean distance from the
 *  ideal pixel offset). Returns null if no legal move in any direction. */
function findSnap(
  game: InstanceType<WasmModule["WasmGame"]>,
  movingCells: number[],
  dx: number,
  dy: number
): SnapResult | null {
  let best: SnapResult | null = null;
  for (let dirIdx = 0; dirIdx < 6; dirIdx++) {
    const moveIdx = game.find_move(new Uint8Array(movingCells), dirIdx);
    if (moveIdx < 0) continue;
    const sd = Math.hypot(dx - DIR_PIXEL[dirIdx].dx, dy - DIR_PIXEL[dirIdx].dy);
    if (!best || sd < best.dist) {
      best = { dirIdx, moveIdx, dist: sd };
    }
  }
  return best;
}

export default function GameView() {
  const [wasm, setWasm] = useState<WasmModule | null>(null);
  const [game, setGame] = useState<InstanceType<WasmModule["WasmGame"]> | null>(
    null
  );
  const [tick, setTick] = useState(0);

  const [selection, setSelection] = useState<number[]>([]);
  const [drag, setDrag] = useState<{
    startCell: number;
    startX: number;
    startY: number;
    currentX: number;
    currentY: number;
  } | null>(null);

  const selectionRef = useRef(selection);
  selectionRef.current = selection;
  const dragRef = useRef(drag);
  dragRef.current = drag;

  useEffect(() => {
    let cancelled = false;
    import("abalone-wasm").then((mod) => {
      if (cancelled) return;
      setWasm(mod);
      setGame(new mod.WasmGame());
    });
    return () => {
      cancelled = true;
    };
  }, []);

  const snapshot = useMemo(() => {
    if (!wasm || !game) return null;
    const cells = new Int8Array(81);
    for (let c = 0; c < 81; c++) cells[c] = game.cell(c);
    const turn = game.turn() as 0 | 1;
    return {
      cells,
      turn,
      state: game.state(),
      ply: game.ply(),
      lostBlack: game.lost(wasm.WasmSide.Black),
      lostWhite: game.lost(wasm.WasmSide.White),
      legalCount: game.legal_indices().length,
    };
    // tick triggers re-derive after mutation
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [wasm, game, tick]);

  const finalizeGesture = useCallback(() => {
    const d = dragRef.current;
    if (!d || !game || !snapshot) {
      setDrag(null);
      return;
    }
    const dx = d.currentX - d.startX;
    const dy = d.currentY - d.startY;
    const dist = Math.hypot(dx, dy);

    if (dist < DRAG_THRESHOLD) {
      // Click: toggle selection.
      const owner = snapshot.cells[d.startCell];
      const cur = selectionRef.current;
      if (owner !== snapshot.turn) {
        setSelection([]);
      } else if (cur.includes(d.startCell)) {
        setSelection(cur.filter((c) => c !== d.startCell));
      } else if (cur.length >= 3) {
        setSelection([d.startCell]);
      } else {
        const next = [...cur, d.startCell];
        setSelection(isValidGroup(next) ? next : [d.startCell]);
      }
    } else {
      // Drag: apply if currently snapped to a legal target.
      const cur = selectionRef.current;
      const movingCells = cur.includes(d.startCell) ? cur : [d.startCell];
      const snap = findSnap(game, movingCells, dx, dy);
      if (snap && snap.dist < SNAP_RADIUS) {
        game.apply_index(snap.moveIdx);
        setSelection([]);
        setTick((t) => t + 1);
      }
    }
    setDrag(null);
  }, [game, snapshot]);

  useEffect(() => {
    if (!drag) return;
    const onMove = (e: PointerEvent) => {
      const cur = dragRef.current;
      if (!cur) return;
      setDrag({ ...cur, currentX: e.clientX, currentY: e.clientY });
    };
    const onUp = () => finalizeGesture();
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    window.addEventListener("pointercancel", onUp);
    return () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      window.removeEventListener("pointercancel", onUp);
    };
  }, [drag, finalizeGesture]);

  const moving: MovingState | null = useMemo(() => {
    if (!drag || !game || !snapshot) return null;
    const dx = drag.currentX - drag.startX;
    const dy = drag.currentY - drag.startY;
    if (Math.hypot(dx, dy) < DRAG_THRESHOLD) return null;

    const turnSide = snapshot.turn;
    const oppSide = (turnSide === 0 ? 1 : 0) as 0 | 1;
    if (snapshot.cells[drag.startCell] !== turnSide) return null;

    const movingCells = selection.includes(drag.startCell)
      ? selection
      : [drag.startCell];

    const snap = findSnap(game, movingCells, dx, dy);

    if (!snap || snap.dist >= SNAP_RADIUS) {
      // Free-drag: marbles trail the cursor.
      const ownPositions = movingCells.map((c) => {
        const center = cellCenter(c);
        return { x: center.x + dx, y: center.y + dy };
      });
      return {
        ownerColor: turnSide,
        oppColor: oppSide,
        ownFromCells: movingCells,
        ownPositions,
        snapped: false,
        ownToCells: [],
        oppFromCells: [],
        oppToPositions: [],
      };
    }

    // Snapped: compute the full sumito preview from a hypothetical apply.
    const dirShift = DIR_SHIFTS[snap.dirIdx];
    const ownToCells = movingCells.map((c) => c + dirShift);
    const ownPositions = ownToCells.map(cellCenter);

    const post = game.move_preview(snap.moveIdx);

    const oppFromCells: number[] = [];
    const oppToCells: number[] = [];
    let preOppCount = 0;
    let postOppCount = 0;
    for (let c = 0; c < 81; c++) {
      const wasOpp = snapshot.cells[c] === oppSide;
      const isOpp = post[c] === oppSide;
      if (wasOpp) preOppCount++;
      if (isOpp) postOppCount++;
      if (wasOpp && !isOpp) oppFromCells.push(c);
      if (!wasOpp && isOpp) oppToCells.push(c);
    }

    const oppToPositions = oppToCells.map((c) => ({
      ...cellCenter(c),
      offBoard: false,
    }));

    // Pushed off: walk forward from any opp_from cell to find the front-most
    // pre-state opp marble, then render past the front edge.
    const pushedOff = preOppCount - postOppCount;
    if (pushedOff > 0 && oppFromCells.length > 0) {
      let front = oppFromCells[0];
      while (
        isValid(front + dirShift) &&
        snapshot.cells[front + dirShift] === oppSide
      ) {
        front += dirShift;
      }
      const fc = cellCenter(front);
      oppToPositions.push({
        x: fc.x + DIR_PIXEL[snap.dirIdx].dx,
        y: fc.y + DIR_PIXEL[snap.dirIdx].dy,
        offBoard: true,
      });
    }

    return {
      ownerColor: turnSide,
      oppColor: oppSide,
      ownFromCells: movingCells,
      ownPositions,
      snapped: true,
      ownToCells,
      oppFromCells,
      oppToPositions,
    };
  }, [drag, game, selection, snapshot]);

  if (!wasm || !game || !snapshot) {
    return <div style={{ color: "var(--muted)" }}>Loading engine…</div>;
  }

  const turnLabel = snapshot.turn === 0 ? "Black" : "White";
  const stateLabel =
    snapshot.state === wasm.WasmGameState.InProgress
      ? "In progress"
      : snapshot.state === wasm.WasmGameState.BlackWins
        ? "Black wins"
        : snapshot.state === wasm.WasmGameState.WhiteWins
          ? "White wins"
          : "Draw";

  const reset = (kind: "standard" | "belgian") => {
    game.free();
    setGame(
      kind === "standard" ? new wasm.WasmGame() : wasm.WasmGame.belgian_daisy()
    );
    setSelection([]);
    setDrag(null);
    setTick((t) => t + 1);
  };

  const onCellPointerDown = (
    cell: number,
    clientX: number,
    clientY: number
  ) => {
    setDrag({
      startCell: cell,
      startX: clientX,
      startY: clientY,
      currentX: clientX,
      currentY: clientY,
    });
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      <div
        style={{
          display: "flex",
          gap: 24,
          alignItems: "center",
          flexWrap: "wrap",
        }}
      >
        <div>
          <span style={{ color: "var(--muted)", fontSize: 12 }}>Turn </span>
          <span style={{ fontWeight: 600 }}>{turnLabel}</span>
        </div>
        <div style={{ color: "var(--muted)", fontSize: 13 }}>
          Ply {snapshot.ply}
        </div>
        <div style={{ fontSize: 13 }}>
          <span style={{ color: "var(--muted)" }}>Lost</span>{" "}
          B={snapshot.lostBlack}{" "}
          <span style={{ color: "var(--muted)" }}>·</span>{" "}
          W={snapshot.lostWhite}
        </div>
        <div style={{ fontSize: 13 }}>
          <span style={{ color: "var(--muted)" }}>State</span> {stateLabel}
        </div>
        <div style={{ fontSize: 13 }}>
          <span style={{ color: "var(--muted)" }}>Legal moves</span>{" "}
          {snapshot.legalCount}
        </div>
        <div style={{ marginLeft: "auto", display: "flex", gap: 8 }}>
          <button onClick={() => reset("standard")} style={btnSecondary}>
            Reset (Standard)
          </button>
          <button onClick={() => reset("belgian")} style={btnSecondary}>
            Belgian Daisy
          </button>
        </div>
      </div>
      <HexBoard
        cells={snapshot.cells}
        selection={selection}
        moving={moving}
        onCellPointerDown={onCellPointerDown}
      />
      <div style={{ color: "var(--muted)", fontSize: 12, lineHeight: 1.5 }}>
        Click a marble to select; click more in line for a 2- or 3-piece group.
        Drag any selected marble — the group drifts with your cursor and snaps
        onto a legal landing when you get close. Releasing while snapped applies
        the move; releasing elsewhere returns the marbles. Pushed opponents
        appear at their new positions; captured marbles fade past the edge.
      </div>
    </div>
  );
}

const btnSecondary: React.CSSProperties = {
  background: "transparent",
  color: "var(--text)",
  border: "1px solid var(--border)",
  borderRadius: 6,
  padding: "8px 12px",
  fontSize: 13,
};
