"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import HexBoard, { isValid } from "./HexBoard";

type WasmModule = typeof import("abalone-wasm");

const DRAG_THRESHOLD = 10; // px, below which we treat the gesture as a click

// Engine `Dir` order: E=0, NE=1, NW=2, W=3, SW=4, SE=5
const DIR_SHIFTS: number[] = [1, 10, 9, -1, -10, -9];
// Positive directions used to validate a 2/3-marble line group.
const POSITIVE_DIR_SHIFTS = [1, 10, 9];

/** Snap a pixel delta to one of 6 hex directions, or null if too small. */
function nearestDirIdx(dx: number, dy: number): number | null {
  const r = Math.hypot(dx, dy);
  if (r < DRAG_THRESHOLD) return null;
  const theta = Math.atan2(dy, dx); // (-π, π]
  // 6 sextants, 60° each. Sextant 0 = E, 1 = SE, 2 = SW, 3 = W, 4 = NW, 5 = NE.
  const sextant = ((Math.round(theta / (Math.PI / 3)) % 6) + 6) % 6;
  // Map to engine Dir index: E=0, NE=1, NW=2, W=3, SW=4, SE=5
  return [0, 5, 4, 3, 2, 1][sextant];
}

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
    const state = game.state();
    const ply = game.ply();
    const lostBlack = game.lost(wasm.WasmSide.Black);
    const lostWhite = game.lost(wasm.WasmSide.White);
    const legalCount = game.legal_indices().length;
    return { cells, turn, state, ply, lostBlack, lostWhite, legalCount };
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
        // Clicked empty or opponent: clear selection.
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
      // Drag: try to apply a move.
      const dirIdx = nearestDirIdx(dx, dy);
      if (dirIdx != null) {
        const cur = selectionRef.current;
        const movingCells = cur.includes(d.startCell) ? cur : [d.startCell];
        const idx = game.find_move(new Uint8Array(movingCells), dirIdx);
        if (idx >= 0) {
          game.apply_index(idx);
          setSelection([]);
          setTick((t) => t + 1);
        }
      }
    }
    setDrag(null);
  }, [game, snapshot]);

  // Window-level move/up listeners while dragging.
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

  const ghost = useMemo(() => {
    if (!drag || !game || !snapshot) return null;
    const dx = drag.currentX - drag.startX;
    const dy = drag.currentY - drag.startY;
    if (Math.hypot(dx, dy) < DRAG_THRESHOLD) return null;
    const dirIdx = nearestDirIdx(dx, dy);
    if (dirIdx == null) return null;

    const movingCells = selection.includes(drag.startCell)
      ? selection
      : [drag.startCell];
    const shift = DIR_SHIFTS[dirIdx];
    const destCells = movingCells
      .map((c) => c + shift)
      .filter((c) => isValid(c));
    const legal =
      destCells.length === movingCells.length &&
      game.find_move(new Uint8Array(movingCells), dirIdx) >= 0;
    return {
      sourceCells: movingCells,
      destCells,
      legal,
      movingOwner: snapshot.turn,
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

  const onCellPointerDown = (cell: number, clientX: number, clientY: number) => {
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
        ghost={ghost}
        onCellPointerDown={onCellPointerDown}
      />
      <div style={{ color: "var(--muted)", fontSize: 12 }}>
        Click a marble to select (up to 3 in a line). Drag a selected marble in
        any of the 6 hex directions to move. Drag from an unselected own marble
        to move just that marble.
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
