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
import EvalBar from "./EvalBar";
import AnalysisPanel, { type AnalysisMove } from "./AnalysisPanel";

type WasmModule = typeof import("abalone-wasm");

const DRAG_THRESHOLD = 10;
const SNAP_RADIUS = HEX_SIZE * 0.85;
const POSITIVE_DIR_SHIFTS = [1, 10, 9];
const ANALYSIS_TOP_N = 5;

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

/** Build a snapped MovingState from a move idx — used for hover preview
 *  on an analysis row. Mirrors the snap-state branch of the drag preview
 *  but doesn't depend on cursor position. */
function buildHoverPreview(
  game: InstanceType<WasmModule["WasmGame"]>,
  cells: Int8Array,
  turnSide: 0 | 1,
  moveIdx: number
): MovingState {
  const oppSide = (turnSide === 0 ? 1 : 0) as 0 | 1;
  const srcRaw = Array.from(game.move_source_cells(moveIdx));
  const ownFromCells = srcRaw.filter((c) => c !== 0xff);
  const post = game.move_preview(moveIdx);

  const ownToCells: number[] = [];
  const oppFromCells: number[] = [];
  const oppToCells: number[] = [];
  let preOppCount = 0,
    postOppCount = 0;
  for (let c = 0; c < 81; c++) {
    const wasOwn = cells[c] === turnSide;
    const isOwn = post[c] === turnSide;
    const wasOpp = cells[c] === oppSide;
    const isOpp = post[c] === oppSide;
    if (wasOpp) preOppCount++;
    if (isOpp) postOppCount++;
    if (!wasOwn && isOwn) ownToCells.push(c);
    if (wasOpp && !isOpp) oppFromCells.push(c);
    if (!wasOpp && isOpp) oppToCells.push(c);
  }

  const ownPositions = ownToCells.map(cellCenter);
  const oppToPositions = oppToCells.map((c) => ({
    ...cellCenter(c),
    offBoard: false,
  }));

  // Pushed off: walk forward along the implied push direction. We can
  // infer the direction from any (own_from, own_to) pair via DIR_SHIFTS.
  if (preOppCount > postOppCount && oppFromCells.length > 0 && ownFromCells.length > 0) {
    const dirShift =
      ownToCells.length > 0 && ownFromCells.length > 0
        ? ownToCells[0] - ownFromCells[0]
        : null;
    const dirIdx = dirShift == null ? -1 : DIR_SHIFTS.indexOf(dirShift);
    if (dirIdx >= 0) {
      let front = oppFromCells[0];
      while (
        isValid(front + DIR_SHIFTS[dirIdx]) &&
        cells[front + DIR_SHIFTS[dirIdx]] === oppSide
      ) {
        front += DIR_SHIFTS[dirIdx];
      }
      const fc = cellCenter(front);
      oppToPositions.push({
        x: fc.x + DIR_PIXEL[dirIdx].dx,
        y: fc.y + DIR_PIXEL[dirIdx].dy,
        offBoard: true,
      });
    }
  }

  return {
    ownerColor: turnSide,
    oppColor: oppSide,
    ownFromCells,
    ownPositions,
    snapped: true,
    ownToCells,
    oppFromCells,
    oppToPositions,
  };
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

  const [showEngine, setShowEngine] = useState(true);
  const [simulations, setSimulations] = useState(500);
  const [hoveredAnalysisIdx, setHoveredAnalysisIdx] = useState<number | null>(
    null
  );

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
    return {
      cells,
      turn: game.turn() as 0 | 1,
      state: game.state(),
      ply: game.ply(),
      lostBlack: game.lost(wasm.WasmSide.Black),
      lostWhite: game.lost(wasm.WasmSide.White),
      legalCount: game.legal_indices().length,
    };
    // tick triggers re-derive after mutation
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [wasm, game, tick]);

  // Compute MCTS analysis. Wasm-side `AnalysisResult` is freed before
  // useMemo returns; we keep only plain JS data.
  const analysis = useMemo(() => {
    if (!wasm || !game || !snapshot || !showEngine) return null;
    if (snapshot.legalCount === 0) return null;
    const r = game.analyze(simulations);
    if (!r) return null;
    const indices = Array.from(r.indices());
    const evals = Array.from(r.evals());
    const visits = Array.from(r.visits());
    const rootEval = r.root_eval();
    r.free();
    const moves: AnalysisMove[] = indices.map((idx, i) => ({
      idx,
      notation: wasm.move_notation(idx),
      evalWhite: evals[i],
      visits: visits[i],
    }));
    moves.sort((a, b) => b.visits - a.visits);
    const totalVisits = visits.reduce((s, v) => s + v, 0);
    return { rootEval, topMoves: moves.slice(0, ANALYSIS_TOP_N), totalVisits };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [wasm, game, snapshot, showEngine, simulations, tick]);

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
      const cur = selectionRef.current;
      const movingCells = cur.includes(d.startCell) ? cur : [d.startCell];
      const snap = findSnap(game, movingCells, dx, dy);
      if (snap && snap.dist < SNAP_RADIUS) {
        game.apply_index(snap.moveIdx);
        setSelection([]);
        setHoveredAnalysisIdx(null);
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

  const dragMoving: MovingState | null = useMemo(() => {
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

    const dirShift = DIR_SHIFTS[snap.dirIdx];
    const ownToCells = movingCells.map((c) => c + dirShift);
    const ownPositions = ownToCells.map(cellCenter);
    const post = game.move_preview(snap.moveIdx);

    const oppFromCells: number[] = [];
    const oppToCells: number[] = [];
    let preOppCount = 0,
      postOppCount = 0;
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

    if (preOppCount > postOppCount && oppFromCells.length > 0) {
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

  const hoverPreview: MovingState | null = useMemo(() => {
    if (drag) return null;
    if (hoveredAnalysisIdx == null || !game || !snapshot) return null;
    return buildHoverPreview(
      game,
      snapshot.cells,
      snapshot.turn,
      hoveredAnalysisIdx
    );
  }, [drag, hoveredAnalysisIdx, game, snapshot]);

  const moving = dragMoving ?? hoverPreview;

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

  // Eval bar source: prefer MCTS root eval; fall back to static heuristic.
  const evalForBar =
    showEngine && analysis
      ? analysis.rootEval
      : showEngine
        ? game.eval_white_pov()
        : 0;

  const reset = (kind: "standard" | "belgian") => {
    game.free();
    setGame(
      kind === "standard" ? new wasm.WasmGame() : wasm.WasmGame.belgian_daisy()
    );
    setSelection([]);
    setDrag(null);
    setHoveredAnalysisIdx(null);
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

  const applyAnalysisMove = (idx: number) => {
    game.apply_index(idx);
    setSelection([]);
    setDrag(null);
    setHoveredAnalysisIdx(null);
    setTick((t) => t + 1);
  };

  return (
    <div
      style={{
        maxWidth: 1100,
        margin: "0 auto",
        display: "flex",
        flexDirection: "column",
        gap: 16,
      }}
    >
      <header
        style={{
          display: "flex",
          gap: 24,
          alignItems: "center",
          flexWrap: "wrap",
          padding: "4px 0",
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
        <label
          style={{
            display: "flex",
            alignItems: "center",
            gap: 6,
            fontSize: 13,
            cursor: "pointer",
            marginLeft: "auto",
          }}
        >
          <input
            type="checkbox"
            checked={showEngine}
            onChange={(e) => setShowEngine(e.target.checked)}
          />
          Engine analysis
        </label>
        <div style={{ display: "flex", gap: 8 }}>
          <button onClick={() => reset("standard")} style={btnSecondary}>
            Standard
          </button>
          <button onClick={() => reset("belgian")} style={btnSecondary}>
            Belgian Daisy
          </button>
        </div>
      </header>

      <div
        style={{
          display: "flex",
          gap: 16,
          alignItems: "stretch",
          justifyContent: "center",
        }}
      >
        {showEngine && <EvalBar evalWhitePov={evalForBar} />}
        <HexBoard
          cells={snapshot.cells}
          selection={selection}
          moving={moving}
          onCellPointerDown={onCellPointerDown}
        />
        {showEngine && (
          <AnalysisPanel
            topMoves={analysis?.topMoves ?? []}
            totalVisits={analysis?.totalVisits ?? 0}
            turnLabel={turnLabel}
            hoveredIdx={hoveredAnalysisIdx}
            onHover={setHoveredAnalysisIdx}
            onApply={applyAnalysisMove}
            simulations={simulations}
            onSimulationsChange={setSimulations}
          />
        )}
      </div>

      <p
        style={{
          color: "var(--muted)",
          fontSize: 12,
          lineHeight: 1.5,
          margin: 0,
          maxWidth: 620,
          alignSelf: "center",
          textAlign: "center",
        }}
      >
        Click a marble to select; click more in line for a 2- or 3-piece group.
        Drag any selected marble — the group drifts with your cursor and snaps
        onto a legal landing when you get close. Releasing while snapped applies
        the move; releasing elsewhere returns the marbles. Pushed opponents
        appear at their new positions; captured marbles fade past the edge.
      </p>
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
