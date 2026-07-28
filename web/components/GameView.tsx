"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import HexBoard, {
  cellCenter,
  DIR_PIXEL,
  DIR_SHIFTS,
  HEX_SIZE,
  isValid,
  type LastMove,
  type MovingState,
} from "./HexBoard";
import EvalBar from "./EvalBar";
import AnalysisPanel, { type AnalysisMove } from "./AnalysisPanel";
import PlayerPlate from "./PlayerPlate";
import { buildHoverPreview } from "@/lib/boardPreview";
import ReviewView, { type ReviewGame } from "./ReviewView";
import { useEngine } from "@/lib/engine/useEngine";
import type { AiSide, Opening, SearchResultMsg } from "@/lib/engine/protocol";

type WasmModule = typeof import("abalone-wasm");

const DRAG_THRESHOLD = 10;
const SNAP_RADIUS = HEX_SIZE * 0.85;
const POSITIVE_DIR_SHIFTS = [1, 10, 9];
const ANALYSIS_TOP_N = 5;

/** Leaves per forward pass. The network costs roughly the same for one
 *  position as for sixteen, so this is close to a free 16x — but each extra
 *  leaf is chosen under virtual loss rather than a real value, so the tree
 *  degrades if it goes much higher. Self-play uses the same order (MODEL §7.1). */
const NN_BATCH_SIZE = 16;

type Mode = "play" | "analysis" | "review";
type EngineKind = "heuristic" | "network";

/** Difficulty is simulations under a friendlier name. The jumps are roughly
 *  geometric because strength goes with the log of the search budget, so
 *  evenly-spaced sim counts would feel like no difference at all at the top.
 *  `Beginner` runs a single simulation, which is the network's raw policy —
 *  a real opponent with no lookahead rather than a crippled one. */
const DIFFICULTIES = [
  { key: "beginner", label: "Beginner", sims: 1, blurb: "raw policy, no search" },
  { key: "casual", label: "Casual", sims: 64, blurb: "64 simulations" },
  { key: "club", label: "Club", sims: 200, blurb: "200 simulations" },
  { key: "strong", label: "Strong", sims: 600, blurb: "600 simulations" },
  { key: "maximum", label: "Maximum", sims: 1600, blurb: "1600 simulations" },
] as const;
type DifficultyKey = (typeof DIFFICULTIES)[number]["key"];

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

export default function GameView() {
  const [wasm, setWasm] = useState<WasmModule | null>(null);

  // The position is (opening, moves played) rather than a mutated handle.
  // Two things need the move list anyway — undo, and the engine worker, which
  // rebuilds the position by replaying it — so deriving the board from it
  // removes the only other copy of the game state.
  const [opening, setOpening] = useState<Opening>("standard");
  const [history, setHistory] = useState<number[]>([]);

  const [selection, setSelection] = useState<number[]>([]);
  const [drag, setDrag] = useState<{
    startCell: number;
    startX: number;
    startY: number;
    currentX: number;
    currentY: number;
  } | null>(null);

  const [mode, setMode] = useState<Mode>("play");
  // Black by default: Black opens in Abalone, so this puts the first move in
  // your hands rather than opening on a position the network already played
  // into. It also makes the flipped board the default orientation.
  const [playerSide, setPlayerSide] = useState<0 | 1>(0);
  const [difficulty, setDifficulty] = useState<DifficultyKey>("club");
  const [engineKind, setEngineKind] = useState<EngineKind>("network");
  const [analysisSims, setAnalysisSims] = useState(400);
  const [hoveredAnalysisIdx, setHoveredAnalysisIdx] = useState<number | null>(
    null
  );

  // Snapshotted when review is entered, so "New game" cannot pull the record
  // out from under the review that is reading it.
  const [review, setReview] = useState<ReviewGame | null>(null);

  const engine = useEngine();
  const [neural, setNeural] = useState<{
    key: string;
    result: SearchResultMsg;
  } | null>(null);

  const selectionRef = useRef(selection);
  selectionRef.current = selection;
  const dragRef = useRef(drag);
  dragRef.current = drag;

  useEffect(() => {
    let cancelled = false;
    import("abalone-wasm").then((mod) => {
      if (!cancelled) setWasm(mod);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  const game = useMemo(() => {
    if (!wasm) return null;
    const g =
      opening === "belgian" ? wasm.WasmGame.belgian_daisy() : new wasm.WasmGame();
    for (const idx of history) g.apply_index(idx);
    return g;
  }, [wasm, opening, history]);

  // Rebuilding on every move leaves the previous handle stranded in wasm
  // memory; drop it once React has committed the replacement.
  const liveGame = useRef<typeof game>(null);
  useEffect(() => {
    const stale = liveGame.current;
    liveGame.current = game;
    if (stale && stale !== game) stale.free();
  }, [game]);
  useEffect(() => () => liveGame.current?.free(), []);

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
  }, [wasm, game]);

  const positionKey = useMemo(
    () => `${opening}:${history.join(",")}`,
    [opening, history]
  );

  const playing = mode === "play";
  const reviewing = mode === "review";
  const simulations = playing
    ? (DIFFICULTIES.find((d) => d.key === difficulty) ?? DIFFICULTIES[2]).sims
    : analysisSims;

  const inProgress =
    !!wasm && !!snapshot && snapshot.state === wasm.WasmGameState.InProgress;
  // In analysis both sides are played by hand; the engine only ever advises.
  const aiSide: AiSide = playing ? (playerSide === 0 ? "white" : "black") : "none";
  const aiTurn = aiSide === "black" ? 0 : aiSide === "white" ? 1 : -1;
  const isAiTurn = inProgress && snapshot?.turn === aiTurn;

  // The board renders Black at the top, which reads correctly when you are
  // White. Playing Black turns it around, so your own marbles are always the
  // near side.
  const flipped = playing && playerSide === 0;
  const bottomSide: 0 | 1 = flipped ? 0 : 1;
  // A flipped board means a drag toward the bottom of the screen is a drag
  // toward the top of the board. Mirroring the pointer delta the moment it is
  // measured fixes both consumers at once: the direction `findSnap` matches
  // against, and the offset the free-drag ghost is drawn at.
  const dragSign = flipped ? -1 : 1;

  const applyMove = useCallback((idx: number) => {
    setHistory((h) => [...h, idx]);
    setSelection([]);
    setDrag(null);
    setHoveredAnalysisIdx(null);
  }, []);

  // `applyMove` is stable, but reading it through a ref keeps the search
  // effect from re-running (and re-searching) for reasons unrelated to the
  // position.
  const applyMoveRef = useRef(applyMove);
  applyMoveRef.current = applyMove;

  // Heuristic analysis is synchronous and cheap enough to run inline.
  const heuristicAnalysis = useMemo(() => {
    if (!wasm || !game || !snapshot) return null;
    if (playing || engineKind !== "heuristic") return null;
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
  }, [wasm, game, snapshot, playing, engineKind, simulations]);

  // In play mode the network searches only on its own turn — it should not be
  // computing (or displaying) an opinion while you are thinking. In analysis it
  // evaluates every position, and that same search fills the panel.
  const wantNetwork =
    mode !== "review" &&
    inProgress &&
    (snapshot?.legalCount ?? 0) > 0 &&
    (playing ? isAiTurn : engineKind === "network");

  const { search: engineSearch } = engine;
  useEffect(() => {
    if (!wantNetwork) return;
    let cancelled = false;
    const key = positionKey;
    engineSearch({
      opening,
      moves: history,
      simulations,
      batchSize: NN_BATCH_SIZE,
    }).then((res) => {
      if (cancelled || !res) return;
      setNeural({ key, result: res });
      if (isAiTurn && res.bestIdx >= 0) applyMoveRef.current(res.bestIdx);
    });
    return () => {
      cancelled = true;
    };
  }, [
    wantNetwork,
    positionKey,
    opening,
    history,
    isAiTurn,
    simulations,
    engineSearch,
  ]);

  const neuralFresh = neural?.key === positionKey ? neural.result : null;

  const analysis =
    engineKind === "network"
      ? neuralFresh
        ? {
            rootEval: neuralFresh.rootEval,
            topMoves: neuralFresh.topMoves,
            totalVisits: neuralFresh.totalVisits,
          }
        : null
      : heuristicAnalysis;

  const thinking = wantNetwork && !neuralFresh;

  const finalizeGesture = useCallback(() => {
    const d = dragRef.current;
    if (!d || !game || !snapshot) {
      setDrag(null);
      return;
    }
    const dx = (d.currentX - d.startX) * dragSign;
    const dy = (d.currentY - d.startY) * dragSign;
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
        applyMove(snap.moveIdx);
      }
    }
    setDrag(null);
  }, [game, snapshot, applyMove, dragSign]);

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
    const dx = (drag.currentX - drag.startX) * dragSign;
    const dy = (drag.currentY - drag.startY) * dragSign;
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
  }, [drag, game, selection, snapshot, dragSign]);

  // Where the previous move came from and went to. `move_source_cells` only
  // decodes the index, so reading it off the post-move position is safe.
  const lastMove: LastMove | null = useMemo(() => {
    if (!wasm || !game || history.length === 0) return null;
    const idx = history[history.length - 1];
    const from = Array.from(game.move_source_cells(idx)).filter(
      (c) => c !== 0xff
    );
    const shift = DIR_SHIFTS[wasm.move_motion_dir(idx)];
    return { fromCells: from, toCells: from.map((c) => c + shift) };
  }, [wasm, game, history]);

  const hoverPreview: MovingState | null = useMemo(() => {
    if (drag) return null;
    if (hoveredAnalysisIdx == null || !game || !snapshot || !wasm) return null;
    return buildHoverPreview(
      game,
      wasm,
      snapshot.cells,
      snapshot.turn,
      hoveredAnalysisIdx
    );
  }, [drag, hoveredAnalysisIdx, game, snapshot, wasm]);

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

  // Eval bar source: prefer the search's root eval; fall back to the static
  // heuristic while the network is still thinking about this position.
  const evalForBar = analysis ? analysis.rootEval : game.eval_white_pov();

  const reset = (kind: Opening) => {
    setOpening(kind);
    setHistory([]);
    setSelection([]);
    setDrag(null);
    setHoveredAnalysisIdx(null);
    setNeural(null);
  };

  // Undo takes back a full round-trip when the network is playing, so you land
  // back on your own move rather than immediately watching it replay the
  // position you just left.
  const undo = () => {
    const back = playing && history.length >= 2 ? 2 : 1;
    setHistory((h) => h.slice(0, Math.max(0, h.length - back)));
    setSelection([]);
    setDrag(null);
    setHoveredAnalysisIdx(null);
  };

  const startReview = () => {
    if (history.length === 0) return;
    setReview({
      opening,
      moves: history,
      playerSide: playing ? playerSide : null,
      difficulty: playing ? difficultyLabel : null,
    });
    setMode("review");
  };

  const onCellPointerDown = (
    cell: number,
    clientX: number,
    clientY: number
  ) => {
    if (isAiTurn) return;
    setDrag({
      startCell: cell,
      startX: clientX,
      startY: clientY,
      currentX: clientX,
      currentY: clientY,
    });
  };

  const applyAnalysisMove = (idx: number) => {
    if (isAiTurn) return;
    applyMove(idx);
  };

  const engineFooter = (() => {
    if (engineKind !== "network") return "Hand-written evaluator, no network.";
    if (engine.status === "error") {
      return `Network unavailable: ${engine.error ?? "unknown error"}`;
    }
    if (engine.status === "loading" || engine.status === "idle") {
      return "Loading best.onnx…";
    }
    const provider = engine.info?.provider ?? "wasm";
    const threads = engine.info?.threads ?? 1;
    const timing = neuralFresh
      ? ` · ${(neuralFresh.elapsedMs / 1000).toFixed(1)}s · ${Math.round(
          (neuralFresh.totalVisits / neuralFresh.elapsedMs) * 1000
        )} sims/s`
      : "";
    return `best.onnx · ${provider}${provider === "wasm" ? ` ×${threads}` : ""}${timing}`;
  })();

  const gameOver = snapshot.state !== wasm.WasmGameState.InProgress;
  const youWon =
    (playerSide === 0 && snapshot.state === wasm.WasmGameState.BlackWins) ||
    (playerSide === 1 && snapshot.state === wasm.WasmGameState.WhiteWins);


  const topSide: 0 | 1 = bottomSide === 0 ? 1 : 0;
  const sideName = (s: 0 | 1) => (s === 0 ? "Black" : "White");
  // Captures are the *opponent's* losses: you score by pushing their marbles
  // off, and six of them ends the game.
  const capturesBy = (s: 0 | 1) =>
    s === 1 ? snapshot.lostBlack : snapshot.lostWhite;
  const difficultyLabel =
    DIFFICULTIES.find((d) => d.key === difficulty)?.label ?? "";

  const plateFor = (s: 0 | 1) => {
    const isYou = playing && s === playerSide;
    const isEngine = playing && s !== playerSide;
    return (
      <PlayerPlate
        side={s}
        name={isYou ? "You" : isEngine ? "Network" : sideName(s)}
        detail={
          isYou
            ? sideName(s)
            : isEngine
              ? `${sideName(s)} · ${difficultyLabel}`
              : undefined
        }
        captures={capturesBy(s)}
        isTurn={!gameOver && snapshot.turn === s}
        thinking={thinking && snapshot.turn === s}
      />
    );
  };

  const resultBanner = (() => {
    if (!gameOver) return null;
    const drawn = snapshot.state === wasm.WasmGameState.Draw;
    const tone = drawn
      ? "var(--muted)"
      : playing
        ? youWon
          ? "var(--legal)"
          : "var(--illegal)"
        : "var(--text)";
    const headline = drawn
      ? "Draw"
      : playing
        ? youWon
          ? "You win"
          : "The network wins"
        : stateLabel;
    return (
      <div
        className="panel"
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          gap: 12,
          padding: "10px 14px",
          borderColor: tone,
          background: "var(--surface-raised)",
        }}
      >
        <span style={{ fontSize: 14, fontWeight: 600, color: tone }}>
          {headline}
        </span>
        <span style={{ fontSize: 12, color: "var(--muted)" }}>
          {drawn ? `${snapshot.ply} plies` : `${stateLabel} in ${snapshot.ply} plies`}
        </span>
        <button className="btn" onClick={() => reset(opening)}>
          Play again
        </button>
        <button className="btn" onClick={startReview}>
          Review game
        </button>
      </div>
    );
  })();

  return (
    <div
      style={{
        minHeight: "calc(100vh - 48px)",
        display: "flex",
        flexDirection: "column",
        gap: 18,
      }}
    >
      {/* The tab bar is the one element that must not move between modes, so
          it lives outside the content column: pinned to the top of the page
          and centred on it, independent of how wide or tall a mode's layout
          is. Everything else centres in the space that is left. */}
      <div className="tabs" role="tablist" style={{ alignSelf: "center" }}>
        {(["play", "analysis"] as const).map((m) => (
          <button
            key={m}
            role="tab"
            aria-selected={mode === m}
            className="tab"
            onClick={() => setMode(m)}
          >
            {m === "play" ? "Play" : "Analysis"}
          </button>
        ))}
        {review && (
          <button
            role="tab"
            aria-selected={mode === "review"}
            className="tab"
            onClick={() => setMode("review")}
          >
            Review
          </button>
        )}
      </div>

      <div
        style={{
          flex: 1,
          display: "flex",
          flexDirection: "column",
          justifyContent: "center",
          gap: 18,
        }}
      >
      {/* `fit-content` is the whole trick behind the alignment: this column is
          exactly as wide as the game area inside it, so the bar above spans
          the same width and its two ends land on the board's edges. It tracks
          the mode automatically — Analysis is wider than Play because of the
          eval bar and panel — with no width hardcoded anywhere. */}
      <div
        style={{
          width: "fit-content",
          maxWidth: "100%",
          margin: "0 auto",
          display: "flex",
          flexDirection: "column",
          gap: 18,
        }}
      >
      {/* One bar of chrome, not three. Centred as a single cluster rather
          than spread to the edges: the controls are wider than the board in
          Play mode, so `space-between` pushed their ends past it and read as
          a misalignment. Centring stays symmetric in both modes and needs no
          width to be hardcoded. */}
      <div
        style={{
          display: "flex",
          gap: 12,
          alignItems: "center",
          justifyContent: "center",
          flexWrap: "wrap",
        }}
      >
        <div
          style={{
            display: "flex",
            gap: 10,
            alignItems: "center",
            flexWrap: "wrap",
          }}
        >
          {reviewing ? null : playing ? (
            <>
              <select
                className="select"
                aria-label="Your colour"
                value={playerSide}
                onChange={(e) => setPlayerSide(Number(e.target.value) as 0 | 1)}
              >
                <option value={0}>Play Black</option>
                <option value={1}>Play White</option>
              </select>
              <select
                className="select"
                aria-label="Difficulty"
                value={difficulty}
                onChange={(e) => setDifficulty(e.target.value as DifficultyKey)}
              >
                {DIFFICULTIES.map((d) => (
                  <option key={d.key} value={d.key}>
                    {d.label}
                  </option>
                ))}
              </select>
            </>
          ) : (
            <select
              className="select"
              aria-label="Evaluator"
              value={engineKind}
              onChange={(e) => setEngineKind(e.target.value as EngineKind)}
            >
              <option value="network">Trained network</option>
              <option value="heuristic">Heuristic</option>
            </select>
          )}

          {!reviewing && (
            <span
              aria-hidden
              style={{
                width: 1,
                height: 22,
                background: "var(--border)",
                margin: "0 2px",
              }}
            />
          )}

          {!reviewing && (
            <>
          <button
            className="btn"
            onClick={undo}
            disabled={history.length === 0}
          >
            Undo
          </button>
          {/* Reachable mid-game as well as from the result banner: "why am I
              losing this?" is the same question as "where did I go wrong?",
              asked earlier. */}
          <button
            className="btn"
            onClick={startReview}
            disabled={history.length === 0}
          >
            Review
          </button>
          {/* Opening + New game are one action, so they sit tighter to each
              other than to Undo. */}
          <span style={{ display: "flex", gap: 4, alignItems: "center" }}>
            <select
              className="select"
              aria-label="Opening"
              value={opening}
              onChange={(e) => reset(e.target.value as Opening)}
            >
              <option value="standard">Standard</option>
              <option value="belgian">Belgian Daisy</option>
            </select>
            <button className="btn" onClick={() => reset(opening)}>
              New game
            </button>
          </span>
            </>
          )}
        </div>
      </div>

      {!reviewing && resultBanner}

      {reviewing && review ? (
        <ReviewView
          wasm={wasm}
          game={review}
          onExit={() => setMode(playing ? "play" : "analysis")}
        />
      ) : (

      <div
        style={{
          display: "flex",
          gap: 18,
          alignItems: "flex-start",
          justifyContent: "center",
        }}
      >
        {!playing && (
          <EvalBar
            evalWhitePov={evalForBar}
            wdlWhite={neuralFresh?.rootNet?.wdlWhite ?? null}
            expectedScoreWhite={neuralFresh?.rootNet?.expectedScoreWhite ?? null}
            bottomSide={bottomSide}
          />
        )}

        {/* Board column: the plates stretch to the board's own width, so the
            player facing you is literally the plate nearest you. */}
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            gap: 10,
            width: "fit-content",
          }}
        >
          {plateFor(topSide)}
          <HexBoard
            cells={snapshot.cells}
            selection={selection}
            moving={moving}
            lastMove={lastMove}
            onCellPointerDown={onCellPointerDown}
            flipped={flipped}
            idle={!!isAiTurn}
          />
          {plateFor(bottomSide)}

          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 10,
              fontSize: 11,
              color: "var(--faint)",
              padding: "0 2px",
            }}
          >
            <span>Ply {snapshot.ply}</span>
            {playing && (
              <>
                <span>·</span>
                <span>{engineFooter}</span>
              </>
            )}
            {engine.error && (
              <span style={{ marginLeft: "auto", color: "var(--illegal)" }}>
                {engine.error}
              </span>
            )}
          </div>
        </div>

        {!playing && (
          <AnalysisPanel
            topMoves={analysis?.topMoves ?? []}
            totalVisits={analysis?.totalVisits ?? 0}
            turnLabel={turnLabel}
            hoveredIdx={hoveredAnalysisIdx}
            onHover={setHoveredAnalysisIdx}
            onApply={applyAnalysisMove}
            simulations={analysisSims}
            onSimulationsChange={setAnalysisSims}
            engineLabel={
              engineKind === "network" ? "Network" : "Heuristic engine"
            }
            busy={thinking}
            busyVisits={engine.progress?.visits ?? 0}
            footer={engineFooter}
            wdlWhite={neuralFresh?.rootNet?.wdlWhite ?? null}
            expectedScoreWhite={neuralFresh?.rootNet?.expectedScoreWhite ?? null}
          />
        )}
      </div>
      )}
      </div>

      <p
        style={{
          color: "var(--faint)",
          fontSize: 12,
          lineHeight: 1.5,
          margin: 0,
          maxWidth: 640,
          alignSelf: "center",
          textAlign: "center",
        }}
      >
        {reviewing
          ? "Scrub with the slider or the ← → keys. Click the graph or a move to jump there; hover a flagged move to see what the engine wanted instead."
          : playing
            ? "Click a marble to select it, or click along a line for a 2- or 3-piece group. Drag to move — the group snaps onto a legal landing when you get close."
            : "Click a marble to select it, or click along a line for a 2- or 3-piece group. Drag to move. Hover a suggested move to preview it, click to play it."}
      </p>
      </div>
    </div>
  );
}
