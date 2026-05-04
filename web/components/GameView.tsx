"use client";

import { useEffect, useMemo, useState } from "react";
import HexBoard from "./HexBoard";

type WasmModule = typeof import("abalone-wasm");

export default function GameView() {
  const [wasm, setWasm] = useState<WasmModule | null>(null);
  const [game, setGame] = useState<InstanceType<WasmModule["WasmGame"]> | null>(
    null
  );
  // Bumped after every mutation so React knows to re-read state from the
  // wasm-side object (which mutates in place).
  const [tick, setTick] = useState(0);
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);

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
    const legal = Array.from(game.legal_indices());
    const turn = game.turn();
    const state = game.state();
    const ply = game.ply();
    const lostBlack = game.lost(wasm.WasmSide.Black);
    const lostWhite = game.lost(wasm.WasmSide.White);
    return { cells, legal, turn, state, ply, lostBlack, lostWhite };
    // tick forces recomputation after mutation
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [wasm, game, tick]);

  const sourceCells = useMemo(() => {
    if (!game || hoverIdx == null) return null;
    return Array.from(game.move_source_cells(hoverIdx)).filter(
      (c) => c !== 0xff
    );
  }, [game, hoverIdx]);

  if (!wasm || !game || !snapshot) {
    return <div style={{ color: "var(--muted)" }}>Loading engine…</div>;
  }

  const turnLabel = snapshot.turn === wasm.WasmSide.Black ? "Black" : "White";
  const stateLabel =
    snapshot.state === wasm.WasmGameState.InProgress
      ? "In progress"
      : snapshot.state === wasm.WasmGameState.BlackWins
        ? "Black wins"
        : snapshot.state === wasm.WasmGameState.WhiteWins
          ? "White wins"
          : "Draw";
  const isTerminal = snapshot.state !== wasm.WasmGameState.InProgress;

  const applyMove = (idx: number) => {
    game.apply_index(idx);
    setHoverIdx(null);
    setTick((t) => t + 1);
  };

  const reset = (kind: "standard" | "belgian") => {
    game.free();
    setGame(
      kind === "standard" ? new wasm.WasmGame() : wasm.WasmGame.belgian_daisy()
    );
    setHoverIdx(null);
    setTick((t) => t + 1);
  };

  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "auto 320px",
        gap: 24,
        maxWidth: 1100,
      }}
    >
      <div>
        <HexBoard
          cells={snapshot.cells}
          highlightedSources={sourceCells ?? []}
        />
      </div>
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          gap: 16,
          background: "var(--panel)",
          border: "1px solid var(--border)",
          borderRadius: 12,
          padding: 16,
          height: "fit-content",
        }}
      >
        <div>
          <div style={{ color: "var(--muted)", fontSize: 12 }}>Turn</div>
          <div style={{ fontSize: 22, fontWeight: 600 }}>{turnLabel}</div>
        </div>
        <div style={{ display: "flex", gap: 12, fontSize: 14 }}>
          <span>Ply {snapshot.ply}</span>
          <span style={{ color: "var(--muted)" }}>·</span>
          <span>State: {stateLabel}</span>
        </div>
        <div style={{ display: "flex", gap: 16, fontSize: 14 }}>
          <span>
            <strong>Black lost:</strong> {snapshot.lostBlack}
          </span>
          <span>
            <strong>White lost:</strong> {snapshot.lostWhite}
          </span>
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          <button onClick={() => reset("standard")} style={btnSecondary}>
            Reset (Standard)
          </button>
          <button onClick={() => reset("belgian")} style={btnSecondary}>
            Belgian Daisy
          </button>
        </div>
        <hr style={{ border: 0, borderTop: "1px solid var(--border)", margin: 0 }} />
        <div style={{ color: "var(--muted)", fontSize: 12 }}>
          Legal moves ({snapshot.legal.length})
        </div>
        <div
          style={{
            display: "flex",
            flexWrap: "wrap",
            gap: 6,
            maxHeight: 360,
            overflowY: "auto",
          }}
        >
          {snapshot.legal.map((idx) => (
            <button
              key={idx}
              disabled={isTerminal}
              onMouseEnter={() => setHoverIdx(idx)}
              onMouseLeave={() => setHoverIdx((h) => (h === idx ? null : h))}
              onClick={() => applyMove(idx)}
              style={{
                ...btnMove,
                outline: hoverIdx === idx ? "1px solid var(--highlight)" : "none",
              }}
            >
              {wasm.move_notation(idx)}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

const btnBase: React.CSSProperties = {
  background: "transparent",
  color: "var(--text)",
  border: "1px solid var(--border)",
  borderRadius: 6,
  padding: "6px 10px",
  fontSize: 12,
};

const btnMove: React.CSSProperties = {
  ...btnBase,
  fontFamily: "ui-monospace, SF Mono, Menlo, monospace",
};

const btnSecondary: React.CSSProperties = {
  ...btnBase,
  padding: "8px 12px",
  fontSize: 13,
};
