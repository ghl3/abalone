"use client";

export interface AnalysisMove {
  idx: number;
  notation: string;
  evalWhite: number;
  visits: number;
}

interface Props {
  topMoves: AnalysisMove[];
  totalVisits: number;
  turnLabel: string;
  hoveredIdx: number | null;
  onHover: (idx: number | null) => void;
  onApply: (idx: number) => void;
  simulations: number;
  onSimulationsChange: (n: number) => void;
}

function formatEval(v: number): string {
  if (v >= 0.999) return "+M";
  if (v <= -0.999) return "-M";
  return `${v >= 0 ? "+" : ""}${v.toFixed(2)}`;
}

function evalColor(v: number): string {
  if (v > 0.05) return "#a4d8a4";
  if (v < -0.05) return "#d8a4a4";
  return "var(--muted)";
}

export default function AnalysisPanel({
  topMoves,
  totalVisits,
  turnLabel,
  hoveredIdx,
  onHover,
  onApply,
  simulations,
  onSimulationsChange,
}: Props) {
  return (
    <div
      style={{
        width: 280,
        alignSelf: "stretch",
        background: "var(--panel)",
        border: "1px solid var(--border)",
        borderRadius: 12,
        padding: 14,
        display: "flex",
        flexDirection: "column",
        gap: 10,
      }}
    >
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "baseline",
        }}
      >
        <div style={{ fontSize: 13, fontWeight: 600 }}>Engine analysis</div>
        <div style={{ color: "var(--muted)", fontSize: 11 }}>
          MCTS · {totalVisits} visits
        </div>
      </div>

      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 8,
          fontSize: 11,
          color: "var(--muted)",
        }}
      >
        <span style={{ minWidth: 28 }}>{simulations}</span>
        <input
          type="range"
          min={50}
          max={2000}
          step={50}
          value={simulations}
          onChange={(e) => onSimulationsChange(parseInt(e.target.value, 10))}
          style={{ flex: 1 }}
        />
      </div>

      <div
        style={{
          color: "var(--muted)",
          fontSize: 11,
          marginTop: 4,
          marginBottom: 2,
        }}
      >
        Top moves for {turnLabel}
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
        {topMoves.length === 0 && (
          <div style={{ color: "var(--muted)", fontSize: 12 }}>No legal moves.</div>
        )}
        {topMoves.map((m, i) => {
          const active = hoveredIdx === m.idx;
          return (
            <div
              key={m.idx}
              onMouseEnter={() => onHover(m.idx)}
              onMouseLeave={() => onHover(null)}
              onClick={() => onApply(m.idx)}
              style={{
                display: "grid",
                gridTemplateColumns: "auto 1fr auto auto",
                gap: 8,
                alignItems: "center",
                padding: "6px 8px",
                background: active ? "var(--accent-soft)" : "transparent",
                borderRadius: 4,
                cursor: "pointer",
                fontSize: 13,
                fontFamily: "ui-monospace, SF Mono, Menlo, monospace",
                transition: "background 60ms",
              }}
            >
              <span style={{ color: "var(--muted)" }}>{i + 1}.</span>
              <span>{m.notation}</span>
              <span style={{ color: "var(--muted)", fontSize: 11 }}>
                {m.visits}
              </span>
              <span style={{ color: evalColor(m.evalWhite), minWidth: 44, textAlign: "right" }}>
                {formatEval(m.evalWhite)}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
