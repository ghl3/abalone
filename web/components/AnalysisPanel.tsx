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
  /** Which evaluator produced these numbers. */
  engineLabel: string;
  /** Search in flight; rows below are for the previous position. */
  busy?: boolean;
  /** Simulations completed so far, for the in-flight search. */
  busyVisits?: number;
  /** One-line provenance under the rows: provider, timing, load state. */
  footer?: string;
  /** Raw value/score heads at the root, White's POV. */
  wdlWhite?: [number, number, number] | null;
  expectedScoreWhite?: number | null;
}

function pct(p: number): string {
  return `${Math.round(p * 100)}%`;
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
  engineLabel,
  busy = false,
  busyVisits = 0,
  footer,
  wdlWhite,
  expectedScoreWhite,
}: Props) {
  return (
    <div
      className="panel"
      style={{
        width: 280,
        alignSelf: "stretch",
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
        <div style={{ fontSize: 13, fontWeight: 600 }}>{engineLabel}</div>
        <div style={{ color: "var(--muted)", fontSize: 11 }}>
          {busy
            ? `thinking · ${busyVisits}/${simulations}`
            : `MCTS · ${totalVisits} visits`}
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
        <span style={{ color: "var(--faint)" }}>depth</span>
        <input
          type="range"
          min={50}
          max={2000}
          step={50}
          value={simulations}
          onChange={(e) => onSimulationsChange(parseInt(e.target.value, 10))}
          style={{ flex: 1, accentColor: "var(--accent)" }}
        />
        <span
          style={{
            minWidth: 34,
            textAlign: "right",
            fontFamily: "var(--mono)",
            color: "var(--text)",
          }}
        >
          {simulations}
        </span>
      </div>

      {wdlWhite && (
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            gap: 3,
            padding: "8px 0",
            borderTop: "1px solid var(--border)",
            borderBottom: "1px solid var(--border)",
            fontSize: 11,
            fontFamily: "ui-monospace, SF Mono, Menlo, monospace",
          }}
        >
          <div style={{ color: "var(--muted)", fontFamily: "inherit" }}>
            Network read · no search
          </div>
          <div style={{ display: "flex", justifyContent: "space-between" }}>
            <span>W {pct(wdlWhite[0])}</span>
            <span style={{ color: "var(--muted)" }}>
              draw {pct(wdlWhite[1])}
            </span>
            <span>B {pct(wdlWhite[2])}</span>
          </div>
          {expectedScoreWhite != null && (
            <div style={{ color: "var(--muted)" }}>
              expected score{" "}
              <span style={{ color: "var(--text)" }}>
                {expectedScoreWhite >= 0 ? "+" : ""}
                {expectedScoreWhite.toFixed(1)}
              </span>{" "}
              marbles (White)
            </div>
          )}
        </div>
      )}

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

      {footer && (
        <div
          style={{
            marginTop: "auto",
            paddingTop: 8,
            borderTop: "1px solid var(--border)",
            color: "var(--muted)",
            fontSize: 11,
            lineHeight: 1.4,
          }}
        >
          {footer}
        </div>
      )}
    </div>
  );
}
