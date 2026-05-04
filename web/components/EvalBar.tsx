"use client";

interface Props {
  /** Eval from White's POV: +1 = White winning, -1 = Black winning, 0 = even. */
  evalWhitePov: number;
}

function format(v: number): string {
  if (v >= 0.999) return "+M";
  if (v <= -0.999) return "-M";
  return `${v >= 0 ? "+" : ""}${v.toFixed(2)}`;
}

export default function EvalBar({ evalWhitePov }: Props) {
  const clamped = Math.max(-1, Math.min(1, evalWhitePov));
  // 0 maps to 50% white; +1 = entire bar white; -1 = entire bar black.
  const whitePct = ((clamped + 1) / 2) * 100;

  return (
    <div
      style={{
        width: 28,
        alignSelf: "stretch",
        background: "var(--black)",
        borderRadius: 6,
        border: "1px solid var(--border)",
        position: "relative",
        overflow: "hidden",
      }}
      title={`Eval (white POV): ${format(evalWhitePov)}`}
    >
      <div
        style={{
          position: "absolute",
          left: 0,
          right: 0,
          top: 0,
          height: `${whitePct}%`,
          background: "var(--white)",
          transition: "height 220ms cubic-bezier(0.2, 0.8, 0.2, 1)",
        }}
      />
      {/* Centerline at 50% to mark the equal-position reference. */}
      <div
        style={{
          position: "absolute",
          left: 0,
          right: 0,
          top: "50%",
          height: 1,
          background: "rgba(255,255,255,0.18)",
          pointerEvents: "none",
        }}
      />
      <div
        style={{
          position: "absolute",
          inset: 0,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          pointerEvents: "none",
        }}
      >
        <span
          style={{
            fontSize: 10,
            fontFamily: "ui-monospace, SF Mono, Menlo, monospace",
            background: "rgba(20,23,26,0.78)",
            color: "var(--text)",
            padding: "2px 4px",
            borderRadius: 3,
            border: "1px solid var(--border)",
          }}
        >
          {format(evalWhitePov)}
        </span>
      </div>
    </div>
  );
}
