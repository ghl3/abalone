"use client";

interface Props {
  /** 0 = Black, 1 = White. */
  side: 0 | 1;
  name: string;
  /** Difficulty, "you", or whatever qualifies the name. */
  detail?: string;
  /** Marbles this player has pushed off — i.e. the *opponent's* losses. Six
   *  ends the game, which is why this is the headline number and not a
   *  footnote: it is the win condition, counted. */
  captures: number;
  isTurn: boolean;
  thinking?: boolean;
}

const TARGET = 6;

export default function PlayerPlate({
  side,
  name,
  detail,
  captures,
  isTurn,
  thinking = false,
}: Props) {
  // The pips show marbles taken *from* the opponent, so they wear the
  // opponent's colour — you collect their marbles, not your own.
  const takenSide = side === 0 ? 1 : 0;
  const takenFill = takenSide === 0 ? "var(--black)" : "var(--white)";

  return (
    <div
      className="panel"
      style={{
        display: "flex",
        alignItems: "center",
        gap: 12,
        padding: "9px 14px",
        // Whose move it is should be readable without hunting: an accent edge
        // plus a lift, not a two-shade-of-grey difference.
        borderColor: isTurn ? "var(--accent)" : "var(--border)",
        background: isTurn ? "var(--surface-raised)" : "var(--surface)",
        boxShadow: isTurn ? "0 0 0 1px rgba(93,173,236,0.25)" : "none",
        transition: "background 160ms, border-color 160ms, box-shadow 160ms",
      }}
    >
      <span
        aria-hidden
        style={{
          width: 20,
          height: 20,
          borderRadius: "50%",
          flexShrink: 0,
          background:
            side === 0
              ? "radial-gradient(circle at 34% 30%, var(--black-soft), var(--black) 62%, var(--black-shade))"
              : "radial-gradient(circle at 34% 30%, var(--white-soft), var(--white) 62%, var(--white-shade))",
          border: `1px solid ${
            side === 0 ? "var(--black-stroke)" : "var(--white-stroke)"
          }`,
        }}
      />

      <div style={{ display: "flex", flexDirection: "column", gap: 1 }}>
        <span style={{ fontSize: 13, fontWeight: 600, lineHeight: 1.25 }}>
          {name}
        </span>
        {detail && (
          <span style={{ fontSize: 11, color: "var(--muted)", lineHeight: 1.2 }}>
            {detail}
          </span>
        )}
      </div>

      {thinking && (
        <span
          style={{
            display: "flex",
            alignItems: "center",
            gap: 6,
            fontSize: 11,
            color: "var(--accent)",
          }}
        >
          <span className="thinking-dot" />
          thinking
        </span>
      )}

      <div
        style={{
          marginLeft: "auto",
          display: "flex",
          alignItems: "center",
          gap: 8,
        }}
        title={`${captures} of ${TARGET} marbles pushed off`}
      >
        <div style={{ display: "flex", gap: 3 }}>
          {Array.from({ length: TARGET }, (_, i) => (
            <span
              key={i}
              style={{
                width: 9,
                height: 9,
                borderRadius: "50%",
                background: i < captures ? takenFill : "transparent",
                border:
                  i < captures
                    ? "1px solid rgba(0,0,0,0.45)"
                    : "1px solid var(--border-strong)",
                // The sixth pip is the one that ends the game; make the last
                // step visibly different from the four before it.
                boxShadow:
                  i < captures && captures === TARGET - 1 && i === captures - 1
                    ? "0 0 0 2px rgba(246, 195, 67, 0.35)"
                    : "none",
                transition: "background 200ms",
              }}
            />
          ))}
        </div>
        <span
          style={{
            fontFamily: "var(--mono)",
            fontSize: 12,
            color: captures > 0 ? "var(--text)" : "var(--faint)",
            minWidth: 24,
            textAlign: "right",
          }}
        >
          {captures}/{TARGET}
        </span>
      </div>
    </div>
  );
}
