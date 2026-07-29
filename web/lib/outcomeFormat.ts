/** One vocabulary for "who is winning", shared by the win bar, the analysis
 *  panel and the review graph.
 *
 *  Outcomes are stated as probabilities and margins as marbles; there is no
 *  signed eval on screen anywhere, because it is the same axis as the
 *  probabilities in a unit that has to be taught. What survives here is the
 *  colour and the margin format, and both follow one rule: **a number is
 *  White-positive, and colour names the side rather than the sentiment.**
 *
 *  The side-not-sentiment part is the load-bearing half. A red/green ramp keyed
 *  to good-versus-bad inverts its meaning every ply — under a heading reading
 *  *Top moves for Black*, Black's strongest move is the most negative number in
 *  the list, and a good/bad ramp paints it the alarm colour. Keying colour to
 *  the side means the ramp agrees with the sign at every ply instead of half of
 *  them.
 */

/** The two poles, lifted from the marble colours until they clear 3:1 against
 *  the panel surface — the board versions are too dark to read as a fill or as
 *  text. Validated for CVD separation (ΔE 22.5 protan) and normal vision
 *  (ΔE 23.8). */
export const SIDE_TINT = {
  white: "#ded6c2",
  black: "#8090ad",
} as const;

/** Below this the position is level and neither pole should be claimed. */
const LEVEL = 0.05;

export function tintFor(evalWhite: number): string {
  if (evalWhite > LEVEL) return SIDE_TINT.white;
  if (evalWhite < -LEVEL) return SIDE_TINT.black;
  return "var(--muted)";
}

export function percent(p: number): string {
  return `${Math.round(p * 100)}%`;
}

/** Expected final capture differential, in marbles, signed for White. Distinct
 *  from the eval in both scale and meaning: six marbles ends the game, so a
 *  margin of 1 is a real but survivable lead where an eval of 1 is over. */
export function formatMargin(marbles: number): string {
  // Round first, then take the sign of the rounded value: -0.04 formatted the
  // other way round prints "-0.0", which reads as a lead too small to have a
  // direction and yet insists on one.
  const rounded = Number(marbles.toFixed(1));
  return `${rounded > 0 ? "+" : rounded < 0 ? "-" : "±"}${Math.abs(rounded).toFixed(1)}`;
}
