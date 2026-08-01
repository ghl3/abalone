/** The session — opening, record, mode, review — persisted to localStorage so
 *  a reload lands where you were instead of on a fresh board.
 *
 *  This exists mostly for the review cache's sake: cached reads are useless if
 *  the game they belong to evaporates with the tab. Serialisation is dumb on
 *  purpose; the *trust* lives in GameView's restore path, which replays the
 *  record against the wasm rules with legality checks before letting any of
 *  it near the board. A blob that merely parses proves nothing.
 */

import type { Opening } from "./engine/protocol";
import type { ReviewGame } from "@/components/ReviewView";

const KEY = "abalone:session";
/** Bump to orphan every saved session when the shape changes; restore already
 *  tolerates absence, so orphaning is free. */
const VERSION = 1;

export interface SavedSession {
  opening: Opening;
  history: number[];
  /** Redo stack, newest last — replays after `history`, reversed. */
  future: number[];
  mode: "play" | "analysis" | "review";
  playerSide: 0 | 1;
  difficulty: string;
  depth: string;
  analysisFlipped: boolean;
  review: ReviewGame | null;
}

export function loadSession(): SavedSession | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = window.localStorage.getItem(KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (parsed?.v !== VERSION || typeof parsed.session !== "object") return null;
    return parsed.session as SavedSession;
  } catch {
    return null;
  }
}

export function saveSession(session: SavedSession): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(KEY, JSON.stringify({ v: VERSION, session }));
  } catch {
    // Storage full or blocked: the session just becomes ephemeral again.
  }
}
