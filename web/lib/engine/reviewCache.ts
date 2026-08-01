/** Finished reviews, cached in localStorage.
 *
 *  A sweep is a minute of compute whose inputs are completely determined: the
 *  model, the opening, the move record and the search budget. Caching on
 *  exactly those four means re-entering a review — including after a page
 *  reload — replays a finished judgement instead of re-running it. The model
 *  is identified by the deployed file's HTTP validators, so promoting a new
 *  best.onnx silently invalidates every review the old network wrote; a stale
 *  cache can never masquerade as the current engine's opinion.
 */

import type { Opening } from "./protocol";
import type { PlyRead } from "./review";
import { DEFAULT_MODEL_PATH } from "./useEngine";

const PREFIX = "abalone:review:";
/** Whole games of reads at a few KB per position: keep a shelf, not a library.
 *  Four covers "the game I am poking at plus the last few", and eviction is by
 *  age because that is also the order the games stop mattering in. */
const KEEP = 4;

export interface ReviewDesc {
  opening: Opening;
  moves: number[];
  sims: number;
}

/** Identity of the deployed model, read from a HEAD request against the
 *  static file — the weights never load here. Memoised for the session: the
 *  file cannot change under a running page in any way this cache must honour.
 *  A failed request degrades to a shared "unknown" tag rather than disabling
 *  the cache, which is the right trade for a purely local, purely advisory
 *  store. */
let tagPromise: Promise<string> | null = null;
function modelTag(): Promise<string> {
  if (!tagPromise) {
    tagPromise = fetch(DEFAULT_MODEL_PATH, { method: "HEAD" })
      .then(
        (res) =>
          res.headers.get("etag") ??
          res.headers.get("last-modified") ??
          `bytes:${res.headers.get("content-length") ?? "?"}`
      )
      .catch(() => "unknown");
  }
  return tagPromise;
}

/** The full identity a cached review answers for. Stored alongside the reads
 *  and compared on load, so a hash collision in the storage key cannot serve
 *  one game's judgement as another's. */
export async function reviewKey(desc: ReviewDesc): Promise<string> {
  return [await modelTag(), desc.opening, desc.sims, desc.moves.join(",")].join(
    "|"
  );
}

/** FNV-1a over the identity, because the identity itself — a whole move
 *  record — is too long to want as a storage key. Collisions are guarded by
 *  the stored copy of the real key, not by the hash. */
function storageKey(key: string): string {
  let h = 0x811c9dc5;
  for (let i = 0; i < key.length; i++) {
    h ^= key.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return PREFIX + (h >>> 0).toString(16);
}

interface Entry {
  key: string;
  at: number;
  reads: PlyRead[];
}

export function loadReview(key: string): PlyRead[] | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = window.localStorage.getItem(storageKey(key));
    if (!raw) return null;
    const entry = JSON.parse(raw) as Entry;
    if (
      entry.key !== key ||
      !Array.isArray(entry.reads) ||
      entry.reads.length === 0 ||
      typeof entry.reads[0].ply !== "number" ||
      !Array.isArray(entry.reads[0].allMoves)
    ) {
      return null;
    }
    return entry.reads;
  } catch {
    return null;
  }
}

export function saveReview(key: string, reads: PlyRead[]): void {
  if (typeof window === "undefined") return;
  const entry: Entry = { key, at: Date.now(), reads };
  const write = () =>
    window.localStorage.setItem(storageKey(key), JSON.stringify(entry));
  try {
    write();
  } catch {
    // Quota. Old reviews are the least valuable bytes in the store; if even an
    // empty shelf cannot take this one, losing the cache costs a re-sweep and
    // nothing else.
    for (const e of entries()) window.localStorage.removeItem(e.storage);
    try {
      write();
    } catch {
      return;
    }
  }
  const stale = entries()
    .sort((a, b) => b.at - a.at)
    .slice(KEEP);
  for (const e of stale) window.localStorage.removeItem(e.storage);
}

function entries(): { storage: string; at: number }[] {
  const out: { storage: string; at: number }[] = [];
  for (let i = 0; i < window.localStorage.length; i++) {
    const storage = window.localStorage.key(i);
    if (!storage || !storage.startsWith(PREFIX)) continue;
    try {
      const entry = JSON.parse(window.localStorage.getItem(storage) ?? "");
      out.push({ storage, at: typeof entry.at === "number" ? entry.at : 0 });
    } catch {
      out.push({ storage, at: 0 }); // unparseable: first against the wall
    }
  }
  return out;
}
