"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type {
  ProgressMsg,
  ReadyMsg,
  SearchRequest,
  SearchResultMsg,
  WorkerResponse,
} from "./protocol";

export type EngineStatus = "idle" | "loading" | "ready" | "error";

/** Called as the search runs, with the tree as it currently stands. Delivered
 *  per *call* rather than through shared state on purpose: a superseded search
 *  can still have a message in flight, and routing progress back to the caller
 *  that asked for it is what stops the old position's rows landing in the new
 *  position's panel. */
export type ProgressHandler = (p: ProgressMsg) => void;

/** The model every part of the app runs unless told otherwise. Exported for
 *  the review cache, which keys on this file's identity. */
export const DEFAULT_MODEL_PATH = "/models/best.onnx";

/** Drives the engine worker. The worker is created on first use rather than on
 *  mount, so a visitor who never turns the network on never downloads the
 *  12 MB runtime or the 12 MB model. */
export function useEngine(modelPath = DEFAULT_MODEL_PATH) {
  const workerRef = useRef<Worker | null>(null);
  const [status, setStatus] = useState<EngineStatus>("idle");
  const [info, setInfo] = useState<ReadyMsg | null>(null);
  const [error, setError] = useState<string | null>(null);

  /** In-flight searches, keyed by request id. A search that is superseded
   *  resolves `null` — the caller wanted the newest answer, not every answer —
   *  and stops receiving progress at the same moment. */
  const pending = useRef(
    new Map<
      number,
      { resolve: (r: SearchResultMsg | null) => void; onProgress?: ProgressHandler }
    >()
  );
  const nextId = useRef(1);

  const ensureWorker = useCallback(() => {
    if (workerRef.current) return workerRef.current;
    const worker = new Worker(new URL("./engine.worker.ts", import.meta.url), {
      type: "module",
    });
    worker.onmessage = (e: MessageEvent<WorkerResponse>) => {
      const msg = e.data;
      if (msg.kind === "ready") {
        setStatus("ready");
        setInfo(msg);
      } else if (msg.kind === "result") {
        // A search that completes clears whatever the last one complained
        // about. Without this a single transient failure stays pinned under
        // the board for the rest of the session, describing an engine that
        // has since recovered.
        setError(null);
        pending.current.get(msg.id)?.resolve(msg);
        pending.current.delete(msg.id);
      } else if (msg.kind === "progress") {
        // Nothing to route to means the search was superseded while this tick
        // was on the wire. Dropping it is the point.
        pending.current.get(msg.id)?.onProgress?.(msg);
      } else if (msg.kind === "error") {
        // A failed search is not a failed engine: keep `ready` so the panel
        // still reports the provider, and surface the message either way.
        setStatus((s) => (s === "ready" && msg.id !== undefined ? s : "error"));
        setError(msg.message);
        if (msg.id !== undefined) {
          pending.current.get(msg.id)?.resolve(null);
          pending.current.delete(msg.id);
        }
      }
    };
    worker.onerror = (e) => {
      setStatus("error");
      setError(e.message || "engine worker failed to start");
    };
    workerRef.current = worker;
    setStatus("loading");
    worker.postMessage({ kind: "init", modelPath });
    return worker;
  }, [modelPath]);

  useEffect(() => {
    return () => {
      workerRef.current?.terminate();
      workerRef.current = null;
      for (const p of pending.current.values()) p.resolve(null);
      pending.current.clear();
    };
  }, []);

  /** Search a position, reporting the tree as it builds through `onProgress`.
   *  Any earlier in-flight search is abandoned, so calling this on every
   *  position change costs one wasted forward pass at most. */
  const search = useCallback(
    (
      req: Omit<SearchRequest, "id">,
      onProgress?: ProgressHandler
    ): Promise<SearchResultMsg | null> => {
      const worker = ensureWorker();
      const id = nextId.current++;
      for (const [oldId, p] of pending.current) {
        p.resolve(null);
        pending.current.delete(oldId);
      }
      return new Promise((resolve) => {
        pending.current.set(id, { resolve, onProgress });
        worker.postMessage({ kind: "search", id, ...req });
      });
    },
    [ensureWorker]
  );

  const cancel = useCallback(() => {
    workerRef.current?.postMessage({ kind: "cancel" });
    for (const p of pending.current.values()) p.resolve(null);
    pending.current.clear();
  }, []);

  return { status, info, error, search, cancel, ensureWorker };
}
