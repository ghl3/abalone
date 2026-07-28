"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type {
  ReadyMsg,
  SearchRequest,
  SearchResultMsg,
  WorkerResponse,
} from "./protocol";

export type EngineStatus = "idle" | "loading" | "ready" | "error";

export interface EngineProgress {
  visits: number;
  simulations: number;
}

/** Drives the engine worker. The worker is created on first use rather than on
 *  mount, so a visitor who never turns the network on never downloads the
 *  12 MB runtime or the 12 MB model. */
export function useEngine(modelPath = "/models/best.onnx") {
  const workerRef = useRef<Worker | null>(null);
  const [status, setStatus] = useState<EngineStatus>("idle");
  const [info, setInfo] = useState<ReadyMsg | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState<EngineProgress | null>(null);

  /** Resolvers for in-flight searches, keyed by request id. A search that is
   *  superseded resolves `null` — the caller wanted the newest answer, not
   *  every answer. */
  const pending = useRef(new Map<number, (r: SearchResultMsg | null) => void>());
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
        setProgress(null);
        pending.current.get(msg.id)?.(msg);
        pending.current.delete(msg.id);
      } else if (msg.kind === "progress") {
        setProgress({ visits: msg.visits, simulations: msg.simulations });
      } else if (msg.kind === "error") {
        // A failed search is not a failed engine: keep `ready` so the panel
        // still reports the provider, and surface the message either way.
        setStatus((s) => (s === "ready" && msg.id !== undefined ? s : "error"));
        setError(msg.message);
        setProgress(null);
        if (msg.id !== undefined) {
          pending.current.get(msg.id)?.(null);
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
      for (const resolve of pending.current.values()) resolve(null);
      pending.current.clear();
    };
  }, []);

  /** Search a position. Any earlier in-flight search is abandoned, so calling
   *  this on every position change costs one wasted forward pass at most. */
  const search = useCallback(
    (req: Omit<SearchRequest, "id">): Promise<SearchResultMsg | null> => {
      const worker = ensureWorker();
      const id = nextId.current++;
      for (const [oldId, resolve] of pending.current) {
        resolve(null);
        pending.current.delete(oldId);
      }
      return new Promise((resolve) => {
        pending.current.set(id, resolve);
        worker.postMessage({ kind: "search", id, ...req });
      });
    },
    [ensureWorker]
  );

  const cancel = useCallback(() => {
    workerRef.current?.postMessage({ kind: "cancel" });
    for (const resolve of pending.current.values()) resolve(null);
    pending.current.clear();
    setProgress(null);
  }, []);

  return { status, info, error, progress, search, cancel, ensureWorker };
}
