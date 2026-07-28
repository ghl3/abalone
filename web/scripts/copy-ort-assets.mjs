// Stage onnxruntime-web's wasm binaries into `public/ort/`.
//
// ORT fetches its runtime binary at session-creation time from
// `ort.env.wasm.wasmPaths`. Left unset it resolves relative to the bundled
// chunk URL, which under webpack is a hashed path that has no .wasm next to
// it — so the first `InferenceSession.create` 404s. Copying to a stable
// public path and pointing `wasmPaths` at it is the supported escape hatch.
//
// Both variants are staged: `.jsep` carries the WebGPU kernels, the plain one
// is the CPU fallback. Which is fetched depends on the execution provider the
// worker settles on, and that is a runtime decision.

import { copyFile, mkdir, stat } from "node:fs/promises";
import { createRequire } from "node:module";
import { join } from "node:path";

const require = createRequire(import.meta.url);
const OUT = join(import.meta.dirname, "..", "public", "ort");

// Each of these is an explicit subpath in onnxruntime-web's `exports` map, so
// resolving them by name survives the package rearranging its `dist/`.
const FILES = [
  "ort-wasm-simd-threaded.wasm",
  "ort-wasm-simd-threaded.mjs",
  "ort-wasm-simd-threaded.jsep.wasm",
  "ort-wasm-simd-threaded.jsep.mjs",
];

await mkdir(OUT, { recursive: true });

for (const f of FILES) {
  const src = require.resolve(`onnxruntime-web/${f}`);
  const dst = join(OUT, f);
  const [s, d] = await Promise.all([
    stat(src),
    stat(dst).catch(() => null),
  ]);
  if (d && d.size === s.size && d.mtimeMs >= s.mtimeMs) continue;
  await copyFile(src, dst);
  console.log(`ort assets: ${f} (${(s.size / 1048576).toFixed(1)} MB)`);
}
