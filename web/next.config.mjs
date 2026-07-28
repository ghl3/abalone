/** @type {import('next').NextConfig} */
const nextConfig = {
  webpack(config) {
    // wasm-bindgen >= 0.2.100 emits glue that calls
    // `wasm.__wbindgen_start()` synchronously at module top, so the wasm
    // namespace must be fully resolved before the next statement runs.
    // `asyncWebAssembly` makes the .wasm import async; `topLevelAwait`
    // lets the generated `import * as wasm from "...wasm"` be awaited
    // before any subsequent line runs.
    config.experiments = {
      ...config.experiments,
      asyncWebAssembly: true,
      topLevelAwait: true,
    };
    config.output.environment = {
      ...config.output.environment,
      asyncFunction: true,
    };
    return config;
  },

  // `onnxruntime-web`'s threaded wasm backend needs `SharedArrayBuffer`, which
  // browsers only hand to a cross-origin-isolated document. Without these two
  // headers ORT is capped at one thread and inference runs several times
  // slower. Everything this app loads is same-origin, so `require-corp` costs
  // us nothing.
  async headers() {
    return [
      {
        source: "/:path*",
        headers: [
          { key: "Cross-Origin-Opener-Policy", value: "same-origin" },
          { key: "Cross-Origin-Embedder-Policy", value: "require-corp" },
        ],
      },
    ];
  },
};

export default nextConfig;
