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
};

export default nextConfig;
