"use strict";

const runtimeOptions = new URLSearchParams(self.location.search);
const bundle = runtimeOptions.get("bundle");
const versioned = (file) => (bundle ? `${file}?bundle=${bundle}` : file);
importScripts(versioned("./assets/runtime/ort.webgpu.min.js"));
if (!globalThis.crypto?.subtle) importScripts(versioned("./assets/runtime/sha256.js"));
importScripts(versioned("./terrain-common.js"), versioned("./model-runtime.js"));

ort.env.wasm.wasmPaths = new URL("./assets/runtime/", self.location.href).href;
ort.env.wasm.numThreads = 1;
const runtime = new TerrainHiGAN.ModelRuntime(ort, runtimeOptions, (data) => postMessage(data));

self.onmessage = async ({ data }) => {
  try {
    const model = await runtime.sessionFor(data.zoom);
    const inputs = Object.fromEntries(
      Object.entries(data.inputs).map(([name, input]) => [
        name,
        runtime.tensor(input.data, input.dims),
      ]),
    );
    const before = runtime.metrics.inferenceMs;
    const result = await runtime.run(model, inputs);
    const output = result.elevation_normalized;
    const values = new Float32Array(output.data);
    postMessage(
      {
        type: "result",
        id: data.id,
        model: model.name,
        provider: model.provider,
        dims: output.dims,
        values: values.buffer,
        inferenceMs: runtime.metrics.inferenceMs - before,
      },
      [values.buffer],
    );
  } catch (error) {
    postMessage({ type: "error", id: data.id, message: error.message || String(error) });
  }
};
