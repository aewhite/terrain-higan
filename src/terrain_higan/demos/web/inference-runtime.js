"use strict";
(() => {
  // Generation and inference share one worker. Serialize model execution and
  // session eviction; there are no nested workers or transferable input copies.
  class InferenceRuntime {
    constructor(options, emit, ort = globalThis.ort) {
      this.size = 1;
      this.lowMemory = options.get("mobile") === "1";
      this.providers = new Map();
      this.pending = Promise.resolve();
      let lastInferencePhase = -Infinity;
      this.runtime = new TerrainHiGAN.ModelRuntime(ort, options, (data) => {
        const now = performance.now();
        if (data.phase !== "inference" || now - lastInferencePhase >= 100) {
          if (data.phase === "inference") lastInferencePhase = now;
          emit(data);
        }
      });
      this.metrics = this.runtime.metrics;
      Object.assign(this.metrics, { inferenceWorkers: 1, maxParallelRuns: 0 });
    }

    async sessionFor(zoom) {
      return { zoom, name: TerrainHiGAN.modelName(zoom) };
    }

    tensor(data, dims) {
      return { data, dims };
    }

    run(model, inputs, assertCurrent = () => {}) {
      const result = this.pending.then(async () => {
        assertCurrent();
        const loaded = await this.runtime.sessionFor(model.zoom);
        assertCurrent();
        const previous = this.providers.get(model.name);
        if (previous && previous.name !== loaded.provider.name)
          throw new Error(
            `Inference provider changed for ${model.name}; reload to rebuild terrain consistently`,
          );
        const feeds = Object.fromEntries(
          Object.entries(inputs).map(([name, input]) => [
            name,
            this.runtime.tensor(input.data, input.dims),
          ]),
        );
        this.metrics.maxParallelRuns = 1;
        const output = await this.runtime.run(loaded, feeds);
        // WASM resolves synchronously. Yield a task so cancel/navigation
        // messages can run between CPU calls instead of waiting for a tile.
        if (loaded.provider.name === "wasm") await new Promise((resolve) => setTimeout(resolve, 0));
        this.providers.set(model.name, loaded.provider);
        return { ...output, model: model.name, provider: loaded.provider };
      });
      this.pending = result.catch(() => {});
      return result;
    }
  }

  function createInferenceRuntime(options, emit) {
    emit({ type: "phase", phase: "runtime", message: "Runtime" });
    const gpu = options.get("backend") !== "wasm" && !!self.navigator?.gpu;
    const bundle = options.get("bundle");
    const versioned = (file) => (bundle ? `${file}?bundle=${bundle}` : file);
    importScripts(
      versioned(gpu ? "./assets/runtime/ort.webgpu.min.js" : "./assets/runtime/ort.wasm.min.js"),
      versioned("./model-runtime.js"),
    );
    const mobile = options.get("mobile") === "1";
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.wasmPaths = mobile
      ? {
          mjs: new URL(
            gpu
              ? "./assets/runtime/ort-wasm-mobile.jsep.mjs"
              : "./assets/runtime/ort-wasm-mobile.mjs",
            self.location.href,
          ).href,
          wasm: new URL(
            gpu
              ? "./assets/runtime/ort-wasm-simd-threaded.jsep.wasm"
              : "./assets/runtime/ort-wasm-simd-threaded.wasm",
            self.location.href,
          ).href,
        }
      : new URL("./assets/runtime/", self.location.href).href;
    return new InferenceRuntime(options, emit);
  }

  Object.assign(TerrainHiGAN, { InferenceRuntime, createInferenceRuntime });
})();
