"use strict";
(() => {
  // Generation and inference share one worker. Serialize model execution and
  // session eviction; there are no nested workers or transferable input copies.
  class InferenceRuntime {
    constructor(options, emit, ort = globalThis.ort) {
      this.lowMemory = options.get("mobile") === "1";
      const requested = Number(options.get("batch"));
      this.batchSize =
        !this.lowMemory && options.get("backend") !== "wasm" && !!self.navigator?.gpu
          ? [1, 2, 4, 8].includes(requested)
            ? requested
            : 4
          : 1;
      this.size = this.batchSize;
      this.providers = new Map();
      this.pending = Promise.resolve();
      this.queued = [];
      this.draining = false;
      this.delayMs = 0;
      let lastInferencePhase = -Infinity;
      this.runtime = new TerrainHiGAN.ModelRuntime(ort, options, (data) => {
        const now = performance.now();
        if (data.phase !== "inference" || now - lastInferencePhase >= 100) {
          if (data.phase === "inference") lastInferencePhase = now;
          emit(data);
        }
      });
      this.metrics = this.runtime.metrics;
      Object.assign(this.metrics, {
        inferenceWorkers: 1,
        maxParallelRuns: 0,
        batchedRuns: 0,
        inferenceBlocks: 0,
        maxBatchSize: 0,
        maxQueuedRuns: 0,
      });
    }

    async sessionFor(zoom) {
      return { zoom, name: TerrainHiGAN.modelName(zoom) };
    }

    tensor(data, dims) {
      return { data, dims };
    }

    setPacing(delayMs) {
      this.delayMs = Math.max(0, Math.min(32, delayMs));
      this.metrics.inferenceDelayMs = this.delayMs;
    }

    run(model, inputs, assertCurrent = () => {}) {
      const result = new Promise((resolve, reject) => {
        this.queued.push({ model, inputs, assertCurrent, resolve, reject });
      });
      this.metrics.maxQueuedRuns = Math.max(this.metrics.maxQueuedRuns, this.queued.length);
      if (!this.draining) {
        this.draining = true;
        this.pending = Promise.resolve().then(() => this.drain());
      }
      return result;
    }

    compatible(first, next) {
      return (
        first.model.zoom === next.model.zoom &&
        Object.keys(first.inputs).length === Object.keys(next.inputs).length &&
        Object.entries(first.inputs).every(
          ([name, input]) =>
            input.dims[0] === 1 && next.inputs[name]?.dims.join(",") === input.dims.join(","),
        )
      );
    }

    async drain() {
      try {
        while (this.queued.length) {
          const first = this.queued.shift();
          const jobs = [first];
          try {
            first.assertCurrent();
            const loaded = await this.runtime.sessionFor(first.model.zoom);
            first.assertCurrent();
            this.size = loaded.provider.name === "webgpu" ? this.batchSize : 1;
            if (loaded.provider.name === "webgpu") {
              for (let index = 0; index < this.queued.length && jobs.length < this.batchSize; ) {
                const next = this.queued[index];
                try {
                  next.assertCurrent();
                } catch (error) {
                  this.queued.splice(index, 1);
                  next.reject(error);
                  continue;
                }
                if (this.compatible(first, next)) jobs.push(...this.queued.splice(index, 1));
                else index++;
              }
            }
            const outputs = await this.execute(loaded, jobs);
            jobs.forEach((job, index) => job.resolve(outputs[index]));
          } catch (error) {
            jobs.forEach((job) => job.reject(error));
          }
        }
      } finally {
        this.draining = false;
      }
    }

    async execute(loaded, jobs) {
      const { model, inputs } = jobs[0];
      const previous = this.providers.get(model.name);
      if (previous && previous.name !== loaded.provider.name)
        throw new Error(
          `Inference provider changed for ${model.name}; reload to rebuild terrain consistently`,
        );
      const feeds = Object.fromEntries(
        Object.entries(inputs).map(([name, input]) => {
          if (jobs.length === 1) return [name, this.runtime.tensor(input.data, input.dims)];
          const values = new Float32Array(input.data.length * jobs.length);
          jobs.forEach((job, index) =>
            values.set(job.inputs[name].data, index * input.data.length),
          );
          return [name, this.runtime.tensor(values, [jobs.length, ...input.dims.slice(1)])];
        }),
      );
      this.metrics.maxParallelRuns = 1;
      const output = await this.runtime.run(loaded, feeds);
      this.metrics.inferenceBlocks += jobs.length;
      this.metrics.maxBatchSize = Math.max(this.metrics.maxBatchSize, jobs.length);
      if (jobs.length > 1) this.metrics.batchedRuns++;
      // Leave GPU time for drawing when the page reports slow frames. WASM
      // also needs a task boundary so navigation/cancel messages can run.
      if (loaded.provider.name === "wasm" || this.delayMs > 0)
        await new Promise((resolve) => setTimeout(resolve, this.delayMs));
      this.providers.set(model.name, loaded.provider);
      if (jobs.length === 1) return [{ ...output, model: model.name, provider: loaded.provider }];
      const elevation = output.elevation_normalized;
      if (elevation.dims[0] !== jobs.length)
        throw new Error("Batched model returned an unexpected batch dimension");
      const count = elevation.data.length / jobs.length;
      return jobs.map((_, index) => ({
        model: model.name,
        provider: loaded.provider,
        elevation_normalized: {
          dims: [1, ...elevation.dims.slice(1)],
          data: elevation.data.slice(index * count, (index + 1) * count),
        },
      }));
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
