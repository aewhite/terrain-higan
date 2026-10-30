"use strict";
(() => {
  const { modelName } = TerrainHiGAN;

  class InferencePool {
    constructor(options, emit, createWorker = (url) => new Worker(url)) {
      this.options = options;
      this.lowMemory = options.get("mobile") === "1";
      this.emit = emit;
      this.createWorker = createWorker;
      this.size = TerrainHiGAN.inferenceWorkerCount(options);
      this.slots = Array.from({ length: this.size }, () => ({ worker: null, job: null }));
      this.queue = [];
      this.nextId = 0;
      this.providers = new Map();
      this.error = null;
      this.lastInferencePhase = -Infinity;
      this.metrics = {
        modelRuns: 0,
        inferenceMs: 0,
        inferenceWorkers: this.size,
        maxParallelRuns: 0,
      };
    }

    async sessionFor(zoom) {
      return { zoom, name: modelName(zoom) };
    }

    tensor(data, dims) {
      return { data, dims };
    }

    run(model, inputs, assertCurrent = () => {}) {
      if (this.error) return Promise.reject(this.error);
      return new Promise((resolve, reject) => {
        this.queue.push({
          id: ++this.nextId,
          zoom: model.zoom,
          inputs,
          assertCurrent,
          resolve,
          reject,
        });
        this.dispatch();
      });
    }

    dispatch() {
      if (this.error) return;
      try {
        for (const slot of this.slots) {
          if (slot.job || !this.queue.length) continue;
          while (this.queue.length) {
            try {
              this.queue[0].assertCurrent();
              break;
            } catch (error) {
              this.queue.shift().reject(error);
            }
          }
          if (!this.queue.length) break;
          if (!slot.worker) {
            const url = new URL("./inference-worker.js", self.location.href);
            url.search = this.options.toString();
            slot.worker = this.createWorker(url.href);
            slot.worker.onmessage = ({ data }) => this.receive(slot, data);
            slot.worker.onerror = (event) =>
              this.fail(new Error(event.message || "Inference worker failed"));
            slot.worker.onmessageerror = () =>
              this.fail(new Error("Inference worker message failed"));
          }
          slot.job = this.queue.shift();
          const { id, zoom, inputs } = slot.job;
          // Inputs are assembled copies, never the canonical cache's buffers.
          const buffers = Object.values(inputs).map((input) => input.data.buffer);
          slot.worker.postMessage({ id, zoom, inputs }, buffers);
        }
        this.metrics.maxParallelRuns = Math.max(
          this.metrics.maxParallelRuns,
          this.slots.filter((slot) => slot.job).length,
        );
      } catch (error) {
        this.fail(error);
      }
    }

    receive(slot, data) {
      if (data.type === "bundle-changed") {
        this.emit(data);
        this.fail(new Error("Model bundle changed"));
        return;
      }
      if (data.type === "phase" && slot.job) {
        const now = performance.now();
        if (data.phase !== "inference" || now - this.lastInferencePhase >= 100) {
          if (data.phase === "inference") this.lastInferencePhase = now;
          this.emit(data);
        }
        return;
      }
      const job = slot.job;
      if (!job || data.id !== job.id || !["result", "error"].includes(data.type)) return;
      if (data.type === "result") {
        const provider = this.providers.get(data.model);
        if (provider && provider !== data.provider.name) {
          this.fail(new Error(`Inference workers selected different providers for ${data.model}`));
          return;
        }
        this.providers.set(data.model, data.provider.name);
      }
      slot.job = null;
      if (data.type === "error") job.reject(new Error(data.message));
      else {
        this.metrics.modelRuns++;
        this.metrics.inferenceMs += data.inferenceMs;
        const provider = { ...data.provider };
        const workers = `${this.size} ${this.size === 1 ? "worker" : "workers"}`;
        provider.label =
          provider.name === "wasm" ? `CPU / WASM (${workers})` : `WebGPU + WASM (${workers})`;
        provider.detail += `; ${workers}, independent runtimes with one CPU thread each`;
        job.resolve({
          elevation_normalized: { data: new Float32Array(data.values), dims: data.dims },
          model: data.model,
          provider,
        });
      }
      this.dispatch();
    }

    fail(error) {
      this.error = error;
      for (const slot of this.slots) {
        slot.job?.reject(error);
        slot.job = null;
        slot.worker?.terminate();
      }
      for (const job of this.queue) job.reject(error);
      this.queue.length = 0;
    }
  }

  Object.assign(TerrainHiGAN, { InferencePool });
})();
