"use strict";
(() => {
  const { modelName } = TerrainHiGAN;

  class ModelRuntime {
    constructor(ort, options, emit) {
      this.ort = ort;
      this.options = options;
      this.emit = emit;
      this.sessions = new Map();
      this.lowMemory = options.get("mobile") === "1";
      if (this.lowMemory) this.size = 1;
      this.metrics = { modelRuns: 0, inferenceMs: 0 };
    }

    modelUrl(zoom) {
      const url = `./assets/models/${modelName(zoom)}.onnx`;
      const bundle = this.options.get("bundle");
      return bundle ? `${url}?bundle=${bundle}` : url;
    }

    async verifiedModelBytes(zoom) {
      const bundle = this.options.get("bundle");
      if (!bundle) return this.modelUrl(zoom);
      const response = await fetch("./assets/models/manifest.json", { cache: "no-store" });
      if (!response.ok) throw new Error(`Model manifest returned HTTP ${response.status}`);
      const manifest = await response.json();
      if (manifest.bundle !== bundle) {
        this.emit({ type: "bundle-changed" });
        throw new Error("Model bundle changed");
      }
      // Persist model files separately from the bounded in-memory ONNX sessions.
      // Use the content hash, not the UI bundle, so code updates reuse weights.
      const expected = manifest.models[modelName(zoom)];
      const key = new URL(
        `./assets/models/${modelName(zoom)}.onnx?sha256=${expected}`,
        self.location.href,
      ).href;
      let cache;
      try {
        cache = await globalThis.caches?.open("terrain-higan-models-v1");
        const cached = await cache?.match(key);
        if (cached) {
          this.emit({ type: "phase", phase: "read", message: `Read ${modelName(zoom)}` });
          const bytes = new Uint8Array(await cached.arrayBuffer());
          if ((await this.modelHash(bytes, zoom)) === expected) return bytes;
          // A damaged local entry must never reach ONNX. Repair it from the server.
          await cache.delete(key);
        }
      } catch (error) {
        console.warn("Model file cache unavailable; using browser HTTP cache:", error);
        cache = null;
      }
      this.emit({
        type: "phase",
        phase: "download",
        message: `Download ${modelName(zoom)}`,
      });
      const modelResponse = await fetch(key, { cache: cache ? "no-store" : "force-cache" });
      if (!modelResponse.ok)
        throw new Error(`${modelName(zoom)} returned HTTP ${modelResponse.status}`);
      const bytes = new Uint8Array(await modelResponse.arrayBuffer());
      if ((await this.modelHash(bytes, zoom)) !== expected) {
        this.emit({ type: "bundle-changed" });
        throw new Error(`${modelName(zoom)} does not match model bundle ${bundle}`);
      }
      if (cache) {
        try {
          await cache.put(
            key,
            new Response(bytes, { headers: { "Content-Type": "application/octet-stream" } }),
          );
        } catch (error) {
          // Storage denial/quota must not prevent rendering verified model data.
          console.warn("Could not retain model file in browser storage:", error);
        }
      }
      return bytes;
    }

    async modelHash(bytes, zoom) {
      this.emit({ type: "phase", phase: "verify", message: `Check ${modelName(zoom)}` });
      const digest = globalThis.crypto?.subtle
        ? await crypto.subtle.digest("SHA-256", bytes)
        : sha256.arrayBuffer(bytes);
      return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join(
        "",
      );
    }

    async sessionFor(zoom) {
      if (this.sessions.has(zoom)) {
        const existing = this.sessions.get(zoom);
        this.sessions.delete(zoom);
        this.sessions.set(zoom, existing);
        return existing;
      }
      // Mobile workers execute one job at a time, so unused sessions can be
      // released before loading the next model without interrupting inference.
      if (this.lowMemory && this.sessions.size >= 4) {
        const oldest = this.sessions.keys().next().value;
        const model = await this.sessions.get(oldest);
        this.sessions.delete(oldest);
        await model.session.release();
      }
      const pending = this.loadSession(zoom);
      this.sessions.set(zoom, pending);
      return pending;
    }

    async loadSession(zoom) {
      const name = modelName(zoom);
      const model = await this.verifiedModelBytes(zoom);
      this.emit({
        type: "phase",
        phase: "initialize",
        message: `Load ${name}`,
      });
      const cpuSelected = this.options.get("backend") === "wasm";
      let detail = cpuSelected ? "CPU selected" : "WebGPU API unavailable";
      if (!cpuSelected && self.navigator?.gpu) {
        try {
          const session = await this.ort.InferenceSession.create(model, {
            executionProviders: ["webgpu"],
          });
          return {
            session,
            name,
            provider: {
              name: "webgpu",
              label: "WebGPU",
              detail: "WebGPU inference",
            },
          };
        } catch (error) {
          detail = String(error);
          console.warn("WebGPU initialization failed; using WASM:", error);
        }
      }
      const session = await this.ort.InferenceSession.create(model, {
        executionProviders: ["wasm"],
        ...(this.lowMemory ? { enableCpuMemArena: false, enableMemPattern: false } : {}),
      });
      return {
        session,
        name,
        provider: {
          name: "wasm",
          label: `CPU / WASM (${this.ort.env.wasm.numThreads} ${this.ort.env.wasm.numThreads === 1 ? "thread" : "threads"})`,
          detail,
        },
      };
    }

    tensor(values, dimensions) {
      return new this.ort.Tensor("float32", values, dimensions);
    }

    async runSmallPatches(model, inputs) {
      const factor = model.name === "z0" ? 1 : 2;
      const first = Object.values(inputs)[0];
      const height = first.dims[2] * factor - 16;
      const width = first.dims[3] * factor - 16;
      if (height <= 32 && width <= 32) return model.session.run(inputs);
      const patchHeight = Math.min(32, height),
        patchWidth = Math.min(32, width);
      const inputHeight = (patchHeight + 16) / factor,
        inputWidth = (patchWidth + 16) / factor;
      const values = new Float32Array(width * height);
      for (let top = 0; top < height; top += patchHeight) {
        const sourceTop = Math.min(top, height - patchHeight);
        for (let left = 0; left < width; left += patchWidth) {
          const sourceLeft = Math.min(left, width - patchWidth);
          const patches = {};
          for (const [name, tensor] of Object.entries(inputs)) {
            const [, channels, rows, columns] = tensor.dims;
            const data = new Float32Array(channels * inputHeight * inputWidth);
            for (let channel = 0; channel < channels; channel++)
              for (let row = 0; row < inputHeight; row++) {
                const offset =
                  (channel * rows + sourceTop / factor + row) * columns + sourceLeft / factor;
                data.set(
                  tensor.data.subarray(offset, offset + inputWidth),
                  (channel * inputHeight + row) * inputWidth,
                );
              }
            patches[name] = this.tensor(data, [1, channels, inputHeight, inputWidth]);
          }
          const result = await model.session.run(patches);
          const output = result.elevation_normalized;
          const copyWidth = Math.min(patchWidth, width - left);
          for (let row = 0; row < Math.min(patchHeight, height - top); row++) {
            const offset = (top - sourceTop + row) * patchWidth + left - sourceLeft;
            values.set(
              output.data.subarray(offset, offset + copyWidth),
              (top + row) * width + left,
            );
          }
        }
      }
      return { elevation_normalized: this.tensor(values, [1, 1, height, width]) };
    }

    async run(model, inputs) {
      this.emit({
        type: "phase",
        phase: "inference",
        message: `Generate ${model.name}`,
      });
      const start = performance.now();
      const output =
        this.lowMemory && model.provider.name === "wasm"
          ? await this.runSmallPatches(model, inputs)
          : await model.session.run(inputs);
      this.metrics.modelRuns++;
      this.metrics.inferenceMs += performance.now() - start;
      return output;
    }
  }
  Object.assign(TerrainHiGAN, { ModelRuntime });
})();
