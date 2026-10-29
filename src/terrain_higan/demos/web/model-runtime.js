"use strict";
(() => {
  const { modelName } = TerrainHiGAN;
  // Early rounding differences grow through repeated upscaling; keep coarse parents canonical.
  const CANONICAL_PARENT_MAX_ZOOM = 15;

  class ModelRuntime {
    constructor(ort, options, emit) {
      this.ort = ort;
      this.options = options;
      this.emit = emit;
      this.sessions = new Map();
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
      const modelResponse = await fetch(this.modelUrl(zoom), { cache: "no-store" });
      if (!modelResponse.ok)
        throw new Error(`${modelName(zoom)} returned HTTP ${modelResponse.status}`);
      const bytes = new Uint8Array(await modelResponse.arrayBuffer());
      const digest = globalThis.crypto?.subtle
        ? await crypto.subtle.digest("SHA-256", bytes)
        : sha256.arrayBuffer(bytes);
      const hash = Array.from(new Uint8Array(digest), (byte) =>
        byte.toString(16).padStart(2, "0"),
      ).join("");
      if (hash !== manifest.models[modelName(zoom)]) {
        this.emit({ type: "bundle-changed" });
        throw new Error(`${modelName(zoom)} does not match model bundle ${bundle}`);
      }
      return bytes;
    }

    sessionFor(zoom) {
      if (!this.sessions.has(zoom)) this.sessions.set(zoom, this.loadSession(zoom));
      return this.sessions.get(zoom);
    }

    async loadSession(zoom) {
      const name = modelName(zoom);
      const model = await this.verifiedModelBytes(zoom);
      const cpuSelected = this.options.get("backend") === "wasm";
      const useWasm = cpuSelected || zoom <= CANONICAL_PARENT_MAX_ZOOM;
      let detail = cpuSelected
        ? "CPU selected"
        : useWasm
          ? `WASM z0-z${CANONICAL_PARENT_MAX_ZOOM} keeps parent terrain consistent across backends`
          : "WebGPU API unavailable";
      if (!useWasm && self.navigator?.gpu) {
        try {
          const session = await this.ort.InferenceSession.create(model, {
            executionProviders: ["webgpu"],
          });
          return {
            session,
            name,
            provider: {
              name: "webgpu",
              label: "WebGPU + WASM",
              detail: `WASM z0-z${CANONICAL_PARENT_MAX_ZOOM}; WebGPU z${CANONICAL_PARENT_MAX_ZOOM + 1}-z20`,
            },
          };
        } catch (error) {
          detail = String(error);
          console.warn("WebGPU initialization failed; using WASM:", error);
        }
      }
      const session = await this.ort.InferenceSession.create(model, {
        executionProviders: ["wasm"],
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

    async run(model, inputs) {
      const start = performance.now();
      const output = await model.session.run(inputs);
      this.metrics.modelRuns++;
      this.metrics.inferenceMs += performance.now() - start;
      return output;
    }
  }
  Object.assign(TerrainHiGAN, { ModelRuntime, CANONICAL_PARENT_MAX_ZOOM });
})();
