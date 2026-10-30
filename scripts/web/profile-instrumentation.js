// Injected only by profile-demo.cjs; production demo files are left unchanged.
(() => {
  if (globalThis.__terrainProfile) return;
  const state = (globalThis.__terrainProfile = {
    stats: {},
    stack: [],
    frames: [],
    longTasks: [],
    memories: [],
    canvases: [],
    renderer: null,
    record(label, wall, sync, self) {
      const s = (this.stats[label] ||= {
        count: 0,
        wallMs: 0,
        syncMs: 0,
        selfSyncMs: 0,
        maxMs: 0,
        durations: [],
      });
      s.count++;
      s.wallMs += wall;
      s.syncMs += sync;
      s.selfSyncMs += self;
      s.maxMs = Math.max(s.maxMs, wall);
      if (s.durations.length < 10000) s.durations.push(wall);
    },
    measure(label, fn) {
      if (fn.__terrainMeasured) return fn;
      const measured = function (...args) {
        const name = typeof label === "function" ? label.apply(this, args) : label;
        const started = performance.now(),
          span = { children: 0 };
        state.stack.push(span);
        let result, error;
        try {
          result = fn.apply(this, args);
        } catch (e) {
          error = e;
        }
        const sync = performance.now() - started;
        state.stack.pop();
        if (state.stack.length) state.stack[state.stack.length - 1].children += sync;
        const finish = () =>
          state.record(name, performance.now() - started, sync, Math.max(0, sync - span.children));
        if (error) {
          finish();
          throw error;
        }
        if (result && typeof result.then === "function")
          return result.then(
            (v) => {
              finish();
              return v;
            },
            (e) => {
              finish();
              throw e;
            },
          );
        finish();
        return result;
      };
      measured.__terrainMeasured = true;
      return measured;
    },
    wrap(object, methods, prefix) {
      for (const name of methods)
        if (typeof object?.[name] === "function")
          object[name] = this.measure(prefix + "." + name, object[name]);
    },
    reset() {
      this.stats = {};
      this.frames = [];
      this.longTasks = [];
    },
    snapshot() {
      const stats = {};
      for (const [name, s] of Object.entries(this.stats)) {
        const d = [...s.durations].sort((a, b) => a - b);
        stats[name] = {
          count: s.count,
          wallMs: s.wallMs,
          syncMs: s.syncMs,
          selfSyncMs: s.selfSyncMs,
          maxMs: s.maxMs,
          p50Ms: d[Math.floor(d.length * 0.5)] || 0,
          p95Ms: d[Math.floor(d.length * 0.95)] || 0,
        };
      }
      const canvases = this.canvases
        .map((r) => r.deref())
        .filter(Boolean)
        .map((c) => ({ id: c.id, width: c.width, height: c.height }));
      return {
        stats,
        frames: this.frames,
        longTasks: this.longTasks,
        memories: this.memories.map(({ memory, maximum }) => ({
          bytes: memory.buffer.byteLength,
          maximumPages: maximum,
        })),
        canvases,
        canvasRgbaBytes: canvases.reduce((n, c) => n + c.width * c.height * 4, 0),
        resources: performance.getEntriesByType("resource").map((r) => ({
          name: r.name,
          duration: r.duration,
          transferSize: r.transferSize,
          encodedBodySize: r.encodedBodySize,
        })),
      };
    },
  });
  const memory = WebAssembly.Memory;
  WebAssembly.Memory = new Proxy(memory, {
    construct(target, args, newTarget) {
      const result = Reflect.construct(target, args, newTarget);
      state.memories.push({ memory: result, maximum: args[0].maximum });
      return result;
    },
  });
  const nativeFetch = globalThis.fetch;
  globalThis.fetch = state.measure(
    function (url) {
      const name = String(url);
      return name.includes(".onnx")
        ? "network.modelHeaders"
        : name.includes(".wasm")
          ? "network.wasmHeaders"
          : "network.otherHeaders";
    },
    async function (...args) {
      const response = await nativeFetch.apply(this, args);
      const url = String(args[0]);
      state.wrap(
        response,
        ["arrayBuffer"],
        url.includes(".onnx")
          ? "network.modelBody"
          : url.includes(".wasm")
            ? "network.wasmBody"
            : "network.otherBody",
      );
      return response;
    },
  );
  if (globalThis.crypto?.subtle) state.wrap(crypto.subtle, ["digest"], "hash");
  if (typeof document !== "undefined") {
    const getContext = HTMLCanvasElement.prototype.getContext;
    const seen = new WeakSet();
    HTMLCanvasElement.prototype.getContext = function (...args) {
      if (!seen.has(this)) {
        seen.add(this);
        state.canvases.push(new WeakRef(this));
      }
      return getContext.apply(this, args);
    };
    new PerformanceObserver((list) => {
      for (const e of list.getEntries())
        state.longTasks.push({ start: e.startTime, duration: e.duration });
    }).observe({ type: "longtask", buffered: true });
    let last = 0;
    const tick = (now) => {
      if (last) state.frames.push(now - last);
      last = now;
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }
})();
