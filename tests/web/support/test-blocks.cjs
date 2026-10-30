const assert = require("node:assert/strict");
const { loadModules } = require("./test-support.cjs");

module.exports = async function checkBlocks() {
  const { api, context } = loadModules();
  const request = { seed: 1n, assertCurrent() {} };
  const region = { zoom: 0, x: -3, y: -5, width: 2, height: 3 };
  const generator = new api.TerrainGenerator({});
  let release, entered;
  const gate = new Promise((resolve) => {
    release = resolve;
  });
  const started = new Promise((resolve) => {
    entered = resolve;
  });
  const calls = [];
  generator.generateUncached = async (block) => {
    calls.push(block);
    entered();
    await gate;
    return {
      values: Float32Array.from(
        { length: block.width * block.height },
        (_, i) => block.x + (i % block.width) + (block.y + Math.floor(i / block.width)) * 1024,
      ),
      model: "z0",
      provider: { name: "wasm" },
    };
  };
  let obsolete = false;
  const first = generator.generate(region, {
    seed: 1n,
    assertCurrent() {
      if (obsolete) throw new Error("cancelled");
    },
  });
  const second = generator.generate({ ...region, x: -2, y: -4, width: 1, height: 1 }, request);
  await started;
  assert.equal(calls.length, 1, "overlapping concurrent crops claim one block");
  assert.equal(generator.cache.bytes, 0, "pending output is not published");
  assert.equal(generator.cache.pending.size, 1);
  assert.equal(generator.cache.coalesced, 1);
  assert.deepEqual({ ...calls[0] }, { zoom: 0, x: -64, y: -64, width: 64, height: 64 });
  obsolete = true;
  release();
  await assert.rejects(first, /cancelled/);
  const value = await second;
  assert.equal(value.values[0], -2 - 4 * 1024);
  assert.equal(generator.cache.pending.size, 0);
  assert.equal(generator.cache.bytes, 64 * 64 * 4);
  value.values.fill(123);
  const repeated = await generator.generate(
    { ...region, x: -2, y: -4, width: 1, height: 1 },
    request,
  );
  assert.equal(repeated.values[0], -2 - 4 * 1024, "caller owns its crop independently");
  const crossing = await generator.generate(
    { zoom: 0, x: -1, y: -1, width: 2, height: 2 },
    request,
  );
  assert.deepEqual(Array.from(crossing.values), [-1025, -1024, -1, 0]);
  assert.equal(calls.length, 4, "negative-coordinate boundary uses four complete blocks");

  const failing = new api.TerrainGenerator({});
  let attempts = 0;
  failing.generateUncached = async () => {
    if (++attempts === 1) throw new Error("model failure");
    return { values: new Float32Array(64 * 64).fill(7) };
  };
  const failures = await Promise.allSettled([
    failing.generate(region, request),
    failing.generate(region, request),
  ]);
  assert(failures.every((result) => result.status === "rejected"));
  assert.equal(attempts, 1);
  assert.equal(failing.cache.pending.size, 0);
  assert.equal(failing.cache.bytes, 0);
  assert.equal((await failing.generate(region, request)).values[0], 7, "failed claims can retry");

  const clearing = new api.TerrainGenerator({});
  const releases = [];
  clearing.generateUncached = () => new Promise((resolve) => releases.push(resolve));
  const old = clearing.generate(region, request);
  await Promise.resolve();
  clearing.cache.clear();
  const fresh = clearing.generate(region, request);
  await Promise.resolve();
  releases[0]({ values: new Float32Array(64 * 64).fill(1) });
  await old;
  assert.equal(clearing.cache.bytes, 0, "cleared producers cannot repopulate the cache");
  assert.equal(clearing.cache.pending.size, 1, "old completion cannot release a newer claim");
  releases[1]({ values: new Float32Array(64 * 64).fill(2) });
  await fresh;
  assert.equal((await clearing.generate(region, request)).values[0], 2);

  {
    const calls = [];
    const feeds = [];
    const fakeOrt = {
      env: { wasm: { numThreads: 4 } },
      Tensor: class {
        constructor(_type, data, dims) {
          this.data = data;
          this.dims = dims;
        }
      },
      InferenceSession: {
        async create(file, settings) {
          calls.push({ file, provider: settings.executionProviders[0] });
          return {
            async run(inputs) {
              feeds.push(inputs);
              return { elevation_normalized: { data: inputs.noise.data, dims: inputs.noise.dims } };
            },
            async release() {},
          };
        },
      },
    };
    const phases = [];
    context.navigator.gpu = {};
    const gpu = new api.InferenceRuntime(
      new URLSearchParams("mobile=1&workers=8"),
      (data) => phases.push(data),
      fakeOrt,
    );
    for (const zoom of [0, 5, 15, 20]) {
      const data = new Float32Array(80 * 80).fill(zoom);
      const result = await gpu.run(await gpu.sessionFor(zoom), {
        noise: gpu.tensor(data, [1, 1, 80, 80]),
      });
      assert.equal(
        result.provider.name,
        "webgpu",
        "all levels, including mobile parents, prefer WebGPU",
      );
      assert.equal(
        result.elevation_normalized.data,
        data,
        "GPU outputs need no worker-transfer copy",
      );
      assert.equal(data.length, 80 * 80, "inputs remain owned by the generation worker");
    }
    assert(calls.every((call) => call.provider === "webgpu"));
    assert.equal(feeds.length, 4, "mobile GPU execution uses whole blocks, not CPU micro-patches");
    const originalSession = await gpu.runtime.sessionFor(0);
    gpu.runtime.sessions.set(
      0,
      Promise.resolve({ ...originalSession, provider: { name: "wasm" } }),
    );
    await assert.rejects(
      gpu.run(await gpu.sessionFor(0), { noise: gpu.tensor(new Float32Array(1), [1, 1, 1, 1]) }),
      /provider changed/,
      "an existing world's cache cannot mix different providers for the same model",
    );
    assert.equal(feeds.length, 4, "a provider change is rejected before running another block");
    assert.equal(gpu.metrics.maxParallelRuns, 1);
    assert(phases.some((phase) => phase.phase === "initialize"));
    assert(phases.some((phase) => phase.phase === "inference"));

    const failingOrt = {
      ...fakeOrt,
      InferenceSession: {
        async create(file, settings) {
          if (settings.executionProviders[0] === "webgpu")
            throw new Error("GPU adapter unavailable");
          return fakeOrt.InferenceSession.create(file, settings);
        },
      },
    };
    const fallback = new api.ModelRuntime(failingOrt, new URLSearchParams(), () => {});
    const cpu = await fallback.sessionFor(0);
    assert.equal(cpu.provider.name, "wasm");
    assert.match(cpu.provider.detail, /GPU adapter unavailable/);
    assert.equal(await fallback.sessionFor(0), cpu, "fallback sessions are reused");

    // A single execution queue skips cancelled work before loading a session and
    // keeps session eviction outside active execution.
    let release, started;
    const entered = new Promise((resolve) => (started = resolve));
    const hold = new Promise((resolve) => (release = resolve));
    let active = 0,
      maximum = 0;
    const ran = [];
    const serialOrt = {
      ...fakeOrt,
      InferenceSession: {
        async create(file) {
          return {
            async run(inputs) {
              active++;
              maximum = Math.max(maximum, active);
              ran.push(file);
              if (ran.length === 1) {
                started();
                await hold;
              }
              active--;
              return { elevation_normalized: { data: inputs.noise.data, dims: inputs.noise.dims } };
            },
            async release() {
              assert.equal(active, 0, "sessions cannot be released during a run");
            },
          };
        },
      },
    };
    const serial = new api.InferenceRuntime(
      new URLSearchParams("backend=wasm&mobile=1"),
      () => {},
      serialOrt,
    );
    const input = () => ({ noise: serial.tensor(new Float32Array([3]), [1, 1, 1, 1]) });
    const first = serial.run(await serial.sessionFor(0), input());
    await entered;
    let obsolete = false;
    const cancelled = assert.rejects(
      serial.run(await serial.sessionFor(1), input(), () => {
        if (obsolete) throw new Error("cancelled");
      }),
      /cancelled/,
    );
    const following = [];
    for (let zoom = 2; zoom < 8; zoom++)
      following.push(serial.run(await serial.sessionFor(zoom), input()));
    setTimeout(() => {
      obsolete = true;
    }, 0);
    release();
    await Promise.all([first, cancelled, ...following]);
    assert.equal(maximum, 1, "ONNX execution never overlaps across sessions");
    assert(
      !ran.some((file) => file.endsWith("z1.onnx")),
      "cancelled queued work never reaches ONNX",
    );
    assert.equal(serial.runtime.sessions.size, 4, "mobile session memory stays bounded");
    assert.equal(serial.metrics.modelRuns, 7);

    const imports = [];
    const loadScripts = context.importScripts;
    context.importScripts = (...files) => {
      imports.push(...files);
      loadScripts(...files);
    };
    context.ort = fakeOrt;
    api.createInferenceRuntime(new URLSearchParams("mobile=1"), () => {});
    assert(imports.some((file) => file.includes("ort.webgpu.min.js")));
    assert.match(fakeOrt.env.wasm.wasmPaths.mjs, /ort-wasm-mobile\.jsep\.mjs$/);
    assert.equal(fakeOrt.env.wasm.numThreads, 1);
    imports.length = 0;
    api.createInferenceRuntime(new URLSearchParams("mobile=1&backend=wasm"), () => {});
    assert(imports.some((file) => file.includes("ort.wasm.min.js")));
    assert.match(fakeOrt.env.wasm.wasmPaths.mjs, /ort-wasm-mobile\.mjs$/);
    delete context.navigator.gpu;
    imports.length = 0;
    api.createInferenceRuntime(new URLSearchParams(), () => {});
    assert(
      imports.some((file) => file.includes("ort.wasm.min.js")),
      "missing WebGPU selects CPU runtime",
    );
    console.log(
      "Canonical blocks, cancellation/ownership, GPU-first execution, CPU fallback and one serialized runtime passed",
    );
  }
};
