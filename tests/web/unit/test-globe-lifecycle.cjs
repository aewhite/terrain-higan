const { loadModules } = require("../support/test-support.cjs");
const assert = require("node:assert/strict");

(async () => {
  const { api, load } = loadModules();
  load("globe-topology.js");
  load("globe-terrain.js");
  const { GlobeGenerator, ObsoleteGlobeRequest } = api.globe;

  const mobileGenerator = new GlobeGenerator({ lowMemory: true });
  let activeDependencies = 0,
    peakDependencies = 0;
  mobileGenerator.block = async (_seed, region) => {
    activeDependencies++;
    peakDependencies = Math.max(peakDependencies, activeDependencies);
    await Promise.resolve();
    activeDependencies--;
    return { ...region, values: new Float32Array(region.width * region.height).fill(region.x) };
  };
  const gathered = await mobileGenerator.gather(
    1n,
    10,
    [0, 64, 128].map((east) => ({ face: 0, east, north: 0 })),
  );
  assert.equal(peakDependencies, 1, "mobile prepares one dependency region at a time");
  assert.deepEqual(Array.from(gathered), [0, 64, 128]);

  function gatedRuntime() {
    let release, entered;
    const gate = new Promise((resolve) => {
      release = resolve;
    });
    const started = new Promise((resolve) => {
      entered = resolve;
    });
    const runs = [];
    const runtime = {
      sessionFor: async (zoom) => ({ zoom, name: api.modelName(zoom) }),
      tensor: (data, dims) => ({ data, dims }),
      async run(model, inputs) {
        runs.push(model.zoom);
        if (runs.length === 1) {
          entered();
          await gate;
        }
        const input = inputs.input_map || inputs.noise;
        const height = input.dims[2] * (model.zoom ? 2 : 1) - 16;
        const width = input.dims[3] * (model.zoom ? 2 : 1) - 16;
        return {
          model: model.name,
          elevation_normalized: {
            dims: [1, 1, height, width],
            data: new Float32Array(width * height),
          },
        };
      },
    };
    return { runtime, started, release: () => release(), runs };
  }

  const cancelledRuntime = gatedRuntime();
  const cancelledGenerator = new GlobeGenerator(cancelledRuntime.runtime);
  let obsolete = false;
  const assertCurrent = () => {
    if (obsolete) throw new ObsoleteGlobeRequest();
  };
  const cancelled = assert.rejects(
    cancelledGenerator.tile(1n, 0, 5, 16, 16, assertCurrent),
    ObsoleteGlobeRequest,
  );
  await cancelledRuntime.started;
  const runsAtCancellation = cancelledRuntime.runs.length;
  obsolete = true;
  cancelledRuntime.release();
  await cancelled;
  while (cancelledGenerator.cache.pending.size)
    await Promise.allSettled([...cancelledGenerator.cache.pending.values()]);
  assert.equal(
    cancelledRuntime.runs.length,
    runsAtCancellation,
    "obsolete dependencies never start another inference",
  );

  const sharedRuntime = gatedRuntime();
  const sharedGenerator = new GlobeGenerator(sharedRuntime.runtime);
  obsolete = false;
  const superseded = assert.rejects(
    sharedGenerator.tile(1n, 0, 5, 16, 16, assertCurrent),
    ObsoleteGlobeRequest,
  );
  await sharedRuntime.started;
  obsolete = true;
  const needed = sharedGenerator.tile(1n, 0, 5, 16, 16);
  sharedRuntime.release();
  const tile = await needed;
  await superseded;
  assert(tile.values.every(Number.isFinite), "current consumers recover shared cancelled claims");
  assert.equal(sharedGenerator.cache.pending.size, 0);
  const cachedRuns = sharedRuntime.runs.length;
  assert.deepEqual((await sharedGenerator.tile(1n, 0, 5, 16, 16)).values, tile.values);
  assert.equal(sharedRuntime.runs.length, cachedRuns, "completed shared blocks remain reusable");

  let worker;
  const pool = new api.InferencePool(
    new URLSearchParams("workers=1"),
    () => {},
    () => {
      worker = {
        messages: [],
        postMessage(message) {
          this.messages.push(message);
        },
        terminate() {},
      };
      return worker;
    },
  );
  const model = await pool.sessionFor(0);
  const inputs = () => ({ noise: pool.tensor(new Float32Array([0]), [1, 1, 1, 1]) });
  const running = pool.run(model, inputs());
  obsolete = false;
  const discarded = assert.rejects(pool.run(model, inputs(), assertCurrent), ObsoleteGlobeRequest);
  const latest = pool.run(model, inputs());
  obsolete = true;
  const reply = (message) =>
    worker.onmessage({
      data: {
        id: message.id,
        type: "result",
        model: "z0",
        provider: { name: "wasm", detail: "test" },
        values: new Float32Array([0]).buffer,
        dims: [1, 1, 1, 1],
        inferenceMs: 1,
      },
    });
  reply(worker.messages[0]);
  assert.equal(worker.messages.length, 2);
  assert.equal(worker.messages[1].id, 3, "queued obsolete inference is skipped before dispatch");
  reply(worker.messages[1]);
  await Promise.all([running, discarded, latest]);
  assert.equal(pool.queue.length, 0);
  console.log(
    "Globe cancellation stops dependency chains and queued inference, preserves shared retries, and reuses completed blocks",
  );
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
