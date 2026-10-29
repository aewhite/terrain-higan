const assert = require("node:assert/strict");
const { loadModules } = require("./test-support.cjs");
const { MessageChannel, receiveMessageOnPort } = require("node:worker_threads");

module.exports = async function checkBlocks() {
  const { api } = loadModules();
  for (const [query, size] of [
    ["", 4],
    ["workers=1", 1],
    ["workers=2", 2],
    ["workers=8", 4],
  ]) {
    assert.equal(new api.InferencePool(new URLSearchParams(query), () => {}).size, size);
  }
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

  const workers = [];
  const pool = new api.InferencePool(
    new URLSearchParams("workers=2"),
    () => {},
    (url) => {
      const worker = {
        url,
        messages: [],
        terminated: false,
        postMessage(data, buffers) {
          const ports = new MessageChannel();
          ports.port1.postMessage(data, buffers);
          this.messages.push(receiveMessageOnPort(ports.port2).message);
          ports.port1.close();
          ports.port2.close();
        },
        terminate() {
          this.terminated = true;
        },
      };
      workers.push(worker);
      return worker;
    },
  );
  const model = await pool.sessionFor(0);
  const input = (value) => ({ noise: pool.tensor(new Float32Array([value]), [1, 1, 1, 1]) });
  const jobs = [1, 2, 3, 4].map((value) => pool.run(model, input(value)));
  assert.equal(workers.length, 2);
  assert.equal(pool.queue.length, 2);
  assert.equal(pool.metrics.maxParallelRuns, 2);
  const reply = (worker, index, provider = "wasm") => {
    const message = worker.messages[index];
    worker.onmessage({
      data: {
        type: "result",
        id: message.id,
        model: "z0",
        provider: { name: provider, label: provider, detail: "test" },
        values: message.inputs.noise.data.buffer,
        dims: [1, 1, 1, 1],
        inferenceMs: 3,
      },
    });
  };
  reply(workers[1], 0);
  reply(workers[0], 0);
  reply(workers[0], 1);
  reply(workers[1], 1);
  assert.deepEqual(
    (await Promise.all(jobs)).map((result) => result.elevation_normalized.data[0]),
    [1, 2, 3, 4],
  );
  assert.equal(pool.metrics.modelRuns, 4);
  assert.equal(pool.metrics.inferenceMs, 12);
  assert.equal(pool.queue.length, 0);
  assert(pool.slots.every((slot) => !slot.job));

  const fatal = [1, 2, 3].map((value) => pool.run(model, input(value)));
  const settled = Promise.allSettled(fatal);
  workers[0].onerror({ message: "worker crashed" });
  assert((await settled).every((result) => result.status === "rejected"));
  assert(workers.every((worker) => worker.terminated));
  await assert.rejects(pool.run(model, input(4)), /worker crashed/);

  const startup = new api.InferencePool(
    new URLSearchParams(),
    () => {},
    () => {
      throw new Error("startup failed");
    },
  );
  await assert.rejects(startup.run(model, input(1)), /startup failed/);
  assert.equal(startup.queue.length, 0);

  const mismatchedWorkers = [];
  const mismatched = new api.InferencePool(
    new URLSearchParams(),
    () => {},
    () => {
      const worker = {
        messages: [],
        postMessage(data) {
          this.messages.push(data);
        },
        terminate() {},
      };
      mismatchedWorkers.push(worker);
      return worker;
    },
  );
  const mismatchJobs = [mismatched.run(model, input(1)), mismatched.run(model, input(2))];
  const mismatchResults = Promise.allSettled(mismatchJobs);
  reply(mismatchedWorkers[0], 0);
  reply(mismatchedWorkers[1], 0, "webgpu");
  const results = await mismatchResults;
  assert.equal(results[0].status, "fulfilled");
  assert.equal(results[1].status, "rejected");
  assert.match(results[1].reason.message, /different providers/);

  const childReplies = [];
  const childOrt = {
    env: { wasm: { numThreads: 4 } },
    Tensor: class {
      constructor(_type, data, dims) {
        this.data = data;
        this.dims = dims;
      }
    },
    InferenceSession: {
      async create() {
        return {
          async run(inputs) {
            assert.deepEqual(Array.from(inputs.noise.dims), [1, 1, 80, 80]);
            return {
              elevation_normalized: {
                data: new Float32Array(64 * 64).fill(9),
                dims: [1, 1, 64, 64],
              },
            };
          },
        };
      },
    },
  };
  const child = loadModules({ ort: childOrt, postMessage: (data) => childReplies.push(data) });
  child.load("inference-worker.js");
  await child.context.onmessage({
    data: {
      id: 42,
      zoom: 0,
      inputs: { noise: { data: new Float32Array(80 * 80), dims: [1, 1, 80, 80] } },
    },
  });
  assert.equal(childOrt.env.wasm.numThreads, 1, "each child owns a single-thread runtime");
  assert.equal(childReplies[0].id, 42);
  assert.equal(childReplies[0].type, "result");
  assert.equal(new Float32Array(childReplies[0].values)[0], 9);
  console.log(
    "Block checks passed: canonical crops, shared claims, cancellation, failure/retry, clear ownership, worker routing",
  );
};
