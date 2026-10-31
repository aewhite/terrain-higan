const assert = require("node:assert/strict");
const { loadModules } = require("../support/test-support.cjs");

(async () => {
  let frameId = 0,
    now = 0;
  const frames = new Map(),
    events = {},
    messages = [];
  const { api, context, load } = loadModules({
    navigator: { userAgent: "iPhone" },
    performance: { now: () => now },
    requestAnimationFrame(callback) {
      frames.set(++frameId, callback);
      return frameId;
    },
  });
  load("refinement.js");
  const pacing = new api.RefinementScheduler({
    dataset: {},
    addEventListener(type, callback) {
      events[type] = callback;
    },
  });
  const frame = (time) => {
    now = time;
    const pending = [...frames.values()];
    frames.clear();
    for (const callback of pending) callback(time);
    assert(frames.size <= 1, "only one frame monitor is scheduled");
  };
  pacing.setWorker({
    postMessage(message) {
      messages.push(message);
    },
  });
  pacing.setBusy(true);
  frame(0);
  frame(16);
  for (let time = 50; time <= 500; time += 50) frame(time);
  assert(pacing.delayMs > 0, "slow drawing reduces inference throughput");
  const limit = pacing.delayMs;
  for (let time = 550; time <= 1500; time += 50) frame(time);
  assert.equal(pacing.delayMs, limit, "sustained slow frames cannot starve inference");
  assert(pacing.meshBudgetMs < 4, "slow frames reduce main-thread mesh work");
  for (let time = 1516; time < 3000; time += 16) frame(time);
  assert.equal(pacing.delayMs, 0, "smooth frames restore throughput");
  const sent = messages.length;
  events.pointermove({ type: "pointermove", buttons: 0, pointerType: "mouse" });
  assert.equal(messages.length, sent, "hover does not throttle refinement");
  events.pointerdown({ type: "pointerdown" });
  assert(
    pacing.delayMs > 0 && pacing.delayMs < 100,
    "interaction paces rather than pauses inference",
  );
  assert(pacing.meshBudgetMs < 4);
  for (let index = 0; index < 30; index++) events.pointermove({ type: "pointermove", buttons: 1 });
  assert(messages.length <= sent + 1, "gesture feedback is coalesced");
  pacing.setBusy(false);
  frame(now + 1000);
  assert.equal(frames.size, 0, "an idle viewer stops monitoring frames");
  assert.equal(pacing.delayMs, 0);
  pacing.setBusy(true);
  for (let index = 0; index < 40; index++) frame(5000 + (index * 1000) / 30);
  assert.equal(pacing.delayMs, 0, "a steady 30 Hz display does not look like missed frames");
  pacing.setBusy(false);
  frame(7000);
  pacing.setWorker({
    postMessage(message) {
      assert.equal(message.delayMs, 0);
    },
  });

  const desktop = loadModules({
    performance: { now: () => 0 },
    requestAnimationFrame: () => 1,
  });
  desktop.load("refinement.js");
  const desktopEvents = {};
  const desktopPacing = new desktop.api.RefinementScheduler({
    dataset: {},
    addEventListener(type, callback) {
      desktopEvents[type] = callback;
    },
  });
  desktopPacing.setBusy(true);
  desktopEvents.pointerdown({ type: "pointerdown" });
  desktopPacing.observe(0);
  desktopPacing.observe(16);
  assert.equal(desktopPacing.delayMs, 0, "healthy desktop gestures have no fixed inference gap");
  desktopPacing.observe(60);
  assert(desktopPacing.delayMs > 0, "desktop frame pressure still throttles inference");
  assert.equal(desktopPacing.meshBudgetMs, 1.5);

  // An active gesture never blocks the queue. Timers also give cancel messages
  // a chance to invalidate queued work before the next ONNX call.
  const ran = [];
  api.ModelRuntime = class {
    constructor() {
      this.metrics = {};
    }
    sessionFor(zoom) {
      return { zoom, provider: { name: "webgpu" } };
    }
    tensor(data, dims) {
      return { data, dims };
    }
    async run(model) {
      ran.push(model.zoom);
      return {};
    }
  };
  let release, reached;
  const yielded = new Promise((resolve) => {
    reached = resolve;
  });
  context.setTimeout = (callback, delay) => {
    assert(delay > 0 && delay <= limit, "inference delay is finite and bounded");
    release = callback;
    reached();
  };
  const runtime = new api.InferenceRuntime(new URLSearchParams(), () => {});
  runtime.setPacing(limit);
  const first = runtime.run(await runtime.sessionFor(0), {});
  await yielded;
  assert.deepEqual(ran, [0], "inference runs during the gesture before yielding GPU time");
  let obsolete = false;
  const cancelled = assert.rejects(
    runtime.run(await runtime.sessionFor(1), {}, () => {
      if (obsolete) throw new Error("obsolete");
    }),
    /obsolete/,
  );
  const following = runtime.run(await runtime.sessionFor(2), {});
  runtime.setPacing(0);
  obsolete = true;
  release();
  await Promise.all([first, cancelled, following]);
  assert.deepEqual(ran, [0, 2], "obsolete work is skipped and current work continues");

  // Pacing is a worker-wide control message, independent of request IDs and
  // cancellation generations in both entry points.
  context.ort = { env: { wasm: {} } };
  load("model-worker.js");
  let delay;
  const handler = api.createWorkerHandler(
    {
      setPacing(value) {
        delay = value;
      },
    },
    {},
    () => assert.fail("no terrain reply"),
  );
  await handler({ data: { type: "frame-budget", delayMs: 7 } });
  assert.equal(delay, 7);
  const globeWorker = loadModules();
  globeWorker.load("globe-topology.js");
  globeWorker.load("globe-terrain.js");
  const replies = [];
  globeWorker.context.importScripts = () => {};
  globeWorker.context.postMessage = (data) => replies.push(data);
  globeWorker.api.createInferenceRuntime = () => ({
    run() {},
    metrics: {},
    providers: new Map(),
    setPacing(value) {
      delay = value;
    },
  });
  let finish;
  const tileReady = new Promise((resolve) => {
    finish = resolve;
  });
  globeWorker.api.globe.GlobeGenerator = class {
    constructor() {
      this.cache = { bytes: 0, hits: 0, coalesced: 0 };
    }
    async tile() {
      await tileReady;
      return { values: new Float32Array(1), vertexValues: new Float32Array(1), model: "z0" };
    }
  };
  globeWorker.load("globe-worker.js");
  const tile = globeWorker.context.onmessage({
    data: { type: "tile", id: 1, key: "root", seed: "1", face: 0, depth: 0, tileX: 0, tileY: 0 },
  });
  await globeWorker.context.onmessage({ data: { type: "frame-budget", delayMs: 9 } });
  assert.equal(delay, 9);
  finish();
  await tile;
  assert.equal(replies.at(-1).type, "tile", "pacing does not cancel an active globe request");
  console.log(
    "Adaptive frame budgets, continuous inference, cancellation, worker feedback and idle cleanup passed",
  );
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
