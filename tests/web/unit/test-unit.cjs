const { loadModules, demoRequire } = require("../support/test-support.cjs");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const vm = require("node:vm");
const { bundleManifest } = demoRequire("./sync-models.cjs");
const fixtures = require("../../fixtures/terrain-contract.json");
const plain = (value) => JSON.parse(JSON.stringify(value));

(async () => {
  const { api, context, load } = loadModules();
  const timestamp = 1792354800123;
  const seeded = loadModules({ Date: { now: () => timestamp } }).api;
  for (const value of [undefined, null, "", "   "])
    assert.equal(seeded.parseSeed(value), BigInt(timestamp));
  for (const value of ["0", "19", "4829347", "18446744073709551615", "-1"])
    assert.equal(seeded.parseSeed(value), BigInt.asUintN(64, BigInt(value)));
  assert.equal(seeded.parseSeed(" 19 "), 19n);
  assert.throws(() => seeded.parseSeed("invalid"), { name: "SyntaxError" });
  await require("../support/test-blocks.cjs")();
  for (const { region, parent } of fixtures.geometry)
    assert.deepEqual(plain(api.parentRegion(region)), parent);
  assert.deepEqual(
    Array.from(api.philox4x32(fixtures.philox.counter, fixtures.philox.key)),
    fixtures.philox.output,
  );
  assert.deepEqual(Array.from(api.coordinateWords(-1)), [4294967295, 4294967295]);
  assert.throws(() => api.validateRegion({ zoom: 21, x: 0, y: 0, width: 1, height: 1 }), /0..20/);
  assert.throws(
    () => api.validateRegion({ zoom: 0, x: Infinity, y: 0, width: 1, height: 1 }),
    /safe integer/,
  );
  assert.throws(() => api.validateRegion({ zoom: 0, x: 0, y: 0, width: 0, height: 1 }), /positive/);
  assert.equal(api.metersPerPixel(20), 0.5);
  assert.deepEqual(
    Array.from({ length: 21 }, (_, zoom) => api.modelName(zoom)),
    [
      "z0",
      ...Array.from({ length: 10 }, (_, i) => `z${i + 1}`),
      ...Array.from({ length: 10 }, (_, i) => `z${i + 11}s`),
    ],
  );

  const keys = [];
  for (const seed of [0n, 4829347n, -1n, 18446744073709551615n]) {
    for (const zoom of [0, 10, 20])
      keys.push([seed, zoom, Array.from(await api.noiseKey(seed, zoom))]);
  }
  const fallback = loadModules({ crypto: {} });
  vm.runInContext(fs.readFileSync(demoRequire.resolve("js-sha256"), "utf8"), fallback.context);
  for (const [seed, zoom, expected] of keys)
    assert.deepEqual(Array.from(await fallback.api.noiseKey(seed, zoom)), expected);
  const full = await api.normalField(1234n, 7, -2, 5, 8, 6);
  const part = await api.normalField(1234n, 7, 1, 7, 3, 2);
  for (let y = 0; y < 2; y++)
    assert.deepEqual(part.slice(y * 3, y * 3 + 3), full.slice((y + 2) * 8 + 3, (y + 2) * 8 + 6));
  const shuffled = await api.upscalerNoise(7n, 3, -2, 1, 2, 2);
  const destination = await api.normalField(7n, 3, -4, 2, 4, 4);
  for (let dy = 0; dy < 2; dy++)
    for (let dx = 0; dx < 2; dx++) {
      for (let y = 0; y < 2; y++)
        for (let x = 0; x < 2; x++) {
          assert.equal(
            shuffled[(dy * 2 + dx) * 4 + y * 2 + x],
            destination[(y * 2 + dy) * 4 + x * 2 + dx],
          );
        }
    }

  for (const { meters, rgb } of fixtures.realistic.filter(({ meters }) => meters >= 0)) {
    const output = [0, 0, 0];
    api.writeRealisticColor(meters, output, 0);
    assert.deepEqual(output, rgb);
  }
  const oceanSamples = [-12000, -4000, -1000, -200, -50, -25, -0.001].map((meters) => {
    const output = [0, 0, 0];
    api.writeRealisticColor(meters, output, 0);
    assert(output[2] > output[1] && output[1] > output[0]);
    return output;
  });
  assert.deepEqual(oceanSamples[0], [29, 56, 129]);
  assert.deepEqual(oceanSamples[1], [42, 81, 164]);
  assert.deepEqual(oceanSamples[5], [64, 141, 199]);
  assert.deepEqual(oceanSamples[6], [89, 172, 207]);
  for (let index = 1; index < oceanSamples.length; index++)
    assert(oceanSamples[index].every((value, channel) => value > oceanSamples[index - 1][channel]));
  const normalized = (meters) => (meters + 2342.974207789) / 2439.687062251;
  const relief = api.renderElevationRelief(
    Float32Array.from([-10000, -1, 0, 9000].map(normalized)),
    4,
    1,
    0,
    1,
  );
  assert.deepEqual(Array.from(relief.slice(0, 8)), [0, 25, 115, 255, 173, 224, 255, 255]);
  assert(relief[9] > relief[8] && relief[9] > relief[10]);
  const east = api.computeHillshade(
    Float32Array.from({ length: 64 }, (_, i) => i % 8),
    8,
    8,
    1,
  );
  const west = api.computeHillshade(
    Float32Array.from({ length: 64 }, (_, i) => 7 - (i % 8)),
    8,
    8,
    1,
  );
  assert(east[36] > west[36]);
  for (const sampleSpacingMeters of [0.5, 512, 524288]) {
    const spacedSlope = api.computeHillshade(
      Float32Array.from({ length: 64 }, (_, index) => (index % 8) * sampleSpacingMeters),
      8,
      8,
      sampleSpacingMeters,
    );
    assert.deepEqual(spacedSlope, east, "the same physical slope has the same hillshade");
  }
  for (const map of ["realistic", "elevation"]) {
    const renderSlope = (zoom) => {
      const spacing = api.metersPerPixel(zoom);
      const values = Float32Array.from({ length: 9 }, (_, index) =>
        normalized(
          2000 - ((index % 3) - 1) * 0.25 * spacing - (Math.floor(index / 3) - 1) * 0.125 * spacing,
        ),
      );
      return api.renderRegion({ values, width: 3, height: 3, zoom }, map, 1);
    };
    const reference = renderSlope(19);
    for (const zoom of [9, 10, 15, 20]) {
      const rendered = renderSlope(zoom);
      for (let channel = 0; channel < rendered.length; channel++)
        assert(
          Math.abs(rendered[channel] - reference[channel]) <= 1,
          `${map} shades the same physical slope consistently at z${zoom}`,
        );
    }
  }
  assert.equal(api.contourInterval(9), 1000);
  assert.equal(api.contourInterval(15), 50);
  assert.deepEqual(Array.from(api.contourBaseColor(0)), [239, 232, 211]);
  assert.deepEqual(Array.from(api.contourBaseColor(-1)), [202, 226, 232]);
  assert.deepEqual(
    Array.from(
      api
        .renderContours(
          Float32Array.from([0, 5, 15, 0, 5, 15, 0, 5, 15].map(normalized)),
          3,
          3,
          1,
          19,
        )
        .slice(0, 4),
    ),
    [165, 126, 84, 255],
  );
  assert.deepEqual(plain(api.summarizeElevations(Float32Array.from([0, 1]))), {
    min: -2342.974207789,
    max: 96.712854462,
    sum: -2246.261353327,
    sumSquares: 5498881.51458268,
    count: 2,
  });
  for (const map of ["realistic", "elevation", "contour"]) {
    const padding = api.renderingPadding(map);
    assert.equal(padding, 1);
    const flat = new Float32Array(81).fill(normalized(100));
    const peak = new Float32Array(flat);
    peak[40] = normalized(1600);
    for (const zoom of [0, 9, 13, 17, 20]) {
      const baseline = api.renderRegion({ values: flat, width: 9, height: 9, zoom }, map, padding);
      const rendered = api.renderRegion({ values: peak, width: 9, height: 9, zoom }, map, padding);
      assert.notDeepEqual(rendered, baseline, `${map} preserves an isolated peak at z${zoom}`);
      if (map === "contour") {
        const centerPixel = (3 * 7 + 3) * 4;
        assert.notDeepEqual(
          rendered.slice(centerPixel, centerPixel + 4),
          baseline.slice(centerPixel, centerPixel + 4),
        );
      }
      for (let row = 0; row < 7; row++)
        for (let column = 0; column < 7; column++) {
          if (Math.abs(row - 3) + Math.abs(column - 3) <= 1) continue;
          const pixel = (row * 7 + column) * 4;
          assert.deepEqual(
            rendered.slice(pixel, pixel + 4),
            baseline.slice(pixel, pixel + 4),
            `${map} does not spread an isolated peak into distant pixels at z${zoom}`,
          );
        }
    }
    assert.equal(peak[40], Math.fround(normalized(1600)));
  }
  const cache = new api.RegionCache(8);
  cache.put("a", { values: new Float32Array(2) });
  cache.put("b", { values: new Float32Array(2) });
  assert.equal(cache.get("a"), undefined);
  assert.equal(cache.bytes, 8);
  cache.put("oversized", { values: new Float32Array(3) });
  assert.equal(cache.bytes, 8);
  cache.clear();
  assert.equal(cache.bytes, 0);
  const disabled = new api.RegionCache(0);
  disabled.put("a", { values: new Float32Array(1) });
  assert.equal(disabled.get("a"), undefined);

  const sent = [],
    received = [],
    errors = [];
  const controller = new api.TileController({
    send: (msg) => sent.push(msg),
    onTile: (data) => {
      received.push(data);
      return data;
    },
    onStatus() {},
    onError: (message) => errors.push(message),
  });
  const tile = (key) => ({ key, x: 0, y: 0, width: 2, height: 2 });
  controller.setView([tile("old")], { seed: "1", zoom: 0, map: "realistic" });
  const old = sent[0];
  controller.setView([tile("new")], { seed: "2", zoom: 1, map: "elevation" });
  assert.equal(controller.phase, "cancelling");
  controller.setView([tile("new")], { seed: "2", zoom: 1, map: "elevation" });
  assert.equal(sent.length, 2, "cancel once while awaiting the terminal reply");
  controller.receive({ ...old, type: "error", message: "obsolete failure" });
  assert.equal(errors.length, 0);
  const current = sent[2];
  controller.receive({ ...old, type: "cancelled" });
  assert.equal(controller.active.id, current.id);
  controller.receive({ ...current, type: "tile", provider: { label: "test" } });
  assert.equal(controller.phase, "ready");
  assert.equal(received.length, 1);
  controller.setView([tile("new")], { seed: "2", zoom: 1, map: "elevation" });
  assert.equal(sent.length, 3, "cached revisit skips inference");
  controller.setView([tile("broken")], { seed: "2", zoom: 1, map: "elevation" });
  controller.receive({ ...sent[3], type: "error", message: "failed" });
  assert.deepEqual(errors, ["failed"]);

  // Exercise the real message handler with a controlled async generation boundary.
  context.ort = { env: { wasm: {} }, Tensor: class {}, InferenceSession: {} };
  load("model-worker.js");
  let release;
  const wait = new Promise((resolve) => {
    release = resolve;
  });
  const replies = [];
  const fake = {
    cache: { hits: 0, bytes: 0 },
    async generate(region, request) {
      await wait;
      request.assertCurrent();
      return {
        ...region,
        values: new Float32Array(region.width * region.height),
        model: "z0",
        provider: { label: "test" },
      };
    },
  };
  const handler = api.createWorkerHandler(
    { metrics: { modelRuns: 0, inferenceMs: 0 } },
    fake,
    (message) => replies.push(message),
  );
  const pending = handler({
    data: {
      type: "generate",
      id: 1,
      key: "one",
      seed: "1",
      zoom: 0,
      x: 0,
      y: 0,
      width: 2,
      height: 2,
      map: "realistic",
    },
  });
  await handler({ data: { type: "cancel", id: 2 } });
  await pending;
  assert.equal(replies[replies.length - 1].type, "cancelled");
  assert.equal(replies[replies.length - 1].id, 1);
  const newer = handler({
    data: {
      type: "generate",
      id: 3,
      key: "two",
      seed: "1",
      zoom: 0,
      x: 0,
      y: 0,
      width: 2,
      height: 2,
      map: "realistic",
    },
  });
  await handler({ data: { type: "cancel", id: 2 } });
  release();
  await newer;
  assert.equal(
    replies[replies.length - 1].type,
    "tile",
    "stale cancellation cannot stop a newer request",
  );
  assert.equal(replies[replies.length - 1].id, 3);

  for (const stage of ["session", "noise", "run"]) {
    const isolated = loadModules();
    let entered, finish;
    const started = new Promise((resolve) => {
      entered = resolve;
    });
    const blocked = new Promise((resolve) => {
      finish = resolve;
    });
    const pause = async (name) => {
      if (stage === name) {
        entered();
        await blocked;
      }
    };
    isolated.api.normalField = async () => {
      await pause("noise");
      return new Float32Array(80 * 80);
    };
    isolated.load("terrain.js");
    const runtime = {
      async sessionFor() {
        await pause("session");
        return { name: "z0", provider: { name: "wasm" } };
      },
      tensor: (values) => values,
      async run() {
        await pause("run");
        return { elevation_normalized: { data: new Float32Array(api.BLOCK_SIZE ** 2) } };
      },
    };
    let obsolete = false;
    const generator = new isolated.api.TerrainGenerator(runtime);
    const pending = generator.generate(
      { zoom: 0, x: 0, y: 0, width: 2, height: 2 },
      {
        seed: 1n,
        assertCurrent() {
          if (obsolete) throw new Error("cancelled");
        },
      },
    );
    await started;
    assert.equal(generator.cache.bytes, 0, stage + " cannot expose unfinished blocks");
    obsolete = true;
    finish();
    await assert.rejects(pending, /cancelled/);
    assert.equal(
      generator.cache.bytes,
      api.BLOCK_SIZE ** 2 * 4,
      stage + " cancellation preserves completed shared blocks",
    );
  }

  const { MessageChannel } = require("node:worker_threads");
  const ports = new MessageChannel();
  let retained, retainedBytes;
  const completed = {
    cache: { hits: 0, bytes: 0 },
    async generate(region) {
      retained = new Float32Array(region.width * region.height).fill(1);
      retainedBytes = retained.byteLength;
      return {
        ...region,
        values: retained,
        model: "z0",
        provider: { name: "wasm", label: "test" },
      };
    },
  };
  const transfer = api.createWorkerHandler(
    { metrics: { modelRuns: 0, inferenceMs: 0 } },
    completed,
    (message, buffers) => {
      if (message.type === "tile") ports.port1.postMessage(message, buffers);
    },
  );
  await transfer({
    data: {
      type: "generate",
      id: 1,
      key: "transfer",
      seed: "1",
      zoom: 0,
      x: 0,
      y: 0,
      width: 2,
      height: 2,
      map: "realistic",
    },
  });
  assert.equal(
    retained.byteLength,
    retainedBytes,
    "transferring page buffers must not detach cached elevation",
  );
  assert(retained.every((value) => value === 1));
  ports.port1.close();
  ports.port2.close();

  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "wfc-bundle-"));
  try {
    for (const name of [
      "index.html",
      "model-worker.js",
      "noise.js",
      "styles.css",
      "package-lock.json",
      "prepare-runtime.cjs",
    ])
      fs.writeFileSync(path.join(dir, name), "baseline");
    const before = bundleManifest({}, dir).bundle;
    fs.writeFileSync(path.join(dir, "noise.js"), "changed");
    assert.notEqual(bundleManifest({}, dir).bundle, before);
  } finally {
    fs.rmSync(dir, { recursive: true });
  }
  console.log(
    "Unit checks passed: geometry, noise, palettes, caches, cancellation, request ownership, bundle coverage",
  );
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
