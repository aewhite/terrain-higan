const { loadModules, near, demoDir, demoRequire } = require("../support/test-support.cjs");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { createHash } = require("node:crypto");
const ort = demoRequire("onnxruntime-node");

(async () => {
  const messages = [];
  const { api, context } = loadModules();
  const options = new URLSearchParams("backend=wasm");
  const nodeOrt = {
    env: { wasm: { numThreads: 1 } },
    Tensor: ort.Tensor,
    InferenceSession: { create: (file) => ort.InferenceSession.create(path.join(demoDir, file)) },
  };
  const runtime = new api.ModelRuntime(nodeOrt, options, (message) => messages.push(message));
  const generator = new api.TerrainGenerator(runtime);
  const request = { seed: 4829347n, assertCurrent() {} };
  const generate = (zoom, x, y, width, height) => {
    const region = { zoom, x, y, width, height };
    api.validateRegion(region);
    return generator.generate(region, request);
  };

  const base = await generate(0, -64, -64, 128, 128);
  const before = runtime.metrics.modelRuns;
  await generate(1, -32, -32, 64, 64);
  assert.equal(
    runtime.metrics.modelRuns - before,
    4,
    "four canonical z1 blocks reuse the cached z0 parents",
  );
  const repeat = runtime.metrics.modelRuns;
  await generate(1, -32, -32, 64, 64);
  assert.equal(runtime.metrics.modelRuns, repeat, "cached blocks run no models");
  const beforeOverlap = runtime.metrics.modelRuns;
  const overlap = await generate(0, 48, -24, 32, 48);
  assert.equal(runtime.metrics.modelRuns - beforeOverlap, 2);
  const independent = await generator.generateUncached(
    { zoom: 0, x: 64, y: 0, width: 64, height: 64 },
    request,
  );
  assert.deepEqual(
    Array.from(
      generator.cache.get(
        generator.cache.key(request.seed, { zoom: 0, x: 64, y: 0, width: 64, height: 64 }),
      ).values,
    ),
    Array.from(independent.values),
  );
  const cropped = await generate(0, -16, -16, 32, 32);
  for (let row = 0; row < 32; row++)
    assert.deepEqual(
      cropped.values.subarray(row * 32, row * 32 + 32),
      base.values.subarray((row + 48) * 128 + 48, (row + 48) * 128 + 80),
    );
  const deep = await generate(20, -7, -3, 16, 16);
  assert([...deep.values].every(Number.isFinite));
  generator.cache.clear();
  const freshDeep = await generate(20, -7, -3, 16, 16);
  assert.deepEqual(
    Array.from(deep.values),
    Array.from(freshDeep.values),
    "z20 is history-invariant",
  );
  assert.notEqual(await runtime.sessionFor(19), await runtime.sessionFor(20));
  assert.equal(deep.model, "z20s");
  console.log("Canonical blocks, cropped cache reuse, and z20 history invariance passed");

  const schema = path.resolve(
    path.dirname(demoRequire.resolve("onnxruntime-web")),
    "../lib/onnxjs/ort-schema/protobuf/onnx.js",
  );
  const { onnx } = require(schema);
  const noise = await api.upscalerNoise(4829347n, 10, 0, 0, 24, 24);
  // Measured residual divisor, elevation bounds, and residual bounds from UPSCALE_STAT_ROWS.
  const residualStats = [
    [
      0.0032727053800782266, -3.4554131031036377, 4.4300537109375, -1.5577173233032227,
      1.1975839138031006,
    ],
    [
      0.005331021131394116, -3.403956651687622, 4.228604316711426, -0.9458143711090088,
      1.1941447257995605,
    ],
    [
      0.0033842522091731206, -3.3716323375701904, 4.100369453430176, -0.5960838794708252,
      0.7548511624336243,
    ],
    [
      0.002113956124704955, -3.3573215007781982, 4.030477046966553, -0.3723407983779907,
      0.471513956785202,
    ],
    [
      0.0013204719862208551, -3.3483824729919434, 3.9898626804351807, -0.2325807809829712,
      0.2945288121700287,
    ],
    [
      0.0008248264235519859, -3.3427984714508057, 3.9644927978515625, -0.14528048038482666,
      0.18397599458694458,
    ],
    [
      0.000515223132090294, -3.339310646057129, 3.948645830154419, -0.09074878692626953,
      0.11491960287094116,
    ],
    [
      0.00032183232183787425, -3.337131977081299, 3.9387471675872803, -0.056685805320739746,
      0.07178404927253723,
    ],
    [
      0.00020103041098602304, -3.335771083831787, 3.9325637817382812, -0.0354083776473999,
      0.04483950138092041,
    ],
    [
      0.00012557297873446306, -3.33492112159729, 3.928701639175415, -0.02211761474609375,
      0.02800881862640381,
    ],
    [
      7.843843945034266e-5, -3.3343899250030518, 3.9262890815734863, -0.013815760612487793,
      0.01749551296234131,
    ],
  ];
  for (const [index, stats] of residualStats.entries()) {
    const name = index === 0 ? "z10" : `z${index + 10}s`;
    const [scale, minimum, maximum, residualMinimum, residualMaximum] = stats;
    const model = onnx.ModelProto.decode(
      fs.readFileSync(path.join(demoDir, `assets/models/${name}.onnx`)),
    );
    assert.deepEqual(
      model.graph.input.map((input) => input.name),
      ["input_map", "noise"],
    );
    const resizeModes = model.graph.node
      .filter((node) => node.opType === "Resize")
      .map((node) =>
        Buffer.from(node.attribute.find((attribute) => attribute.name === "mode").s).toString(),
      )
      .sort();
    assert.deepEqual(resizeModes, ["nearest"]);
    const sinh = model.graph.node.find((node) => node.opType === "Sinh");
    const multiply = model.graph.node.find(
      (node) => node.opType === "Mul" && node.input.includes(sinh.output[0]),
    );
    const scaleTensor = model.graph.initializer.find(
      (tensor) => tensor.name === multiply.input.find((input) => input !== sinh.output[0]),
    );
    assert.equal(
      Buffer.from(scaleTensor.rawData).readFloatLE(),
      Math.fround(scale),
      `${name} embeds its measured residual scale`,
    );
    model.graph.output.push(
      onnx.ValueInfoProto.create({ name: sinh.output[0], type: model.graph.output[0].type }),
    );
    const session = await ort.InferenceSession.create(onnx.ModelProto.encode(model).finish());
    const parents = [0, -3.5, 4.42].map((elevation) => new Float32Array(24 * 24).fill(elevation));
    parents.push(Float32Array.from({ length: 24 * 24 }, (_, pixel) => Math.sin(pixel * 0.7)));
    for (const parent of parents) {
      const inputs = {
        input_map: new ort.Tensor("float32", parent, [1, 1, 24, 24]),
        noise: new ort.Tensor("float32", noise, [1, 4, 24, 24]),
      };
      const output = await session.run(inputs);
      const outputWidth = output.elevation_normalized.dims[3];
      const expected = Float32Array.from(output[sinh.output[0]].data, (value, pixel) => {
        const row = Math.floor(pixel / outputWidth) + 8;
        const column = (pixel % outputWidth) + 8;
        const parentRow = Math.floor(row / 2);
        const parentColumn = Math.floor(column / 2);
        const referenceElevation = parent[parentRow * 24 + parentColumn];
        const gx = Math.fround(
          Math.fround(
            parent[parentRow * 24 + parentColumn + 1] - parent[parentRow * 24 + parentColumn - 1],
          ) * 0.5,
        );
        const gy = Math.fround(
          Math.fround(
            parent[(parentRow + 1) * 24 + parentColumn] -
              parent[(parentRow - 1) * 24 + parentColumn],
          ) * 0.5,
        );
        const horizontal = Math.fround(referenceElevation + (column % 2 ? 1 : -1) * gx * 0.25);
        const elevation = Math.fround(horizontal + (row % 2 ? 1 : -1) * gy * 0.25);
        const interpolationDelta = Math.fround(elevation - referenceElevation);
        const totalResidual = Math.fround(
          Math.fround(value * Math.fround(scale)) + interpolationDelta,
        );
        const residual = Math.max(residualMinimum, Math.min(residualMaximum, totalResidual));
        return Math.max(minimum, Math.min(maximum, Math.fround(referenceElevation + residual)));
      });
      near(output.elevation_normalized.data, expected, 1e-6);
    }
    await session.release();
  }
  console.log("z10-z20 residual formula passed");

  const bundle = "1234567890abcdef";
  const bytes = Uint8Array.from([1, 2, 3, 4]);
  const hash = createHash("sha256").update(bytes).digest("hex");
  options.set("bundle", bundle);
  context.fetch = async (url) =>
    url.endsWith("manifest.json")
      ? { ok: true, json: async () => ({ bundle, models: { z20s: hash } }) }
      : { ok: true, arrayBuffer: async () => bytes.buffer };
  assert.deepEqual(Array.from(await runtime.verifiedModelBytes(20)), Array.from(bytes));
  context.fetch = async (url) =>
    url.endsWith("manifest.json")
      ? { ok: true, json: async () => ({ bundle, models: { z20s: hash } }) }
      : { ok: true, arrayBuffer: async () => Uint8Array.from([1, 2, 3, 5]).buffer };
  await assert.rejects(runtime.verifiedModelBytes(20), /does not match model bundle/);
  assert.equal(messages[messages.length - 1].type, "bundle-changed");

  const providers = [];
  nodeOrt.InferenceSession.create = async (_model, settings) => {
    providers.push(settings.executionProviders[0]);
    return {};
  };
  context.navigator.gpu = {};
  options.delete("bundle");
  options.set("backend", "webgpu");
  runtime.sessions.clear();
  await runtime.sessionFor(15);
  const gpu = await runtime.sessionFor(16);
  assert.deepEqual(providers, ["webgpu", "webgpu"]);
  assert.equal(gpu.provider.name, "webgpu");
  options.set("backend", "wasm");
  runtime.sessions.clear();
  await runtime.sessionFor(16);
  assert.equal(providers[2], "wasm");
  console.log("Bundle integrity and actual provider routing passed");
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
