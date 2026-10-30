const { loadModules, demoDir, demoRequire, near } = require("../support/test-support.cjs");
const assert = require("node:assert/strict");
const path = require("node:path");
const ort = demoRequire("onnxruntime-node");
(async () => {
  const { api } = loadModules();
  const shapes = [];
  const wrapper = {
    env: { wasm: { numThreads: 1 } },
    Tensor: ort.Tensor,
    InferenceSession: {
      async create(file, settings) {
        assert.equal(settings.enableCpuMemArena, false);
        assert.equal(settings.enableMemPattern, false);
        const session = await ort.InferenceSession.create(path.join(demoDir, file), {
          intraOpNumThreads: 1,
          interOpNumThreads: 1,
          enableCpuMemArena: false,
          enableMemPattern: false,
        });
        return {
          run(inputs) {
            shapes.push(Object.values(inputs).map((t) => Array.from(t.dims)));
            return session.run(inputs);
          },
          release: () => session.release(),
        };
      },
    },
  };
  const runtime = new api.ModelRuntime(wrapper, new URLSearchParams("mobile=1"), () => {});
  for (const zoom of [0, 1, 10, 20]) {
    const height = zoom === 0 ? 80 : 44,
      width = zoom === 0 ? 96 : 48;
    const values = await api.normalField(7n, zoom, 0, 0, width, height);
    const inputs =
      zoom === 0
        ? { noise: new ort.Tensor("float32", values, [1, 1, height, width]) }
        : {
            input_map: new ort.Tensor("float32", values, [1, 1, height, width]),
            noise: new ort.Tensor(
              "float32",
              await api.upscalerNoise(7n, zoom, 0, 0, width, height),
              [1, 4, height, width],
            ),
          };
    const reference = await ort.InferenceSession.create(
      path.join(demoDir, "assets/models/" + api.modelName(zoom) + ".onnx"),
      { intraOpNumThreads: 1, interOpNumThreads: 1 },
    );
    const expected = await reference.run(inputs);
    const before = shapes.length;
    const output = await runtime.run(await runtime.sessionFor(zoom), inputs);
    assert.deepEqual(
      Array.from(output.elevation_normalized.dims),
      Array.from(expected.elevation_normalized.dims),
    );
    near(output.elevation_normalized.data, expected.elevation_normalized.data, 2e-5);
    const maximum = zoom === 0 ? 48 : 24;
    assert(
      shapes
        .slice(before)
        .flat()
        .every((d) => d[2] <= maximum && d[3] <= maximum),
      "all inference windows fit the mobile memory bound",
    );
    await reference.release();
  }
  for (const pending of runtime.sessions.values()) await (await pending).session.release();
  const released = [];
  const fake = {
    env: { wasm: { numThreads: 1 } },
    InferenceSession: {
      async create(file) {
        return { release: async () => released.push(file) };
      },
    },
  };
  const bounded = new api.ModelRuntime(fake, new URLSearchParams("mobile=1"), () => {});
  for (let level = 0; level < 8; level++) await bounded.sessionFor(level);
  assert.equal(bounded.sessions.size, 4);
  assert.equal(
    released.length,
    4,
    "mobile releases old sessions instead of retaining the entire chain",
  );
  assert(released.every((file) => !file.endsWith("z7.onnx")));
  console.log(
    "Mobile inference: bounded 32px output patches match full z0/z1/z10/z20 runs, including edges; disabled memory arenas and released LRU sessions",
  );
})().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});
