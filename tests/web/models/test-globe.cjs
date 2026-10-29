const { loadModules, near, demoDir, demoRequire, rootDir } = require("../support/test-support.cjs");
const assert = require("node:assert/strict");
const path = require("node:path");
const fs = require("node:fs");
const ort = demoRequire("onnxruntime-node");
const { bundleManifest } = demoRequire("./sync-models.cjs");

(async () => {
  const { api, load } = loadModules();
  load("globe-topology.js");
  load("globe-terrain.js");
  load("globe-tile-controller.js");
  const globe = api.globe;
  const sent = [],
    states = [];
  let disposed = 0;
  const renderer = {
    camera: { position: {} },
    bounds: (tile) => ({ ...tile, center: { distanceTo: () => 0 } }),
    visible: (bounds) => bounds.face === 0 || bounds.face === 4,
    span: (bounds) => (bounds.depth === 0 ? 1000 : 32),
    prepareView() {},
    show() {},
    createTile: (tile) => tile,
    dispose: () => disposed++,
  };
  const controller = new globe.GlobeTileController(
    renderer,
    (data) => sent.push(data),
    (state) => states.push(state),
  );
  controller.reset("1");
  const desired = controller.sequence.map((node) => node.key);
  const obsolete = { ...controller.active };
  controller.receive({
    ...obsolete,
    id: obsolete.id + 100,
    type: "tile",
    values: new Float32Array(1).buffer,
  });
  assert.equal(controller.tiles.size, 0);
  while (controller.active) {
    const request = controller.active;
    controller.receive({ ...request, type: "tile", values: new Float32Array(1).buffer });
    assert.deepEqual(
      controller.sequence.map((node) => node.key),
      desired.filter((key) => !controller.tiles.has(key)),
      "desired LOD is independent of loading order",
    );
  }
  assert.equal(states[states.length - 1].ready, true);
  assert.equal(controller.display.length, 8);
  for (let index = 0; index < 700; index++) controller.tiles.set(`old:${index}`, {});
  controller.update();
  assert(controller.tiles.size <= controller.capacity);
  assert(disposed > 0);
  controller.reset("2");
  controller.receive({ ...obsolete, type: "tile", values: new Float32Array(1).buffer });
  assert.equal(controller.tiles.size, 0, "old seed replies cannot populate the new world");
  const current = controller.active;
  const stateCount = states.length;
  controller.receive({ ...current, type: "tile", values: new Float32Array(1).buffer }, false);
  assert.equal(states.length, stateCount, "streamed replies defer viewport work to the frame");
  assert.notEqual(
    controller.active.id,
    current.id,
    "inference continues while the frame is queued",
  );
  assert(controller.tiles.has(current.key));
  controller.update();
  controller.fail("test failure");
  assert.equal(states[states.length - 1].error, "test failure");
  for (const side of [16, 64, 16384, globe.sideAt(20)])
    for (let face = 0; face < globe.FACE_COUNT; face++)
      for (const east of [0.5 / side, 0.123, 0.501, 0.873, 1 - 0.5 / side])
        for (const north of [0.5 / side, 0.123, 0.501, 0.873, 1 - 0.5 / side]) {
          const vector = globe.faceVector(face, east, north);
          const roundTrip = globe.vectorFace(vector);
          assert.equal(roundTrip.face, face);
          assert(Math.abs(roundTrip.east - east) < 1e-10);
          assert(Math.abs(roundTrip.north - north) < 1e-10);
        }
  for (let face = 0; face < globe.FACE_COUNT; face++)
    for (let edge = 0; edge < 4; edge++) {
      const neighbor = globe.edges[face][edge];
      assert.equal(globe.edges[neighbor.face][neighbor.edge].face, face);
      for (let along = 0; along < 16; along++) {
        const east = edge === 0 ? -1 : edge === 1 ? 16 : along;
        const north = edge === 2 ? -1 : edge === 3 ? 16 : along;
        const point = globe.canonicalPixel(face, east, north, 16);
        assert.equal(point.face, neighbor.face);
        assert(point.east >= 0 && point.east < 16 && point.north >= 0 && point.north < 16);
      }
    }
  for (let face = 0; face < globe.FACE_COUNT; face++)
    for (const east of [-12, 0, 15, 27])
      for (const north of [-12, 0, 15, 27]) {
        const point = globe.canonicalPixel(face, east, north, 16);
        assert(point.east >= 0 && point.east < 16 && point.north >= 0 && point.north < 16);
      }
  assert(Math.abs(globe.greatCircle([1, 0, 0], [0, 1, 0]) - (Math.PI * globe.RADIUS) / 2) < 1e-6);
  assert.equal(globe.greatCircle([1, 0, 0], [1, 0, 0]), 0);
  for (const end of [
    [-1, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
  ]) {
    const arc = globe.arcPoints([1, 0, 0], end);
    assert(arc.every((point) => point.every(Number.isFinite)));
    near(arc[arc.length - 1], end, 1e-10);
  }
  const nodeOrt = {
    env: { wasm: { numThreads: 1 } },
    Tensor: ort.Tensor,
    InferenceSession: {
      create: (file) =>
        ort.InferenceSession.create(path.join(demoDir, file), {
          intraOpNumThreads: 1,
          interOpNumThreads: 1,
        }),
    },
  };
  const runtime = new api.ModelRuntime(nodeOrt, new URLSearchParams("backend=wasm"), () => {});
  const generator = new globe.GlobeGenerator(runtime);
  for (let face = 0; face < globe.FACE_COUNT; face++) {
    const alias = globe.canonicalPixel(face, -1, 6, 16);
    const original = await generator.noise(19n, 0, face, -1, 6, 1, 1);
    const neighbor = await generator.noise(19n, 0, alias.face, alias.east, alias.north, 1, 1);
    assert.equal(original[0], neighbor[0]);
  }
  const seed = 4829347n;
  const started = performance.now();
  const roots = await Promise.all(
    Array.from({ length: globe.FACE_COUNT }, (_, face) => generator.tile(seed, face, 0, 0, 0)),
  );
  console.log("Generated complete coarse globe", runtime.metrics);
  for (const face of [0, 2, 5]) {
    const reference = await generator.vertices(seed, face, 2, 0, 0);
    assert.deepEqual(
      roots[face].vertexValues,
      reference,
      "reused halo samples preserve canonical vertices exactly",
    );
  }
  assert(roots.every((tile) => tile.values.every(Number.isFinite)));
  const native = await generator.block(seed, generator.address(2, 0, 0, 0));
  for (let row = 0; row < 64; row++)
    for (let column = 0; column < 64; column++)
      assert.equal(
        roots[0].values[(row + 2) * 68 + column + 2],
        native.values[row * 64 + column],
        "display textures retain native model samples without smoothing",
      );
  const snapshot = roots[0].values.slice();
  const before = runtime.metrics.modelRuns;
  await generator.tile(seed, 0, 0, 0, 0);
  assert.equal(runtime.metrics.modelRuns, before, "repeat tile reuses canonical model blocks");
  let maximumEdgeDifference = 0;
  let worstEdge;
  const edgeValue = (face, edge, along) => {
    const column = edge === 0 ? 0 : edge === 1 ? 64 : along;
    const row = edge === 2 ? 0 : edge === 3 ? 64 : along;
    return roots[face].vertexValues[(row + 1) * 67 + column + 1];
  };
  for (let face = 0; face < globe.FACE_COUNT; face++)
    for (let edge = 0; edge < 4; edge++) {
      const neighbor = globe.edges[face][edge];
      for (let along = 0; along <= 64; along++) {
        const difference = Math.abs(
          edgeValue(face, edge, along) -
            edgeValue(neighbor.face, neighbor.edge, neighbor.reverse ? 64 - along : along),
        );
        if (difference > maximumEdgeDifference) {
          maximumEdgeDifference = difference;
          worstEdge = { face, edge, along, neighbor };
        }
      }
    }
  assert(
    maximumEdgeDifference < 2e-5,
    `face boundary mismatch: ${maximumEdgeDifference} ${JSON.stringify(worstEdge)}`,
  );
  for (let face = 0; face < globe.FACE_COUNT; face++)
    for (let firstEdge = 0; firstEdge < 4; firstEdge++) {
      const tileX = firstEdge === 1 ? 1 : 0;
      const tileY = firstEdge === 3 ? 1 : 0;
      const adjacent = globe.canonicalPixel(
        face,
        tileX + (firstEdge === 0 ? -1 : firstEdge === 1 ? 1 : 0),
        tileY + (firstEdge === 2 ? -1 : firstEdge === 3 ? 1 : 0),
        2,
      );
      const edge = globe.edges[face][firstEdge];
      const firstAddress = [face, tileX, tileY];
      const secondAddress = [adjacent.face, adjacent.east, adjacent.north];
      const secondEdge = edge.edge;
      const [first, second] = await Promise.all(
        [firstAddress, secondAddress].map(([face, tileX, tileY]) =>
          generator.tile(seed, face, 1, tileX, tileY),
        ),
      );
      const atEdge = (tile, edge, along) => {
        const column = edge === 0 ? 0 : edge === 1 ? 64 : along;
        const row = edge === 2 ? 0 : edge === 3 ? 64 : along;
        return tile.vertexValues[(row + 1) * 67 + column + 1];
      };
      for (let along = 0; along <= 64; along++)
        maximumEdgeDifference = Math.max(
          maximumEdgeDifference,
          Math.abs(
            atEdge(first, firstEdge, along) -
              atEdge(second, secondEdge, edge.reverse ? 64 - along : along),
          ),
        );
    }
  assert(maximumEdgeDifference < 2e-5, "higher-level polar/equatorial seams must agree");
  const parent = await generator.tile(seed, 0, 0, 0, 0);
  const child = await generator.tile(seed, 0, 1, 0, 0);
  assert(child.values.every(Number.isFinite));
  assert.notDeepEqual(Array.from(child.values), Array.from(parent.values));
  const deep = await generator.tile(seed, 5, 18, 100000, 100000);
  assert.equal(deep.model, "z20s");
  assert.deepEqual(deep.vertexValues, await generator.vertices(seed, 5, 20, 100000, 100000));
  assert(deep.values.every(Number.isFinite));
  generator.cache.clear();
  const rebuilt = await generator.tile(seed, 0, 0, 0, 0);
  near(rebuilt.values, snapshot, 1e-7);
  const other = await generator.tile(seed + 1n, 0, 0, 0, 0);
  assert.notDeepEqual(Array.from(other.values), Array.from(snapshot));
  const manifest = bundleManifest({});
  for (const file of [
    "globe.html",
    "globe-app.js",
    "globe-renderer.js",
    "globe-walk.js",
    "globe-export.js",
    "globe-topology.js",
    "globe-mesh.js",
    "globe-terrain.js",
    "globe-worker.js",
    "globe-tile-controller.js",
    "assets/runtime/three.module.js",
    "assets/runtime/three.core.js",
    "assets/runtime/OrbitControls.js",
    "assets/runtime/GLTFExporter.js",
  ])
    assert(manifest.code[file], `${file} participates in bundle invalidation`);
  const report = {
    topology: globe.GLOBE_VERSION,
    baseNside: globe.BASE_NSIDE,
    edgeDifferenceMeters: maximumEdgeDifference * 2439.687062251,
    modelRuns: runtime.metrics.modelRuns,
    elapsedMs: performance.now() - started,
    cacheBytes: generator.cache.bytes,
    minimumAltitude: globe.MIN_ALTITUDE,
  };
  const output = path.join(rootDir, "runs/globe-viewer");
  fs.mkdirSync(output, { recursive: true });
  fs.writeFileSync(
    path.join(output, "numerical-checks.json"),
    JSON.stringify(report, null, 2) + "\n",
  );
  console.log(
    "Globe topology, noise aliases, native samples, all face edges/corners, real z0–z20 inference, cache history, and bundle coverage passed",
    report,
  );
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
