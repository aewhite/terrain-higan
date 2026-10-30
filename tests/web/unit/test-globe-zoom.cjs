const { loadModules, demoDir, loadThree } = require("../support/test-support.cjs");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const { performance } = require("node:perf_hooks");

(async () => {
  const THREE = await loadThree();
  const { api, context, load } = loadModules({ THREE, innerWidth: 1280, innerHeight: 900 });
  load("globe-topology.js");
  load("globe-mesh.js");
  load("globe-tile-controller.js");
  vm.runInContext(
    fs
      .readFileSync(path.join(demoDir, "globe-renderer.js"), "utf8")
      .replace('import * as THREE from "three";', ""),
    context,
  );
  const globe = api.globe;
  for (const side of [1, 64, globe.sideAt(20)])
    for (let face = 0; face < globe.FACE_COUNT; face++)
      for (const east of [-0.5, side - 0.5])
        for (const north of [-0.5, side - 0.5]) {
          const actual = vm.runInContext(
            `TerrainHiGAN.globe.canonicalPixel(${face}, ${east}, ${north}, ${side})`,
            context,
            { timeout: 1000 },
          );
          assert.deepEqual(
            [actual.face, actual.east, actual.north],
            [face, east, north],
            "exact cube corners remain on their face without repeated wrapping",
          );
        }
  for (let face = 0; face < globe.FACE_COUNT; face++)
    for (let edge = 0; edge < 4; edge++) {
      const east = edge === 0 ? -0.75 : edge === 1 ? 63.75 : 31.5;
      const north = edge === 2 ? -0.75 : edge === 3 ? 63.75 : 31.5;
      const expected = globe.acrossEdge(face, east, north, edge, 64);
      const actual = vm.runInContext(
        `TerrainHiGAN.globe.canonicalPixel(${face}, ${east}, ${north}, 64)`,
        context,
        { timeout: 1000 },
      );
      assert.deepEqual(actual, expected, "fractional edge samples wrap exactly once");
    }
  const renderer = Object.create(globe.GlobeRenderer.prototype);
  Object.assign(renderer, {
    canvas: { width: 2560, height: 1800 },
    camera: new THREE.PerspectiveCamera(42, 1280 / 900, 0.001, 12),
    sphere: new THREE.Sphere(new THREE.Vector3(), 1),
    frustum: new THREE.Frustum(),
    raycaster: new THREE.Raycaster(),
    scene: new THREE.Scene(),
    tiles: new Set(),
    terrainTiles: [],
    map: "realistic",
    exaggeration: 1,
    onChange() {},
  });
  renderer.setView(20, -25, globe.RADIUS * 2);
  const controller = new globe.GlobeTileController(
    renderer,
    () => {},
    () => {},
  );
  for (let face = 0; face < globe.FACE_COUNT; face++) {
    const root = controller.node(face, 0, 0, 0);
    controller.tiles.set(root.key, {
      ...root,
      level: 2,
      values: new Float32Array(68 * 68).fill(-1),
      vertexValues: new Float32Array(67 * 67).fill(-1),
    });
  }
  controller.tilesVersion++;
  context.zoomRenderer = renderer;
  context.zoomController = controller;
  const timings = [];
  let maxSliceMs = 0;
  for (let iteration = 0; iteration < 3; iteration++) {
    const started = performance.now();
    vm.runInContext(
      "zoomRenderer.zoom(Math.exp(-0.5), 640, 450); zoomController.update()",
      context,
      { timeout: 5000 },
    );
    while (renderer.meshPending) {
      const slice = performance.now();
      vm.runInContext("zoomController.update()", context, { timeout: 5000 });
      maxSliceMs = Math.max(maxSliceMs, performance.now() - slice);
    }
    timings.push(Math.round(performance.now() - started));
    assert(renderer.patchTiles.size > 0);
    for (const patch of renderer.patchTiles.values()) {
      assert(patch.mesh.geometry.getAttribute("position").array.every(Number.isFinite));
      assert(patch.mesh.geometry.getAttribute("normal").array.every(Number.isFinite));
    }
  }
  console.log("Fractional cube-edge wrapping and rapid Retina zoom regression passed", {
    settleMs: timings,
    maxSliceMs: Math.round(maxSliceMs),
  });
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
