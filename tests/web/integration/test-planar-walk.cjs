const { loadModules, demoDir, loadThree } = require("../support/test-support.cjs");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

(async () => {
  const THREE = await loadThree();
  const { api, context, load } = loadModules({
    THREE,
    requestAnimationFrame: () => 1,
    cancelAnimationFrame() {},
  });
  for (const file of ["globe-topology.js", "globe-mesh.js", "globe-tile-controller.js"]) load(file);
  for (const file of ["globe-renderer.js", "globe-walk.js", "globe-export.js", "planar-walk.js"])
    vm.runInContext(
      "{\n" +
        fs
          .readFileSync(path.join(demoDir, file), "utf8")
          .replace('import * as THREE from "three";', "") +
        "\n}",
      context,
    );
  const { PlanarRenderer, PlanarWalk, GlobeTileController, RADIUS, walkingScene } = api.globe;
  for (const origin of [
    { x: 0, y: 0 },
    { x: -293988345, y: 31759354 },
  ]) {
    const renderer = Object.create(PlanarRenderer.prototype);
    Object.assign(renderer, {
      origin,
      canvas: { width: 320, height: 240 },
      camera: new THREE.PerspectiveCamera(42, 4 / 3, 0.001, 12),
      scene: new THREE.Scene(),
      frustum: new THREE.Frustum(),
      sphere: new THREE.Sphere(),
      tiles: new Set(),
      map: "realistic",
      exaggeration: 1,
      renderer: { setClearColor() {} },
    });
    const controller = new GlobeTileController(
      renderer,
      () => {},
      () => {},
    );
    controller.seed = "4829347";
    const walk = new PlanarWalk(renderer, controller, () => {});
    walk.enter();
    controller.update();
    assert.equal(Math.max(...walk.selection.leaves.map((node) => node.depth)), 14);
    assert(walk.selection.leaves.length < 6000, "z14 refinement keeps the scene bounded");
    assert(walk.selection.leaves.some((node) => node.depth < walk.maxDepth()));
    assert.equal(api.metersPerPixel(walk.maxDepth()), 32);
    assert(
      walk.selection.leaves.some(
        (node) => walk.distance(node.bounds).nearest <= api.globe.WALK_RADIUS && node.depth < 14,
      ),
      "terrain becomes coarser within the walking area away from the entry point",
    );
    for (const node of walk.selection.leaves)
      if (walk.distance(node.bounds).nearest <= api.globe.WALK_FINE_RADIUS)
        assert.equal(node.depth, 14, "z14s covers the area around the entry point");
    for (const node of walk.selection.leaves) {
      const spacing = api.metersPerPixel(node.depth);
      const values = new Float32Array(68 * 68),
        vertices = new Float32Array(67 * 67);
      for (let row = 0; row < 67; row++)
        for (let column = 0; column < 67; column++) {
          const east = (node.tileX * 64 + column - 1) * spacing - origin.x;
          const south = (node.tileY * 64 + row - 1) * spacing - origin.y;
          vertices[row * 67 + column] =
            (100 + east * 0.15 + south * 0.2 + 2342.974207789) / 2439.687062251;
        }
      values.fill(vertices[32 * 67 + 32]);
      controller.tiles.set(node.key, {
        ...node,
        level: node.depth,
        values,
        vertexValues: vertices,
      });
    }
    controller.tilesVersion++;
    let iterations = 0;
    while (!walk.prepared && iterations++ < 10000) controller.update();
    assert(walk.prepared);
    assert(renderer.camera.far * RADIUS > walk.selection.viewRadius + api.globe.WALK_RADIUS);
    const distantPeak = new THREE.Vector3(0, 3000 / RADIUS, -100000 / RADIUS).project(
      renderer.camera,
    );
    assert(
      Math.abs(distantPeak.x) < 1 &&
        Math.abs(distantPeak.y) < 1 &&
        distantPeak.z < 1 &&
        distantPeak.z > -1,
      "distant mountains are not clipped at the detail radius",
    );
    for (const tile of renderer.patchTiles.values()) {
      assert(tile.source.level <= 14, "walking uses no models above z14s");
      assert.equal(tile.waterMesh.geometry.index.count, tile.mesh.userData.segments ** 2 * 6);
    }
    assert(Math.abs(renderer.camera.position.y * RADIUS - 102) < 0.001);
    const ray = new THREE.Raycaster();
    for (const distance of [0, 20, 127, 129, 256, 1000, 19990, 100000, 490000])
      for (let bearing = 0; bearing < 12; bearing++) {
        const east = Math.cos((bearing * Math.PI) / 6) * distance,
          south = Math.sin((bearing * Math.PI) / 6) * distance;
        const position = walk.mapPosition(east, south);
        const height = walk.groundAt(position);
        assert(height !== null, "full 20 km coverage in every direction");
        assert(
          Math.abs(height - (100 + east * 0.15 + south * 0.2)) <
            (distance <= api.globe.WALK_RADIUS ? 0.01 : 0.1),
        );
        const tile = walk.patchAt(position);
        ray.set(new THREE.Vector3(position.x, 1, position.z), new THREE.Vector3(0, -1, 0));
        const hit = ray.intersectObject(tile.mesh)[0];
        if (hit) assert(Math.abs(hit.point.y * RADIUS - height) < 0.01);
      }
    let builds = 0,
      requests = 0;
    renderer.createTile = () => {
      builds++;
      throw Error("Unexpected mesh rebuild");
    };
    controller.send = () => {
      requests++;
    };
    walk.paused = false;
    walk.setKey("KeyW", true);
    walk.requestedSpeed = 1000;
    const previous = renderer.camera.position.clone();
    walk.tick(0.05);
    const traveled = renderer.camera.position.distanceTo(previous) * RADIUS;
    assert(traveled > 2.49 && traveled <= 2.500001);
    assert(
      Math.abs(renderer.camera.position.y * RADIUS - walk.groundAt(renderer.camera.position) - 2) <
        1e-8,
    );
    walk.mouse.x = 0.7;
    for (let iteration = 0; iteration < 20; iteration++) {
      walk.tick(0.05);
      controller.update();
    }
    assert.equal(builds, 0);
    assert.equal(requests, 0);
    assert(walk.heading.x > 0, "mouse right turns east from north");
    const scene = walkingScene(walk);
    assert.equal(scene.userData.topology, "planar");
    assert.equal(scene.userData.originX, origin.x);
    assert.equal(scene.userData.units, "meters");
    const detailedTiles = [...renderer.patchTiles.values()].filter(
      (tile) => walk.distance(tile.bounds).nearest <= api.globe.WALK_RADIUS,
    );
    assert.equal(scene.children.length, detailedTiles.length * 2);
    const maximumTileSpan = Math.max(
      ...detailedTiles.map((tile) => 64 * api.metersPerPixel(tile.depth)),
    );
    const bounds = new THREE.Box3().setFromObject(scene);
    assert(
      Math.max(
        Math.abs(bounds.min.x),
        Math.abs(bounds.max.x),
        Math.abs(bounds.min.z),
        Math.abs(bounds.max.z),
      ) <
        api.globe.WALK_RADIUS + Math.SQRT2 * maximumTileSpan,
      "export stays within the detailed disk's full boundary tiles, not the 500 km scenery",
    );
    scene.traverse((object) => {
      object.geometry?.dispose();
      object.material?.dispose();
    });
    renderer.restyle(controller.tiles.values(), "realistic", 1);
    console.log(
      "Planar walking: metric scale, negative coordinates, mixed LOD, 20 km coverage, ground pin, slope speed cap, frozen scene and export passed",
      { origin, tiles: walk.selection.leaves.length },
    );
  }
  const worker = loadModules({ location: { search: "?surface=planar" } });
  const messages = [];
  let region;
  worker.context.importScripts = () => {};
  worker.context.postMessage = (data) => messages.push(data);
  worker.api.createInferenceRuntime = () => ({
    metrics: {},
    providers: new Map([["test", { name: "wasm" }]]),
    async run() {},
  });
  worker.api.TerrainGenerator = class {
    constructor() {
      this.cache = {};
    }
    async generate(request) {
      region = request;
      return { values: Float32Array.from({ length: 68 * 68 }, (_, index) => index), model: "test" };
    }
  };
  worker.load("globe-topology.js");
  worker.load("globe-worker.js");
  await worker.context.onmessage({
    data: {
      type: "tile",
      id: 1,
      key: "planar",
      seed: "1",
      face: 0,
      depth: 20,
      tileX: -10,
      tileY: 15,
    },
  });
  assert.deepEqual(JSON.parse(JSON.stringify(region)), {
    zoom: 20,
    x: -642,
    y: 958,
    width: 68,
    height: 68,
  });
  assert.equal(messages[0].type, "tile");
  const vertices = new Float32Array(messages[0].vertexValues);
  assert.equal(vertices[0], 34.5);
  assert.equal(vertices[66 * 67 + 66], 4588.5);
  console.log("Planar worker uses the flat map's seed, zoom and canonical sample coordinates");
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
