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
  const renderer = Object.create(globe.GlobeRenderer.prototype);
  Object.assign(renderer, {
    canvas: { width: 1280, height: 900 },
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
    sun: new THREE.DirectionalLight(),
    renderer: { render() {} },
    animationFrame: 1,
  });
  const sample = renderer.sample;
  const show = renderer.show;
  renderer.sample = () => 0;
  renderer.show = () => {};
  const controller = new globe.GlobeTileController(
    renderer,
    () => {},
    () => {},
  );
  const timings = {};
  const time = (name, callback, count = 1) => {
    const start = performance.now();
    for (let iteration = 0; iteration < count; iteration++) callback(iteration);
    timings[name] = Number(((performance.now() - start) / count).toFixed(3));
  };
  const assertCoverage = () => {
    const leaves = controller.sequence.filter(
      (node) => !controller.selection.expanded.has(node.key),
    );
    for (let row = 0; row < 18; row++)
      for (let column = 0; column < 24; column++) {
        const point = renderer.pick(((column + 0.5) * 1280) / 24, ((row + 0.5) * 900) / 18);
        if (!point) continue;
        const coordinate = globe.vectorFace(point);
        assert(
          leaves.some((node) => {
            const east = coordinate.east * 2 ** node.depth - node.tileX;
            const north = coordinate.north * 2 ** node.depth - node.tileY;
            return (
              node.face === coordinate.face && east >= 0 && east <= 1 && north >= 0 && north <= 1
            );
          }),
          "visible globe rays are covered without offscreen siblings",
        );
      }
  };
  renderer.setView(20, -25, globe.RADIUS * 2);
  time("overviewColdMs", () => controller.reset("1"));
  assertCoverage();
  assert(
    controller.sequence.every((node) => renderer.visible(node.bounds)),
    "only visible tiles are queued",
  );
  const selection = controller.selection;
  controller.update();
  assert.equal(controller.selection, selection, "unchanged viewport reuses selection");
  assert.equal(controller.active.children?.length > 0, true);
  let message;
  const originalSend = controller.send;
  controller.send = (data) => {
    message = data;
  };
  controller.active = null;
  controller.dispatch();
  assert.equal(message.children, undefined, "worker requests never clone the quadtree");
  controller.send = originalSend;
  time(
    "overviewMovedMs",
    (index) => {
      renderer.setView(20, -25 + (index + 1) * 0.001, globe.RADIUS * 2);
      controller.update();
    },
    30,
  );
  time("overviewUnchangedMs", () => controller.update(), 30);
  renderer.zoom(1e-12);
  time("z20ColdMs", () => controller.update());
  time(
    "z20MovedMs",
    (index) => {
      renderer.drag(640, 450, 640 + (index % 2 ? 1 : -1), 450);
      controller.update();
    },
    30,
  );
  assert.equal(Math.max(...controller.sequence.map((node) => node.depth)), 18);
  const selectedDirection = renderer.camera.position.clone().normalize().toArray();
  assertCoverage();
  assert(controller.sequence.every((node) => renderer.visible(node.bounds)));
  for (const node of controller.sequence) {
    let pixels = 0;
    const samples = renderer.projectionSamples(node.bounds);
    for (let offset = 0; offset < samples.length; offset += 9) {
      const center = new THREE.Vector3()
        .fromArray(samples, offset)
        .multiplyScalar(renderer.sphere.radius);
      if (
        center.clone().applyMatrix4(renderer.camera.matrixWorldInverse).z >= -renderer.camera.near
      )
        continue;
      center.project(renderer.camera);
      for (const neighbor of [offset + 3, offset + 6]) {
        const point = new THREE.Vector3()
          .fromArray(samples, neighbor)
          .multiplyScalar(renderer.sphere.radius)
          .project(renderer.camera);
        pixels = Math.max(
          pixels,
          Math.hypot(
            ((point.x - center.x) * renderer.canvas.width) / 2,
            ((point.y - center.y) * renderer.canvas.height) / 2,
          ),
        );
      }
    }
    const expected = pixels ? pixels * 64 : Infinity;
    const actual = renderer.span(node.bounds);
    assert(
      expected === actual || Math.abs(actual / expected - 1) < 1e-5,
      "optimized projection agrees with Three.js",
    );
  }
  renderer.sample = sample;
  renderer.show = show;
  const fixture = (depth, tileX, tileY) => ({
    face: 5,
    depth,
    tileX,
    tileY,
    level: depth + 2,
    values: new Float32Array(68 * 68).fill(0.2),
    vertexValues: new Float32Array(67 * 67).fill(0.2),
  });
  let parent;
  time("createRootMs", () => {
    parent = renderer.createTile(fixture(0, 0, 0));
  });
  time(
    "createChildMs",
    () => {
      const tile = renderer.createTile(fixture(1, 0, 0));
      assert(tile.mesh.geometry.getAttribute("position").count === 353);
      renderer.dispose(tile);
    },
    20,
  );
  const child = renderer.createTile(fixture(1, 0, 0));
  const positions = child.mesh.geometry.getAttribute("position");
  const normals = child.mesh.geometry.getAttribute("normal");
  for (const [column, row] of [
    [0, 0],
    [32, 32],
    [64, 64],
    [12, 51],
  ]) {
    const nativeColumn = Math.floor(column / 4) * 4,
      nativeRow = Math.floor(row / 4) * 4;
    const index = (nativeRow / 4) * 17 + nativeColumn / 4;
    const point = renderer.point(child, nativeColumn, nativeRow);
    const expected = point.clone().sub(child.mesh.position);
    assert(new THREE.Vector3().fromBufferAttribute(positions, index).distanceTo(expected) < 1e-7);
    const normal = point.clone().normalize();
    assert(new THREE.Vector3().fromBufferAttribute(normals, index).distanceTo(normal) < 1e-7);
  }
  const childCenter = renderer.point(child, 32, 32).normalize();
  renderer.setView(
    (Math.asin(childCenter.y) * 180) / Math.PI,
    (Math.atan2(-childCenter.z, childCenter.x) * 180) / Math.PI,
    globe.RADIUS * 2,
  );
  child.mesh.visible = true;
  renderer.scene.updateMatrixWorld();
  assert(renderer.surfacePick(640, 450), "generated geometry can be picked");
  renderer.dispose(child);
  renderer.dispose(parent);
  time(
    "createNativeChildMs",
    () => {
      const tile = renderer.createTile(fixture(18, 200000, 200000));
      assert.equal(tile.mesh.userData.segments, 64);
      assert.equal(tile.mesh.geometry.getAttribute("position").count, 4481);
      assert.equal(tile.mesh.geometry.index.count, 26112);
      assert.equal(tile.mesh.material.map.image.width, 68);
      renderer.dispose(tile);
    },
    10,
  );
  const medium = renderer.createTile(fixture(9, 1, 1));
  assert.equal(medium.mesh.userData.segments, 32);
  assert.equal(medium.mesh.geometry.getAttribute("position").count, 1217);
  assert.equal(medium.mesh.material.map.image.width, 68);
  renderer.dispose(medium);
  const tiles = [...controller.selection.required.values()].map((node) => ({
    ...node,
    ...fixture(node.depth, node.tileX, node.tileY),
    face: node.face,
  }));
  renderer.prepareView(tiles);
  renderer.prepareView(tiles, 1234);
  const prepared = renderer.terrainTiles;
  renderer.prepareView(tiles, 1234);
  assert.equal(renderer.terrainTiles, prepared, "camera updates reuse the height index");
  renderer.prepareView(tiles, 1235);
  assert.notEqual(renderer.terrainTiles, prepared, "tile changes invalidate the height index");
  const direction = selectedDirection;
  assert.notEqual(
    renderer.sample(direction, tiles),
    null,
    "sampling benchmark hits selected terrain",
  );
  time(
    "sample1000Ms",
    () => {
      for (let count = 0; count < 1000; count++) renderer.sample(direction, tiles);
    },
    10,
  );
  console.log(JSON.stringify({ desiredTiles: tiles.length, timings }, null, 2));
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
