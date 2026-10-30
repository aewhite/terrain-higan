const { loadModules, demoDir, loadThree } = require("../support/test-support.cjs");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

(async () => {
  const THREE = await loadThree();
  const { api, context, load } = loadModules({ THREE, innerWidth: 320, innerHeight: 240 });
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
  const previewStates = [],
    previewRequests = [];
  const previewRenderer = {
    camera: { position: {} },
    selectionKey: "overview",
    bounds: (tile) => ({ ...tile, center: { distanceTo: () => 0 } }),
    visible: (bounds) => bounds.face < 2,
    span: (bounds) => 256 / 2 ** bounds.depth,
    prepareView() {},
    show() {},
    dispose() {},
  };
  const previewController = new globe.GlobeTileController(
    previewRenderer,
    (request) => previewRequests.push(request),
    (status) => previewStates.push(status),
  );
  const reply = () =>
    previewController.receive({
      ...previewController.active,
      type: "tile",
      level: previewController.active.depth + 2,
      values: new Float32Array(68 * 68).buffer,
      vertexValues: new Float32Array(67 * 67).buffer,
    });
  previewController.reset("1");
  assert.equal(previewStates.at(-1).previewReady, false);
  reply();
  assert.equal(previewStates.at(-1).previewReady, false, "one face leaves a partial preview");
  assert.equal(previewController.active.depth, 0, "visible roots precede fine inference");
  reply();
  assert.equal(previewStates.at(-1).previewReady, true, "coarse roots cover the visible globe");
  assert.equal(previewStates.at(-1).ready, false, "preview completion does not imply full detail");
  assert(previewController.active.depth > 0, "refinement starts automatically");
  assert(previewController.patches.every((patch) => patch.source.depth === 0));
  const rootRequests = previewRequests.filter((request) => request.depth === 0).length;
  while (previewController.active) {
    reply();
    assert.equal(previewStates.at(-1).previewReady, true, "refinement retains full coverage");
  }
  assert.equal(previewStates.at(-1).ready, true);
  assert(previewController.patches.every((patch) => patch.source.depth === patch.depth));
  assert.equal(
    previewRequests.filter((request) => request.depth === 0).length,
    rootRequests,
    "refinement reuses the preview roots",
  );
  previewController.reset("2");
  assert.equal(previewStates.at(-1).previewReady, false, "a new seed resets preview coverage");
  for (const [samplePixels, expectedDepth] of [
    [1.01, 0],
    [1.4, 0],
    [1.5, 0],
    [1.5001, 1],
    [2.5, 1],
    [3.01, 2],
  ]) {
    const selectionRenderer = {
      camera: { position: {} },
      bounds: (tile) => ({ ...tile, center: { distanceTo: () => 0 } }),
      visible: (bounds) => bounds.face === 0,
      span: (bounds) => (64 * samplePixels) / 2 ** bounds.depth,
      prepareView() {},
      show() {},
    };
    const selectionController = new globe.GlobeTileController(
      selectionRenderer,
      () => {},
      () => {},
    );
    selectionController.reset("1");
    const leaves = selectionController.selection.leaves;
    assert.equal(leaves.length, 4 ** expectedDepth);
    assert(
      leaves.every((node) => node.depth === expectedDepth),
      "only refine model samples exceeding 1.5 physical pixels",
    );
    assert(leaves.every((node) => selectionRenderer.span(node.bounds) / 64 <= 1.5));
  }
  const renderer = Object.create(globe.GlobeRenderer.prototype);
  let renders = 0;
  Object.assign(renderer, {
    canvas: { width: 320, height: 240 },
    camera: new THREE.PerspectiveCamera(42, 320 / 240, 0.001, 12),
    sphere: new THREE.Sphere(new THREE.Vector3(), 1),
    frustum: new THREE.Frustum(),
    raycaster: new THREE.Raycaster(),
    scene: new THREE.Scene(),
    tiles: new Set(),
    terrainTiles: [],
    map: "realistic",
    exaggeration: 1,
    onChange() {},
    updateStars() {},
    measurement: new THREE.Group(),
    sun: new THREE.DirectionalLight(),
    renderer: {
      render() {
        renders++;
      },
    },
  });
  renderer.setView(20, -25, globe.RADIUS * 2);
  const requests = [];
  const controller = new globe.GlobeTileController(
    renderer,
    (request) => requests.push(request),
    () => {},
  );
  controller.reset("1");
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
  controller.active = null;
  controller.update();
  const overviewDepth = Math.max(...controller.patches.map((patch) => patch.depth));
  renderer.zoom(1e-12);
  controller.update();
  const deepest = Math.max(...controller.patches.map((patch) => patch.depth));
  assert.equal(deepest, 18, "mesh refinement reaches the camera before any fine-model replies");
  assert(deepest > overviewDepth);
  assert(controller.patches.every((patch) => patch.source.depth === 0));
  assert(
    [...controller.tiles.values()].every((tile) => !tile.mesh),
    "model data does not allocate hidden meshes",
  );
  controller.active = null;
  controller.dispatch();
  assert.equal(
    controller.active.depth,
    18,
    "the nearest final tile takes priority over intermediate levels",
  );
  assert(
    controller.sequence.every(
      (node) => node.depth === 0 || !controller.selection.expanded.has(node.key),
    ),
  );
  assert(renderer.patchTiles.size < 250, "geometry remains camera-local");
  for (let row = 0; row < 7; row++)
    for (let column = 0; column < 9; column++) {
      const horizontal = ((column + 0.5) * 320) / 9,
        vertical = ((row + 0.5) * 240) / 7;
      const sphere = renderer.pick(horizontal, vertical);
      const mesh = renderer.surfacePick(horizontal, vertical);
      assert(mesh, "pending inference leaves no holes in the spherical surface");
      assert(
        globe.greatCircle(sphere, mesh) < 0.05,
        "coarse textures cannot flatten the near-camera globe",
      );
    }
  const snapshots = new Map(renderer.patchTiles);
  renderer.draw();
  const before = renders;
  controller.update();
  renderer.draw();
  assert.equal(renders, before, "unchanged viewport/data does not redraw");
  for (const [key, patch] of renderer.patchTiles) assert.equal(patch, snapshots.get(key));
  const current = controller.active;
  controller.receive(
    {
      ...current,
      type: "tile",
      level: 20,
      values: new Float32Array(68 * 68).fill(-1).buffer,
      vertexValues: new Float32Array(67 * 67).fill(-1).buffer,
    },
    false,
  );
  controller.update();
  const upgraded = renderer.patchTiles.get(current.key);
  assert.equal(upgraded.source.depth, 18);
  assert.equal(upgraded.mesh.userData.segments, 64);
  assert.equal(upgraded.mesh.material.map.image.width, 68);
  for (const [key, patch] of renderer.patchTiles)
    if (key !== current.key)
      assert.equal(patch, snapshots.get(key), "one reply only changes its visible patch");
  const oldTextures = new Set(
    [...controller.tiles.values()].map((tile) => tile.texture).filter(Boolean),
  );
  let disposed = 0;
  for (const texture of oldTextures) texture.addEventListener("dispose", () => disposed++);
  renderer.restyle(controller.tiles.values(), "elevation", 1);
  assert.equal(disposed, oldTextures.size, "shared textures are released exactly once");
  controller.update();
  assert(
    [...renderer.patchTiles.values()].every((patch) => !oldTextures.has(patch.mesh.material.map)),
  );
  for (const [key, tile] of controller.tiles) {
    renderer.dispose(tile);
    if (tile.depth) controller.tiles.delete(key);
    else
      controller.tiles.set(key, {
        ...tile,
        vertexValues: Float32Array.from(
          { length: 67 * 67 },
          (_, index) => 3 + 0.0001 * (Math.floor(index / 67) + (index % 67)),
        ),
        curvature: undefined,
      });
  }
  controller.tilesVersion++;
  renderer.restyle(controller.tiles.values(), "realistic", 10);
  renderer.setView(20, -25, 10);
  controller.update();
  controller.update();
  const center = renderer.pick(160, 120);
  const raisedPatch = [...renderer.patchTiles.values()].find((patch) => {
    const coordinate = globe.vectorFace(center);
    const east = coordinate.east * 2 ** patch.depth - patch.tileX;
    const north = coordinate.north * 2 ** patch.depth - patch.tileY;
    return coordinate.face === patch.face && east >= 0 && east <= 1 && north >= 0 && north <= 1;
  });
  const raisedCoordinate = globe.vectorFace(center);
  const raisedHeight =
    renderer.vertexHeight(
      raisedPatch.source,
      (raisedCoordinate.east * 2 ** raisedPatch.source.depth - raisedPatch.source.tileX) * 64,
      (raisedCoordinate.north * 2 ** raisedPatch.source.depth - raisedPatch.source.tileY) * 64,
    ) * 10;
  assert(raisedHeight > 0);
  assert(
    renderer.view().altitude >= raisedHeight + renderer.minimumClearance() - 1e-5,
    "collision uses displayed vertex elevations, not just raw center samples",
  );
  assert(renderer.surfacePick(160, 120), "exaggerated coarse geometry cannot engulf the camera");
  const obsolete = { ...controller.active };
  renderer.setView(20, 155, renderer.view().altitude);
  controller.update();
  assert(
    requests.some((request) => request.type === "cancel"),
    "offscreen inference cannot hold up the new viewport",
  );
  assert.notEqual(controller.active.id, obsolete.id);
  const tileCount = controller.tiles.size;
  controller.receive({ ...obsolete, type: "tile", values: new Float32Array(1).buffer });
  assert.equal(controller.tiles.size, tileCount, "cancelled viewport replies are ignored");
  controller.reset("2");
  assert.equal(renderer.tiles.size, 0);
  assert.equal(renderer.patchTiles.size, 0);
  renderer.surfaceAt = () => 0;
  renderer.exaggeration = 1;
  const start = [1, 0, 0],
    end = [0.5, 0, -Math.sqrt(3) / 2];
  const segmentCounts = [];
  for (const altitude of [500, 20000]) {
    renderer.setView(0, 0, altitude);
    renderer.measure(start, end, false);
    const positions = renderer.measurement.children[0].geometry.getAttribute("position");
    segmentCounts.push(positions.count);
    for (let index = 1; index < positions.count; index++) {
      const midpoint = new THREE.Vector3()
        .fromBufferAttribute(positions, index - 1)
        .add(new THREE.Vector3().fromBufferAttribute(positions, index))
        .multiplyScalar(0.5);
      assert(midpoint.length() > 1, "measurement chords stay above a flat surface at close zoom");
    }
  }
  assert(segmentCounts[0] > segmentCounts[1], "measurement geometry adapts to zoom");
  renderer.measure(null, null, false);
  console.log(
    "Camera-local spherical refinement without fine inference, direct leaf scheduling, mesh reuse, idle rendering and shared-texture lifetime passed",
  );
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
