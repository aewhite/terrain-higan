const { loadModules, demoDir, loadThree } = require("../support/test-support.cjs");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

(async () => {
  const THREE = await loadThree();
  const { api, context, load } = loadModules({
    THREE,
    innerWidth: 320,
    innerHeight: 240,
    devicePixelRatio: 1,
  });
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
  assert.equal(globe.FACE_COUNT, 6);
  const centers = [
    [1, 0, 0],
    [0, 0, -1],
    [0, 1, 0],
    [-1, 0, 0],
    [0, 0, 1],
    [0, -1, 0],
  ];
  const quarterPoints = [
    [12, 0, 5],
    [5, 0, -12],
    [5, 12, 0],
    [-12, 5, 0],
    [0, 5, 12],
    [0, -12, 5],
  ];
  for (let face = 0; face < globe.FACE_COUNT; face++) {
    nearVector(globe.faceVector(face, 0.5, 0.5), centers[face]);
    nearVector(
      globe.faceVector(face, 0.25, 0.5),
      quarterPoints[face].map((value) => value / 13),
    );
    const inverse = globe.vectorFace(quarterPoints[face]);
    assert.equal(inverse.face, face);
    assert(Math.abs(inverse.east - 0.25) < 1e-15);
    assert(Math.abs(inverse.north - 0.5) < 1e-15);
  }
  function nearVector(actual, expected) {
    assert(Math.hypot(...actual.map((value, axis) => value - expected[axis])) < 1e-14);
  }
  const corners = new Map();
  for (let face = 0; face < globe.FACE_COUNT; face++) {
    for (const east of [0, 1])
      for (const north of [0, 1]) {
        const direction = globe.faceVector(face, east, north);
        const key = direction.map(Math.sign).join(":");
        const canonical = globe.vectorFace(direction);
        const previous = corners.get(key);
        if (previous) assert.deepEqual(canonical, previous.coordinate);
        corners.set(key, { coordinate: canonical, count: (previous?.count || 0) + 1 });
      }
    for (let edge = 0; edge < 4; edge++) {
      const neighbor = globe.edges[face][edge];
      const reciprocal = globe.edges[neighbor.face][neighbor.edge];
      assert.equal(reciprocal.face, face);
      assert.equal(reciprocal.edge, edge);
      assert.equal(reciprocal.reverse, neighbor.reverse);
      for (const along of [0, 1 / globe.sideAt(20), 0.25, 0.5, 0.75, 1]) {
        const edgePoint = (edge, along) => (edge < 2 ? [edge, along] : [along, edge - 2]);
        nearVector(
          globe.faceVector(face, ...edgePoint(edge, along)),
          globe.faceVector(
            neighbor.face,
            ...edgePoint(neighbor.edge, neighbor.reverse ? 1 - along : along),
          ),
        );
      }
    }
  }
  assert.equal(corners.size, 8);
  assert([...corners.values()].every((corner) => corner.count === 3));
  for (const face of [0, 1, 2, 3, 4, 5]) {
    const coordinate = face < 4 ? 1 - 0.5 / globe.sideAt(20) : 0.5 / globe.sideAt(20);
    const result = globe.vectorFace(globe.faceVector(face, coordinate, coordinate));
    assert.equal(result.face, face);
    assert(Math.abs(result.east - coordinate) * globe.sideAt(20) < 0.0001);
    assert(Math.abs(result.north - coordinate) * globe.sideAt(20) < 0.0001);
  }
  const renderer = Object.create(globe.GlobeRenderer.prototype);
  Object.assign(renderer, {
    canvas: { width: 320, height: 240 },
    camera: new THREE.PerspectiveCamera(42, 320 / 240, 0.001, 12),
    sphere: new THREE.Sphere(new THREE.Vector3(), 1),
    frustum: new THREE.Frustum(),
    raycaster: new THREE.Raycaster(),
    terrainTiles: [],
    map: "realistic",
    exaggeration: 1,
    onChange() {},
    sample: () => 3000,
    show() {},
    dispose() {},
    createTile: (tile) => tile,
  });
  renderer.setView(20, -25, globe.RADIUS * 2);
  renderer.stars = renderer.createStars();
  const starPositions = renderer.stars.geometry.getAttribute("position");
  const latitudes = Array.from({ length: starPositions.count }, (_, index) =>
    starPositions.getY(index),
  ).sort((first, second) => first - second);
  const latitudeGaps = latitudes.slice(1).map((value, index) => value - latitudes[index]);
  assert(
    Math.max(...latitudeGaps) > (4 * 2) / starPositions.count,
    "stars have irregular spacing rather than evenly spaced latitude bands",
  );
  renderer.backgroundColor = new THREE.Color(0x08151a);
  renderer.spaceColor = new THREE.Color(0x02040a);
  let clearColor;
  renderer.renderer = { setClearColor: (color) => (clearColor = color.clone()) };
  renderer.updateStars();
  assert(renderer.stars.visible);
  assert.equal(renderer.stars.material.opacity, 1);
  assert(renderer.stars.material.depthTest, "terrain occludes background stars");
  assert.equal(renderer.stars.material.depthWrite, false);
  assert(renderer.stars.position.equals(renderer.camera.position));
  assert.equal(clearColor.getHex(), renderer.spaceColor.getHex());
  renderer.setView(20, -25, globe.RADIUS * 0.5);
  renderer.updateStars();
  assert(renderer.stars.material.opacity > 0 && renderer.stars.material.opacity < 1);
  renderer.setView(20, -25, 20000);
  renderer.updateStars();
  assert.equal(renderer.stars.visible, false);
  assert(clearColor.equals(renderer.backgroundColor));
  renderer.setView(20, -25, globe.RADIUS * 2);
  renderer.walk = { active: true };
  const walkingSky = new THREE.Color(0x87bce6);
  renderer.renderer.setClearColor(walkingSky);
  renderer.updateStars();
  assert.equal(renderer.stars.visible, false);
  assert(clearColor.equals(walkingSky), "walking retains its own sky color");
  renderer.walk = null;
  const tracked = renderer.pick(120, 100);
  const before = renderer.camera.position.clone().normalize();
  renderer.drag(120, 100, 160, 120);
  const orbitAngle = before.angleTo(renderer.camera.position.clone().normalize());
  const projected = new THREE.Vector3(...tracked)
    .multiplyScalar(renderer.sphere.radius)
    .project(renderer.camera);
  assert(Math.abs((projected.x + 1) * 160 - 160) < 0.01, "grabbed point follows horizontal drag");
  assert(Math.abs((1 - projected.y) * 120 - 120) < 0.01, "grabbed point follows vertical drag");
  renderer.setView(20, -25, 4000);
  const nearBefore = renderer.camera.position.clone().normalize();
  renderer.drag(120, 100, 160, 120);
  const nearAngle = nearBefore.angleTo(renderer.camera.position.clone().normalize());
  assert(nearAngle < orbitAngle / 1000, "surface rotation slows with ground resolution");
  renderer.zoom(1e-12);
  assert(renderer.view().altitude > 3000, "camera stays above mountain geometry");
  assert(Math.abs(renderer.view().clearance - renderer.minimumClearance()) < 1e-5);
  assert(renderer.view().clearance < 1000, "z20 is reachable below the former regional limit");
  assert(
    renderer.camera.near < renderer.view().clearance / globe.RADIUS,
    "near clipping plane does not remove close terrain",
  );
  const controller = new globe.GlobeTileController(
    renderer,
    () => {},
    () => {},
  );
  controller.reset("1");
  const deepest = Math.max(...controller.sequence.map((node) => node.depth + 2));
  assert.equal(deepest, 20, "physical-pixel selection reaches the z20 model");
  assert(controller.sequence.length < 2500, "close-up culling remains local");
  console.log("z20 desired tiles", controller.sequence.length);
  renderer.setView(20, -25, 250000);
  renderer.pendingAltitude = null;
  controller.update();
  const normalLevel = Math.max(...controller.sequence.map((node) => node.depth + 2));
  renderer.canvas.width *= 2;
  renderer.canvas.height *= 2;
  renderer.resize = () => {};
  controller.update();
  const retinaLevel = Math.max(...controller.sequence.map((node) => node.depth + 2));
  assert.equal(retinaLevel, normalLevel + 1, "Retina resolution selects the next model");
  for (const node of controller.nodes.values()) {
    if (node.depth !== 18 || !renderer.visible(node.bounds)) continue;
    assert(Number.isFinite(renderer.span(node.bounds)));
  }
  for (const height of [-6000, 8500]) {
    renderer.map = "elevation";
    renderer.sample = () => height;
    renderer.setView(20, -25, height + 1000);
    renderer.zoom(1e-9);
    assert(renderer.view().altitude >= height + renderer.minimumClearance() - 1e-5);
  }
  renderer.exaggeration = 10;
  renderer.setView(20, -25, 100000);
  renderer.zoom(1e-9);
  controller.update();
  assert(renderer.view().altitude > 85000, "exaggerated mountains also block the camera");
  const finest = controller.sequence.filter(
    (node) => node.depth === 18 && renderer.visible(node.bounds),
  );
  assert(finest.length > 0);
  assert(
    finest.every((node) => renderer.span(node.bounds) / 64 <= 1.01),
    "native zoom includes the displaced sphere's physical sample scale",
  );
  renderer.map = "realistic";
  renderer.exaggeration = 1;
  renderer.sample = () => 0;
  for (const latitude of [-90, -89.9999, 89.9999, 90]) {
    renderer.setView(latitude, 37, 1000);
    const view = renderer.view();
    assert(
      Math.abs(view.latitude - latitude) < 1e-8,
      "near-pole camera coordinates survive restoration",
    );
    assert(renderer.camera.matrixWorld.elements.every(Number.isFinite));
    const expected = renderer.camera.position.clone().normalize();
    assert(
      Math.abs(expected.dot(renderer.camera.up)) < 1e-12,
      "polar camera up is tangent to the surface",
    );
    assert(new THREE.Vector3(...renderer.pick(160, 120)).distanceTo(expected) < 1e-9);
    const up = renderer.camera.up.clone();
    renderer.setView(view.latitude, view.longitude, view.altitude);
    renderer.camera.up.copy(up);
    renderer.updateCamera();
    assert(Math.abs(renderer.view().latitude - latitude) < 1e-8);
  }
  for (let view = 0; view < 101; view++) {
    renderer.setView(20, -25 + view * 0.02, 1000);
    controller.update();
    assert(
      controller.nodes.size <= controller.nodeCapacity,
      "navigation bounds the geometry cache",
    );
    for (const node of controller.nodes.values())
      for (const child of node.children || [])
        assert.equal(
          controller.nodes.get(child.key),
          child,
          "cached children cannot retain evicted subtrees",
        );
  }
  console.log(
    "Globe grab direction, zoom sensitivity, ground collision, clipping, z20 selection, DPR and deep culling passed",
  );
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
