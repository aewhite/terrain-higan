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
    requestAnimationFrame: () => 1,
    cancelAnimationFrame() {},
  });
  for (const file of ["globe-topology.js", "globe-mesh.js", "globe-tile-controller.js"]) load(file);
  for (const file of ["globe-renderer.js", "globe-walk.js", "globe-export.js"])
    vm.runInContext(
      "{\n" +
        fs
          .readFileSync(path.join(demoDir, file), "utf8")
          .replace('import * as THREE from "three";', "") +
        "\n}",
      context,
    );
  const globe = api.globe;
  function fixture(latitude = 20, longitude = 0) {
    const renderer = Object.create(globe.GlobeRenderer.prototype);
    Object.assign(renderer, {
      canvas: { width: 320, height: 240 },
      camera: new THREE.PerspectiveCamera(42, 4 / 3, 0.001, 12),
      sphere: new THREE.Sphere(new THREE.Vector3(), 1),
      frustum: new THREE.Frustum(),
      raycaster: new THREE.Raycaster(),
      scene: new THREE.Scene(),
      tiles: new Set(),
      terrainTiles: [],
      map: "realistic",
      exaggeration: 5,
      onChange() {},
      updateStars() {},
      renderer: { setClearColor() {}, render() {} },
      sun: new THREE.DirectionalLight(),
      measurement: new THREE.Group(),
    });
    renderer.setView(latitude, longitude, 1000);
    const sent = [];
    const controller = new globe.GlobeTileController(
      renderer,
      (message) => sent.push(message),
      () => {},
    );
    const walk = new globe.GlobeWalk(renderer, controller, () => {});
    return { renderer, controller, walk, sent };
  }
  function fill(instance, elevation) {
    const { controller, walk } = instance;
    for (const node of walk.selection.leaves) {
      const side = 64 * 2 ** node.depth;
      const vertices = new Float32Array(67 * 67),
        values = new Float32Array(68 * 68);
      for (let row = 0; row < 67; row++)
        for (let column = 0; column < 67; column++) {
          const direction = globe.faceVector(
            node.face,
            (node.tileX * 64 + column - 1) / side,
            (node.tileY * 64 + row - 1) / side,
          );
          vertices[row * 67 + column] =
            (elevation(new THREE.Vector3(...direction)) + 2342.974207789) / 2439.687062251;
        }
      values.fill(vertices[32 * 67 + 32]);
      controller.tiles.set(node.key, {
        ...node,
        level: node.depth + 2,
        values,
        vertexValues: vertices,
      });
    }
    controller.tilesVersion++;
    let iterations = 0;
    while (!walk.prepared && iterations++ < 10000) controller.update();
    assert(walk.prepared, "complete scene builds before walking is enabled");
  }
  const instance = fixture(),
    { renderer, controller, walk, sent } = instance;
  assert(!walk.enter({ altitude: 20000 }));
  assert(walk.enter());
  walk.resume();
  assert.equal(walk.frame, 0, "preparation cannot start the input loop");
  walk.setKey("KeyW", true);
  assert.equal(walk.keys.size, 0, "held keys during preparation are ignored");
  const selectionStarted = performance.now();
  walk.selection = walk.select();
  const selectionMs = performance.now() - selectionStarted;
  assert(
    walk.selection.leaves.length < 6000,
    "scene has a bounded detailed disk and coarse backdrop",
  );
  assert.equal(globe.WALK_MAX_LEVEL, 14);
  assert.equal(Math.max(...walk.selection.leaves.map((node) => node.depth + 2)), 14);
  assert(walk.selection.leaves.some((node) => node.depth < walk.maxDepth()));
  assert(
    walk.selection.leaves.some(
      (node) => walk.distance(node.bounds).nearest <= globe.WALK_RADIUS && node.depth + 2 < 14,
    ),
    "terrain becomes coarser within the walking area away from the entry point",
  );
  for (const node of walk.selection.leaves)
    if (walk.distance(node.bounds).nearest <= globe.WALK_FINE_RADIUS)
      assert.equal(node.depth + 2, 14, "z14s covers the area around the entry point");
  const anchor = walk.anchor.clone();
  const north = new THREE.Vector3(0, 1, 0).addScaledVector(anchor, -anchor.y).normalize();
  const east = new THREE.Vector3().crossVectors(north, anchor).normalize();
  const started = performance.now();
  fill(
    instance,
    (direction) => 100 + direction.clone().sub(anchor).dot(north) * globe.RADIUS * 0.5,
  );
  assert.equal(renderer.exaggeration, 1);
  assert.equal(walk.aboveGround, 2);
  for (let bearing = 0; bearing < 8; bearing++) {
    const angle = 900000 / globe.RADIUS;
    const direction = anchor
      .clone()
      .multiplyScalar(Math.cos(angle))
      .addScaledVector(north, Math.sin(angle) * Math.cos((bearing * Math.PI) / 4))
      .addScaledVector(east, Math.sin(angle) * Math.sin((bearing * Math.PI) / 4));
    assert.notEqual(walk.groundAt(direction), null, "high-observer scenery extends beyond 500 km");
  }
  assert.equal(sent.length, 0);
  assert.equal(controller.sequence.length, 0);
  assert(
    renderer.camera.far * globe.RADIUS > walk.selection.viewRadius + globe.WALK_RADIUS,
    "the far plane covers the entire backdrop even at the walking boundary",
  );
  const distantPeak = anchor
    .clone()
    .multiplyScalar(Math.cos(100000 / globe.RADIUS))
    .addScaledVector(north, Math.sin(100000 / globe.RADIUS))
    .multiplyScalar(1 + 3000 / globe.RADIUS)
    .project(renderer.camera);
  assert(
    Math.abs(distantPeak.x) < 1 &&
      Math.abs(distantPeak.y) < 1 &&
      distantPeak.z < 1 &&
      distantPeak.z > -1,
    "a mountain 100 km away remains in the walking camera's view",
  );
  const snapshot = renderer.patchTiles;
  const meshes = [...snapshot.values()].map((tile) => tile.mesh);
  for (const tile of snapshot.values()) {
    assert(tile.source.level <= 14, "walking never requests models above z14s");
    assert.equal(
      tile.waterMesh.geometry.index.count,
      tile.mesh.userData.segments ** 2 * 6,
      "sea surfaces contain no overlapping skirt triangles",
    );
  }
  const shader = {
    uniforms: {},
    vertexShader: "#include <logdepthbuf_vertex>",
    fragmentShader: "#include <logdepthbuf_fragment>",
  };
  meshes[0].material.onBeforeCompile(shader);
  assert.equal(meshes[0].material.defines.USE_LOGARITHMIC_DEPTH_BUFFER, "");
  assert(!shader.vertexShader.includes("1.0 + gl_Position.w"));
  assert(shader.fragmentShader.includes("log2( vFragDepth / terrainDepthRange.x )"));
  assert.equal(shader.uniforms.terrainDepthRange, renderer.depthRange);
  renderer.draw();
  const depth = (distance) =>
    Math.log2(distance / (renderer.camera.near * globe.RADIUS)) * renderer.depthRange.value.y;
  assert(
    depth(20001) - depth(20000) > 1 / (2 ** 24 - 1),
    "distant surfaces one meter apart retain distinct 24-bit depth values",
  );
  for (let bearing = 0; bearing < 12; bearing++)
    for (const radius of [0, 64, 127, 129, 256, 1000, 10000, 19990, 100000, 490000]) {
      const angle = radius / globe.RADIUS;
      const tangent = north
        .clone()
        .multiplyScalar(Math.cos((bearing * Math.PI) / 6))
        .addScaledVector(east, Math.sin((bearing * Math.PI) / 6));
      const point = anchor
        .clone()
        .multiplyScalar(Math.cos(angle))
        .addScaledVector(tangent, Math.sin(angle));
      const height = walk.groundAt(point);
      assert.notEqual(height, null, "all bearings of the 20 km disk are prepared");
      assert(
        Math.abs(height - (100 + point.clone().sub(anchor).dot(north) * globe.RADIUS * 0.5)) <
          (radius <= globe.WALK_RADIUS ? 0.05 : 20),
        "ground sampling tolerates floating-point cracks at tile boundaries",
      );
      if (bearing % 3 === 0 && radius <= globe.WALK_RADIUS) {
        renderer.raycaster.set(point.clone().multiplyScalar(2), point.clone().negate());
        const reference = renderer.raycaster.intersectObjects(meshes, false)[0];
        if (reference)
          assert(
            Math.abs(height - (reference.point.length() - 1) * globe.RADIUS) < 0.01,
            "fast ground sampling agrees with rendered triangles",
          );
      }
    }
  let creates = 0;
  const createTile = renderer.createTile.bind(renderer);
  renderer.createTile = (...args) => {
    creates++;
    return createTile(...args);
  };
  walk.paused = false;
  walk.setKey("KeyW", true);
  walk.requestedSpeed = 1000;
  walk.pitch = 1.3;
  const initial = renderer.camera.position.clone();
  for (let frame = 0; frame < 100; frame++) {
    const previous = renderer.camera.position.clone();
    walk.tick(0.02);
    controller.update();
    assert(
      renderer.camera.position.distanceTo(previous) * globe.RADIUS <= 1.000001,
      "3D walking speed stays below 50 m/s on slopes",
    );
    const actual =
      (renderer.camera.position.length() - 1) * globe.RADIUS -
      walk.groundAt(renderer.camera.position);
    assert(Math.abs(actual - 2) < 1e-5, "eye height follows terrain uphill and downhill");
  }
  assert(renderer.camera.position.distanceTo(initial) * globe.RADIUS > 80);
  walk.heading.applyAxisAngle(renderer.camera.position.clone().normalize(), Math.PI);
  walk.pitch = -1.3;
  walk.updateCamera();
  controller.update();
  assert.equal(renderer.patchTiles, snapshot, "rotation keeps the prebuilt scene");
  assert.equal(walk.selection, controller.selection);
  assert.equal(creates, 0, "movement and rotation build no meshes");
  assert.equal(sent.length, 0, "movement and rotation request no inference");
  assert.equal(controller.sequence.length, 0);
  walk.requestedSpeed = 0;
  const stationary = renderer.camera.position.clone();
  for (let frame = 0; frame < 20; frame++) walk.tick(0.02);
  assert(renderer.camera.position.distanceTo(stationary) * globe.RADIUS < 1e-6);
  walk.requestedSpeed = 1;
  walk.adjustSpeed(100000);
  assert.equal(walk.requestedSpeed, 50);
  const boundary = anchor
    .clone()
    .multiplyScalar(Math.cos((globe.WALK_RADIUS - 0.01) / globe.RADIUS))
    .addScaledVector(north, Math.sin((globe.WALK_RADIUS - 0.01) / globe.RADIUS));
  renderer.camera.position.copy(boundary);
  walk.heading.copy(north).addScaledVector(boundary, -north.dot(boundary)).normalize();
  walk.protect();
  walk.tick(0.05);
  assert.equal(walk.status, "Scene boundary");
  assert.equal(walk.speed, 0);
  assert(
    globe.greatCircle(anchor.toArray(), renderer.camera.position.clone().normalize().toArray()) <=
      globe.WALK_RADIUS,
  );
  renderer.camera.position.copy(initial);
  walk.protect();
  const exportStarted = performance.now();
  const exported = globe.walkingScene(walk);
  assert.equal(exported.userData.units, "meters");
  assert.equal(exported.userData.radiusMeters, 20000);
  const detailedTiles = [...snapshot.values()].filter(
    (tile) => walk.distance(tile.bounds).nearest <= globe.WALK_RADIUS,
  );
  assert.equal(exported.children.length, detailedTiles.length * 2);
  assert(
    exported.children.length < snapshot.size * 2,
    "coarse background scenery is not included in the 20 km export",
  );
  const bounds = new THREE.Box3().setFromObject(exported);
  assert(bounds.min.x < -19000 && bounds.max.x > 19000);
  assert(bounds.min.z < -19000 && bounds.max.z > 19000);
  assert(
    Math.max(...bounds.min.toArray().map(Math.abs), ...bounds.max.toArray().map(Math.abs)) <
      globe.WALK_RADIUS +
        2 * Math.max(...detailedTiles.map((tile) => walk.distance(tile.bounds).extent)),
    "export is local meters, not unit-sphere or ECEF coordinates",
  );
  for (const child of exported.children) {
    child.geometry.dispose();
    child.material.dispose();
  }
  const exportMs = performance.now() - exportStarted;
  walk.exit();
  assert.equal(
    renderer.depthMaterial(new THREE.MeshBasicMaterial()).defines?.USE_LOGARITHMIC_DEPTH_BUFFER,
    undefined,
    "orbit materials keep the standard depth path",
  );
  assert.equal(renderer.exaggeration, 5);
  const radial = renderer.camera.position.clone().normalize();
  assert(Math.abs(renderer.camera.up.dot(radial)) < 1e-12);
  const grabbed = renderer.pick(140, 100);
  renderer.drag(140, 100, 180, 120);
  const projected = new THREE.Vector3(...grabbed)
    .multiplyScalar(renderer.sphere.radius)
    .project(renderer.camera);
  assert(Math.abs((projected.x + 1) * 160 - 180) < 0.01);
  assert(Math.abs((1 - projected.y) * 120 - 120) < 0.01);
  for (const latitude of [-90, 0, 90]) {
    const local = fixture(latitude, latitude === 0 ? 44.999999 : 0);
    local.walk.enter();
    const selected = local.walk.select();
    const leaves = selected.leaves.filter((node) => node.depth === local.walk.maxDepth());
    local.walk.selection = { ...selected, leaves, ordered: leaves };
    fill(local, () => -1000);
    assert(local.renderer.view().altitude < 0, "walking follows the seabed underwater");
    local.walk.paused = false;
    local.walk.requestedSpeed = 50;
    local.walk.setKey("KeyW", true);
    for (let step = 0; step < 5; step++) local.walk.tick(0.05);
    assert(local.renderer.camera.matrixWorld.elements.every(Number.isFinite));
    assert(Math.abs(local.renderer.view().altitude + 998) < 0.001);
    const lowLight = local.renderer.camera.position.clone().normalize();
    local.renderer.draw();
    assert(
      Math.abs(local.renderer.sun.position.clone().normalize().dot(lowLight) - Math.SQRT1_2) <
        1e-12,
    );
    local.walk.exit();
    assert(
      Math.abs(local.renderer.camera.up.dot(local.renderer.camera.position.clone().normalize())) <
        1e-12,
    );
  }
  console.log(
    "Walking: detailed 20 km coverage, spherical-horizon backdrop, staged preparation, rendered ground pinning, slope speed cap, frozen meshes/inference, poles/seams, underwater, meter export and orbit controls passed",
    { tiles: snapshot.size, selectionMs, preparationMs: performance.now() - started, exportMs },
  );
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
