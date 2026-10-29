const { loadModules, demoDir, loadThree } = require("../support/test-support.cjs");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

(async () => {
  const THREE = await loadThree();
  const { api, context, load } = loadModules({ THREE });
  load("globe-topology.js");
  for (const file of ["globe-renderer.js", "globe-walk.js", "planar-walk.js"])
    vm.runInContext(
      "{\n" +
        fs
          .readFileSync(path.join(demoDir, file), "utf8")
          .replace('import * as THREE from "three";', "") +
        "\n}",
      context,
    );
  const {
    RADIUS,
    horizonAngle,
    horizonDistance,
    GlobeWalk,
    GlobeRenderer,
    PlanarWalk,
    PlanarRenderer,
  } = api.globe;
  assert.equal(RADIUS, 6371000);
  for (const observerHeight of [0, 2, 1000, 8849, 20000])
    for (const targetHeight of [0, 2, 1000, 8849, 20000]) {
      const expected =
        RADIUS *
        (Math.acos(RADIUS / (RADIUS + observerHeight)) +
          Math.acos(RADIUS / (RADIUS + targetHeight)));
      assert(Math.abs(horizonDistance(observerHeight, targetHeight) - expected) < 0.00001);
      assert.equal(
        horizonDistance(observerHeight, targetHeight),
        horizonDistance(targetHeight, observerHeight),
      );
    }
  assert.equal(horizonDistance(0, 0), 0);
  assert(horizonAngle(1e-12) > 0, "stable spherical evaluation retains tiny positive heights");
  assert(horizonDistance(2, 8849) > 330000 && horizonDistance(2, 8849) < 350000);
  assert(
    horizonDistance(8849, 8849) > 660000,
    "elevated observers can see peaks beyond the old 500 km limit",
  );
  for (const [observerHeight, targetHeight] of [
    [2, 2],
    [2, 1000],
    [1000, 10],
    [8849, 8849],
  ])
    for (const factor of [0.99, 1.01]) {
      const distance = horizonDistance(observerHeight, targetHeight) * factor;
      const observer = new THREE.Vector3(RADIUS + observerHeight, 0, 0);
      const target = new THREE.Vector3(
        Math.cos(distance / RADIUS),
        Math.sin(distance / RADIUS),
        0,
      ).multiplyScalar(RADIUS + targetHeight);
      const delta = target.clone().sub(observer);
      const fraction = THREE.MathUtils.clamp(-observer.dot(delta) / delta.lengthSq(), 0, 1);
      const unobstructed = observer.addScaledVector(delta, fraction).length() >= RADIUS;
      assert.equal(
        unobstructed,
        distance <= horizonDistance(observerHeight, targetHeight),
        "both-height horizon agrees with line-of-sight intersection against the reference sphere",
      );
    }
  const renderer = Object.create(GlobeRenderer.prototype);
  renderer.camera = new THREE.PerspectiveCamera();
  const walk = new GlobeWalk(renderer, {}, () => {});
  walk.active = walk.prepared = true;
  walk.patchIndex = new Map();
  const tile = (name, distance, minimum, maximum) => {
    const center = new THREE.Vector3(Math.cos(distance / RADIUS), Math.sin(distance / RADIUS), 0);
    const positions = new Float32Array(5 * 3);
    [minimum, minimum, maximum, minimum, 50000].forEach((height, index) =>
      center
        .clone()
        .multiplyScalar(height / RADIUS)
        .toArray(positions, index * 3),
    );
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    const mesh = new THREE.Mesh(geometry);
    mesh.position.copy(center);
    mesh.userData.segments = 1;
    const result = {
      mesh,
      waterMesh: new THREE.Mesh(),
      bounds: new THREE.Sphere(center, 100 / RADIUS),
    };
    walk.cacheHorizon(result);
    assert(
      Math.abs(result.horizon.maximumHeight - maximum) < 0.01,
      "the actual highest surface vertex, not a deep or elevated skirt, bounds visibility",
    );
    walk.patchIndex.set(name, result);
    return result;
  };
  const low = tile("low", 80000, 10, 10);
  const mountain = tile("mountain", 80000, 10, 1500);
  const farPeak = tile("farPeak", 650000, 10, 9000);
  const farLow = tile("farLow", 650000, 10, 1500);
  const belowSea = tile("belowSea", 80000, -1000, -900);
  const observe = (height) => {
    renderer.camera.position.set(1 + height / RADIUS, 0, 0);
    walk.updateVisibility();
  };
  observe(2);
  assert.equal(low.mesh.visible, false);
  assert.equal(
    mountain.mesh.visible,
    true,
    "a high feature remains visible even though low ground at the same location is below the horizon",
  );
  assert.equal(
    mountain.waterMesh.visible,
    false,
    "sea level and mountain peaks have different visibility limits",
  );
  assert.equal(farPeak.mesh.visible, false);
  observe(1000);
  assert.equal(
    low.mesh.visible,
    true,
    "observer terrain elevation counts, not just 2 m eye clearance",
  );
  assert.equal(low.waterMesh.visible, true);
  observe(9000);
  assert.equal(farPeak.mesh.visible, true);
  assert.equal(
    farLow.mesh.visible,
    false,
    "individual feature heights affect visibility at the same range",
  );
  assert(walk.sceneryRadius() >= api.globe.WALK_RADIUS + horizonDistance(20000, 20000));
  renderer.renderDirty = false;
  walk.updateVisibility();
  assert.equal(
    renderer.renderDirty,
    false,
    "stationary camera rotation does not repeat height-bound visibility work",
  );
  observe(-1000);
  assert(
    [...walk.patchIndex.values()].every((patch) => patch.mesh.visible && patch.waterMesh.visible),
  );
  assert.equal(
    belowSea.mesh.visible,
    true,
    "above-sphere horizon culling does not remove underwater terrain",
  );
  for (const level of [5, 10, 14, 15])
    assert.equal(
      renderer.meshSegments({ level }),
      64,
      "walking retains every native geometry sample, including narrow peaks",
    );
  const planarRenderer = Object.create(PlanarRenderer.prototype);
  const planarWalk = new PlanarWalk(planarRenderer, {}, () => {});
  assert.equal(planarWalk.sceneryRadius(), 500000);
  planarWalk.prepared = true;
  planarWalk.updateVisibility();
  for (const patch of walk.patchIndex.values()) {
    patch.mesh.geometry.dispose();
    patch.mesh.material.dispose();
    patch.waterMesh.geometry.dispose();
    patch.waterMesh.material.dispose();
  }
  console.log(
    "Spherical horizons: exact two-height geometry, independent sphere LOS, actual terrain peaks, high observers beyond 500 km, sea/subsea behavior, native peak geometry and unchanged flat visibility passed",
  );
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
