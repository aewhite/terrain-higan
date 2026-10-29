const { loadModules, demoDir, loadThree } = require("../support/test-support.cjs");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

(async () => {
  const THREE = await loadThree();
  let now = 0,
    nextFrame = 0;
  const frames = new Map();
  const { api, context, load } = loadModules({
    THREE,
    performance: { now: () => now },
    requestAnimationFrame: (callback) => {
      frames.set(++nextFrame, callback);
      return nextFrame;
    },
    cancelAnimationFrame: (frame) => frames.delete(frame),
  });
  for (const file of ["globe-topology.js", "globe-mesh.js"]) load(file);
  for (const file of ["globe-renderer.js", "globe-walk.js", "planar-walk.js"])
    vm.runInContext(
      "{\n" +
        fs
          .readFileSync(path.join(demoDir, file), "utf8")
          .replace('import * as THREE from "three";', "") +
        "\n}",
      context,
    );
  const { GlobeWalk, GlobeRenderer, PlanarWalk, PlanarRenderer, RADIUS } = api.globe;
  const advance = () => {
    now += 50;
    const pending = [...frames.values()];
    frames.clear();
    for (const callback of pending) callback(now);
  };
  for (const planar of [false, true]) {
    const renderer = Object.create((planar ? PlanarRenderer : GlobeRenderer).prototype);
    Object.assign(renderer, {
      camera: new THREE.PerspectiveCamera(42, 1, 0.001, 12),
      canvas: { width: 320, height: 240 },
      frustum: new THREE.Frustum(),
      sphere: new THREE.Sphere(),
      renderer: { setClearColor() {} },
    });
    const walk = new (planar ? PlanarWalk : GlobeWalk)(renderer, { failed: false }, () => {});
    walk.active = walk.prepared = true;
    walk.patchIndex = new Map();
    walk.anchor = planar ? new THREE.Vector3() : new THREE.Vector3(0, 1, 0);
    walk.groundAt = () => 100;
    walk.requestedSpeed = 1000;
    const origin = new THREE.Vector3(0, planar ? 102 / RADIUS : 1 + 102 / RADIUS, 0);
    const reset = () => {
      walk.pause();
      renderer.camera.position.copy(origin);
      walk.heading.set(0, 0, -1);
      walk.mouseBound = true;
      walk.pitch = 0;
      walk.updateCamera();
      walk.resume();
      advance();
      assert.equal(frames.size, 0, "ready, idle controls run no continuous animation");
    };
    for (const [codes, east, north] of [
      [["KeyW"], 0, 1],
      [["ArrowUp"], 0, 1],
      [["KeyS"], 0, -1],
      [["ArrowDown"], 0, -1],
      [["KeyA"], -1, 0],
      [["ArrowLeft"], -1, 0],
      [["KeyD"], 1, 0],
      [["ArrowRight"], 1, 0],
      [["KeyW", "KeyD"], 1, 1],
    ]) {
      reset();
      for (const code of codes) walk.setKey(code, true);
      advance();
      const delta = renderer.camera.position.clone().sub(origin).multiplyScalar(RADIUS);
      assert(
        delta.length() > 2.49 && delta.length() <= 2.500001,
        "all directions including diagonals share the 50 m/s cap",
      );
      if (east) assert(delta.x * east > 0.17);
      else assert(Math.abs(delta.x) < 1e-6);
      if (north) assert(-delta.z * north > 0.17);
      else assert(Math.abs(delta.z) < 1e-6);
      for (const code of codes) walk.setKey(code, false);
      const stopped = renderer.camera.position.clone();
      advance();
      assert.equal(walk.speed, 0);
      assert(renderer.camera.position.distanceTo(stopped) * RADIUS < 1e-6);
      assert.equal(frames.size, 0, "key release stops the input animation");
    }
    reset();
    walk.setKey("KeyW", true);
    walk.setKey("ArrowUp", true);
    walk.setKey("KeyW", false);
    advance();
    assert(walk.speed > 49.9, "releasing one alias leaves the other key held");
    walk.setKey("KeyS", true);
    advance();
    assert.equal(walk.speed, 0, "opposing directions cancel");
    reset();
    walk.mouse.x = 0.8;
    walk.wake();
    advance();
    assert(walk.heading.x > 0, "mouse can steer while standing still");
    assert(renderer.camera.position.distanceTo(origin) * RADIUS < 1e-6);
    walk.toggleMouse();
    assert.equal(walk.mouseBound, false);
    walk.mouse.x = 1;
    const heading = walk.heading.clone();
    advance();
    assert(walk.heading.distanceTo(heading) < 1e-12, "free mouse cannot turn the camera");
    walk.setKey("KeyW", true);
    advance();
    assert(walk.speed > 49.9, "freeing the mouse leaves keyboard movement enabled");
    walk.toggleMouse();
    assert.equal(walk.input().yaw, 0, "rebinding starts neutral");
    walk.pause();
    assert.equal(walk.keys.size, 0);
    assert.equal(frames.size, 0);
    walk.resume();
    const stopped = renderer.camera.position.clone();
    advance();
    assert(
      renderer.camera.position.distanceTo(stopped) * RADIUS < 1e-6,
      "resume does not restore stale held keys or auto-walk",
    );
    walk.pause();
    walk.active = false;
    assert.equal(walk.setKey("KeyW", true), false, "normal map/orbit controls are untouched");
    console.log(
      `${planar ? "Planar" : "Globe"} keyboard walking: WASD/arrows, backward/strafe, diagonal cap, key release, mouse binding, pause clearing and idle frames passed`,
    );
  }
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
