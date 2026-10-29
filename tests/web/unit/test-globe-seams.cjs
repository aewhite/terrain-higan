const { loadModules, demoDir, loadThree } = require("../support/test-support.cjs");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

(async () => {
  const THREE = await loadThree();
  const { api, context, load } = loadModules({ THREE, innerWidth: 1280, innerHeight: 900 });
  load("globe-topology.js");
  load("globe-mesh.js");
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
    scene: new THREE.Scene(),
    tiles: new Set(),
    terrainTiles: [],
    map: "realistic",
    exaggeration: 1,
    onChange() {},
  });
  const fixture = (face, depth, tileX, tileY) => ({
    key: `${face}:${depth}:${tileX}:${tileY}`,
    face,
    depth,
    tileX,
    tileY,
    level: depth + 2,
    values: new Float32Array(68 * 68).fill(-1),
    vertexValues: new Float32Array(67 * 67).fill(-1),
  });
  const interpolationError = (tile) => {
    const segments = tile.mesh.userData.segments,
      stride = 64 / segments,
      width = segments + 1;
    const positions = tile.mesh.geometry.getAttribute("position");
    let maximum = 0;
    for (let row = 0; row < segments; row++)
      for (let column = 0; column < segments; column++)
        for (const [fractionEast, fractionNorth] of [
          [0.5, 0],
          [0, 0.5],
          [0.5, 0.5],
          [0.25, 0.25],
          [0.75, 0.75],
        ]) {
          const exact = renderer.point(
            tile,
            (column + fractionEast) * stride,
            (row + fractionNorth) * stride,
          );
          const projected = exact.clone().project(renderer.camera);
          if (
            Math.abs(projected.x) > 1 ||
            Math.abs(projected.y) > 1 ||
            projected.z < -1 ||
            projected.z > 1
          )
            continue;
          const top = row * width + column;
          let approximation;
          if (fractionEast + fractionNorth <= 1) {
            approximation = new THREE.Vector3()
              .fromBufferAttribute(positions, top)
              .multiplyScalar(1 - fractionEast - fractionNorth)
              .addScaledVector(
                new THREE.Vector3().fromBufferAttribute(positions, top + 1),
                fractionEast,
              )
              .addScaledVector(
                new THREE.Vector3().fromBufferAttribute(positions, top + width),
                fractionNorth,
              );
          } else {
            approximation = new THREE.Vector3()
              .fromBufferAttribute(positions, top + width + 1)
              .multiplyScalar(fractionEast + fractionNorth - 1)
              .addScaledVector(
                new THREE.Vector3().fromBufferAttribute(positions, top + 1),
                1 - fractionNorth,
              )
              .addScaledVector(
                new THREE.Vector3().fromBufferAttribute(positions, top + width),
                1 - fractionEast,
              );
          }
          approximation.add(tile.mesh.position).project(renderer.camera);
          maximum = Math.max(
            maximum,
            Math.hypot(
              (approximation.x - projected.x) * 640,
              (approximation.y - projected.y) * 450,
            ),
          );
        }
    return maximum;
  };
  const report = [];
  for (const [face, depth, tileX, tileY] of [
    [0, 1, 1, 1],
    [0, 3, 7, 7],
    [0, 3, 0, 4],
    [4, 3, 7, 4],
    [5, 3, 0, 0],
    [0, 12, 4095, 4095],
    [5, 12, 0, 0],
  ]) {
    const source = fixture(face, 0, 0, 0);
    const patch = { ...fixture(face, depth, tileX, tileY), source };
    patch.bounds = renderer.bounds(patch);
    const center = patch.bounds.center;
    renderer.setView(
      (Math.asin(center.y) * 180) / Math.PI,
      (Math.atan2(-center.z, center.x) * 180) / Math.PI,
      globe.RADIUS / 2 ** depth,
    );
    renderer.createTile(patch);
    const normals = patch.mesh.geometry.getAttribute("normal"),
      positions = patch.mesh.geometry.getAttribute("position");
    let normalError = 0;
    for (let index = 0; index < (patch.mesh.userData.segments + 1) ** 2; index++) {
      const radial = new THREE.Vector3()
        .fromBufferAttribute(positions, index)
        .add(patch.mesh.position)
        .normalize();
      normalError = Math.max(
        normalError,
        radial.distanceTo(new THREE.Vector3().fromBufferAttribute(normals, index)),
      );
    }
    report.push({
      face,
      depth,
      segments: patch.mesh.userData.segments,
      pixels: interpolationError(patch),
      normalError,
    });
    renderer.dispose(patch);
    renderer.dispose(source);
  }
  assert(
    report.every((entry) => entry.pixels <= 0.25),
    "polar projection interpolation stays below a quarter pixel",
  );
  assert(
    report.every((entry) => entry.normalError < 1e-6),
    "constant-height seams have radial normals",
  );
  renderer.map = "elevation";
  const patchFor = (face, depth, tileX, tileY, segments, source) => {
    const patch = { ...fixture(face, depth, tileX, tileY), segments, source };
    patch.bounds = renderer.bounds(patch);
    return patch;
  };
  const sourceFor = (face, depth = 0, tileX = 0, tileY = 0) => ({
    ...fixture(face, depth, tileX, tileY),
    values: new Float32Array(68 * 68).fill(1 + face / 10),
    vertexValues: new Float32Array(67 * 67).fill(1 + face / 10),
  });
  const edgePosition = (tile, edge, along) => {
    const segments = tile.mesh.userData.segments,
      width = segments + 1;
    const offset = Math.min(segments - 1, Math.floor(along * segments));
    const index = (step) =>
      edge === 0
        ? step * width
        : edge === 1
          ? step * width + segments
          : edge === 2
            ? step
            : segments * width + step;
    const positions = tile.mesh.geometry.getAttribute("position");
    return new THREE.Vector3()
      .fromBufferAttribute(positions, index(offset))
      .lerp(
        new THREE.Vector3().fromBufferAttribute(positions, index(offset + 1)),
        along * segments - offset,
      )
      .add(tile.mesh.position);
  };
  let maximumGap = 0;
  for (const depth of [0, 3, 18])
    for (let face = 0; face < globe.FACE_COUNT; face++)
      for (let edge = 0; edge < 4; edge++) {
        const side = 2 ** depth,
          along = Math.floor(side / 3);
        const tileX = edge === 0 ? 0 : edge === 1 ? side - 1 : along;
        const tileY = edge === 2 ? 0 : edge === 3 ? side - 1 : along;
        const neighbor = globe.canonicalPixel(
          face,
          tileX + (edge === 0 ? -1 : edge === 1 ? 1 : 0),
          tileY + (edge === 2 ? -1 : edge === 3 ? 1 : 0),
          side,
        );
        const first = patchFor(face, depth, tileX, tileY, 4, sourceFor(face));
        const second = patchFor(
          neighbor.face,
          depth,
          neighbor.east,
          neighbor.north,
          16,
          sourceFor(neighbor.face),
        );
        const stitched = globe.stitchPatches(
          [first, second],
          undefined,
          renderer.meshVertex.bind(renderer),
        );
        const meshes = stitched.patches.map((patch) => renderer.createTile(patch));
        const other = globe.edges[face][edge];
        for (let sample = 0; sample <= 16; sample++) {
          const fraction = sample / 16;
          const gap =
            edgePosition(meshes[0], edge, fraction).distanceTo(
              edgePosition(meshes[1], other.edge, other.reverse ? 1 - fraction : fraction),
            ) * globe.RADIUS;
          maximumGap = Math.max(maximumGap, gap);
          assert(
            gap < 0.5 / side + 0.002,
            `shared face edge ${face}:${edge} at depth ${depth}: ${gap} m`,
          );
        }
        for (const mesh of meshes) renderer.dispose(mesh);
        renderer.dispose(first.source);
        renderer.dispose(second.source);
      }
  const coarse = patchFor(4, 1, 0, 0, 4, sourceFor(4));
  const strip = Array.from({ length: 16 }, (_, north) =>
    patchFor(4, 5, 16, north, north % 2 ? 16 : 4, sourceFor(4, 5, 16, north)),
  );
  strip.forEach((patch, index) => patch.source.vertexValues.fill(2 + index / 100));
  const mixed = globe.stitchPatches(
    [coarse, ...strip],
    undefined,
    renderer.meshVertex.bind(renderer),
  );
  const mixedMeshes = mixed.patches.map((patch) => renderer.createTile(patch));
  for (let index = 0; index < strip.length; index++) {
    const fine = mixedMeshes[index + 1];
    for (let sample = 0; sample <= fine.segments; sample++) {
      const along = sample / fine.segments;
      assert(
        edgePosition(fine, 0, along).distanceTo(
          edgePosition(mixedMeshes[0], 1, (index + along) / 16),
        ) *
          globe.RADIUS <
          0.3,
        "fine edges follow the actual coarse polyline, including T-junction corners",
      );
    }
    if (index)
      for (let sample = 0; sample <= 16; sample++)
        assert(
          edgePosition(fine, 2, sample / 16).distanceTo(
            edgePosition(mixedMeshes[index], 3, sample / 16),
          ) *
            globe.RADIUS <
            0.3,
          "T-junction corrections propagate to perpendicular edges",
        );
  }
  const reused = globe.stitchPatches(
    [coarse, ...strip],
    mixed.cache,
    renderer.meshVertex.bind(renderer),
  );
  reused.patches.forEach((patch, index) =>
    patch.edgeBindings.forEach((binding, edge) =>
      assert.equal(
        binding.line,
        mixed.patches[index].edgeBindings[edge].line,
        "unchanged seam geometry is cached",
      ),
    ),
  );
  for (const mesh of mixedMeshes) {
    renderer.dispose(mesh);
    renderer.dispose(mesh.source);
  }
  const roots = Array.from({ length: globe.FACE_COUNT }, (_, face) =>
    patchFor(face, 0, 0, 0, [4, 8, 16][face % 3], sourceFor(face)),
  );
  const world = globe.stitchPatches(roots, undefined, renderer.meshVertex.bind(renderer));
  const worldMeshes = world.patches.map((patch) => renderer.createTile(patch));
  for (const first of worldMeshes)
    for (let edge = 0; edge < 4; edge++) {
      const other = globe.edges[first.face][edge];
      const second = worldMeshes[other.face];
      for (let sample = 0; sample <= 16; sample++) {
        const along = sample / 16;
        assert(
          edgePosition(first, edge, along).distanceTo(
            edgePosition(second, other.edge, other.reverse ? 1 - along : along),
          ) *
            globe.RADIUS <
            0.5,
          "three/four-face corner ownership is shared across the full globe",
        );
      }
    }
  for (const mesh of worldMeshes) {
    renderer.dispose(mesh);
    renderer.dispose(mesh.source);
  }
  for (const face of [0, 1, 2, 3, 4, 5]) {
    const coordinate = face < 4 ? 2 ** 18 - 1 : 0;
    const tile = renderer.createTile(fixture(face, 18, coordinate, coordinate));
    assert(
      tile.mesh.geometry.getAttribute("normal").array.every(Number.isFinite),
      "native z20 polar normals are finite",
    );
    renderer.dispose(tile);
  }
  const axis = new THREE.Vector3(0.3, 0.4, 0.5),
    amplitude = 0.002;
  const elevated = (east, north) => {
    const direction = new THREE.Vector3(...globe.faceVector(4, east, north));
    return direction.multiplyScalar(1 + amplitude * direction.dot(axis));
  };
  const point = elevated(0.7, 0.2),
    radial = point.clone().normalize();
  const slope = axis
    .clone()
    .addScaledVector(radial, -axis.dot(radial))
    .multiplyScalar(amplitude / point.length());
  const expectedNormal = radial.clone().sub(slope).normalize();
  const slopeNormal = renderer.surfaceNormal(
    point,
    elevated(0.700001, 0.2),
    elevated(0.699999, 0.2),
    elevated(0.7, 0.200001),
    elevated(0.7, 0.199999),
  );
  assert(
    slopeNormal.distanceTo(expectedNormal) < 1e-7,
    "metric-correct normals retain terrain gradients",
  );
  console.log(
    "Shared vertices across all S2 edges, reversed charts, mixed mesh/model LOD and T-junctions passed",
    { maximumGapMeters: maximumGap },
  );
  console.log("S2 seam diagnostics", report);
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
