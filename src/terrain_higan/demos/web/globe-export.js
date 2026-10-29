import * as THREE from "three";

const { RADIUS, WALK_RADIUS } = TerrainHiGAN.globe;

function walkingScene(walk) {
  if (!walk.prepared) throw new Error("Prepare the walking scene before exporting");
  const { up, north, east, origin, metadata } = walk.exportFrame();
  const scene = new THREE.Group();
  scene.name = "Terrain HiGAN walking scene";
  scene.userData = {
    units: "meters",
    seed: walk.controller.seed,
    radiusMeters: WALK_RADIUS,
    ...metadata,
    axes: "+X east, +Y up, -Z north",
    boundary: "Complete tiles covering the 20 km disk",
  };
  const point = new THREE.Vector3(),
    normal = new THREE.Vector3();
  const tiles = [...walk.renderer.patchTiles.values()].filter(
    (tile) => walk.distance(tile.bounds).nearest <= WALK_RADIUS,
  );
  for (const tile of tiles)
    for (const [name, source] of [
      ["Terrain", tile.mesh],
      ["Sea", tile.waterMesh],
    ]) {
      if (!source) continue;
      const count = (tile.mesh.userData.segments + 1) ** 2;
      const geometry = new THREE.BufferGeometry();
      const positions = new Float32Array(count * 3);
      const normals = source.geometry.getAttribute("normal") ? new Float32Array(count * 3) : null;
      const offset = source.position.clone().sub(origin);
      for (let index = 0; index < count; index++) {
        point.fromBufferAttribute(source.geometry.getAttribute("position"), index).add(offset);
        positions.set(
          [point.dot(east) * RADIUS, point.dot(up) * RADIUS, -point.dot(north) * RADIUS],
          index * 3,
        );
        if (normals) {
          normal.fromBufferAttribute(source.geometry.getAttribute("normal"), index);
          normals.set([normal.dot(east), normal.dot(up), -normal.dot(north)], index * 3);
        }
      }
      geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
      if (normals) geometry.setAttribute("normal", new THREE.BufferAttribute(normals, 3));
      const uvs = source.geometry.getAttribute("uv");
      if (uvs)
        geometry.setAttribute("uv", new THREE.BufferAttribute(uvs.array.slice(0, count * 2), 2));
      geometry.setIndex(
        new THREE.BufferAttribute(
          source.geometry.index.array.slice(0, tile.mesh.userData.segments ** 2 * 6),
          1,
        ),
      );
      const material = new THREE.MeshStandardMaterial({
        color: source.material.color,
        map: source.material.map,
        roughness: 1,
        metalness: 0,
        transparent: source.material.transparent,
        opacity: source.material.opacity,
        side: source.material.side,
      });
      const mesh = new THREE.Mesh(geometry, material);
      mesh.name = `${name} ${tile.key}`;
      scene.add(mesh);
    }
  return scene;
}

async function downloadWalkingScene(walk) {
  const { GLTFExporter } = await import("./assets/runtime/GLTFExporter.js");
  const scene = walkingScene(walk);
  try {
    const buffer = await new GLTFExporter().parseAsync(scene, { binary: true });
    const url = URL.createObjectURL(new Blob([buffer], { type: "model/gltf-binary" }));
    const link = document.createElement("a");
    link.href = url;
    link.download = `terrain-${scene.userData.seed}-20km.glb`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  } finally {
    scene.traverse((object) => {
      object.geometry?.dispose();
      object.material?.dispose();
    });
  }
}

Object.assign(TerrainHiGAN.globe, { walkingScene, downloadWalkingScene });
