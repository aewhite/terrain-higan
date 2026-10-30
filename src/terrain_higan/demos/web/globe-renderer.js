import * as THREE from "three";

const {
  elevationMeters,
  writeRealisticColor,
  terrainColor,
  oceanColor,
  contourBaseColor,
  contourInterval,
} = TerrainHiGAN;
const {
  RADIUS,
  MIN_ALTITUDE,
  MAX_ALTITUDE,
  MAX_GLOBE_LEVEL,
  faceVector,
  vectorFace,
  canonicalPixel,
  arcPoints,
  greatCircle,
  stitchPatches,
  edgeValue,
} = TerrainHiGAN.globe;

const tileLayouts = new Map(
  [4, 8, 16, 32, 64].map((segments) => {
    const width = segments + 1,
      skirtStart = width * width;
    const uvs = new Float32Array((skirtStart + segments * 4) * 2);
    const indices = new Uint16Array(segments * segments * 6 + segments * 24);
    const boundary = [];
    let triangle = 0;
    for (let row = 0; row <= segments; row++)
      for (let column = 0; column <= segments; column++) {
        const vertex = row * width + column;
        uvs[vertex * 2] = ((column * 64) / segments + 2) / 68;
        uvs[vertex * 2 + 1] = ((row * 64) / segments + 2) / 68;
        if (column < segments && row < segments) {
          indices.set(
            [vertex, vertex + 1, vertex + width, vertex + 1, vertex + width + 1, vertex + width],
            triangle,
          );
          triangle += 6;
        }
      }
    for (let column = 0; column < segments; column++) boundary.push(column);
    for (let row = 0; row < segments; row++) boundary.push(row * width + segments);
    for (let column = segments; column > 0; column--) boundary.push(segments * width + column);
    for (let row = segments; row > 0; row--) boundary.push(row * width);
    for (let offset = 0; offset < boundary.length; offset++) {
      const index = boundary[offset],
        next = (offset + 1) % boundary.length;
      uvs.set(uvs.subarray(index * 2, index * 2 + 2), (skirtStart + offset) * 2);
      indices.set(
        [
          index,
          skirtStart + offset,
          boundary[next],
          boundary[next],
          skirtStart + offset,
          skirtStart + next,
        ],
        triangle,
      );
      triangle += 6;
    }
    return [segments, { uvs, indices, boundary, skirtStart, width }];
  }),
);

class GlobeRenderer {
  constructor(canvas, onChange, initialize = true) {
    this.canvas = canvas;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
    this.renderer.setClearColor(0x08151a);
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.scene = new THREE.Scene();
    this.backgroundColor = new THREE.Color(0x08151a);
    this.spaceColor = new THREE.Color(0x02040a);
    this.stars = this.createStars();
    this.scene.add(this.stars);
    this.camera = new THREE.PerspectiveCamera(42, 1, 0.001, 12);
    this.camera.up.set(0, 1, 0);
    this.onChange = onChange;
    this.terrainTiles = [];
    this.surfaceHeight = 0;
    this.scene.add(new THREE.AmbientLight(0xffffff, 1.4));
    this.sun = new THREE.DirectionalLight(0xffffff, 1.7);
    this.scene.add(this.sun);
    this.loadingPlanet = new THREE.Mesh(
      new THREE.SphereGeometry(0.9999, 64, 32),
      this.depthMaterial(new THREE.MeshLambertMaterial({ color: 0x345467 })),
    );
    this.scene.add(this.loadingPlanet);
    this.canvas.dataset.loadingPlanet = "true";
    this.raycaster = new THREE.Raycaster();
    this.sphere = new THREE.Sphere(new THREE.Vector3(), 1);
    this.frustum = new THREE.Frustum();
    this.tiles = new Set();
    this.measurement = new THREE.Group();
    this.scene.add(this.measurement);
    this.map = "realistic";
    this.exaggeration = 1;
    if (initialize) {
      this.setView(20, -25, RADIUS * 2);
      this.resize();
    }
  }

  setLoading(loading) {
    if (this.loadingPlanet.visible === loading) return;
    this.loadingPlanet.visible = loading;
    this.canvas.dataset.loadingPlanet = String(loading);
    this.renderDirty = true;
  }

  createStars() {
    const count = 8500;
    const positions = new Float32Array(count * 3);
    const colors = new Float32Array(count * 3);
    const color = new THREE.Color();
    let seed = 0x51a7f13d;
    const random = () => {
      seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
      return seed / 4294967296;
    };
    for (let index = 0; index < count; index++) {
      const vertical = random() * 2 - 1;
      const radius = Math.sqrt(1 - vertical * vertical);
      const angle = random() * Math.PI * 2;
      positions.set([radius * Math.cos(angle), vertical, radius * Math.sin(angle)], index * 3);
      const brightness = 0.12 + 0.88 * random() ** 4;
      const tint = random();
      color.setHex(tint < 0.14 ? 0xffe2bd : tint < 0.47 ? 0xc4d9ff : 0xf2f5ff);
      color.multiplyScalar(brightness).toArray(colors, index * 3);
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    const stars = new THREE.Points(
      geometry,
      new THREE.PointsMaterial({
        size: 1.5,
        sizeAttenuation: false,
        vertexColors: true,
        transparent: true,
        opacity: 0,
        depthWrite: false,
        toneMapped: false,
      }),
    );
    stars.renderOrder = -1;
    stars.frustumCulled = false;
    return stars;
  }

  updateStars() {
    if (this.walk?.active) {
      this.stars.visible = false;
      this.stars.material.opacity = 0;
      return;
    }
    const altitude = (this.camera.position.length() - 1) * RADIUS;
    const opacity = THREE.MathUtils.smoothstep(altitude, RADIUS * 0.15, RADIUS);
    this.stars.visible = opacity > 0;
    this.stars.material.opacity = opacity;
    this.stars.position.copy(this.camera.position);
    this.stars.scale.setScalar(this.camera.far * 0.9);
    this.renderer.setClearColor(this.backgroundColor.clone().lerp(this.spaceColor, opacity));
  }

  setView(latitude, longitude, altitude) {
    const lat = THREE.MathUtils.degToRad(latitude);
    const lon = THREE.MathUtils.degToRad(longitude);
    this.pendingAltitude = altitude;
    this.clearanceTarget = null;
    const radius = 1 + Math.max(-200000, Math.min(MAX_ALTITUDE, altitude)) / RADIUS;
    this.camera.up.set(
      -Math.sin(lat) * Math.cos(lon),
      Math.cos(lat),
      Math.sin(lat) * Math.sin(lon),
    );
    this.camera.position
      .set(Math.cos(lat) * Math.cos(lon), Math.sin(lat), -Math.cos(lat) * Math.sin(lon))
      .multiplyScalar(radius);
    this.updateCamera();
  }

  minimumClearance() {
    const coordinate = vectorFace(this.camera.position.toArray());
    const step = 1 / (16 * 2 ** MAX_GLOBE_LEVEL);
    const center = new THREE.Vector3(
      ...faceVector(coordinate.face, coordinate.east, coordinate.north),
    );
    const east = faceVector(
      coordinate.face,
      coordinate.east + (coordinate.east < 0.5 ? step : -step),
      coordinate.north,
    );
    const north = faceVector(
      coordinate.face,
      coordinate.east,
      coordinate.north + (coordinate.north < 0.5 ? step : -step),
    );
    const spacing =
      RADIUS *
      Math.max(
        center.distanceTo(new THREE.Vector3(...east)),
        center.distanceTo(new THREE.Vector3(...north)),
      );
    return Math.max(
      MIN_ALTITUDE,
      (spacing * this.sphere.radius * this.canvas.height * 1.01) /
        (2 * Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2))),
    );
  }

  surfaceAt(direction) {
    const coordinate = vectorFace(direction);
    const displayed = this.sample(direction, this.displayTiles || [], coordinate);
    let height = this.sample(direction, this.terrainTiles, coordinate) ?? displayed ?? 0;
    if (displayed !== null) height = Math.max(height, displayed);
    for (const patch of this.patchTiles?.values() || []) {
      if (patch.face !== coordinate.face) continue;
      const east = coordinate.east * 2 ** patch.depth - patch.tileX;
      const north = coordinate.north * 2 ** patch.depth - patch.tileY;
      if (east < 0 || east > 1 || north < 0 || north > 1) continue;
      const source = patch.source;
      height = Math.max(
        height,
        this.vertexHeight(
          source,
          (coordinate.east * 2 ** source.depth - source.tileX) * 64,
          (coordinate.north * 2 ** source.depth - source.tileY) * 64,
        ),
      );
      if (patch.edgeBindings) {
        const distances = [east, 1 - east, north, 1 - north];
        for (let edge = 0; edge < 4; edge++) {
          if (distances[edge] > 1 / patch.mesh.userData.segments) continue;
          const value = edgeValue(patch.edgeBindings[edge], (edge < 2 ? north : east) * 64);
          const rendered =
            ((Math.hypot(value[0], value[1], value[2]) - 1) * RADIUS) / this.exaggeration;
          height = Math.max(height, rendered);
        }
      }
    }
    return (this.map === "realistic" ? Math.max(0, height) : height) * this.exaggeration;
  }

  prepareView(tiles, version) {
    if (version === undefined || version !== this.terrainVersion) {
      this.terrainTiles = [...tiles];
      this.terrainVersion = version;
    }
    if (this.walk?.active) {
      this.walk.protect();
      this.updateCamera();
      return;
    }
    if (
      this.pendingAltitude !== null &&
      this.sample(this.camera.position.toArray(), this.terrainTiles) !== null
    ) {
      this.camera.position.setLength(1 + this.pendingAltitude / RADIUS);
      this.pendingAltitude = null;
    }
    this.updateCamera();
  }

  updateCamera() {
    if (this.walk?.active) {
      this.walk.updateCamera();
      return;
    }
    const direction = this.camera.position.clone().normalize();
    this.surfaceHeight = this.surfaceAt(direction.toArray());
    this.sphere.radius = 1 + this.surfaceHeight / RADIUS;
    const floor = this.sphere.radius + this.minimumClearance() / RADIUS;
    const length = this.camera.position.length();
    const requested =
      this.clearanceTarget === null || this.clearanceTarget === undefined
        ? length
        : this.sphere.radius + Math.max(this.minimumClearance(), this.clearanceTarget) / RADIUS;
    const bounded = Math.max(floor, Math.min(1 + MAX_ALTITUDE / RADIUS, requested));
    if (Math.abs(bounded - length) > 1e-13) this.camera.position.multiplyScalar(bounded / length);
    this.camera.lookAt(0, 0, 0);
    const clearance = (this.camera.position.length() - this.sphere.radius) * RADIUS;
    this.camera.near = Math.max(0.01, clearance * 0.02) / RADIUS;
    this.camera.far = Math.max(2, this.camera.position.length() + 1.1);
    this.camera.updateProjectionMatrix();
    this.camera.updateMatrixWorld();
    this.viewProjection ||= new THREE.Matrix4();
    this.viewProjection.multiplyMatrices(
      this.camera.projectionMatrix,
      this.camera.matrixWorldInverse,
    );
    this.frustum.setFromProjectionMatrix(this.viewProjection);
    this.selectionKey = [
      ...this.camera.position.toArray(),
      ...this.camera.up.toArray(),
      this.sphere.radius,
      this.canvas.width,
      this.canvas.height,
    ].join(":");
  }

  view() {
    const position = this.camera.position;
    return {
      latitude: THREE.MathUtils.radToDeg(
        Math.atan2(position.y, Math.hypot(position.x, position.z)),
      ),
      longitude: THREE.MathUtils.radToDeg(Math.atan2(-position.z, position.x)),
      altitude: (position.length() - 1) * RADIUS,
      clearance: this.walk?.active
        ? this.walk.aboveGround
        : (position.length() - this.sphere.radius) * RADIUS,
    };
  }

  resize() {
    this.renderer.setPixelRatio(TerrainHiGAN.displayPixelRatio());
    this.renderer.setSize(innerWidth, innerHeight);
    this.camera.aspect = innerWidth / innerHeight;
    this.updateCamera();
  }

  pick(clientX, clientY) {
    if (this.walk?.active) return this.surfacePick(clientX, clientY);
    this.raycaster.setFromCamera(
      new THREE.Vector2((2 * clientX) / innerWidth - 1, 1 - (2 * clientY) / innerHeight),
      this.camera,
    );
    const hit = this.raycaster.ray.intersectSphere(this.sphere, new THREE.Vector3());
    return hit ? hit.normalize().toArray() : null;
  }

  surfacePick(clientX, clientY) {
    this.raycaster.setFromCamera(
      new THREE.Vector2((2 * clientX) / innerWidth - 1, 1 - (2 * clientY) / innerHeight),
      this.camera,
    );
    const surfaces = [...this.tiles].filter((mesh) => mesh.visible);
    const hit = this.raycaster.intersectObjects(surfaces, false)[0];
    return hit ? hit.point.normalize().toArray() : null;
  }

  zoom(factor, clientX = innerWidth / 2, clientY = innerHeight / 2) {
    this.pendingAltitude = null;
    const before = this.pick(clientX, clientY);
    const clearance = Math.max(this.minimumClearance(), this.view().clearance * factor);
    this.clearanceTarget = clearance;
    this.camera.position.setLength(this.sphere.radius + clearance / RADIUS);
    this.camera.lookAt(0, 0, 0);
    this.camera.updateMatrixWorld();
    const after = this.pick(clientX, clientY);
    if (before && after) {
      const rotation = new THREE.Quaternion().setFromUnitVectors(
        new THREE.Vector3(...after),
        new THREE.Vector3(...before),
      );
      this.camera.position.applyQuaternion(rotation);
      this.camera.up.applyQuaternion(rotation);
    }
    this.updateCamera();
  }

  drag(fromX, fromY, toX, toY) {
    this.pendingAltitude = null;
    const before = this.pick(fromX, fromY);
    const after = this.pick(toX, toY);
    const clearance = this.view().clearance;
    this.clearanceTarget = clearance;
    let rotation;
    if (before && after) {
      rotation = new THREE.Quaternion().setFromUnitVectors(
        new THREE.Vector3(...after),
        new THREE.Vector3(...before),
      );
    } else {
      const scale =
        (2 * Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2)) * clearance) /
        (RADIUS * innerHeight);
      const right = new THREE.Vector3().setFromMatrixColumn(this.camera.matrixWorld, 0);
      const up = new THREE.Vector3().setFromMatrixColumn(this.camera.matrixWorld, 1);
      rotation = new THREE.Quaternion()
        .setFromAxisAngle(up, -(toX - fromX) * scale)
        .multiply(new THREE.Quaternion().setFromAxisAngle(right, -(toY - fromY) * scale));
    }
    this.camera.position.applyQuaternion(rotation);
    this.camera.up.applyQuaternion(rotation);
    const height = this.surfaceAt(this.camera.position.clone().normalize().toArray());
    this.camera.position.setLength(1 + (height + clearance) / RADIUS);
    this.updateCamera();
    this.onChange();
  }

  bounds(tile) {
    const side = 2 ** tile.depth;
    const center = new THREE.Vector3(
      ...faceVector(tile.face, (tile.tileX + 0.5) / side, (tile.tileY + 0.5) / side),
    );
    let radius = 0;
    const centers = new Float64Array(27);
    let offset = 0;
    for (const east of [0, 0.5, 1])
      for (const north of [0, 0.5, 1]) {
        const point = faceVector(
          tile.face,
          (tile.tileX + east) / side,
          (tile.tileY + north) / side,
        );
        centers.set(point, offset);
        offset += 3;
        radius = Math.max(
          radius,
          Math.hypot(point[0] - center.x, point[1] - center.y, point[2] - center.z),
        );
      }
    const bounds = new THREE.Sphere(center, radius * 1.05);
    bounds.surfaceRadius = radius * 1.05;
    bounds.centers = centers;
    bounds.tile = { face: tile.face, depth: tile.depth, tileX: tile.tileX, tileY: tile.tileY };
    return bounds;
  }

  projectionSamples(bounds) {
    if (bounds.samples) return bounds.samples;
    const tile = bounds.tile;
    const side = 2 ** tile.depth;
    const step = 1 / (16 * 2 ** (tile.depth + 2));
    bounds.samples = new Float64Array(81);
    let offset = 0,
      center = 0;
    for (const east of [0, 0.5, 1])
      for (const north of [0, 0.5, 1]) {
        const eastCoordinate = (tile.tileX + east) / side;
        const northCoordinate = (tile.tileY + north) / side;
        for (const point of [
          bounds.centers.subarray(center, center + 3),
          faceVector(
            tile.face,
            eastCoordinate + (eastCoordinate < 0.5 ? step : -step),
            northCoordinate,
          ),
          faceVector(
            tile.face,
            eastCoordinate,
            northCoordinate + (northCoordinate < 0.5 ? step : -step),
          ),
        ]) {
          bounds.samples.set(point, offset);
          offset += 3;
        }
        center += 3;
      }
    return bounds.samples;
  }

  visible(bounds) {
    if (this.walk?.active) return this.walk.visible(bounds);
    if (bounds.visibilityKey === this.selectionKey) return bounds.visibility;
    bounds.visibilityKey = this.selectionKey;
    const radius = this.camera.position.length();
    const facing = bounds.center.dot(this.camera.position) / radius;
    if (facing + bounds.radius < this.sphere.radius / radius - 1e-12) {
      bounds.visibility = false;
      return false;
    }
    const extent =
      bounds.radius * this.sphere.radius +
      Math.min(bounds.radius * 0.1, (radius - this.sphere.radius) * 0.25);
    bounds.visibility = this.frustum.planes.every(
      (plane) => plane.normal.dot(bounds.center) * this.sphere.radius + plane.constant >= -extent,
    );
    return bounds.visibility;
  }

  span(bounds) {
    if (this.walk?.active) {
      const sphere = this.walk.bounds(bounds);
      const distance = Math.max(
        this.camera.near,
        sphere.center.distanceTo(this.camera.position) - sphere.radius,
      );
      const focal =
        this.canvas.height / (2 * Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2)));
      return (bounds.radius * 2 * focal) / distance;
    }
    if (bounds.pixelKey === this.selectionKey) return bounds.pixelSpan;
    let pixels = 0;
    const matrix = this.viewProjection.elements;
    const samples = this.projectionSamples(bounds);
    const radius = this.sphere.radius;
    for (let offset = 0; offset < samples.length; offset += 9) {
      const depth =
        radius *
          (matrix[3] * samples[offset] +
            matrix[7] * samples[offset + 1] +
            matrix[11] * samples[offset + 2]) +
        matrix[15];
      if (depth <= this.camera.near) continue;
      const horizontal =
        (radius *
          (matrix[0] * samples[offset] +
            matrix[4] * samples[offset + 1] +
            matrix[8] * samples[offset + 2]) +
          matrix[12]) /
        depth;
      const vertical =
        (radius *
          (matrix[1] * samples[offset] +
            matrix[5] * samples[offset + 1] +
            matrix[9] * samples[offset + 2]) +
          matrix[13]) /
        depth;
      for (const neighbor of [offset + 3, offset + 6]) {
        const divisor =
          radius *
            (matrix[3] * samples[neighbor] +
              matrix[7] * samples[neighbor + 1] +
              matrix[11] * samples[neighbor + 2]) +
          matrix[15];
        const projectedX =
          (radius *
            (matrix[0] * samples[neighbor] +
              matrix[4] * samples[neighbor + 1] +
              matrix[8] * samples[neighbor + 2]) +
            matrix[12]) /
          divisor;
        const projectedY =
          (radius *
            (matrix[1] * samples[neighbor] +
              matrix[5] * samples[neighbor + 1] +
              matrix[9] * samples[neighbor + 2]) +
            matrix[13]) /
          divisor;
        pixels = Math.max(
          pixels,
          Math.hypot(
            ((projectedX - horizontal) * this.canvas.width) / 2,
            ((projectedY - vertical) * this.canvas.height) / 2,
          ),
        );
      }
    }
    bounds.pixelKey = this.selectionKey;
    bounds.pixelSpan = pixels ? pixels * 64 : Infinity;
    return bounds.pixelSpan;
  }

  point(tile, column, row) {
    const side = 2 ** tile.depth;
    let east = (tile.tileX + column / 64) / side;
    let north = (tile.tileY + row / 64) / side;
    let face = tile.face;
    if (east < 0 || east > 1 || north < 0 || north > 1) {
      const point = canonicalPixel(
        face,
        east * side * 64 - 0.5,
        north * side * 64 - 0.5,
        side * 64,
      );
      face = point.face;
      east = (point.east + 0.5) / (side * 64);
      north = (point.north + 0.5) / (side * 64);
    }
    const vector = new THREE.Vector3(...faceVector(face, east, north));
    let normalized;
    if (tile.source && tile.source.depth !== tile.depth) {
      const scale = 2 ** (tile.source.depth - tile.depth);
      const sourceEast = (tile.tileX * 64 + column) * scale - tile.source.tileX * 64;
      const sourceNorth = (tile.tileY * 64 + row) * scale - tile.source.tileY * 64;
      const height = this.vertexHeight(tile.source, sourceEast, sourceNorth);
      const displacement =
        this.map === "realistic" && !this.walk?.active ? Math.max(0, height) : height;
      return vector.multiplyScalar(1 + (displacement * this.exaggeration) / RADIUS);
    } else if (!Number.isInteger(column) || !Number.isInteger(row)) {
      const height = this.vertexHeight(tile, column, row);
      return vector.multiplyScalar(
        1 +
          ((this.map === "realistic" && !this.walk?.active ? Math.max(0, height) : height) *
            this.exaggeration) /
            RADIUS,
      );
    } else normalized = tile.vertexValues[(row + 1) * 67 + column + 1];
    const height = elevationMeters(normalized);
    const displacement =
      this.map === "realistic" && !this.walk?.active ? Math.max(0, height) : height;
    return vector.multiplyScalar(1 + (displacement * this.exaggeration) / RADIUS);
  }

  colors(tile) {
    const pixels = new Uint8Array(68 * 68 * 4);
    for (let row = 0; row < 68; row++)
      for (let column = 0; column < 68; column++) {
        const index = row * 68 + column;
        const height = elevationMeters(tile.values[index]);
        if (this.map === "realistic") writeRealisticColor(height, pixels, index * 4);
        else if (this.map === "elevation") {
          if (height < 0) oceanColor(height, pixels, index * 4);
          else
            terrainColor(
              0.25 + 0.75 * Math.min(1, Math.max(0, height) / 9000) ** 0.7,
              pixels,
              index * 4,
            );
        } else {
          const interval = contourInterval(tile.level);
          const right = elevationMeters(tile.values[row * 68 + Math.min(67, column + 1)]);
          const below = elevationMeters(tile.values[Math.min(67, row + 1) * 68 + column]);
          const crosses = (step) =>
            Math.floor(height / step) !== Math.floor(right / step) ||
            Math.floor(height / step) !== Math.floor(below / step);
          const color = crosses(interval * 5)
            ? [113, 78, 45]
            : crosses(interval)
              ? height < 0
                ? [47, 100, 128]
                : [165, 126, 84]
              : contourBaseColor(height);
          pixels.set(color, index * 4);
        }
        pixels[index * 4 + 3] = 255;
      }
    return pixels;
  }

  vertexHeight(tile, east, north) {
    const left = Math.min(64, Math.floor(east)),
      top = Math.min(64, Math.floor(north));
    const fractionEast = east - left,
      fractionNorth = north - top;
    const index = (top + 1) * 67 + left + 1,
      values = tile.vertexValues;
    return elevationMeters(
      (values[index] * (1 - fractionEast) + values[index + 1] * fractionEast) *
        (1 - fractionNorth) +
        (values[index + 67] * (1 - fractionEast) + values[index + 68] * fractionEast) *
          fractionNorth,
    );
  }

  meshSegments(tile) {
    const source = tile.source;
    const level = source ? source.level : tile.level;
    if (this.walk?.active) return 64;
    const native = !source || source.depth === tile.depth;
    const minimum = native ? (level <= 8 ? 16 : level <= 14 ? 32 : 64) : 4;
    if (minimum === 64 || !tile.bounds) return minimum;
    if (!native && source.curvature === undefined) {
      let curvature = 0;
      const values = source.vertexValues;
      for (let row = 1; row < 66; row++)
        for (let column = 1; column < 66; column++) {
          const index = row * 67 + column;
          curvature = Math.max(
            curvature,
            Math.abs(values[index - 1] - 2 * values[index] + values[index + 1]),
            Math.abs(values[index - 67] - 2 * values[index] + values[index + 67]),
            Math.abs(values[index] + values[index + 68] - values[index + 1] - values[index + 67]),
          );
        }
      source.curvature = Math.abs(elevationMeters(curvature) - elevationMeters(0));
    }
    const bounds = tile.bounds;
    const distance = Math.max(
      this.camera.near,
      this.camera.position.distanceTo(bounds.center.clone().multiplyScalar(this.sphere.radius)) -
        bounds.radius * this.sphere.radius,
    );
    const focal =
      this.canvas.height / (2 * Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2)));
    const scale = source ? 2 ** (source.depth - tile.depth) : 1;
    for (const segments of [4, 8, 16, 32]) {
      if (segments < minimum) continue;
      const spacing = (64 * scale) / segments;
      const heightError = native
        ? 0
        : (source.curvature * spacing ** 2 * this.exaggeration) / (2 * RADIUS);
      if (
        ((this.sphere.radius * (bounds.radius / segments) ** 2 + heightError) * focal) / distance >
        0.25
      )
        continue;
      const error = this.chartError(bounds, segments) * this.sphere.radius + heightError;
      if ((error * focal) / distance <= 0.25) return segments;
    }
    return 64;
  }

  chartError(bounds, segments) {
    bounds.chartErrors ||= new Map();
    if (bounds.chartErrors.has(segments)) return bounds.chartErrors.get(segments);
    const tile = bounds.tile,
      side = 2 ** tile.depth;
    const point = (east, north) =>
      faceVector(
        tile.face,
        (tile.tileX + east / segments) / side,
        (tile.tileY + north / segments) / side,
      );
    let error = 0;
    for (const row of [0, Math.floor((segments - 1) / 2), segments - 1])
      for (const column of [0, Math.floor((segments - 1) / 2), segments - 1]) {
        const corners = [
          point(column, row),
          point(column + 1, row),
          point(column, row + 1),
          point(column + 1, row + 1),
        ];
        for (const [east, north] of [
          [0.5, 0],
          [0, 0.5],
          [0.5, 1],
          [1, 0.5],
          [1 / 3, 1 / 3],
          [2 / 3, 2 / 3],
        ]) {
          const exact = point(column + east, row + north);
          const weights =
            east + north <= 1
              ? [1 - east - north, east, north, 0]
              : [0, 1 - north, 1 - east, east + north - 1];
          const delta = exact.map(
            (value, axis) =>
              value -
              corners.reduce((sum, corner, index) => sum + corner[axis] * weights[index], 0),
          );
          error = Math.max(error, Math.hypot(...delta));
        }
      }
    bounds.chartErrors.set(segments, error);
    return error;
  }

  surfaceNormal(point, eastPlus, eastMinus, northPlus, northMinus, target = new THREE.Vector3()) {
    this.normalScratch ||= [new THREE.Vector3(), new THREE.Vector3(), new THREE.Vector3()];
    const [radial, east, north] = this.normalScratch;
    const radius = point.length(),
      eastHigh = eastPlus.length(),
      eastLow = eastMinus.length();
    const northHigh = northPlus.length(),
      northLow = northMinus.length();
    radial.copy(point).divideScalar(radius);
    east
      .copy(eastPlus)
      .divideScalar(eastHigh)
      .addScaledVector(eastMinus, -1 / eastLow);
    north
      .copy(northPlus)
      .divideScalar(northHigh)
      .addScaledVector(northMinus, -1 / northLow);
    east.addScaledVector(radial, -east.dot(radial));
    north.addScaledVector(radial, -north.dot(radial));
    const eastLength = east.lengthSq(),
      northLength = north.lengthSq(),
      cross = east.dot(north);
    const determinant = eastLength * northLength - cross * cross;
    const deltaEast = eastHigh - eastLow,
      deltaNorth = northHigh - northLow;
    return target
      .copy(radial)
      .addScaledVector(
        east,
        -(deltaEast * northLength - deltaNorth * cross) / (determinant * radius),
      )
      .addScaledVector(
        north,
        -(deltaNorth * eastLength - deltaEast * cross) / (determinant * radius),
      )
      .normalize();
  }

  meshVertex(source, east, north) {
    const point = this.point(source, east, north);
    const normal = this.surfaceNormal(
      point,
      this.point(source, east + 1, north),
      this.point(source, east - 1, north),
      this.point(source, east, north + 1),
      this.point(source, east, north - 1),
    );
    return Float64Array.from([...point.toArray(), ...normal.toArray()]);
  }

  tileOrigin(tile) {
    return new THREE.Vector3(
      ...faceVector(
        tile.face,
        (tile.tileX + 0.5) / 2 ** tile.depth,
        (tile.tileY + 0.5) / 2 ** tile.depth,
      ),
    );
  }

  skirtPoint(point, depth) {
    return point.setLength(Math.max(0.9, point.length() - Math.min(0.02, 0.08 / 2 ** depth)));
  }

  seaPoint(point) {
    return point.normalize();
  }

  upAt(position) {
    return position.clone().normalize();
  }

  depthMaterial(material) {
    if (!this.walk?.active) return material;
    material.defines = { ...material.defines, USE_LOGARITHMIC_DEPTH_BUFFER: "" };
    this.depthRange ||= { value: new THREE.Vector2() };
    material.onBeforeCompile = (shader) => {
      shader.uniforms.terrainDepthRange = this.depthRange;
      shader.vertexShader = shader.vertexShader.replace(
        "#include <logdepthbuf_vertex>",
        THREE.ShaderChunk.logdepthbuf_vertex.replace("1.0 + gl_Position.w", "gl_Position.w"),
      );
      shader.fragmentShader =
        "uniform vec2 terrainDepthRange;\n" +
        shader.fragmentShader.replace(
          "#include <logdepthbuf_fragment>",
          THREE.ShaderChunk.logdepthbuf_fragment.replace(
            "log2( vFragDepth ) * logDepthBufFC * 0.5",
            "log2( vFragDepth / terrainDepthRange.x ) * terrainDepthRange.y",
          ),
        );
    };
    return material;
  }

  createTile(tile) {
    const segments = tile.segments ?? this.meshSegments(tile);
    const stride = 64 / segments;
    const { uvs, indices, boundary, skirtStart, width } = tileLayouts.get(segments);
    const positions = new Float32Array((skirtStart + boundary.length) * 3),
      normals = new Float32Array(positions.length);
    const grid = [];
    const pointAt = (column, row) => {
      const index = (row + 1) * 67 + column + 1;
      return (grid[index] ||= this.point(tile, column, row));
    };
    const normal = new THREE.Vector3(),
      shared = new Float64Array(6);
    const center = this.tileOrigin(tile);
    for (let row = 0; row <= segments; row++)
      for (let column = 0; column <= segments; column++) {
        const vertex = row * width + column;
        const nativeColumn = column * stride,
          nativeRow = row * stride;
        const point = pointAt(nativeColumn, nativeRow);
        const edge =
          column === 0 ? 0 : column === segments ? 1 : row === 0 ? 2 : row === segments ? 3 : -1;
        if (tile.edgeBindings && edge >= 0) {
          edgeValue(tile.edgeBindings[edge], edge < 2 ? nativeRow : nativeColumn, shared);
          positions[vertex * 3] = shared[0] - center.x;
          positions[vertex * 3 + 1] = shared[1] - center.y;
          positions[vertex * 3 + 2] = shared[2] - center.z;
          normal.fromArray(shared, 3).normalize();
        } else {
          positions[vertex * 3] = point.x - center.x;
          positions[vertex * 3 + 1] = point.y - center.y;
          positions[vertex * 3 + 2] = point.z - center.z;
          this.surfaceNormal(
            point,
            pointAt(nativeColumn + 1, nativeRow),
            pointAt(nativeColumn - 1, nativeRow),
            pointAt(nativeColumn, nativeRow + 1),
            pointAt(nativeColumn, nativeRow - 1),
            normal,
          );
        }
        normal.toArray(normals, vertex * 3);
      }
    for (let offset = 0; offset < boundary.length; offset++) {
      const index = boundary[offset];
      const point = new THREE.Vector3().fromArray(positions, index * 3).add(center);
      this.skirtPoint(point, tile.depth);
      point.sub(center).toArray(positions, (skirtStart + offset) * 3);
      normals.set(normals.subarray(index * 3, index * 3 + 3), (skirtStart + offset) * 3);
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute("normal", new THREE.BufferAttribute(normals, 3));
    let textureUVs = uvs;
    if (tile.source && tile.source.depth !== tile.depth) {
      const scale = 2 ** (tile.source.depth - tile.depth);
      const east = tile.tileX * scale - tile.source.tileX;
      const north = tile.tileY * scale - tile.source.tileY;
      textureUVs = new Float32Array(uvs.length);
      for (let index = 0; index < uvs.length; index += 2) {
        textureUVs[index] = ((uvs[index] * 68 - 2) * scale + east * 64 + 2) / 68;
        textureUVs[index + 1] = ((uvs[index + 1] * 68 - 2) * scale + north * 64 + 2) / 68;
      }
    }
    geometry.setAttribute("uv", new THREE.BufferAttribute(textureUVs, 2));
    geometry.setIndex(new THREE.BufferAttribute(indices, 1));
    const source = tile.source || tile;
    if (!source.texture) {
      source.texture = new THREE.DataTexture(this.colors(source), 68, 68);
      source.texture.colorSpace = THREE.SRGBColorSpace;
      source.texture.magFilter = source.texture.minFilter = THREE.LinearFilter;
      source.texture.needsUpdate = true;
    }
    const texture = source.texture;
    const material = this.depthMaterial(new THREE.MeshLambertMaterial({ map: texture }));
    const mesh = new THREE.Mesh(geometry, material);
    mesh.userData.segments = segments;
    geometry.computeBoundingBox();
    geometry.boundingSphere = geometry.boundingBox.getBoundingSphere(new THREE.Sphere());
    mesh.position.copy(center);
    mesh.visible = false;
    this.scene.add(mesh);
    if (this.walk?.active) {
      const waterPositions = new Float32Array(positions.length);
      const point = new THREE.Vector3();
      for (let index = 0; index < positions.length; index += 3) {
        point.fromArray(positions, index).add(center);
        this.seaPoint(point).sub(center).toArray(waterPositions, index);
      }
      const waterGeometry = new THREE.BufferGeometry();
      waterGeometry.setAttribute("position", new THREE.BufferAttribute(waterPositions, 3));
      waterGeometry.setIndex(
        new THREE.BufferAttribute(indices.subarray(0, segments * segments * 6), 1),
      );
      const water = new THREE.Mesh(
        waterGeometry,
        this.depthMaterial(
          new THREE.MeshBasicMaterial({
            color: 0x24758a,
            transparent: true,
            opacity: 0.3,
            side: THREE.DoubleSide,
            depthWrite: false,
          }),
        ),
      );
      water.position.copy(center);
      this.scene.add(water);
      tile.waterMesh = water;
    }
    this.tiles.add(mesh);
    tile.mesh = mesh;
    return tile;
  }

  dispose(tile) {
    if (tile.waterMesh) {
      this.scene.remove(tile.waterMesh);
      tile.waterMesh.geometry.dispose();
      tile.waterMesh.material.dispose();
    }
    if (tile.mesh) {
      this.scene.remove(tile.mesh);
      tile.mesh.geometry.dispose();
      tile.mesh.material.dispose();
      this.tiles.delete(tile.mesh);
    }
    if (!tile.source && tile.texture) {
      tile.texture.dispose();
      delete tile.texture;
    }
  }

  restyle(tiles, map, exaggeration) {
    this.renderDirty = true;
    this.displayPatches = null;
    this.stitchCache = undefined;
    for (const patch of this.patchTiles?.values() || []) this.dispose(patch);
    this.patchTiles = new Map();
    this.map = map;
    this.exaggeration = exaggeration;
    for (const tile of tiles) {
      this.dispose(tile);
      delete tile.mesh;
    }
    if (this.walk?.active) this.walk.resetMeshes();
  }

  show(tiles, patches) {
    if (this.displayPatches === patches) return;
    this.displayPatches = patches;
    this.patchTiles ||= new Map();
    this.displayTiles = tiles;
    const layout = patches.map((patch) => ({ ...patch, segments: this.meshSegments(patch) }));
    if (
      layout.length === this.patchTiles.size &&
      layout.every((patch) => {
        const existing = this.patchTiles.get(patch.key);
        return (
          existing?.source === patch.source && existing.mesh.userData.segments === patch.segments
        );
      })
    )
      return;
    const stitched = stitchPatches(layout, this.stitchCache, this.meshVertex.bind(this));
    this.stitchCache = stitched.cache;
    const required = new Map(stitched.patches.map((patch) => [patch.key, patch]));
    for (const [key, tile] of this.patchTiles) {
      const patch = required.get(key);
      if (
        !patch ||
        patch.source !== tile.source ||
        patch.segments !== tile.mesh.userData.segments ||
        patch.edgeBindings.some(
          (binding, index) =>
            binding.line !== tile.edgeBindings[index].line ||
            binding.start !== tile.edgeBindings[index].start ||
            binding.end !== tile.edgeBindings[index].end,
        )
      ) {
        this.dispose(tile);
        this.patchTiles.delete(key);
        this.renderDirty = true;
      }
    }
    for (const patch of stitched.patches) {
      if (this.patchTiles.has(patch.key)) continue;
      const tile = this.createTile({
        ...patch,
        level: patch.source.level,
        values: patch.source.values,
        vertexValues: patch.source.vertexValues,
      });
      tile.mesh.visible = true;
      tile.mesh.updateMatrixWorld();
      this.patchTiles.set(patch.key, tile);
      this.renderDirty = true;
    }
    const previousKey = this.selectionKey;
    this.updateCamera();
    if (previousKey !== this.selectionKey) this.onChange();
  }

  sample(direction, tiles, coordinate = vectorFace(direction)) {
    this.sampleIndexes ||= new WeakMap();
    let faces = this.sampleIndexes.get(tiles);
    if (!faces) {
      faces = Array.from({ length: TerrainHiGAN.globe.FACE_COUNT }, () => []);
      for (const tile of tiles) {
        faces[tile.face][tile.depth] ||= new Map();
        faces[tile.face][tile.depth].set(tile.tileY * 2 ** tile.depth + tile.tileX, tile);
      }
      this.sampleIndexes.set(tiles, faces);
    }
    const levels = faces[coordinate.face];
    for (let depth = levels.length - 1; depth >= 0; depth--) {
      if (!levels[depth]) continue;
      const side = 2 ** depth;
      const east = coordinate.east * side,
        north = coordinate.north * side;
      const tileX = Math.min(side - 1, Math.floor(east)),
        tileY = Math.min(side - 1, Math.floor(north));
      const tile = levels[depth].get(tileY * side + tileX);
      if (tile) return this.sampleTile(tile, (east - tileX) * 64, (north - tileY) * 64);
    }
    return null;
  }

  sampleTile(tile, east, north) {
    const left = Math.floor(east - 0.5);
    const top = Math.floor(north - 0.5);
    const fractionEast = east - 0.5 - left,
      fractionNorth = north - 0.5 - top;
    const index = (top + 2) * 68 + left + 2;
    return elevationMeters(
      (tile.values[index] * (1 - fractionEast) + tile.values[index + 1] * fractionEast) *
        (1 - fractionNorth) +
        (tile.values[index + 68] * (1 - fractionEast) + tile.values[index + 69] * fractionEast) *
          fractionNorth,
    );
  }

  measure(start, end, preview) {
    if (!start && !this.measurement.children.length) return;
    this.renderDirty = true;
    for (const child of [...this.measurement.children]) {
      this.measurement.remove(child);
      child.geometry.dispose();
      child.material.dispose();
    }
    if (!start) return;
    const clearance = this.view().clearance;
    const lift = Math.max(0.02, clearance * 0.001);
    const radiusAt = (point) => 1 + (this.surfaceAt(point) + lift) / RADIUS;
    if (end) {
      const focal =
        this.canvas.height / (2 * Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2)));
      const tolerance = Math.min(lift / 2, clearance / (2 * focal));
      const segments = Math.max(
        1,
        Math.ceil(greatCircle(start, end) / Math.sqrt(8 * RADIUS * tolerance)),
      );
      const points = arcPoints(start, end, segments).map((point) =>
        new THREE.Vector3(...point).multiplyScalar(radiusAt(point)),
      );
      const material = this.depthMaterial(
        preview
          ? new THREE.LineDashedMaterial({ color: 0xffe07a, dashSize: 0.01, gapSize: 0.006 })
          : new THREE.LineBasicMaterial({ color: 0xffe07a }),
      );
      const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(points), material);
      line.computeLineDistances();
      this.measurement.add(line);
    }
    for (const point of end ? [start, end] : [start]) {
      const marker = new THREE.Mesh(
        new THREE.SphereGeometry(Math.min(0.006, (this.view().clearance / RADIUS) * 0.006), 12, 8),
        this.depthMaterial(new THREE.MeshBasicMaterial({ color: 0xffe07a })),
      );
      marker.position.set(...point).multiplyScalar(radiusAt(point));
      this.measurement.add(marker);
    }
  }

  draw() {
    if (!this.renderDirty && this.lastDrawKey === this.selectionKey) return;
    this.renderDirty = false;
    this.lastDrawKey = this.selectionKey;
    this.depthRange ||= { value: new THREE.Vector2() };
    this.depthRange.value.set(this.camera.near, 1 / Math.log2(this.camera.far / this.camera.near));
    const radial = this.upAt(this.camera.position);
    const right = new THREE.Vector3().setFromMatrixColumn(this.camera.matrixWorld, 0);
    right.addScaledVector(radial, -right.dot(radial)).normalize();
    const tangentUp = new THREE.Vector3().crossVectors(radial, right).normalize();
    this.sun.position
      .copy(radial)
      .multiplyScalar(Math.SQRT1_2)
      .addScaledVector(tangentUp, 0.5)
      .addScaledVector(right, -0.5);
    this.updateStars();
    this.renderer.render(this.scene, this.camera);
  }
}

TerrainHiGAN.globe.GlobeRenderer = GlobeRenderer;
