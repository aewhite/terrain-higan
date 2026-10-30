import * as THREE from "three";

const {
  RADIUS,
  greatCircle,
  horizonAngle,
  horizonDistance,
  vectorFace,
  canonicalPixel,
  faceVector,
  stitchPatches,
} = TerrainHiGAN.globe;
const { clamp } = TerrainHiGAN;
const WALK_CEILING = 20000;
const WALK_RADIUS = 20000;
const WALK_VIEW_RADIUS = 500000;
const WALK_FINE_RADIUS = 128;
const WALK_MAX_LEVEL = 14;
const EYE_HEIGHT = 2;
const WALK_MAX_SPEED = 50;
const MOVEMENT_KEYS = new Set([
  "KeyW",
  "KeyA",
  "KeyS",
  "KeyD",
  "ArrowUp",
  "ArrowLeft",
  "ArrowDown",
  "ArrowRight",
]);

class GlobeWalk {
  constructor(renderer, controller, onChange) {
    Object.assign(this, { renderer, controller, onChange });
    this.active = false;
    this.paused = true;
    this.prepared = false;
    this.waiting = true;
    this.requestedSpeed = 1.4;
    this.speed = 0;
    this.aboveGround = EYE_HEIGHT;
    this.heading = new THREE.Vector3();
    this.mouse = { x: 0, y: 0 };
    this.mouseBound = true;
    this.keys = new Set();
    this.pitch = 0;
    this.frame = 0;
    this.status = "Paused";
    renderer.walk = this;
  }

  enter(pose = {}) {
    if ((pose.altitude ?? this.renderer.view().altitude) >= WALK_CEILING) return false;
    this.anchor = pose.anchor
      ? new THREE.Vector3(...pose.anchor).normalize()
      : this.renderer.camera.position.clone().normalize();
    if (
      greatCircle(
        this.anchor.toArray(),
        this.renderer.camera.position.clone().normalize().toArray(),
      ) > WALK_RADIUS
    )
      return false;
    const radial = this.renderer.camera.position.clone().normalize();
    this.heading.copy(pose.heading ? new THREE.Vector3(...pose.heading) : this.renderer.camera.up);
    this.heading.addScaledVector(radial, -this.heading.dot(radial));
    if (this.heading.lengthSq() < 0.5) {
      this.heading.set(0, 1, 0);
      if (Math.abs(radial.y) > 0.99) this.heading.set(1, 0, 0);
      this.heading.addScaledVector(radial, -this.heading.dot(radial));
    }
    this.heading.normalize();
    this.pitch = clamp(pose.pitch || 0, -1.4, 1.4);
    this.previousRelief = this.renderer.exaggeration;
    this.active = true;
    this.paused = false;
    this.clearInput();
    this.mouseBound = true;
    this.selection = null;
    this.renderer.restyle(this.controller.tiles.values(), this.renderer.map, 1);
    this.updateCamera();
    this.onChange();
    return true;
  }

  resetMeshes() {
    this.prepared = false;
    this.waiting = true;
    this.speed = 0;
    this.build = null;
    this.patchIndex = null;
    this.horizonPosition = null;
    this.status = "Preparing terrain";
  }

  reset() {
    if (!this.active) return;
    this.pause();
    this.selection = null;
    this.renderer.restyle(this.controller.tiles.values(), this.renderer.map, 1);
  }

  exit() {
    this.pause();
    this.active = false;
    this.resetMeshes();
    this.status = "Paused";
    this.selection = null;
    const renderer = this.renderer;
    renderer.pendingAltitude = null;
    renderer.clearanceTarget = null;
    const radial = renderer.camera.position.clone().normalize();
    renderer.camera.up
      .copy(this.heading)
      .addScaledVector(radial, -this.heading.dot(radial))
      .normalize();
    renderer.restyle(this.controller.tiles.values(), renderer.map, this.previousRelief);
    renderer.updateCamera();
    renderer.renderer?.setClearColor(0x08151a);
    this.onChange();
  }

  pause() {
    this.paused = true;
    this.speed = 0;
    this.clearInput();
    if (this.prepared) this.status = "Paused";
    if (this.frame) cancelAnimationFrame(this.frame);
    this.frame = 0;
    this.onChange();
  }

  resume() {
    if (!this.active || !this.prepared || this.controller.failed) return;
    this.paused = false;
    this.status = "Ready";
    this.wake();
    this.onChange();
  }

  wake() {
    if (this.frame || !this.active || this.paused || !this.prepared || this.controller.failed)
      return;
    let previous = performance.now();
    const animate = (now) => {
      this.frame = 0;
      if (!this.active || this.paused) return;
      const elapsed = (now - previous) / 1000;
      const key = this.renderer.selectionKey,
        status = this.status,
        speed = this.speed;
      this.tick(Math.min(0.05, elapsed));
      this.speed *= Math.min(1, 0.05 / Math.max(0.001, elapsed));
      previous = now;
      if (key !== this.renderer.selectionKey || status !== this.status || speed !== this.speed)
        this.onChange();
      if (this.active && !this.paused && Object.values(this.input()).some(Boolean))
        this.frame = requestAnimationFrame(animate);
    };
    if (!this.frame) this.frame = requestAnimationFrame(animate);
  }

  clearInput() {
    this.keys.clear();
    this.mouse.x = this.mouse.y = 0;
  }

  input() {
    const held = (...codes) => codes.some((code) => this.keys.has(code));
    const steering = (value) => (this.mouseBound && Math.abs(value) >= 0.06 ? value : 0);
    return {
      forward: Number(held("KeyW", "ArrowUp")) - Number(held("KeyS", "ArrowDown")),
      sideways: Number(held("KeyD", "ArrowRight")) - Number(held("KeyA", "ArrowLeft")),
      yaw: steering(this.mouse.x),
      pitch: steering(this.mouse.y),
    };
  }

  setKey(code, pressed) {
    if (!this.active || !MOVEMENT_KEYS.has(code)) return false;
    if (pressed) {
      if (!this.prepared || this.controller.failed || this.keys.has(code)) return true;
      this.keys.add(code);
      if (this.paused) this.resume();
      else this.wake();
    } else {
      this.keys.delete(code);
      this.wake();
    }
    return true;
  }

  toggleMouse() {
    if (!this.active) return;
    this.mouseBound = !this.mouseBound;
    this.mouse.x = this.mouse.y = 0;
    this.wake();
    this.onChange();
  }

  bindInput(canvas, signal) {
    canvas.tabIndex = 0;
    const listen = (target, type, action) => target.addEventListener(type, action, { signal });
    const editing = (target) =>
      target.closest("input, textarea, select, [contenteditable]:not([contenteditable=false])");
    listen(window, "keydown", (event) => {
      if (
        !this.active ||
        event.defaultPrevented ||
        editing(event.target) ||
        event.ctrlKey ||
        event.metaKey ||
        event.altKey ||
        document.getElementById("shortcuts-dialog").open
      )
        return;
      if (!MOVEMENT_KEYS.has(event.code)) return;
      event.preventDefault();
      if (!event.repeat || this.keys.has(event.code)) this.setKey(event.code, true);
    });
    listen(window, "keyup", (event) => this.setKey(event.code, false));
    listen(document, "focusin", (event) => {
      if (this.active && editing(event.target)) this.pause();
    });
    listen(canvas, "pointermove", (event) => {
      if (!this.active || !this.mouseBound) return;
      this.mouse.x = (2 * event.clientX) / innerWidth - 1;
      this.mouse.y = (2 * event.clientY) / innerHeight - 1;
      this.wake();
    });
    listen(canvas, "pointerleave", () => {
      this.mouse.x = this.mouse.y = 0;
      this.wake();
    });
    listen(canvas, "click", (event) => {
      if (this.active && event.button === 0) {
        canvas.focus({ preventScroll: true });
        this.toggleMouse();
      }
    });
  }

  adjustSpeed(factor) {
    this.requestedSpeed = clamp(this.requestedSpeed * factor, 0.1, WALK_MAX_SPEED);
    this.onChange();
  }

  distance(bounds) {
    const center = greatCircle(this.anchor.toArray(), bounds.center.toArray());
    const extent = 2 * Math.asin(Math.min(1, bounds.radius / 2)) * RADIUS;
    return { center, extent, nearest: Math.max(0, center - extent) };
  }

  visible(bounds) {
    return this.distance(bounds).nearest <= this.sceneryRadius();
  }

  sceneryRadius() {
    return Math.max(WALK_VIEW_RADIUS, WALK_RADIUS + horizonDistance(WALK_CEILING, WALK_CEILING));
  }

  cacheHorizon(tile) {
    const positions = tile.mesh.geometry.getAttribute("position");
    const count = (tile.mesh.userData.segments + 1) ** 2;
    const point = new THREE.Vector3();
    let minimumHeight = Infinity,
      maximumHeight = -Infinity;
    for (let index = 0; index < count; index++) {
      point.fromBufferAttribute(positions, index).add(tile.mesh.position);
      const height = (point.length() - 1) * RADIUS;
      minimumHeight = Math.min(minimumHeight, height);
      maximumHeight = Math.max(maximumHeight, height);
    }
    const extent = 2 * Math.asin(Math.min(1, tile.bounds.radius / 2));
    const envelope = (height) => {
      const angle = Math.min(Math.PI, extent + horizonAngle(height));
      return { cosine: Math.cos(angle), sine: Math.sin(angle) };
    };
    tile.horizon = {
      minimumHeight,
      maximumHeight,
      terrain: envelope(maximumHeight),
      sea: envelope(0),
    };
  }

  updateVisibility() {
    if (!this.prepared) return;
    const renderer = this.renderer,
      position = renderer.camera.position;
    if (this.horizonPosition?.equals(position)) return;
    this.horizonPosition ||= new THREE.Vector3();
    this.horizonPosition.copy(position);
    const observerHeight = (position.length() - 1) * RADIUS;
    const observer = position.clone().normalize();
    const angle = horizonAngle(observerHeight);
    const cosine = Math.cos(angle),
      sine = Math.sin(angle);
    for (const tile of this.patchIndex.values()) {
      const facing = observer.dot(tile.bounds.center);
      const visible = (envelope) =>
        facing >= cosine * envelope.cosine - sine * envelope.sine - 1e-12;
      const terrainVisible =
        observerHeight < 0 || tile.horizon.minimumHeight < 0 || visible(tile.horizon.terrain);
      const seaVisible = observerHeight < 0 || visible(tile.horizon.sea);
      if (tile.mesh.visible !== terrainVisible || tile.waterMesh.visible !== seaVisible)
        renderer.renderDirty = true;
      tile.mesh.visible = terrainVisible;
      tile.waterMesh.visible = seaVisible;
    }
  }

  bounds(bounds) {
    return bounds;
  }

  roots() {
    return Array.from({ length: 6 }, (_, face) => this.controller.node(face, 0, 0, 0));
  }

  maxDepth() {
    return WALK_MAX_LEVEL - 2;
  }

  select() {
    const required = new Map(),
      expanded = new Set(),
      leaves = [];
    const sceneryRadius = this.sceneryRadius();
    const visit = (node) => {
      const distance = this.distance(node.bounds);
      if (distance.nearest > sceneryRadius) return;
      required.set(node.key, node);
      const span = distance.extent * 2;
      const spanLimit =
        distance.nearest <= WALK_RADIUS
          ? Math.max(WALK_FINE_RADIUS, distance.nearest) * 0.5
          : Math.max(1024, distance.nearest * 0.5);
      if (
        node.depth < this.maxDepth() &&
        (distance.nearest <= WALK_FINE_RADIUS || span > spanLimit)
      ) {
        expanded.add(node.key);
        for (let north = 0; north < 2; north++)
          for (let east = 0; east < 2; east++)
            visit(
              this.controller.node(
                node.face,
                node.depth + 1,
                node.tileX * 2 + east,
                node.tileY * 2 + north,
              ),
            );
      } else leaves.push(node);
    };
    for (const root of this.roots()) visit(root);
    const ordered = [...leaves].sort(
      (first, second) => this.distance(first.bounds).center - this.distance(second.bounds).center,
    );
    const viewRadius = leaves.reduce((radius, node) => {
      const distance = this.distance(node.bounds);
      return Math.max(radius, distance.center + distance.extent);
    }, 0);
    const selection = {
      key: this.anchor.toArray().join(":"),
      required,
      expanded,
      leaves,
      ordered,
      viewRadius,
    };
    const controller = this.controller;
    controller.nodeCapacity = Math.max(4096, required.size * 4);
    for (const [key, node] of controller.nodes) {
      if (controller.nodes.size <= controller.nodeCapacity) break;
      if (required.has(key)) continue;
      delete node.children;
      controller.nodes.delete(key);
    }
    for (const node of controller.nodes.values())
      if (node.children?.some((child) => !controller.nodes.has(child.key))) delete node.children;
    return selection;
  }

  prepare() {
    const controller = this.controller,
      renderer = this.renderer;
    if (!this.selection) this.selection = this.select();
    controller.selection = this.selection;
    if (controller.active && !this.selection.required.has(controller.active.key)) {
      controller.send({ type: "cancel" });
      controller.active = null;
    }
    controller.sequence = this.selection.ordered.filter((node) => !controller.tiles.has(node.key));
    controller.capacity = Math.max(512, this.selection.required.size + 128);
    for (const [key, tile] of controller.tiles) {
      if (controller.tiles.size <= controller.capacity) break;
      if (this.selection.required.has(key)) continue;
      // Keep the globe's six coarse faces for a complete sphere on orbit exit.
      if (tile.depth === 0 && tile.level === 2) continue;
      renderer.dispose(tile);
      controller.tiles.delete(key);
      controller.tilesVersion++;
    }
    if (controller.sequence.length) {
      this.status = `Preparing terrain ${this.selection.leaves.length - controller.sequence.length}/${this.selection.leaves.length}`;
    } else if (!this.prepared) {
      if (!this.build) {
        const patches = this.selection.leaves.map((node) => ({
          ...node,
          source: controller.tiles.get(node.key),
          segments: renderer.meshSegments({ ...node, level: node.depth + 2 }),
        }));
        const stitched = stitchPatches(
          patches,
          undefined,
          renderer.meshVertex.bind(renderer),
          this.stitchTopology,
        );
        this.build = { patches: stitched.patches, index: 0 };
        renderer.stitchCache = stitched.cache;
        renderer.patchTiles = new Map();
        renderer.displayTiles = patches.map((patch) => patch.source);
      }
      const started = performance.now();
      while (this.build.index < this.build.patches.length && performance.now() - started < 8) {
        const patch = this.build.patches[this.build.index++];
        const tile = renderer.createTile({
          ...patch,
          level: patch.source.level,
          values: patch.source.values,
          vertexValues: patch.source.vertexValues,
        });
        this.cacheHorizon(tile);
        tile.mesh.updateMatrixWorld();
        if (tile.waterMesh) tile.waterMesh.visible = false;
        renderer.patchTiles.set(tile.key, tile);
      }
      this.status = `Building scene ${this.build.index}/${this.build.patches.length}`;
      if (this.build.index === this.build.patches.length) {
        this.patchIndex = renderer.patchTiles;
        this.depths = [...new Set(this.selection.leaves.map((node) => node.depth))].sort(
          (first, second) => second - first,
        );
        for (const tile of this.patchIndex.values()) {
          tile.mesh.visible = true;
          if (tile.waterMesh) tile.waterMesh.visible = true;
        }
        this.prepared = true;
        this.waiting = false;
        this.protect();
        this.updateCamera();
        this.status = this.paused ? "Paused" : "Ready";
        this.wake();
        renderer.renderDirty = true;
      }
      this.onChange();
    }
    controller.display = this.prepared ? renderer.displayTiles : [];
    controller.patches = this.build?.patches || [];
    const building = this.build ? this.build.patches.length - this.build.index : 0;
    const stateKey = `${controller.tilesVersion}:${building}:${this.prepared}`;
    if (stateKey !== this.stateKey) {
      this.stateKey = stateKey;
      controller.onState({
        ready: this.prepared,
        pending: controller.sequence.length + building,
        visible: renderer.patchTiles?.size || 0,
        cached: controller.tiles.size,
        capacity: controller.capacity,
        targetLevel: WALK_MAX_LEVEL,
        samplePixels: 0,
      });
    }
    controller.dispatch();
  }

  patchAt(direction) {
    const coordinate = vectorFace(direction.toArray());
    for (const depth of this.depths) {
      const side = 2 ** depth;
      const east = Math.min(side - 1, Math.floor(coordinate.east * side));
      const north = Math.min(side - 1, Math.floor(coordinate.north * side));
      const tile = this.patchIndex.get(`${coordinate.face}:${depth}:${east}:${north}`);
      if (tile) return tile;
    }
    return null;
  }

  groundAt(position) {
    if (!this.prepared) return null;
    const direction = position.clone().normalize();
    const patch = this.patchAt(direction);
    if (!patch) return null;
    const coordinate = vectorFace(direction.toArray());
    const side = 2 ** patch.depth,
      segments = patch.mesh.userData.segments;
    const east = Math.min(
      segments - 1,
      Math.floor((coordinate.east * side - patch.tileX) * segments),
    );
    const north = Math.min(
      segments - 1,
      Math.floor((coordinate.north * side - patch.tileY) * segments),
    );
    this.groundRay ||= new THREE.Ray();
    this.groundVertices ||= [
      new THREE.Vector3(),
      new THREE.Vector3(),
      new THREE.Vector3(),
      new THREE.Vector3(),
    ];
    this.groundTriangle ||= new THREE.Triangle();
    this.groundPlane ||= new THREE.Plane();
    this.groundBarycentric ||= new THREE.Vector3();
    const [first, second, third, hit] = this.groundVertices;
    const ray = this.groundRay;
    ray.origin.copy(direction).multiplyScalar(2).sub(patch.mesh.position);
    ray.direction.copy(direction).negate();
    const geometry = patch.mesh.geometry,
      positions = geometry.getAttribute("position"),
      indices = geometry.index;
    let radius = -Infinity;
    for (let row = Math.max(0, north - 1); row <= Math.min(segments - 1, north + 1); row++)
      for (let column = Math.max(0, east - 1); column <= Math.min(segments - 1, east + 1); column++)
        for (let triangle = 0; triangle < 2; triangle++) {
          const offset = (row * segments + column) * 6 + triangle * 3;
          first.fromBufferAttribute(positions, indices.getX(offset));
          second.fromBufferAttribute(positions, indices.getX(offset + 1));
          third.fromBufferAttribute(positions, indices.getX(offset + 2));
          let intersection = ray.intersectTriangle(first, second, third, false, hit);
          if (!intersection) {
            this.groundTriangle.set(first, second, third).getPlane(this.groundPlane);
            if (ray.intersectPlane(this.groundPlane, hit)) {
              this.groundTriangle.getBarycoord(hit, this.groundBarycentric);
              if (
                Math.min(
                  this.groundBarycentric.x,
                  this.groundBarycentric.y,
                  this.groundBarycentric.z,
                ) >= -0.0001
              )
                intersection = hit;
            }
          }
          if (intersection) radius = Math.max(radius, hit.add(patch.mesh.position).length());
        }
    if (radius !== -Infinity) return (radius - 1) * RADIUS;
    const candidates = new Set();
    for (let row = -1; row <= 1; row++)
      for (let column = -1; column <= 1; column++) {
        const neighbor = canonicalPixel(patch.face, patch.tileX + column, patch.tileY + row, side);
        const tile = this.patchAt(
          new THREE.Vector3(
            ...faceVector(
              neighbor.face,
              (neighbor.east + 0.5) / side,
              (neighbor.north + 0.5) / side,
            ),
          ),
        );
        if (tile) candidates.add(tile.mesh);
      }
    const raycaster = this.renderer.raycaster;
    raycaster.set(direction.clone().multiplyScalar(2), direction.clone().negate());
    const intersection = raycaster.intersectObjects([...candidates], false)[0];
    return intersection ? (intersection.point.length() - 1) * RADIUS : null;
  }

  protect() {
    const height = this.groundAt(this.renderer.camera.position);
    this.waiting = height === null;
    if (height === null) {
      this.speed = 0;
      return false;
    }
    this.placeAtHeight(this.renderer.camera.position, height + EYE_HEIGHT);
    this.aboveGround = EYE_HEIGHT;
    if (this.paused) this.status = "Paused";
    return true;
  }

  ahead(distance, direction = this.heading) {
    const position = this.renderer.camera.position;
    const radial = position.clone().normalize();
    const angle = distance / (position.length() * RADIUS);
    return radial
      .multiplyScalar(Math.cos(angle))
      .addScaledVector(direction, Math.sin(angle))
      .normalize();
  }

  placeAtHeight(position, height) {
    return position.setLength(1 + height / RADIUS);
  }

  distanceFromAnchor(position) {
    return greatCircle(this.anchor.toArray(), position.toArray());
  }

  mapFrame() {
    const up = this.anchor.clone();
    const north = new THREE.Vector3(0, 1, 0).addScaledVector(up, -up.y);
    if (north.lengthSq() < 1e-12) north.set(1, 0, 0).addScaledVector(up, -up.x);
    north.normalize();
    return {
      up,
      north,
      east: new THREE.Vector3().crossVectors(north, up).normalize(),
      origin: up.clone(),
    };
  }

  mapPosition(east, south) {
    const frame = this.minimapFrame;
    return frame.origin
      .clone()
      .addScaledVector(frame.east, east / RADIUS)
      .addScaledVector(frame.north, -south / RADIUS)
      .normalize();
  }

  mapHeight(position) {
    const tile = this.patchAt(position);
    if (!tile) return null;
    const coordinate = vectorFace(position.toArray());
    return this.renderer.vertexHeight(
      tile.source,
      (coordinate.east * 2 ** tile.depth - tile.tileX) * 64,
      (coordinate.north * 2 ** tile.depth - tile.tileY) * 64,
    );
  }

  exportFrame() {
    const frame = this.mapFrame();
    const height = this.groundAt(this.anchor);
    frame.origin.multiplyScalar(1 + height / RADIUS);
    return {
      ...frame,
      metadata: {
        topology: TerrainHiGAN.globe.GLOBE_VERSION,
        latitude: THREE.MathUtils.radToDeg(Math.asin(frame.up.y)),
        longitude: THREE.MathUtils.radToDeg(Math.atan2(-frame.up.z, frame.up.x)),
        originAltitudeMeters: height,
      },
    };
  }

  tick(seconds) {
    if (!this.active || this.paused) return;
    const radial = this.renderer.upAt(this.renderer.camera.position);
    const input = this.input();
    this.heading.applyAxisAngle(radial, -input.yaw * seconds);
    this.pitch = clamp(this.pitch - input.pitch * seconds, -1.4, 1.4);
    if (!this.protect()) {
      this.updateCamera();
      return;
    }
    const direction = this.heading
      .clone()
      .multiplyScalar(input.forward)
      .addScaledVector(new THREE.Vector3().crossVectors(this.heading, radial), input.sideways);
    const moving = direction.lengthSq() > 0;
    direction.normalize();
    const budget = moving
      ? clamp(this.requestedSpeed, 0, WALK_MAX_SPEED) * Math.min(seconds, 0.05)
      : 0;
    this.speed = 0;
    this.status = budget > 0 ? "Walking" : "Ready";
    if (budget > 0) {
      let low = 0,
        high = budget,
        accepted = null;
      for (let attempt = 0; attempt < 12; attempt++) {
        const distance = attempt ? (low + high) / 2 : high;
        const next = this.ahead(distance, direction);
        if (this.distanceFromAnchor(next) > WALK_RADIUS) {
          this.status = "Scene boundary";
          break;
        }
        const height = this.groundAt(next);
        if (height === null) {
          this.status = "Scene boundary";
          break;
        }
        this.placeAtHeight(next, height + EYE_HEIGHT);
        const traveled = next.distanceTo(this.renderer.camera.position) * RADIUS;
        if (traveled <= budget + 1e-7) {
          low = distance;
          accepted = next;
          if (!attempt) break;
        } else high = distance;
      }
      if (accepted) {
        this.speed =
          (accepted.distanceTo(this.renderer.camera.position) * RADIUS) / Math.max(seconds, 0.001);
        this.heading.applyQuaternion(
          new THREE.Quaternion().setFromUnitVectors(radial, this.renderer.upAt(accepted)),
        );
        this.renderer.camera.position.copy(accepted);
      }
    }
    this.updateCamera();
  }

  updateCamera() {
    const renderer = this.renderer,
      camera = renderer.camera;
    const radial = renderer.upAt(camera.position);
    camera.up.copy(radial);
    camera.lookAt(
      camera.position
        .clone()
        .add(
          this.heading
            .clone()
            .multiplyScalar(Math.cos(this.pitch))
            .addScaledVector(radial, Math.sin(this.pitch)),
        ),
    );
    renderer.sphere.radius = 1;
    camera.near = 0.05 / RADIUS;
    camera.far =
      ((this.selection?.viewRadius ?? this.sceneryRadius()) + WALK_RADIUS + WALK_CEILING * 2) /
      RADIUS;
    camera.updateProjectionMatrix();
    camera.updateMatrixWorld();
    renderer.viewProjection ||= new THREE.Matrix4();
    renderer.viewProjection.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    renderer.frustum.setFromProjectionMatrix(renderer.viewProjection);
    renderer.selectionKey = [
      ...camera.position.toArray(),
      ...camera.quaternion.toArray(),
      renderer.canvas.width,
      renderer.canvas.height,
      "walk",
    ].join(":");
    renderer.renderer?.setClearColor(renderer.view().altitude < 0 ? 0x103b4b : 0x9bbfd2);
    this.updateVisibility();
  }
}

Object.assign(TerrainHiGAN.globe, {
  GlobeWalk,
  WALK_CEILING,
  WALK_RADIUS,
  WALK_VIEW_RADIUS,
  WALK_FINE_RADIUS,
  WALK_MAX_LEVEL,
  WALK_MAX_SPEED,
});
