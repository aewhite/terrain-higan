import * as THREE from "three";

const {
  GlobeRenderer,
  GlobeWalk,
  GlobeTileController,
  WalkMinimap,
  RADIUS,
  WALK_VIEW_RADIUS,
  WALK_MAX_LEVEL,
  WALK_MAX_SPEED,
  downloadWalkingScene,
} = TerrainHiGAN.globe;
const { metersPerPixel } = TerrainHiGAN;

class PlanarRenderer extends GlobeRenderer {
  constructor(canvas, onChange, origin) {
    super(canvas, onChange, false);
    this.origin = origin;
    this.camera.position.set(0, 2 / RADIUS, 0);
  }

  point(tile, column, row) {
    const spacing = metersPerPixel(tile.depth);
    return new THREE.Vector3(
      ((tile.tileX * 64 + column) * spacing - this.origin.x) / RADIUS,
      this.vertexHeight(tile.source || tile, column, row) / RADIUS,
      ((tile.tileY * 64 + row) * spacing - this.origin.y) / RADIUS,
    );
  }

  tileOrigin(tile) {
    const span = 64 * metersPerPixel(tile.depth);
    return new THREE.Vector3(
      ((tile.tileX + 0.5) * span - this.origin.x) / RADIUS,
      0,
      ((tile.tileY + 0.5) * span - this.origin.y) / RADIUS,
    );
  }

  surfaceNormal(point, eastPlus, eastMinus, southPlus, southMinus, target = new THREE.Vector3()) {
    return target
      .crossVectors(southPlus.clone().sub(southMinus), eastPlus.clone().sub(eastMinus))
      .normalize();
  }

  createTile(tile) {
    super.createTile(tile);
    const indices = tile.mesh.geometry.index.array.slice();
    for (let offset = 0; offset < indices.length; offset += 3) {
      const second = indices[offset + 1];
      indices[offset + 1] = indices[offset + 2];
      indices[offset + 2] = second;
    }
    tile.mesh.geometry.setIndex(new THREE.BufferAttribute(indices, 1));
    return tile;
  }

  skirtPoint(point, depth) {
    point.y -= Math.max(3, metersPerPixel(depth) * 2) / RADIUS;
    return point;
  }

  seaPoint(point) {
    point.y = 0;
    return point;
  }

  upAt() {
    return new THREE.Vector3(0, 1, 0);
  }

  bounds(tile) {
    const sphere = new THREE.Sphere(
      this.tileOrigin(tile),
      (Math.SQRT2 * 32 * metersPerPixel(tile.depth)) / RADIUS,
    );
    return sphere;
  }

  prepareView() {
    this.walk.protect();
    this.walk.updateCamera();
  }

  updateCamera() {
    this.walk?.updateCamera();
  }

  view() {
    return { altitude: this.camera.position.y * RADIUS, clearance: 2 };
  }
}

class PlanarWalk extends GlobeWalk {
  constructor(...args) {
    super(...args);
    this.stitchTopology = {
      canonicalPixel: (face, east, north) => ({ face, east, north }),
      cornerKey: (face, east, north) =>
        `${Math.round(east * 2 ** 26)}:${Math.round(north * 2 ** 26)}`,
    };
  }

  enter() {
    this.anchor = new THREE.Vector3();
    this.heading.set(0, 0, -1);
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

  exit() {
    this.pause();
    this.active = false;
    this.resetMeshes();
    this.onChange();
  }

  roots() {
    const span = 64 * metersPerPixel(0),
      roots = [];
    const { x, y } = this.renderer.origin;
    for (
      let row = Math.floor((y - WALK_VIEW_RADIUS) / span);
      row <= Math.floor((y + WALK_VIEW_RADIUS) / span);
      row++
    )
      for (
        let column = Math.floor((x - WALK_VIEW_RADIUS) / span);
        column <= Math.floor((x + WALK_VIEW_RADIUS) / span);
        column++
      )
        roots.push(this.controller.node(0, 0, column, row));
    return roots;
  }

  maxDepth() {
    return WALK_MAX_LEVEL;
  }

  sceneryRadius() {
    return WALK_VIEW_RADIUS;
  }

  cacheHorizon() {}

  updateVisibility() {}

  distance(bounds) {
    const center = Math.hypot(bounds.center.x, bounds.center.z) * RADIUS;
    const extent = bounds.radius * RADIUS;
    return { center, extent, nearest: Math.max(0, center - extent) };
  }

  distanceFromAnchor(position) {
    return Math.hypot(position.x, position.z) * RADIUS;
  }

  ahead(distance, direction = this.heading) {
    return this.renderer.camera.position.clone().addScaledVector(direction, distance / RADIUS);
  }

  placeAtHeight(position, height) {
    position.y = height / RADIUS;
    return position;
  }

  coordinate(position, depth) {
    const span = 64 * metersPerPixel(depth);
    return {
      east: (this.renderer.origin.x + position.x * RADIUS) / span,
      south: (this.renderer.origin.y + position.z * RADIUS) / span,
    };
  }

  patchAt(position) {
    for (const depth of this.depths) {
      const { east, south } = this.coordinate(position, depth);
      const tile = this.patchIndex.get(`0:${depth}:${Math.floor(east)}:${Math.floor(south)}`);
      if (tile) return tile;
    }
    return null;
  }

  groundAt(position) {
    if (!this.prepared) return null;
    const tile = this.patchAt(position);
    if (!tile) return null;
    const { east, south } = this.coordinate(position, tile.depth);
    const segments = tile.mesh.userData.segments;
    const column = (east - tile.tileX) * segments,
      row = (south - tile.tileY) * segments;
    const left = Math.min(segments - 1, Math.max(0, Math.floor(column)));
    const top = Math.min(segments - 1, Math.max(0, Math.floor(row)));
    const fractionEast = column - left,
      fractionSouth = row - top;
    const vertex = top * (segments + 1) + left;
    const positions = tile.mesh.geometry.getAttribute("position");
    const heights = [vertex, vertex + 1, vertex + segments + 1, vertex + segments + 2].map(
      (index) => positions.getY(index),
    );
    return (
      (fractionEast + fractionSouth <= 1
        ? heights[0] * (1 - fractionEast - fractionSouth) +
          heights[1] * fractionEast +
          heights[2] * fractionSouth
        : heights[1] * (1 - fractionSouth) +
          heights[2] * (1 - fractionEast) +
          heights[3] * (fractionEast + fractionSouth - 1)) * RADIUS
    );
  }

  mapFrame() {
    return {
      origin: this.anchor.clone(),
      up: new THREE.Vector3(0, 1, 0),
      north: new THREE.Vector3(0, 0, -1),
      east: new THREE.Vector3(1, 0, 0),
    };
  }

  mapPosition(east, south) {
    return new THREE.Vector3(east / RADIUS, 0, south / RADIUS);
  }

  mapHeight(position) {
    const tile = this.patchAt(position);
    if (!tile) return null;
    const { east, south } = this.coordinate(position, tile.depth);
    return this.renderer.vertexHeight(
      tile.source,
      (east - tile.tileX) * 64,
      (south - tile.tileY) * 64,
    );
  }

  exportFrame() {
    const frame = this.mapFrame();
    const height = this.groundAt(this.anchor);
    frame.origin.y = height / RADIUS;
    return {
      ...frame,
      metadata: {
        topology: "planar",
        originX: this.renderer.origin.x,
        originY: this.renderer.origin.y,
        originAltitudeMeters: height,
      },
    };
  }
}

class PlanarWalkView {
  constructor(state, onExit) {
    const element = (id) => document.getElementById(id);
    this.canvas = document.createElement("canvas");
    this.canvas.id = "walk-world";
    this.canvas.setAttribute("aria-label", "Walk over the planar terrain");
    document.getElementById("world").after(this.canvas);
    const queue = () => {
      if (this.disposed || this.frame) return;
      this.frame = requestAnimationFrame(() => {
        this.frame = 0;
        this.update();
      });
    };
    this.renderer = new PlanarRenderer(this.canvas, queue, { x: state.x, y: state.y });
    this.renderer.map = state.colormap;
    this.controller = new GlobeTileController(
      this.renderer,
      (data) => this.worker.postMessage(data),
      (status) => {
        element("status").textContent =
          status.error || (status.ready ? "Ready" : "Preparing terrain");
        element("render-indicator").dataset.state = status.error
          ? "failed"
          : status.ready
            ? "ready"
            : "rendering";
        element("visible-tiles").textContent =
          `${status.visible || 0} / ${(status.visible || 0) + (status.pending || 0)}`;
        if (status.ready) {
          const levels = this.controller.display.map((tile) => tile.level);
          const low = Math.min(...levels),
            high = Math.max(...levels);
          element("model").textContent =
            low === high ? TerrainHiGAN.modelName(high) : `z${low}–${high}s`;
          Object.assign(this.canvas.dataset, {
            maxModelLevel: high,
            targetModelLevel: status.targetLevel,
          });
        }
        if (status.error) this.walk.pause();
      },
    );
    this.walk = new PlanarWalk(this.renderer, this.controller, queue);
    this.minimap = new WalkMinimap(this.walk);
    const options = new URLSearchParams(location.search);
    options.set("bundle", TerrainHiGAN.bundle);
    options.set("surface", "planar");
    this.worker = new Worker(`./globe-worker.js?${options}`);
    this.worker.onmessage = ({ data }) => {
      if (data.type === "bundle-changed") {
        location.reload();
        return;
      }
      this.controller.receive(data, false);
      if (data.metrics) this.canvas.dataset.modelRuns = data.metrics.modelRuns;
      queue();
    };
    this.worker.onerror = (event) => this.controller.fail(event.message || "Terrain worker failed");
    this.events = new AbortController();
    const listen = (target, type, action, options = {}) =>
      target.addEventListener(type, action, { ...options, signal: this.events.signal });
    this.walk.bindInput(this.canvas, this.events.signal);
    listen(
      this.canvas,
      "wheel",
      (event) => {
        event.preventDefault();
        this.walk.adjustSpeed(event.deltaY < 0 ? 1.15 : 1 / 1.15);
      },
      { passive: false },
    );
    listen(element("pause-walk"), "click", () => {
      if (this.walk.paused) this.walk.resume();
      else this.walk.pause();
      queue();
    });
    listen(element("walk-speed"), "input", (event) => {
      this.walk.requestedSpeed = Math.max(0, Math.min(WALK_MAX_SPEED, Number(event.target.value)));
      queue();
    });
    listen(element("export-scene"), "click", async () => {
      this.walk.pause();
      this.exporting = true;
      queue();
      try {
        await downloadWalkingScene(this.walk);
      } catch (error) {
        element("status").textContent = error.message;
      } finally {
        this.exporting = false;
        queue();
      }
    });
    listen(window, "keydown", (event) => {
      if (event.target.matches("input") || element("shortcuts-dialog").open) return;
      if (event.code === "Space") {
        event.preventDefault();
        if (event.repeat) return;
        if (this.walk.paused) this.walk.resume();
        else this.walk.pause();
        queue();
      }
      if (event.key === "Escape") this.walk.pause();
      if (event.key === "+" || event.key === "=") this.walk.adjustSpeed(1.15);
      if (event.key === "-") this.walk.adjustSpeed(1 / 1.15);
      if (event.key === "Home") onExit();
    });
    listen(window, "blur", () => this.walk.pause());
    listen(document, "visibilitychange", () => {
      if (document.hidden) this.walk.pause();
    });
    listen(element("shortcuts-dialog"), "toggle", () => {
      if (element("shortcuts-dialog").open) this.walk.pause();
    });
    listen(element("toggle-shortcuts"), "click", () => this.walk.pause());
    listen(window, "resize", () => {
      this.renderer.resize();
      queue();
    });
    this.controller.seed = state.seed.toString();
    this.walk.enter();
    this.renderer.resize();
    document.body.classList.add("walk-active");
    element("toggle-walk").textContent = "Exit walk";
    element("walk-hint").textContent = "z14s near entry · 20 km walk · 500 km scenery";
    element("walk-controls").hidden = false;
    this.update();
  }

  update() {
    this.controller.update();
    const walk = this.walk,
      element = (id) => document.getElementById(id);
    element("walk-status").textContent = this.controller.failed ? "Terrain failed" : walk.status;
    element("walk-mouse").textContent = walk.mouseBound ? "Bound" : "Free";
    element("walk-requested").textContent = `${walk.requestedSpeed.toFixed(1)} m/s`;
    element("walk-actual").textContent = `${walk.speed.toFixed(1)} m/s`;
    element("walk-ground").textContent = walk.prepared ? "2.0 m" : "Loading";
    element("walk-speed").value = walk.requestedSpeed;
    element("pause-walk").disabled = !walk.prepared || this.controller.failed;
    element("pause-walk").textContent = walk.paused ? "Resume" : "Pause";
    element("export-scene").disabled = !walk.prepared || this.exporting;
    element("export-scene").textContent = this.exporting ? "Exporting…" : "Export terrain GLB";
    const position = this.renderer.camera.position;
    element("position").textContent =
      `${Math.round(this.renderer.origin.x + position.x * RADIUS)}, ${Math.round(this.renderer.origin.y + position.z * RADIUS)}`;
    element("elevation").textContent = walk.prepared
      ? `${Math.round(position.y * RADIUS - 2)} m`
      : "Loading";
    for (const id of [
      "resolution",
      "scale-label",
      "view-min",
      "view-max",
      "view-mean",
      "view-stddev",
    ])
      element(id).textContent = "--";
    Object.assign(this.canvas.dataset, {
      walk: walk.active,
      walkPrepared: walk.prepared,
      walkPaused: walk.paused,
      walkMouseBound: walk.mouseBound,
      walkSpeed: walk.speed,
      walkClearance: walk.aboveGround,
      walkSceneTiles: walk.selection?.leaves.length || 0,
    });
    this.minimap.draw();
    this.renderer.draw();
  }

  restyle(map) {
    this.walk.pause();
    this.renderer.restyle(this.controller.tiles.values(), map, 1);
    this.update();
  }

  dispose() {
    this.walk.exit();
    this.disposed = true;
    cancelAnimationFrame(this.frame);
    this.worker.terminate();
    this.events.abort();
    this.minimap.dispose();
    this.renderer.restyle(this.controller.tiles.values(), this.renderer.map, 1);
    this.renderer.renderer.dispose();
    this.renderer.renderer.forceContextLoss();
    const position = this.renderer.camera.position;
    this.canvas.remove();
    document.body.classList.remove("walk-active");
    document.getElementById("toggle-walk").textContent = "Walk";
    document.getElementById("walk-hint").textContent = "Walk over this terrain";
    document.getElementById("walk-controls").hidden = true;
    return {
      x: this.renderer.origin.x + position.x * RADIUS,
      y: this.renderer.origin.y + position.z * RADIUS,
    };
  }
}

Object.assign(TerrainHiGAN.globe, { PlanarRenderer, PlanarWalk, PlanarWalkView });
