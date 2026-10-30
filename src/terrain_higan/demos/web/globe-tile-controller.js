"use strict";
(() => {
  const MODEL_SAMPLE_PIXELS = 1.5;

  class GlobeTileController {
    constructor(renderer, send, onState) {
      this.renderer = renderer;
      this.mobile = !!TerrainHiGAN.mobileView;
      this.send = send;
      this.onState = onState;
      this.tiles = new Map();
      this.nodes = new Map();
      this.sequence = [];
      this.display = [];
      this.active = null;
      this.nextId = 0;
      this.failed = false;
      this.seed = "";
      this.tilesVersion = 0;
    }

    node(face, depth, tileX, tileY) {
      const key = `${face}:${depth}:${tileX}:${tileY}`;
      if (!this.nodes.has(key)) {
        const parentKey = depth
          ? `${face}:${depth - 1}:${Math.floor(tileX / 2)}:${Math.floor(tileY / 2)}`
          : null;
        const node = { key, face, depth, tileX, tileY, parentKey };
        node.bounds = this.renderer.bounds(node);
        this.nodes.set(key, node);
      }
      const node = this.nodes.get(key);
      this.nodes.delete(key);
      this.nodes.set(key, node);
      return node;
    }

    reset(seed) {
      this.renderer.walk?.reset();
      for (const tile of this.tiles.values()) this.renderer.dispose(tile);
      this.tiles.clear();
      this.tilesVersion++;
      this.active = null;
      this.failed = false;
      this.seed = seed;
      this.update();
    }

    update() {
      if (this.failed) return;
      this.renderer.prepareView(this.tiles.values(), this.tilesVersion);
      if (this.renderer.walk?.active) {
        this.renderer.walk.prepare();
        return;
      }
      const sequence = [],
        protect = new Set();
      let display = [],
        patches = [];
      const roots = Array.from({ length: TerrainHiGAN.globe.FACE_COUNT }, (_, face) =>
        this.node(face, 0, 0, 0),
      );
      const childrenOf = (node) => {
        if (node.children) return node.children;
        const children = [];
        for (let north = 0; north < 2; north++)
          for (let east = 0; east < 2; east++)
            children.push(
              this.node(node.face, node.depth + 1, node.tileX * 2 + east, node.tileY * 2 + north),
            );
        node.children = children;
        return children;
      };
      if (!this.selection || this.selection.key !== this.renderer.selectionKey) {
        const expanded = new Set();
        const required = new Map();
        const refine = (node) => {
          if (!this.renderer.visible(node.bounds)) return;
          required.set(node.key, node);
          if (
            node.depth >= TerrainHiGAN.globe.MAX_GLOBE_LEVEL - 2 ||
            this.renderer.span(node.bounds) <= 64 * (this.mobile ? 2 : MODEL_SAMPLE_PIXELS)
          )
            return;
          expanded.add(node.key);
          for (const child of childrenOf(node)) refine(child);
        };
        for (const root of roots) refine(root);
        const leaves = [...required.values()].filter((node) => !expanded.has(node.key));
        const ordered = [...required.values()]
          .filter((node) => node.depth === 0 || !expanded.has(node.key))
          .sort(
            (first, second) =>
              Number(first.depth !== 0) - Number(second.depth !== 0) ||
              first.bounds.center.distanceTo(this.renderer.camera.position) -
                second.bounds.center.distanceTo(this.renderer.camera.position),
          );
        this.selection = { key: this.renderer.selectionKey, expanded, required, ordered, leaves };
        this.nodeCapacity = Math.max(4096, this.selection.required.size * 4);
        if (this.nodes.size > this.nodeCapacity) {
          for (const [key, node] of this.nodes) {
            if (this.nodes.size <= this.nodeCapacity) break;
            if (this.selection.required.has(key)) continue;
            delete node.children;
            this.nodes.delete(key);
          }
          for (const node of this.nodes.values())
            if (node.children?.some((child) => !this.nodes.has(child.key))) delete node.children;
        }
      }
      if (this.active && !this.selection.required.has(this.active.key)) {
        this.send({ type: "cancel" });
        this.active = null;
      }
      for (const node of this.selection.required.values()) protect.add(node.key);
      for (const node of this.selection.ordered) {
        if (!this.tiles.has(node.key)) sequence.push(node);
      }
      if (this.displaySelection === this.selection && this.displayVersion === this.tilesVersion) {
        display = this.display;
        patches = this.patches;
      } else {
        const available = new Map();
        for (const node of this.selection.required.values()) {
          const source = this.tiles.get(node.key) || available.get(node.parentKey);
          if (!source) continue;
          available.set(node.key, source);
          if (this.selection.expanded.has(node.key)) continue;
          patches.push({ ...node, source });
        }
        const sources = new Set(patches.map((patch) => patch.source));
        for (const source of sources) {
          this.tiles.delete(source.key);
          this.tiles.set(source.key, source);
          display.push(source);
        }
      }
      this.patches = patches;
      this.sequence = sequence;
      this.display = display;
      this.capacity = Math.max(this.mobile ? 128 : 512, protect.size + (this.mobile ? 32 : 128));
      for (const [key, tile] of this.tiles) {
        if (this.tiles.size <= this.capacity) break;
        if (!protect.has(key) && this.active?.key !== key) {
          this.renderer.dispose(tile);
          this.tiles.delete(key);
          this.tilesVersion++;
        }
      }
      this.displaySelection = this.selection;
      this.displayVersion = this.tilesVersion;
      this.renderer.show(display, patches);
      this.onState({
        ready: !this.sequence.length,
        pending: this.sequence.length,
        visible: patches.length,
        cached: this.tiles.size,
        capacity: this.capacity,
        targetLevel: Math.max(2, ...[...protect].map((key) => Number(key.split(":")[1]) + 2)),
        samplePixels: Math.max(0, ...display.map((tile) => this.renderer.span(tile.bounds) / 64)),
      });
      this.dispatch();
    }

    dispatch() {
      if (this.failed || this.active || !this.sequence.length) return;
      const node = this.sequence[0];
      this.active = { ...node, id: ++this.nextId };
      this.send({
        type: "tile",
        seed: this.seed,
        ...this.active,
        bounds: undefined,
        children: undefined,
      });
    }

    receive(data, update = true) {
      if (!this.active || data.id !== this.active.id || data.key !== this.active.key) return;
      const node = this.active;
      this.active = null;
      if (data.type === "error") {
        this.fail(data.message);
        return;
      }
      if (data.type === "tile") {
        if (!update) {
          this.sequence = this.sequence.filter((candidate) => candidate.key !== node.key);
          this.dispatch();
        }
        const tile = {
          ...node,
          ...data,
          values: new Float32Array(data.values),
          vertexValues: new Float32Array(data.vertexValues),
        };
        this.tiles.set(node.key, tile);
        this.tilesVersion++;
      }
      if (update) this.update();
    }

    fail(message) {
      this.failed = true;
      this.onState({ error: message });
    }
  }

  TerrainHiGAN.globe.GlobeTileController = GlobeTileController;
})();
