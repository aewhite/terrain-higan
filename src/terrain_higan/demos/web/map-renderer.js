"use strict";
(() => {
  class MapRenderer {
    constructor(canvas) {
      this.canvas = canvas;
      this.context = canvas.getContext("2d", { alpha: false });
      this.front = null;
      this.job = null;
      this.spare = null;
      this.revision = -1;
      this.pending = false;
      this.builds = 0;
    }

    region(view) {
      const ratio = 2 ** (view.zoom - view.level);
      return {
        identity: `${view.seed}:${view.map}:${view.level}`,
        left: Math.floor(view.left) - 1,
        top: Math.floor(view.top) - 1,
        width: Math.ceil(this.canvas.width / ratio) + 3,
        height: Math.ceil(this.canvas.height / ratio) + 3,
      };
    }

    covers(mosaic, region) {
      return (
        mosaic &&
        mosaic.identity === region.identity &&
        region.left >= mosaic.left &&
        region.top >= mosaic.top &&
        region.left + region.width <= mosaic.left + mosaic.surface.width &&
        region.top + region.height <= mosaic.top + mosaic.surface.height
      );
    }

    ready(view, revision) {
      return (
        !this.pending && this.revision === revision && this.covers(this.front, this.region(view))
      );
    }

    rectangle(mosaic, tile, size) {
      const scale = 2 ** (mosaic.level - tile.zoom);
      const x = Math.round(tile.x * scale - mosaic.left);
      const y = Math.round(tile.y * scale - mosaic.top);
      return {
        x,
        y,
        width: Math.round((tile.x + (tile.width ?? size)) * scale - mosaic.left) - x,
        height: Math.round((tile.y + (tile.height ?? size)) * scale - mosaic.top) - y,
      };
    }

    intersects(mosaic, rect) {
      return (
        rect.width > 0 &&
        rect.height > 0 &&
        rect.x < mosaic.surface.width &&
        rect.y < mosaic.surface.height &&
        rect.x + rect.width > 0 &&
        rect.y + rect.height > 0
      );
    }

    paint(mosaic, tile, size) {
      if (!tile.image) {
        const surface = document.createElement("canvas");
        surface.width = size;
        surface.height = size;
        surface
          .getContext("2d")
          .putImageData(new ImageData(new Uint8ClampedArray(tile.pixels), size, size), 0, 0);
        tile.image = surface;
        delete tile.pixels;
      }
      const rect = this.rectangle(mosaic, tile, size);
      mosaic.context.drawImage(tile.image, rect.x, rect.y, rect.width, rect.height);
      if (tile.key && tile.zoom === mosaic.level) mosaic.applied.add(tile.key);
    }

    *build(mosaic, tiles, size) {
      mosaic.context.fillStyle = "#082c50";
      mosaic.context.fillRect(0, 0, mosaic.surface.width, mosaic.surface.height);
      mosaic.context.imageSmoothingEnabled = false;
      yield;
      for (const tile of tiles) {
        this.paint(mosaic, tile, size);
        yield true;
      }
    }

    update(view, tiles, revision, budgetMs = 4) {
      const region = this.region(view);
      if (!this.covers(this.front, region)) {
        if (!this.covers(this.job?.mosaic, region)) {
          this.job?.steps.return();
          const surface =
            this.job?.mosaic.surface || this.spare || document.createElement("canvas");
          this.spare = null;
          const width = Math.ceil((region.width + 256) / view.size) * view.size;
          const height = Math.ceil((region.height + 256) / view.size) * view.size;
          if (surface.width !== width) surface.width = width;
          if (surface.height !== height) surface.height = height;
          const mosaic = {
            ...region,
            level: view.level,
            seed: view.seed,
            map: view.map,
            left: region.left - 128,
            top: region.top - 128,
            surface,
            context: surface.getContext("2d", { alpha: false }),
            applied: new Set(),
          };
          const sources = [...tiles.values()].filter(
            (tile) =>
              tile.seed === view.seed &&
              tile.map === view.map &&
              this.intersects(mosaic, this.rectangle(mosaic, tile, view.size)),
          );
          // Keep committed pixels after their source tiles leave the LRU, and
          // place them above coarser cached data but below new native tiles.
          const previous = this.front;
          const retained =
            previous && previous.seed === view.seed && previous.map === view.map
              ? {
                  image: previous.surface,
                  zoom: previous.level,
                  x: previous.left,
                  y: previous.top,
                  width: previous.surface.width,
                  height: previous.surface.height,
                }
              : null;
          if (retained) sources.push(retained);
          sources.sort(
            (a, b) =>
              Math.abs(b.zoom - view.level) - Math.abs(a.zoom - view.level) ||
              Number(b === retained) - Number(a === retained),
          );
          this.job = { mosaic, steps: this.build(mosaic, sources, view.size) };
        }
        const deadline = performance.now() + budgetMs;
        let step;
        do {
          step = this.job.steps.next();
          // A tile image upload/copy cannot yield halfway through. Limit it
          // to one per frame, independently of the cooperative CPU budget.
          if (step.value) break;
        } while (!step.done && performance.now() < deadline);
        if (step.done) {
          this.spare = this.front?.surface || null;
          this.front = this.job.mosaic;
          this.job = null;
          this.canvas.dataset.mosaicBuilds = ++this.builds;
        }
        this.pending = true;
      } else {
        if (this.job) {
          this.job.steps.return();
          this.spare = this.job.mosaic.surface;
          this.job = null;
        }
        // Native tiles share integer edges. Paint only new tiles into the
        // existing mosaic instead of clearing and redrawing the whole cache.
        const missing = [...tiles.values()].filter(
          (tile) =>
            tile.seed === view.seed &&
            tile.map === view.map &&
            tile.zoom === view.level &&
            !this.front.applied.has(tile.key) &&
            this.intersects(this.front, this.rectangle(this.front, tile, view.size)),
        );
        if (missing.length) this.paint(this.front, missing[0], view.size);
        this.pending = missing.length > 1;
        if (!this.pending) this.revision = revision;
      }
      this.canvas.dataset.mosaicPending = this.pending;
    }

    draw(view) {
      const mosaic = this.front;
      this.context.fillStyle = "#082c50";
      this.context.fillRect(0, 0, this.canvas.width, this.canvas.height);
      if (!mosaic || mosaic.seed !== view.seed || mosaic.map !== view.map) return;
      // Transform the committed image using its own model level, even while
      // a different zoom level's replacement is still being prepared.
      const scale = 2 ** (view.zoom - mosaic.level);
      const levelScale = 2 ** (mosaic.level - view.level);
      const left = view.left * levelScale;
      const top = view.top * levelScale;
      const x = Math.max(0, Math.floor(left) - 1 - mosaic.left);
      const y = Math.max(0, Math.floor(top) - 1 - mosaic.top);
      const width = Math.min(mosaic.surface.width - x, Math.ceil(this.canvas.width / scale) + 3);
      const height = Math.min(mosaic.surface.height - y, Math.ceil(this.canvas.height / scale) + 3);
      if (width <= 0 || height <= 0) return;
      this.context.imageSmoothingEnabled = view.zoom !== view.level;
      this.context.drawImage(
        mosaic.surface,
        x,
        y,
        width,
        height,
        (x + mosaic.left - left) * scale,
        (y + mosaic.top - top) * scale,
        width * scale,
        height * scale,
      );
    }
  }
  TerrainHiGAN.MapRenderer = MapRenderer;
})();
