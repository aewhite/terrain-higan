"use strict";
(() => {
  class TileController {
    constructor({ send, onTile, onStatus, onError, maxCacheTiles = 512 }) {
      this.send = send;
      this.maxCacheTiles = maxCacheTiles;
      this.onTile = onTile;
      this.onStatus = onStatus;
      this.onError = onError;
      this.tiles = new Map();
      this.sequence = [];
      this.active = null;
      this.queued = [];
      this.pipelineSize = 1;
      this.requestId = 0;
      this.error = null;
      this.backend = "";
      this.phase = "idle";
    }

    setView(sequence, identity) {
      this.sequence = sequence;
      this.identity = identity;
      if (
        this.active &&
        [this.active, ...this.queued].some(
          (request) => !sequence.some((tile) => tile.key === request.key),
        ) &&
        !this.active.cancelled
      ) {
        this.active.cancelled = true;
        this.queued.forEach((request) => {
          request.cancelled = true;
        });
        this.phase = "cancelling";
        this.send({ type: "cancel", id: ++this.requestId });
      }
      this.dispatch();
    }

    dispatch() {
      if (this.error || this.active) return;
      const waiting = this.sequence.filter((candidate) => !this.tiles.has(candidate.key));
      const tile = waiting[0];
      if (!tile) {
        this.phase = "ready";
        const provider = this.sequence
          .map((item) => this.tiles.get(item.key)?.provider)
          .find(Boolean);
        if (provider) this.backend = provider.label;
        this.onStatus(
          `Model terrain ready · ${this.backend} · ${this.sequence.length} visible tiles`,
          true,
        );
        return;
      }
      const request = { ...this.identity, ...tile, type: "generate", id: ++this.requestId };
      this.queued = waiting
        .slice(1, this.pipelineSize)
        .map((next) => ({ ...this.identity, ...next, type: "generate", id: request.id }));
      this.active = request;
      this.phase = "generating";
      const loaded = this.sequence.filter((item) => this.tiles.has(item.key)).length;
      this.onStatus(`Generating ${loaded + 1} / ${this.sequence.length} · ${this.backend}`);
      this.send({ ...request, ...(this.queued.length ? { ahead: this.queued } : {}) });
    }

    receive(data, dispatch = true) {
      // Keys can recur after navigation; only the unique request ID owns the active slot.
      if (!this.active || data.id !== this.active.id || data.key !== this.active.key) return;
      if (data.type === "status") {
        if (!this.active.cancelled) this.onStatus(data.message);
        return;
      }
      if (!["tile", "cancelled", "error"].includes(data.type)) return;
      const obsolete = this.active.cancelled;
      this.active = this.queued.shift() || null;
      if (data.type === "error" && !obsolete) {
        this.fail(data.message);
        return;
      }
      const requested = this.sequence.find((tile) => tile.key === data.key);
      if (data.type === "tile" && !obsolete && requested) {
        this.tiles.set(data.key, this.onTile(data, requested, this.identity));
        this.backend = data.provider.label;
        this.trim();
      }
      if (dispatch) this.dispatch();
    }

    trim() {
      const protectedKeys = new Set(this.sequence.map((tile) => tile.key));
      for (const key of this.tiles.keys()) {
        if (this.tiles.size <= Math.max(this.maxCacheTiles, this.sequence.length)) break;
        if (!protectedKeys.has(key)) this.tiles.delete(key);
      }
    }

    fail(message) {
      this.queued = [];
      this.active = null;
      this.error = message;
      this.phase = "failed";
      this.onError(message);
    }
  }
  TerrainHiGAN.TileController = TileController;
})();
