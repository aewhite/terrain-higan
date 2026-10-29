"use strict";
(() => {
  class TileController {
    constructor({ send, onTile, onStatus, onError }) {
      this.send = send;
      this.onTile = onTile;
      this.onStatus = onStatus;
      this.onError = onError;
      this.tiles = new Map();
      this.sequence = [];
      this.active = null;
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
        !sequence.some((tile) => tile.key === this.active.key) &&
        !this.active.cancelled
      ) {
        this.active.cancelled = true;
        this.phase = "cancelling";
        this.send({ type: "cancel", id: ++this.requestId });
      }
      this.dispatch();
    }

    dispatch() {
      if (this.error || this.active) return;
      const tile = this.sequence.find((candidate) => !this.tiles.has(candidate.key));
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
      this.active = request;
      this.phase = "generating";
      const loaded = this.sequence.filter((item) => this.tiles.has(item.key)).length;
      this.onStatus(`Generating ${loaded + 1} / ${this.sequence.length} · ${this.backend}`);
      this.send(request);
    }

    receive(data) {
      // Keys can recur after navigation; only the unique request ID owns the active slot.
      if (!this.active || data.id !== this.active.id || data.key !== this.active.key) return;
      if (data.type === "status") {
        if (!this.active.cancelled) this.onStatus(data.message);
        return;
      }
      if (!["tile", "cancelled", "error"].includes(data.type)) return;
      const obsolete = this.active.cancelled;
      this.active = null;
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
      this.dispatch();
    }

    trim() {
      const protectedKeys = new Set(this.sequence.map((tile) => tile.key));
      for (const key of this.tiles.keys()) {
        if (this.tiles.size <= Math.max(512, this.sequence.length)) break;
        if (!protectedKeys.has(key)) this.tiles.delete(key);
      }
    }

    fail(message) {
      this.error = message;
      this.active = null;
      this.phase = "failed";
      this.onError(message);
    }
  }
  TerrainHiGAN.TileController = TileController;
})();
