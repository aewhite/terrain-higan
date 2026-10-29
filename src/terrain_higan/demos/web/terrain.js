"use strict";
(() => {
  const { HALO, parentRegion, padRegion, crop, normalField, upscalerNoise } = TerrainHiGAN;
  const BLOCK_SIZE = 64;

  class RegionCache {
    constructor(capacity = 128 * 1024 * 1024, maxEntries = 8192) {
      this.capacity = capacity;
      this.maxEntries = maxEntries;
      this.entries = new Map();
      this.bytes = 0;
      this.hits = 0;
      this.pending = new Map();
      this.coalesced = 0;
      this.generation = 0;
    }

    key(seed, region) {
      const { zoom, x, y, width, height } = region;
      return `${seed}:${zoom}:${x}:${y}:${width}:${height}`;
    }

    get(key) {
      const value = this.entries.get(key);
      if (value) {
        this.hits++;
        this.entries.delete(key);
        this.entries.set(key, value);
      }
      return value;
    }

    put(key, region) {
      if (region.values.byteLength > this.capacity || this.maxEntries === 0) return;
      const old = this.entries.get(key);
      if (old) this.bytes -= old.values.byteLength;
      this.entries.delete(key);
      this.entries.set(key, region);
      this.bytes += region.values.byteLength;
      while (this.bytes > this.capacity || this.entries.size > this.maxEntries) {
        const oldest = this.entries.keys().next().value;
        this.bytes -= this.entries.get(oldest).values.byteLength;
        this.entries.delete(oldest);
      }
    }

    clear() {
      this.entries.clear();
      this.bytes = 0;
      this.pending.clear();
      this.generation++;
    }
  }

  class TerrainGenerator {
    constructor(runtime, cache = new RegionCache()) {
      this.runtime = runtime;
      this.cache = cache;
    }

    async generate(region, request) {
      request.assertCurrent();
      const { zoom, x, y, width, height } = region;
      const blocks = [];
      for (let top = Math.floor(y / BLOCK_SIZE) * BLOCK_SIZE; top < y + height; top += BLOCK_SIZE)
        for (
          let left = Math.floor(x / BLOCK_SIZE) * BLOCK_SIZE;
          left < x + width;
          left += BLOCK_SIZE
        )
          blocks.push({ zoom, x: left, y: top, width: BLOCK_SIZE, height: BLOCK_SIZE });
      const values = new Float32Array(width * height);
      let model, provider;
      const concurrency = this.runtime.size || 2;
      // Bound speculative work so cancellation leaves only a small batch of producers running.
      for (let index = 0; index < blocks.length; index += concurrency) {
        request.assertCurrent();
        const batch = await Promise.all(
          blocks
            .slice(index, index + concurrency)
            .map((block) => this.generateBlock(block, request.seed)),
        );
        request.assertCurrent();
        for (const block of batch) {
          const left = Math.max(x, block.x),
            top = Math.max(y, block.y);
          const right = Math.min(x + width, block.x + BLOCK_SIZE);
          const bottom = Math.min(y + height, block.y + BLOCK_SIZE);
          for (let row = top; row < bottom; row++) {
            const start = (row - block.y) * BLOCK_SIZE + left - block.x;
            values.set(
              block.values.subarray(start, start + right - left),
              (row - y) * width + left - x,
            );
          }
          model = block.model;
          provider = block.provider;
        }
      }
      return { ...region, values, model, provider };
    }

    generateBlock(region, seed) {
      const key = this.cache.key(seed, region);
      const cached = this.cache.get(key);
      if (cached) return Promise.resolve(cached);
      const pending = this.cache.pending.get(key);
      if (pending) {
        this.cache.coalesced++;
        return pending;
      }
      const generation = this.cache.generation;
      // Claim synchronously before starting dependency work. The producer belongs to
      // the cache, not any one viewport waiter, and publishes only complete blocks.
      const work = Promise.resolve()
        .then(async () => {
          const generated = await this.generateUncached(region, { seed, assertCurrent() {} });
          const result = { ...region, ...generated, values: new Float32Array(generated.values) };
          if (generation === this.cache.generation) this.cache.put(key, result);
          return result;
        })
        .finally(() => {
          if (this.cache.pending.get(key) === work) this.cache.pending.delete(key);
        });
      this.cache.pending.set(key, work);
      return work;
    }

    async generateUncached(region, request) {
      const { zoom, x, y, width, height } = region;
      request.assertCurrent();
      if (zoom === 0) {
        const model = await this.runtime.sessionFor(0);
        request.assertCurrent();
        const input = padRegion(region, HALO);
        const noise = await normalField(
          request.seed,
          0,
          input.x,
          input.y,
          input.width,
          input.height,
        );
        request.assertCurrent();
        const result = await this.runtime.run(model, {
          noise: this.runtime.tensor(noise, [1, 1, input.height, input.width]),
        });
        request.assertCurrent();
        return {
          values: result.elevation_normalized.data,
          model: result.model || model.name,
          provider: result.provider || model.provider,
        };
      }
      const parent = await this.generate(parentRegion(region), request);
      request.assertCurrent();
      const model = await this.runtime.sessionFor(zoom);
      request.assertCurrent();
      const noise = await upscalerNoise(
        request.seed,
        zoom,
        parent.x,
        parent.y,
        parent.width,
        parent.height,
      );
      request.assertCurrent();
      const result = await this.runtime.run(model, {
        input_map: this.runtime.tensor(parent.values, [1, 1, parent.height, parent.width]),
        noise: this.runtime.tensor(noise, [1, 4, parent.height, parent.width]),
      });
      request.assertCurrent();
      const output = result.elevation_normalized;
      const left = x - (2 * parent.x + HALO),
        top = y - (2 * parent.y + HALO);
      return {
        values: crop(output.data, output.dims[3], left, top, width, height),
        model: result.model || model.name,
        provider: result.provider || model.provider,
      };
    }
  }

  Object.assign(TerrainHiGAN, { BLOCK_SIZE, RegionCache, TerrainGenerator });
})();
