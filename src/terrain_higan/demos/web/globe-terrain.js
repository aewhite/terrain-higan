"use strict";
(() => {
  const { RegionCache, noiseKey, philox4x32, crop } = TerrainHiGAN;
  const { sideAt, canonicalPixel, acrossEdge, faceVector, vectorFace, GLOBE_VERSION } =
    TerrainHiGAN.globe;
  const BLOCK = 64;
  const COLLAR = 4;
  const TILE_GRID = BLOCK + 4;
  const globeSeed = (seed) => BigInt.asUintN(64, seed ^ 0x676c6f62652d7631n);

  class ObsoleteGlobeRequest extends Error {
    constructor() {
      super("Obsolete globe request");
    }
  }

  function predictionCharts(face, east, north, side) {
    const candidates = [{ face, east, north, depth: 0 }];
    const seen = new Set([face]);
    for (let index = 0; index < candidates.length; index++) {
      const point = candidates[index];
      if (point.depth === 2) continue;
      const distances = [
        point.east + 0.5,
        side - point.east - 0.5,
        point.north + 0.5,
        side - point.north - 0.5,
      ];
      for (let edge = 0; edge < 4; edge++) {
        if (distances[edge] >= COLLAR) continue;
        const next = acrossEdge(point.face, point.east, point.north, edge, side);
        if (seen.has(next.face)) continue;
        const insideCollar =
          next.east >= -COLLAR &&
          next.east < side + COLLAR &&
          next.north >= -COLLAR &&
          next.north < side + COLLAR;
        if (insideCollar) {
          seen.add(next.face);
          candidates.push({ ...next, depth: point.depth + 1 });
        }
      }
    }
    for (const point of candidates) {
      const smooth = (distance) => {
        const value = Math.max(0, Math.min(1, (distance + COLLAR) / (2 * COLLAR)));
        return value * value * (3 - 2 * value);
      };
      point.weight =
        smooth(point.east + 0.5) *
        smooth(side - point.east - 0.5) *
        smooth(point.north + 0.5) *
        smooth(side - point.north - 0.5);
    }
    return candidates;
  }

  class GlobeGenerator {
    constructor(runtime, cache = new RegionCache()) {
      this.runtime = runtime;
      this.cache = cache;
    }

    claim(key, produce, assertCurrent) {
      assertCurrent();
      const cached = this.cache.get(key);
      if (cached) return Promise.resolve(cached);
      const pending = this.cache.pending.get(key);
      if (pending) {
        this.cache.coalesced++;
        return pending.catch((error) => {
          assertCurrent();
          if (!(error instanceof ObsoleteGlobeRequest)) throw error;
          return this.claim(key, produce, assertCurrent);
        });
      }
      const generation = this.cache.generation;
      const work = Promise.resolve()
        .then(() => {
          assertCurrent();
          return produce();
        })
        .then((result) => {
          if (generation === this.cache.generation) this.cache.put(key, result);
          return result;
        })
        .finally(() => {
          if (this.cache.pending.get(key) === work) this.cache.pending.delete(key);
        });
      this.cache.pending.set(key, work);
      return work;
    }

    address(level, face, east, north, raw = false) {
      const side = sideAt(level);
      const size = Math.min(side, BLOCK);
      const left = Math.max(0, Math.min(side - size, Math.floor(east / size) * size));
      const top = Math.max(0, Math.min(side - size, Math.floor(north / size) * size));
      const padding = raw ? COLLAR : 0;
      return {
        face,
        level,
        x: left - padding,
        y: top - padding,
        width: size + padding * 2,
        height: size + padding * 2,
      };
    }

    key(seed, region, raw) {
      return `${GLOBE_VERSION}:${seed}:${raw ? "raw" : "final"}:${region.level}:${region.face}:${region.x}:${region.y}`;
    }

    block(seed, region, raw = false, assertCurrent = () => {}) {
      return this.claim(
        this.key(seed, region, raw),
        () =>
          raw
            ? this.rawBlock(seed, region, assertCurrent)
            : this.finalBlock(seed, region, assertCurrent),
        assertCurrent,
      );
    }

    async gather(seed, level, points, raw = false, assertCurrent = () => {}) {
      assertCurrent();
      const regions = new Map();
      const addresses = points.map((point) => {
        const region = this.address(level, point.face, point.east, point.north, raw);
        const key = this.key(seed, region, raw);
        if (!regions.has(key)) regions.set(key, region);
        return { point, key };
      });
      const blocks = new Map();
      const entries = [...regions];
      // Limit dependency production too: one inference worker alone does not
      // bound the tensors retained by an eagerly prepared dependency tree.
      const batchSize = this.runtime.lowMemory ? 1 : Math.max(1, entries.length);
      for (let index = 0; index < entries.length; index += batchSize) {
        assertCurrent();
        const batch = await Promise.all(
          entries
            .slice(index, index + batchSize)
            .map(async ([key, region]) => [
              key,
              await this.block(seed, region, raw, assertCurrent),
            ]),
        );
        for (const [key, block] of batch) blocks.set(key, block);
      }
      return addresses.map(({ point, key }) => {
        const block = blocks.get(key);
        return block.values[(point.north - block.y) * block.width + point.east - block.x];
      });
    }

    async noise(seed, level, face, left, top, width, height, shuffled = false) {
      const key = await noiseKey(globeSeed(seed), level);
      const side = sideAt(level);
      const normal = (east, north) => {
        const point = canonicalPixel(face, east, north, side);
        const words = philox4x32([point.east, point.face, point.north, 0], key);
        return (
          Math.sqrt(-2 * Math.log((words[0] + 0.5) / 4294967296)) *
          Math.cos((2 * Math.PI * (words[1] + 0.5)) / 4294967296)
        );
      };
      const count = width * height;
      const values = new Float32Array(count * (shuffled ? 4 : 1));
      for (let row = 0; row < height; row++)
        for (let column = 0; column < width; column++) {
          const offset = row * width + column;
          if (shuffled) {
            for (let channel = 0; channel < 4; channel++)
              values[channel * count + offset] = normal(
                2 * (left + column) + (channel % 2),
                2 * (top + row) + Math.floor(channel / 2),
              );
          } else values[offset] = normal(left + column, top + row);
        }
      return values;
    }

    async rawBlock(seed, region, assertCurrent) {
      const { face, level, x, y, width, height } = region;
      const model = await this.runtime.sessionFor(level);
      assertCurrent();
      let inputs, left, top;
      if (level === 0) {
        const noise = await this.noise(seed, 0, face, x - 8, y - 8, width + 16, height + 16);
        inputs = { noise: this.runtime.tensor(noise, [1, 1, height + 16, width + 16]) };
        left = top = 0;
      } else {
        const parentLeft = Math.floor((x - 8) / 2);
        const parentTop = Math.floor((y - 8) / 2);
        const parentWidth = Math.ceil((x + width + 8) / 2) - parentLeft;
        const parentHeight = Math.ceil((y + height + 8) / 2) - parentTop;
        const parentSide = sideAt(level - 1);
        const points = [];
        for (let row = 0; row < parentHeight; row++)
          for (let column = 0; column < parentWidth; column++)
            points.push(canonicalPixel(face, parentLeft + column, parentTop + row, parentSide));
        const parent = new Float32Array(
          await this.gather(seed, level - 1, points, false, assertCurrent),
        );
        assertCurrent();
        const noise = await this.noise(
          seed,
          level,
          face,
          parentLeft,
          parentTop,
          parentWidth,
          parentHeight,
          true,
        );
        inputs = {
          input_map: this.runtime.tensor(parent, [1, 1, parentHeight, parentWidth]),
          noise: this.runtime.tensor(noise, [1, 4, parentHeight, parentWidth]),
        };
        left = x - 2 * parentLeft - 8;
        top = y - 2 * parentTop - 8;
      }
      assertCurrent();
      const result = await this.runtime.run(model, inputs, assertCurrent);
      const output = result.elevation_normalized;
      return {
        ...region,
        values: crop(output.data, output.dims[3], left, top, width, height),
        model: result.model || model.name,
        provider: result.provider || model.provider,
      };
    }

    async finalBlock(seed, region, assertCurrent) {
      const side = sideAt(region.level);
      const charts = [];
      const offsets = [0];
      for (let row = 0; row < region.height; row++)
        for (let column = 0; column < region.width; column++) {
          charts.push(...predictionCharts(region.face, region.x + column, region.y + row, side));
          offsets.push(charts.length);
        }
      const predictions = await this.gather(seed, region.level, charts, true, assertCurrent);
      const values = new Float32Array(region.width * region.height);
      for (let index = 0; index < values.length; index++) {
        let sum = 0,
          weight = 0;
        for (let chart = offsets[index]; chart < offsets[index + 1]; chart++) {
          sum += predictions[chart] * charts[chart].weight;
          weight += charts[chart].weight;
        }
        values[index] = sum / weight;
      }
      return { ...region, values };
    }

    async vertices(seed, face, level, tileX, tileY, values, assertCurrent = () => {}) {
      const side = sideAt(level);
      const points = [],
        fractions = [],
        indices = [];
      const result = new Float32Array(67 * 67);
      for (let row = 0; row < 67; row++)
        for (let column = 0; column < 67; column++) {
          let east = tileX * BLOCK + column - 1;
          let north = tileY * BLOCK + row - 1;
          if (values && east > 0 && east < side && north > 0 && north < side) {
            const index = row * TILE_GRID + column;
            result[row * 67 + column] =
              (values[index] * 0.5 + values[index + 1] * 0.5) * 0.5 +
              (values[index + TILE_GRID] * 0.5 + values[index + TILE_GRID + 1] * 0.5) * 0.5;
            continue;
          }
          indices.push(row * 67 + column);
          let chart = face;
          if (east <= 0 || east >= side || north <= 0 || north >= side) {
            if (east < 0 || east > side || north < 0 || north > side) {
              const point = canonicalPixel(face, east - 0.5, north - 0.5, side);
              chart = point.face;
              east = point.east + 0.5;
              north = point.north + 0.5;
            }
            const point = vectorFace(faceVector(chart, east / side, north / side));
            chart = point.face;
            east = point.east * side;
            north = point.north * side;
          }
          const left = Math.floor(east - 0.5),
            top = Math.floor(north - 0.5);
          fractions.push([east - 0.5 - left, north - 0.5 - top]);
          for (let offsetY = 0; offsetY < 2; offsetY++)
            for (let offsetX = 0; offsetX < 2; offsetX++)
              points.push(canonicalPixel(chart, left + offsetX, top + offsetY, side));
        }
      const samples = await this.gather(seed, level, points, false, assertCurrent);
      fractions.forEach(([east, north], index) => {
        const start = index * 4;
        result[indices[index]] =
          (samples[start] * (1 - east) + samples[start + 1] * east) * (1 - north) +
          (samples[start + 2] * (1 - east) + samples[start + 3] * east) * north;
      });
      return result;
    }

    async tile(seed, face, depth, tileX, tileY, assertCurrent = () => {}) {
      const level = depth + 2;
      const side = sideAt(level);
      const points = [];
      for (let row = 0; row < TILE_GRID; row++)
        for (let column = 0; column < TILE_GRID; column++) {
          points.push(
            canonicalPixel(face, tileX * BLOCK + column - 2, tileY * BLOCK + row - 2, side),
          );
        }
      assertCurrent();
      const samples = await this.gather(seed, level, points, false, assertCurrent);
      assertCurrent();
      const values = new Float32Array(samples);
      const vertexValues = await this.vertices(
        seed,
        face,
        level,
        tileX,
        tileY,
        values,
        assertCurrent,
      );
      assertCurrent();
      return {
        face,
        depth,
        tileX,
        tileY,
        level,
        width: TILE_GRID,
        values,
        vertexValues,
        model: TerrainHiGAN.modelName(level),
      };
    }
  }

  Object.assign(TerrainHiGAN.globe, {
    GlobeGenerator,
    ObsoleteGlobeRequest,
    predictionCharts,
    TILE_GRID,
    BLOCK,
  });
})();
