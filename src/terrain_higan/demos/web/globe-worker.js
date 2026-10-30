"use strict";
const options = new URLSearchParams(self.location.search);
const bundle = options.get("bundle");
const versioned = (file) => `${file}${bundle ? `?bundle=${bundle}` : ""}`;
if (!globalThis.crypto?.subtle) importScripts(versioned("./assets/runtime/sha256.js"));
importScripts(
  ...[
    "terrain-common.js",
    "noise.js",
    "terrain.js",
    "inference-pool.js",
    "globe-topology.js",
    "globe-terrain.js",
  ].map((file) => versioned("./" + file)),
);

(() => {
  const emit = (message, transfers) => postMessage(message, transfers);
  const runtime = new TerrainHiGAN.InferencePool(options, emit);
  const run = runtime.run.bind(runtime);
  runtime.run = async (model, inputs, assertCurrent) => {
    const result = await run(model, inputs, assertCurrent);
    emit({ type: "progress", model: result.model, metrics: { ...runtime.metrics } });
    return result;
  };
  const planar = options.get("surface") === "planar";
  const cache =
    options.get("mobile") === "1"
      ? new TerrainHiGAN.RegionCache(16 * 1024 * 1024, 1024)
      : new TerrainHiGAN.RegionCache();
  const generator = planar
    ? new TerrainHiGAN.TerrainGenerator(runtime, cache)
    : new TerrainHiGAN.globe.GlobeGenerator(runtime, cache);
  let generation = 0;
  self.onmessage = async ({ data }) => {
    if (data.type === "reset") {
      generation++;
      generator.cache.clear();
      return;
    }
    if (data.type === "cancel") {
      generation++;
      return;
    }
    const current = ++generation;
    const reply = (message, transfers) =>
      emit({ id: data.id, key: data.key, ...message }, transfers);
    try {
      if (data.type !== "tile") throw new Error("Unknown globe request");
      const { face, depth, tileX, tileY } = data;
      if (
        !planar &&
        (![face, depth, tileX, tileY].every(Number.isInteger) ||
          face < 0 ||
          face >= TerrainHiGAN.globe.FACE_COUNT ||
          depth < 0 ||
          depth > TerrainHiGAN.globe.MAX_GLOBE_LEVEL - 2 ||
          tileX < 0 ||
          tileY < 0 ||
          tileX >= 2 ** depth ||
          tileY >= 2 ** depth)
      )
        throw new RangeError("Invalid S2 tile address");
      const seed = BigInt(data.seed);
      if (seed < 0n || seed >= 2n ** 64n) throw new RangeError("Seed must be unsigned 64-bit");
      const assertCurrent = () => {
        if (current !== generation) throw new TerrainHiGAN.globe.ObsoleteGlobeRequest();
      };
      let tile;
      if (planar) {
        const region = { zoom: depth, x: tileX * 64 - 2, y: tileY * 64 - 2, width: 68, height: 68 };
        TerrainHiGAN.validateRegion(region);
        const generated = await generator.generate(region, { seed, assertCurrent });
        const values = generated.values;
        const vertexValues = new Float32Array(67 * 67);
        for (let row = 0; row < 67; row++)
          for (let column = 0; column < 67; column++) {
            const index = row * 68 + column;
            vertexValues[row * 67 + column] =
              (values[index] + values[index + 1] + values[index + 68] + values[index + 69]) / 4;
          }
        tile = {
          face,
          depth,
          tileX,
          tileY,
          level: depth,
          width: 68,
          values,
          vertexValues,
          model: generated.model,
        };
      } else tile = await generator.tile(seed, face, depth, tileX, tileY, assertCurrent);
      reply(
        {
          type: "tile",
          ...tile,
          values: tile.values.buffer,
          vertexValues: tile.vertexValues.buffer,
          provider: { name: runtime.providers.get(tile.model) || "wasm" },
          metrics: {
            ...runtime.metrics,
            cacheBytes: generator.cache.bytes,
            cacheHits: generator.cache.hits,
            coalesced: generator.cache.coalesced,
          },
        },
        [tile.values.buffer, tile.vertexValues.buffer],
      );
    } catch (error) {
      reply({ type: current === generation ? "error" : "cancelled", message: error.message });
    }
  };
})();
