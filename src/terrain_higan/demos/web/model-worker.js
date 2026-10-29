"use strict";

const runtimeOptions = new URLSearchParams(self.location.search);
const bundle = runtimeOptions.get("bundle");
const versioned = (file) => (bundle ? `${file}?bundle=${bundle}` : file);
if (!globalThis.crypto?.subtle) importScripts(versioned("./assets/runtime/sha256.js"));
importScripts(
  ...["terrain-common.js", "noise.js", "inference-pool.js", "terrain.js", "rendering.js"].map(
    (file) => versioned("./" + file),
  ),
);

(() => {
  const {
    InferencePool,
    TerrainGenerator,
    validateRegion,
    padRegion,
    crop,
    renderRegion,
    renderingPadding,
    summarizeElevations,
  } = TerrainHiGAN;
  class ObsoleteRequest extends Error {}

  /**
   * One page owns this worker. Requests are generate/cancel; terminal replies are
   * tile/cancelled/error and echo id/key. Only bundle-changed is worker-wide.
   * Cancellation is cooperative, so the page waits for a terminal reply before dispatching again.
   */
  function createWorkerHandler(runtime, generator, emit) {
    let newestRequestId = 0;
    let cancelActive = null;
    return async ({ data }) => {
      if (!Number.isSafeInteger(data.id) || data.id <= 0) {
        emit({
          type: "error",
          id: data.id,
          key: data.key,
          message: "Request ID must be a positive safe integer",
        });
        return;
      }
      if (data.type === "cancel") {
        if (data.id > newestRequestId) {
          newestRequestId = data.id;
          cancelActive?.();
        }
        return;
      }
      newestRequestId = Math.max(newestRequestId, data.id);
      const cancelled = new Promise((resolve) => {
        cancelActive = () => resolve(null);
      });
      const reply = (message) => emit({ id: data.id, key: data.key, ...message });
      const request = {
        seed: 0n,
        assertCurrent() {
          if (data.id !== newestRequestId) throw new ObsoleteRequest();
        },
      };
      try {
        if (data.type !== "generate") throw new Error("Unknown worker request type");
        validateRegion(data);
        if (!["contour", "elevation", "realistic"].includes(data.map))
          throw new RangeError("Unknown colormap");
        request.seed = BigInt(data.seed);
        if (request.seed < 0n || request.seed >= 2n ** 64n)
          throw new RangeError("seed must be unsigned 64-bit");
        request.assertCurrent();
        reply({ type: "status", message: `Generating z${data.zoom} terrain` });
        const beforeRuns = runtime.metrics.modelRuns;
        const padding = renderingPadding(data.map);
        const region = padRegion(
          { zoom: data.zoom, x: data.x, y: data.y, width: data.width, height: data.height },
          padding,
        );
        validateRegion(region);
        // Release the display request promptly; shared block producers may still
        // finish for other requests and populate the cache.
        const generated = await Promise.race([generator.generate(region, request), cancelled]);
        request.assertCurrent();
        const pixels = renderRegion(generated, data.map, padding);
        // Transfer copies, never the cache buffer: transfer detaches the sender's storage.
        const elevations = crop(
          generated.values,
          generated.width,
          padding,
          padding,
          data.width,
          data.height,
        );
        emit(
          {
            type: "tile",
            id: data.id,
            key: data.key,
            model: generated.model,
            provider: generated.provider,
            width: data.width,
            height: data.height,
            pixels: pixels.buffer,
            elevations: elevations.buffer,
            elevationStats: summarizeElevations(elevations),
            metrics: {
              ...runtime.metrics,
              cacheHits: generator.cache.hits,
              cacheBytes: generator.cache.bytes,
              coalescedBlocks: generator.cache.coalesced,
              pendingBlocks: generator.cache.pending?.size || 0,
              tileModelRuns: runtime.metrics.modelRuns - beforeRuns,
            },
          },
          [pixels.buffer, elevations.buffer],
        );
      } catch (error) {
        if (error instanceof ObsoleteRequest) reply({ type: "cancelled" });
        else
          reply({ type: "error", message: error instanceof Error ? error.message : String(error) });
      }
    };
  }

  TerrainHiGAN.createWorkerHandler = createWorkerHandler;
  const emit = (message, transfers) => postMessage(message, transfers);
  const runtime = new InferencePool(runtimeOptions, emit);
  self.onmessage = createWorkerHandler(runtime, new TerrainGenerator(runtime), emit);
})();
