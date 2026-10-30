"use strict";
globalThis.TerrainHiGAN ||= {};
/** @typedef {{zoom: number, x: number, y: number, width: number, height: number}} Region
 * Region coordinates are global integer pixels, not the camera's meter coordinates.
 */
(() => {
  const MIN_ZOOM = 0;
  const MAX_ZOOM = 20;
  const HALO = 8;
  const MAX_COORDINATE = 2 ** 50;
  const clamp = (value, low, high) => Math.max(low, Math.min(high, value));
  const modelName = (zoom) => (zoom > 10 ? `z${zoom}s` : `z${zoom}`);
  const metersPerPixel = (zoom) => 512 * 2 ** (10 - zoom);
  const elevationMeters = (value) => value * 2439.687062251 - 2342.974207789;
  const parseSeed = (value) => BigInt.asUintN(64, BigInt(value?.trim() || Date.now()));

  function isMobileDevice(device = globalThis.navigator) {
    return !!(
      device?.userAgentData?.mobile ||
      /Android|iPhone|iPad|iPod|Mobile/i.test(device?.userAgent || "") ||
      (device?.platform === "MacIntel" && device?.maxTouchPoints > 1)
    );
  }

  function displayPixelRatio() {
    const ratio = globalThis.devicePixelRatio || 1;
    const mobile = isMobileDevice() || globalThis.matchMedia?.("(pointer: coarse)").matches;
    return mobile ? Math.min(2, ratio) : ratio;
  }

  function inferenceWorkerCount(options, device = globalThis.navigator) {
    // Each worker owns an ONNX runtime and its model sessions.
    const mobile = isMobileDevice(device);
    return Math.max(1, Math.min(4, Math.trunc(Number(options.get("workers")) || (mobile ? 1 : 4))));
  }

  function validateRegion(region) {
    for (const key of ["zoom", "x", "y", "width", "height"]) {
      if (!Number.isSafeInteger(region[key])) throw new RangeError(key + " must be a safe integer");
    }
    if (region.zoom < MIN_ZOOM || region.zoom > MAX_ZOOM)
      throw new RangeError("zoom must be in 0..20");
    if (region.width <= 0 || region.height <= 0)
      throw new RangeError("Region dimensions must be positive");
    // Leave room for render halos and doubled noise coordinates inside Number's exact integer range.
    if (
      region.x < -MAX_COORDINATE ||
      region.y < -MAX_COORDINATE ||
      region.x + region.width > MAX_COORDINATE ||
      region.y + region.height > MAX_COORDINATE
    ) {
      throw new RangeError("Region exceeds the browser coordinate range");
    }
  }

  function parentRegion({ zoom, x, y, width, height }) {
    // Floor/ceil retain the correct registration for negative and odd coordinates.
    const left = Math.floor((x - HALO) / 2);
    const top = Math.floor((y - HALO) / 2);
    return {
      zoom: zoom - 1,
      x: left,
      y: top,
      width: Math.ceil((x + width + HALO) / 2) - left,
      height: Math.ceil((y + height + HALO) / 2) - top,
    };
  }

  function padRegion(region, padding) {
    return {
      ...region,
      x: region.x - padding,
      y: region.y - padding,
      width: region.width + 2 * padding,
      height: region.height + 2 * padding,
    };
  }

  function crop(source, sourceWidth, left, top, width, height) {
    const output = new Float32Array(width * height);
    for (let row = 0; row < height; row++) {
      const start = (top + row) * sourceWidth + left;
      output.set(source.subarray(start, start + width), row * width);
    }
    return output;
  }

  function summarizeElevations(values, width = values.length, bounds = {}) {
    const { left = 0, top = 0, right = width, bottom = values.length / width } = bounds;
    let min = Infinity,
      max = -Infinity,
      sum = 0,
      sumSquares = 0,
      count = 0;
    for (let y = top; y < bottom; y++)
      for (let x = left; x < right; x++) {
        const value = elevationMeters(values[y * width + x]);
        min = Math.min(min, value);
        max = Math.max(max, value);
        sum += value;
        sumSquares += value * value;
        count++;
      }
    return { min, max, sum, sumSquares, count };
  }

  Object.assign(TerrainHiGAN, {
    MIN_ZOOM,
    MAX_ZOOM,
    HALO,
    MAX_COORDINATE,
    clamp,
    modelName,
    metersPerPixel,
    elevationMeters,
    parseSeed,
    isMobileDevice,
    displayPixelRatio,
    inferenceWorkerCount,
    validateRegion,
    parentRegion,
    padRegion,
    crop,
    summarizeElevations,
  });
})();
