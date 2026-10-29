"use strict";
globalThis.TerrainHiGAN ||= {};
(() => {
  const noiseKeys = new Map();
  function multiplyHighLow(left, right) {
    const leftLow = left & 0xffff,
      leftHigh = left >>> 16;
    const rightLow = right & 0xffff,
      rightHigh = right >>> 16;
    const low = Math.imul(leftLow, rightLow) >>> 0;
    const middleLeft = Math.imul(leftHigh, rightLow) >>> 0;
    const middleRight = Math.imul(leftLow, rightHigh) >>> 0;
    const middle = (low >>> 16) + (middleLeft & 0xffff) + (middleRight & 0xffff);
    return [
      (Math.imul(leftHigh, rightHigh) +
        (middleLeft >>> 16) +
        (middleRight >>> 16) +
        (middle >>> 16)) >>>
        0,
      ((low & 0xffff) | ((middle & 0xffff) << 16)) >>> 0,
    ];
  }

  function philox4x32(counter, key) {
    let [c0, c1, c2, c3] = counter;
    let [k0, k1] = key;
    for (let round = 0; round < 10; round++) {
      const [hi0, lo0] = multiplyHighLow(c0, 0xd2511f53);
      const [hi1, lo1] = multiplyHighLow(c2, 0xcd9e8d57);
      [c0, c1, c2, c3] = [(hi1 ^ c1 ^ k0) >>> 0, lo1, (hi0 ^ c3 ^ k1) >>> 0, lo0];
      k0 = (k0 + 0x9e3779b9) >>> 0;
      k1 = (k1 + 0xbb67ae85) >>> 0;
    }
    return [c0, c1, c2, c3];
  }

  function coordinateWords(value) {
    const words = BigInt.asUintN(64, BigInt(Math.trunc(value)));
    return [Number(words & 0xffffffffn), Number(words >> 32n)];
  }

  async function noiseKey(seed, zoom) {
    const cacheKey = `${seed}:${zoom}`;
    if (!noiseKeys.has(cacheKey)) {
      const prefix = new TextEncoder().encode("wfcai.inference.noise.v1");
      const bytes = new Uint8Array(prefix.length + 10);
      bytes.set(prefix);
      const view = new DataView(bytes.buffer);
      view.setBigUint64(prefix.length, BigInt.asUintN(64, seed), true);
      view.setUint16(prefix.length + 8, zoom, true);
      // Plain HTTP LAN origins do not expose Web Crypto.
      const digestPromise = globalThis.crypto?.subtle
        ? crypto.subtle.digest("SHA-256", bytes)
        : Promise.resolve(sha256.arrayBuffer(bytes));
      noiseKeys.set(
        cacheKey,
        digestPromise.then((digest) => {
          const keyView = new DataView(digest);
          return [keyView.getUint32(0, true), keyView.getUint32(4, true)];
        }),
      );
    }
    return noiseKeys.get(cacheKey);
  }

  async function normalField(seed, zoom, x, y, width, height) {
    const key = await noiseKey(seed, zoom);
    const values = new Float32Array(width * height);
    let index = 0;
    const [firstXLow, firstXHigh] = coordinateWords(x);
    let [yLow, yHigh] = coordinateWords(y);
    for (let row = 0; row < height; row++) {
      let xLow = firstXLow,
        xHigh = firstXHigh;
      for (let column = 0; column < width; column++) {
        const [word0, word1] = philox4x32([xLow, xHigh, yLow, yHigh], key);
        const first = (word0 + 0.5) / 4294967296;
        const second = (word1 + 0.5) / 4294967296;
        values[index++] = Math.sqrt(-2 * Math.log(first)) * Math.cos(2 * Math.PI * second);
        xLow = (xLow + 1) >>> 0;
        if (xLow === 0) xHigh = (xHigh + 1) >>> 0;
      }
      yLow = (yLow + 1) >>> 0;
      if (yLow === 0) yHigh = (yHigh + 1) >>> 0;
    }
    return values;
  }

  async function upscalerNoise(seed, zoom, x, y, width, height) {
    const source = await normalField(seed, zoom, 2 * x, 2 * y, 2 * width, 2 * height);
    const output = new Float32Array(4 * width * height);
    for (let row = 0; row < height; row++)
      for (let column = 0; column < width; column++) {
        const sourceRow = row * 2,
          sourceColumn = column * 2,
          destination = row * width + column;
        output[destination] = source[sourceRow * 2 * width + sourceColumn];
        output[width * height + destination] = source[sourceRow * 2 * width + sourceColumn + 1];
        output[2 * width * height + destination] =
          source[(sourceRow + 1) * 2 * width + sourceColumn];
        output[3 * width * height + destination] =
          source[(sourceRow + 1) * 2 * width + sourceColumn + 1];
      }
    return output;
  }

  Object.assign(TerrainHiGAN, {
    normalField,
    upscalerNoise,
    philox4x32,
    coordinateWords,
    noiseKey,
  });
})();
