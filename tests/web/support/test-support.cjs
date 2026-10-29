const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const { webcrypto } = require("node:crypto");
const { performance } = require("node:perf_hooks");
const { createRequire } = require("node:module");
const { pathToFileURL } = require("node:url");

const rootDir = path.resolve(__dirname, "../../..");
const demoDir = path.join(rootDir, "src/terrain_higan/demos/web");
const demoRequire = createRequire(path.join(demoDir, "package.json"));
const loadThree = () =>
  import(
    pathToFileURL(path.join(path.dirname(demoRequire.resolve("three")), "three.module.js")).href
  );

function loadModules(overrides = {}) {
  const context = vm.createContext({
    URL,
    URLSearchParams,
    TextEncoder,
    Float32Array,
    Uint8Array,
    Uint8ClampedArray,
    DataView,
    crypto: webcrypto,
    performance,
    console,
    postMessage() {},
    location: { href: "http://localhost/model-worker.js", search: "" },
    navigator: { hardwareConcurrency: 4 },
    ...overrides,
  });
  context.self = context;
  const load = (file) =>
    vm.runInContext(fs.readFileSync(path.join(demoDir, file), "utf8"), context, {
      filename: file,
    });
  context.importScripts = (...files) =>
    files.forEach((file) => {
      const name = file.split("?")[0].replace(/^\.\//, "");
      if (!name.startsWith("assets/runtime/")) load(name);
    });
  for (const file of [
    "terrain-common.js",
    "noise.js",
    "model-runtime.js",
    "inference-pool.js",
    "terrain.js",
    "rendering.js",
    "tile-controller.js",
  ])
    load(file);
  return { api: context.TerrainHiGAN, context, load };
}

function near(actual, expected, tolerance = 2e-5) {
  const assert = require("node:assert/strict");
  assert.equal(actual.length, expected.length);
  for (let i = 0; i < actual.length; i++) {
    assert(
      Math.abs(actual[i] - expected[i]) < tolerance,
      `pixel ${i}: ${actual[i]} vs ${expected[i]}`,
    );
  }
}

module.exports = { loadModules, near, rootDir, demoDir, demoRequire, loadThree };
