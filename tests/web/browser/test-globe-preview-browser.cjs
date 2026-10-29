const { demoRequire, rootDir } = require("../support/test-support.cjs");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { chromium } = demoRequire("playwright");
const { browserOptions } = demoRequire("./browser-options.cjs");
const baseUrl = process.env.DEMO_URL || "http://127.0.0.1:4173/src/terrain_higan/demos/web/";
const output = path.join(rootDir, "runs/globe-viewer");

(async () => {
  const browser = await chromium.launch(
    browserOptions({
      headless: process.env.HEADED !== "1",
      args:
        process.env.HEADED === "1"
          ? []
          : ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
    }),
  );
  try {
    const page = await browser.newPage({ viewport: { width: 640, height: 480 } });
    const errors = [];
    page.on("pageerror", (error) => errors.push(error.message));
    await page.addInitScript(() => {
      const NativeWorker = globalThis.Worker;
      globalThis.blockedFineTiles = 0;
      globalThis.Worker = class extends NativeWorker {
        constructor(url, options) {
          super(url, options);
          this.globe = String(url).includes("globe-worker.js");
        }
        postMessage(message, transfer) {
          if (this.globe && message.type === "tile" && message.depth > 0) {
            globalThis.blockedFineTiles++;
            return;
          }
          return super.postMessage(message, transfer);
        }
      };
    });
    await page.goto(`${baseUrl}globe.html?workers=2&backend=wasm`);
    await page.waitForFunction(() => globalThis.TerrainHiGAN?.globe?.GlobeRenderer);
    await page.evaluate(() => {
      const prototype = TerrainHiGAN.globe.GlobeRenderer.prototype;
      const draw = prototype.draw;
      prototype.draw = function (...args) {
        globalThis.previewRenderer = this;
        return draw.apply(this, args);
      };
      const show = prototype.show;
      globalThis.meshUpdates = [];
      prototype.show = function (...args) {
        const started = performance.now();
        const result = show.apply(this, args);
        meshUpdates.push(performance.now() - started);
        return result;
      };
    });
    await page.waitForFunction(
      () => blockedFineTiles > 0 && globalThis.previewRenderer?.patchTiles?.size > 0,
      {},
      { timeout: 120000 },
    );
    assert.equal(await page.locator("#status").innerText(), "Loading");
    const runs = await page.locator("#world").getAttribute("data-model-runs");
    await page.keyboard.press("f");
    await page.mouse.move(320, 240);
    const started = Date.now();
    for (let index = 0; index < 30; index++) await page.mouse.wheel(0, -700);
    await page.waitForFunction(() =>
      [...previewRenderer.patchTiles.values()].some((patch) => patch.depth === 18),
    );
    const report = await page.evaluate(() => {
      const renderer = previewRenderer;
      const globe = TerrainHiGAN.globe;
      let maximumRadialError = 0;
      for (let row = 0; row < 5; row++)
        for (let column = 0; column < 7; column++) {
          renderer.surfacePick(((column + 0.5) * 640) / 7, ((row + 0.5) * 480) / 5);
          const hit = renderer.raycaster.intersectObjects([...renderer.tiles], false)[0];
          if (!hit) throw new Error("Missing preview surface ray");
          const coordinate = globe.vectorFace(hit.point.toArray());
          const patch = [...renderer.patchTiles.values()].find((tile) => {
            const east = coordinate.east * 2 ** tile.depth - tile.tileX;
            const north = coordinate.north * 2 ** tile.depth - tile.tileY;
            return (
              coordinate.face === tile.face && east >= 0 && east <= 1 && north >= 0 && north <= 1
            );
          });
          if (!patch) throw new Error("Preview ray has no spatial patch");
          const source = patch.source;
          const height = renderer.vertexHeight(
            source,
            (coordinate.east * 2 ** source.depth - source.tileX) * 64,
            (coordinate.north * 2 ** source.depth - source.tileY) * 64,
          );
          maximumRadialError = Math.max(
            maximumRadialError,
            Math.abs((hit.point.length() - 1) * globe.RADIUS - Math.max(0, height)),
          );
        }
      return {
        sourceLevels: [
          ...new Set([...renderer.patchTiles.values()].map((patch) => patch.source.level)),
        ],
        maximumGeometryDepth: Math.max(
          ...[...renderer.patchTiles.values()].map((patch) => patch.depth),
        ),
        patches: renderer.patchTiles.size,
        maximumRadialError,
        maximumMeshUpdateMs: Math.max(...meshUpdates),
        clearance: renderer.view().clearance,
      };
    });
    report.zoomMs = Date.now() - started;
    assert.deepEqual(report.sourceLevels, [2], "fine inference remains blocked");
    assert.equal(report.maximumGeometryDepth, 18);
    assert(
      report.maximumRadialError < 0.1,
      "near-camera triangles stay on the curved preview surface",
    );
    assert(report.clearance > 0);
    assert.equal(await page.locator("#world").getAttribute("data-model-runs"), runs);
    assert.deepEqual(errors, []);
    fs.mkdirSync(output, { recursive: true });
    await page.screenshot({ path: path.join(output, "pending-refinement.png") });
    fs.writeFileSync(
      path.join(output, "preview-checks.json"),
      JSON.stringify(report, null, 2) + "\n",
    );
    console.log("Globe delayed-inference preview checks passed", report);
  } finally {
    await browser.close();
  }
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
