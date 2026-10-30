const { demoRequire, rootDir } = require("../support/test-support.cjs");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { chromium } = demoRequire("playwright");
const { browserOptions } = demoRequire("./browser-options.cjs");
const checkWalkingInput = require("../support/test-walk-input-browser.cjs");
const baseUrl = process.env.DEMO_URL || "http://127.0.0.1:4173/src/terrain_higan/demos/web/";

(async () => {
  const browser = await chromium.launch(
    browserOptions({
      headless: true,
      args: ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
    }),
  );
  try {
    const page = await browser.newPage({ viewport: { width: 800, height: 600 } });
    const errors = [];
    page.on("pageerror", (error) => {
      errors.push(error.message);
      console.error(error.message);
    });
    if (!process.env.REAL_TERRAIN)
      await page.addInitScript(() => {
        window.Worker = class {
          constructor() {
            this.generation = 0;
            this.runs = 0;
          }
          terminate() {
            this.generation++;
          }
          postMessage(request) {
            if (["cancel", "reset"].includes(request.type)) {
              this.generation++;
              return;
            }
            const generation = this.generation;
            setTimeout(() => {
              if (generation !== this.generation) return;
              const normalized = (100 + 2342.974207789) / 2439.687062251;
              const data = {
                type: "tile",
                id: request.id,
                key: request.key,
                provider: { name: "wasm" },
                model: "test",
                metrics: { modelRuns: ++this.runs, inferenceWorkers: 1 },
              };
              if (request.type === "generate")
                Object.assign(data, {
                  width: request.width,
                  height: request.height,
                  pixels: new Uint8ClampedArray(request.width * request.height * 4).fill(255)
                    .buffer,
                  elevations: new Float32Array(request.width * request.height).fill(normalized)
                    .buffer,
                  elevationStats: { min: 100, max: 100, mean: 100, stddev: 0 },
                });
              else
                Object.assign(data, {
                  face: 0,
                  depth: request.depth,
                  tileX: request.tileX,
                  tileY: request.tileY,
                  level: request.depth,
                  values: new Float32Array(68 * 68).fill(normalized).buffer,
                  vertexValues: new Float32Array(67 * 67).fill(normalized).buffer,
                });
              this.onmessage({ data });
            }, 0);
          }
        };
      });
    await page.goto(`${baseUrl}?backend=wasm&workers=2#seed=1&z=18&x=-293988345&y=31759354`);
    await page.waitForFunction(
      () => Number(document.getElementById("world").dataset.modelRuns) > 0,
    );
    assert.equal(await page.locator("#experimental").getAttribute("open"), null);
    assert(!(await page.locator("#toggle-walk").isVisible()));
    await page.locator("#experimental > summary").click();
    assert(await page.locator("#toggle-walk").isVisible());
    await page.locator("#toggle-walk").click();
    await page.waitForFunction(
      () => document.getElementById("walk-world")?.dataset.walkPrepared === "true",
      {},
      { timeout: 300000 },
    );
    const minimap = page.locator(".walk-minimap canvas");
    assert(await minimap.isVisible());
    const initial = await page.locator("#walk-world").evaluate((canvas) => ({ ...canvas.dataset }));
    assert.equal(initial.maxModelLevel, "14");
    assert.equal(initial.targetModelLevel, "14");
    assert.equal(Number(await page.locator("#walk-speed").getAttribute("max")), 50);
    await page.locator("#walk-speed").evaluate((input) => {
      input.value = "50";
      input.dispatchEvent(new Event("input", { bubbles: true }));
    });
    await page.waitForFunction(
      () => document.getElementById("walk-requested").textContent === "50.0 m/s",
    );
    const initialCone = await minimap.evaluate((canvas) => ({ ...canvas.dataset }));
    assert.equal(initial.walkClearance, "2");
    assert.equal(initial.walkPaused, "false");
    assert(Math.abs(Number(initialCone.heading) + Math.PI / 2) < 0.0001);
    assert(
      Math.abs(Number(initialCone.fov) - 2 * Math.atan((Math.tan((42 * Math.PI) / 360) * 4) / 3)) <
        1e-8,
    );
    await checkWalkingInput(page, "#walk-world");
    assert(await minimap.isVisible(), "minimap remains visible with controls hidden");
    const moved = await page.locator("#walk-world").evaluate((canvas) => ({ ...canvas.dataset }));
    assert(
      await page.evaluate(() =>
        [...walkInstance.renderer.patchTiles.values()].every((tile) => tile.source.level <= 14),
      ),
    );
    assert(
      await page.evaluate(() => {
        const walk = walkInstance;
        return (
          walk.groundAt(walk.mapPosition(0, -100000)) !== null &&
          walk.renderer.camera.far * TerrainHiGAN.globe.RADIUS > 500000 &&
          walk.renderer.patchTiles.size < 6000 &&
          [...walk.renderer.patchTiles.values()].some(
            (tile) => walk.distance(tile.bounds).nearest <= 20000 && tile.source.level < 14,
          ) &&
          [...walk.renderer.patchTiles.values()].every(
            (tile) =>
              walk.distance(tile.bounds).nearest > TerrainHiGAN.globe.WALK_FINE_RADIUS ||
              tile.source.level === 14,
          )
        );
      }),
      "z14s covers the entry area with bounded, progressively coarser scenery",
    );
    assert.equal(moved.modelRuns, initial.modelRuns, "motion/rotation never infer new terrain");
    assert.equal(moved.walkSceneTiles, initial.walkSceneTiles);
    assert(
      await page.evaluate(
        () =>
          Math.abs(
            walkInstance.renderer.camera.position.y * TerrainHiGAN.globe.RADIUS -
              walkInstance.groundAt(walkInstance.renderer.camera.position) -
              2,
          ) < 1e-7,
      ),
    );
    await page.setViewportSize({ width: 900, height: 600 });
    await page.waitForFunction(
      (fov) => Number(document.querySelector(".walk-minimap canvas").dataset.fov) > fov,
      Number(initialCone.fov),
    );
    await page.keyboard.press("f");
    const downloadPromise = page.waitForEvent("download", { timeout: 120000 });
    await page.locator("#export-scene").click();
    const buffer = fs.readFileSync(await (await downloadPromise).path());
    assert.equal(buffer.readUInt32LE(0), 0x46546c67);
    const gltf = JSON.parse(buffer.subarray(20, 20 + buffer.readUInt32LE(12)).toString());
    assert(
      gltf.nodes.some(
        (node) => node.extras?.topology === "planar" && node.extras.originX === -293988345,
      ),
    );
    const output = path.join(rootDir, "runs/globe-viewer");
    fs.mkdirSync(output, { recursive: true });
    await page.screenshot({ path: path.join(output, "planar-walk.png") });
    await page.locator('[data-map="elevation"]').click();
    await page.waitForFunction(
      () => document.getElementById("walk-world")?.dataset.walkPrepared === "true",
    );
    assert.equal(
      await page.locator("#walk-world").getAttribute("data-model-runs"),
      initial.modelRuns,
      "restyling reuses the prepared terrain",
    );
    await page.locator("#toggle-walk").click();
    assert.equal(await page.locator("#walk-world").count(), 0);
    assert.equal(await page.locator(".walk-minimap").count(), 0);
    await page.keyboard.press("Home");
    await page.waitForFunction(() => {
      const params = new URLSearchParams(location.hash.slice(1));
      const zoom = Number(params.get("z"));
      const area =
        innerWidth * innerHeight * devicePixelRatio ** 2 * TerrainHiGAN.metersPerPixel(zoom) ** 2;
      const ratio = area / (4 * Math.PI * 6371000 ** 2);
      return params.get("x") === "0" && params.get("y") === "0" && ratio >= 0.5 && ratio <= 2;
    });
    await page.locator("#toggle-walk").click();
    await page.waitForFunction(
      () => document.getElementById("walk-world")?.dataset.walkPrepared === "true",
      {},
      { timeout: 300000 },
    );
    await page.locator("#seed").fill("2");
    await page.locator("#seed").dispatchEvent("change");
    assert.equal(await page.locator("#walk-world").count(), 0);
    assert.equal(await page.locator("#seed").inputValue(), "2");
    assert.equal(
      await page.evaluate(() => new URLSearchParams(location.hash.slice(1)).get("seed")),
      "2",
    );
    assert.deepEqual(errors, []);
    console.log(
      "Planar walk browser: preparation, mini map, FOV heading/aspect, motion, GLB, cached restyle, exit and reentry passed",
      { tiles: initial.walkSceneTiles, realTerrain: Boolean(process.env.REAL_TERRAIN) },
    );
  } finally {
    await browser.close();
  }
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
