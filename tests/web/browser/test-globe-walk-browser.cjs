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
    const page = await browser.newPage({ viewport: { width: 640, height: 480 } });
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
            this.requests = 0;
          }
          terminate() {
            this.generation++;
          }
          postMessage(request) {
            if (request.type === "reset" || request.type === "cancel") {
              this.generation++;
              return;
            }
            const generation = this.generation;
            setTimeout(() => {
              if (generation !== this.generation) return;
              const height = request.seed === "2" ? -1000 : 100;
              const normalized = (height + 2342.974207789) / 2439.687062251;
              const values = new Float32Array(68 * 68).fill(normalized);
              const vertexValues = new Float32Array(67 * 67).fill(normalized);
              const tile = { ...request };
              delete tile.bounds;
              delete tile.children;
              this.onmessage({
                data: {
                  ...tile,
                  type: "tile",
                  level: request.depth + 2,
                  values: values.buffer,
                  vertexValues: vertexValues.buffer,
                  provider: { name: "wasm" },
                  metrics: {
                    modelRuns: ++this.requests,
                    inferenceWorkers: 1,
                    cacheBytes: 0,
                    cacheHits: 0,
                    maxParallelRuns: 1,
                  },
                },
              });
            }, 0);
          }
        };
      });
    await page.goto(`${baseUrl}globe.html#seed=1&lat=20&lon=-25&altitude=1000`);
    assert.equal(await page.locator("#experimental").getAttribute("open"), null);
    assert(!(await page.locator("#toggle-walk").isVisible()));
    await page.locator("#experimental > summary").click();
    assert(await page.locator("#toggle-walk").isVisible());
    await page.goto(
      `${baseUrl}globe.html?backend=wasm&workers=2#seed=1&mode=walk&lat=20&lon=-25&altitude=1000&speed=50`,
    );
    await page.waitForFunction(() => document.getElementById("world").dataset.walk === "true");
    assert.equal(await page.locator("#world").getAttribute("data-walk-paused"), "true");
    assert(await page.locator("#toggle-walk").isVisible());
    assert(await page.locator("#export-scene").isDisabled());
    await page.waitForFunction(
      () => document.getElementById("world").dataset.walkPrepared === "true",
      {},
      { timeout: 300000 },
    );
    assert(!(await page.locator("#export-scene").isDisabled()));
    assert.equal(Number(await page.locator("#walk-speed").getAttribute("max")), 50);
    assert.equal(Number(await page.locator("#walk-speed").inputValue()), 50);
    await page.locator("#walk-speed").evaluate((input) => {
      input.value = "49";
      input.dispatchEvent(new Event("input", { bubbles: true }));
    });
    await page.waitForFunction(
      () => document.getElementById("walk-requested").textContent === "49.0 m/s",
    );
    await page.locator("#walk-speed").evaluate((input) => {
      input.value = "50";
      input.dispatchEvent(new Event("input", { bubbles: true }));
    });
    assert.equal(Number(await page.locator("#world").getAttribute("data-walk-clearance")), 2);
    if (!process.env.REAL_TERRAIN)
      assert(
        Math.abs(Number(await page.locator("#world").getAttribute("data-walk-altitude")) - 102) <
          0.01,
      );
    const initial = await page.locator("#world").evaluate((canvas) => ({ ...canvas.dataset }));
    assert.equal(initial.maxModelLevel, "14");
    assert.equal(initial.targetModelLevel, "14");
    const minimap = page.locator(".walk-minimap canvas");
    assert(await minimap.isVisible());
    const initialCone = await minimap.evaluate((canvas) => ({ ...canvas.dataset }));
    assert(Number(initialCone.fov) > 0);
    await checkWalkingInput(page, "#world");
    const moved = await page.locator("#world").evaluate((canvas) => ({ ...canvas.dataset }));
    const movedCone = await minimap.evaluate((canvas) => ({ ...canvas.dataset }));
    assert(Math.abs(Number(movedCone.heading) - Number(initialCone.heading)) > 0.01);
    assert.equal(
      moved.modelRuns,
      initial.modelRuns,
      "walking and turning request no terrain generation",
    );
    assert.equal(
      moved.visibleTiles,
      initial.visibleTiles,
      "turning retains all-direction scene geometry",
    );
    assert.equal(moved.walkSceneTiles, initial.walkSceneTiles);
    const depthChecks = await page.evaluate(async () => {
      const THREE = await import("three");
      const { GlobeRenderer, RADIUS } = TerrainHiGAN.globe;
      const renderer = new GlobeRenderer(document.createElement("canvas"), () => {}, false);
      renderer.walk = { active: true };
      renderer.renderer.setSize(64, 64, false);
      renderer.scene.clear();
      const results = [];
      for (const scale of [1 / RADIUS, 1]) {
        for (const distance of [20000, 100000, 300000, 650000, 900000]) {
          const geometry = new THREE.PlaneGeometry(distance * 0.1 * scale, distance * 0.1 * scale);
          const front = new THREE.Mesh(
            geometry,
            renderer.depthMaterial(new THREE.MeshBasicMaterial({ color: 0xff0000 })),
          );
          const back = new THREE.Mesh(
            geometry,
            renderer.depthMaterial(new THREE.MeshBasicMaterial({ color: 0x00ff00 })),
          );
          front.position.z = -distance * scale;
          back.position.z = -(distance + 1) * scale;
          back.renderOrder = 1;
          renderer.scene.add(front, back);
          const camera = renderer.camera;
          camera.aspect = 1;
          camera.near = 0.05 * scale;
          camera.far = (walkInstance.sceneryRadius() + 60000) * scale;
          camera.updateProjectionMatrix();
          for (const offset of [0, 0.1, 1, 5]) {
            camera.position.set(offset * scale, 2 * scale, 0);
            camera.lookAt(offset * scale, 2 * scale, -scale);
            camera.updateMatrixWorld();
            renderer.renderDirty = true;
            renderer.draw();
            const pixel = new Uint8Array(4);
            const gl = renderer.renderer.getContext();
            gl.readPixels(32, 32, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, pixel);
            results.push({ color: [...pixel], error: gl.getError() });
          }
          renderer.scene.remove(front, back);
          geometry.dispose();
          front.material.dispose();
          back.material.dispose();
        }
      }
      renderer.renderer.dispose();
      renderer.renderer.forceContextLoss();
      return results;
    });
    for (const result of depthChecks) {
      assert.equal(result.error, 0, "walking depth shaders compile and render");
      assert(
        result.color[0] > 240 && result.color[1] < 10,
        "polygons at 20–900 km stay visible and separate from surfaces one meter behind as the camera moves",
      );
    }
    assert(
      await page.evaluate(
        () =>
          Math.abs(
            (walkInstance.renderer.camera.position.length() - 1) * TerrainHiGAN.globe.RADIUS -
              walkInstance.groundAt(walkInstance.renderer.camera.position) -
              2,
          ) < 1e-5,
      ),
    );
    assert(
      await page.evaluate(() =>
        [...walkInstance.renderer.patchTiles.values()].every((tile) => tile.source.level <= 14),
      ),
    );
    assert(
      await page.evaluate(() => {
        const walk = walkInstance;
        const direction = walk.mapPosition(0, -900000);
        return (
          walk.groundAt(direction) !== null &&
          walk.renderer.camera.far * TerrainHiGAN.globe.RADIUS > walk.sceneryRadius() &&
          walk.renderer.patchTiles.size < 6000 &&
          [...walk.renderer.patchTiles.values()].some(
            (tile) => walk.distance(tile.bounds).nearest <= 20000 && tile.source.level < 14,
          ) &&
          [...walk.renderer.patchTiles.values()].every(
            (tile) =>
              tile.mesh.userData.segments === 64 &&
              Number.isFinite(tile.horizon.maximumHeight) &&
              (walk.distance(tile.bounds).nearest > TerrainHiGAN.globe.WALK_FINE_RADIUS ||
                tile.source.level === 14),
          )
        );
      }),
      "native geometry covers the full walking area and height-aware distant backdrop",
    );
    if (!process.env.REAL_TERRAIN)
      assert(
        await page.evaluate(() => {
          const walk = walkInstance;
          const patches = [...walk.patchIndex.values()];
          const peakReach = TerrainHiGAN.globe.horizonDistance(102, 100);
          return (
            patches.some((tile) => tile.mesh.visible) &&
            patches.some((tile) => !tile.mesh.visible) &&
            patches.every(
              (tile) =>
                TerrainHiGAN.globe.greatCircle(
                  walk.renderer.camera.position.clone().normalize().toArray(),
                  tile.bounds.center.toArray(),
                ) <=
                  peakReach + walk.distance(tile.bounds).extent + 1 || !tile.mesh.visible,
            )
          );
        }),
        "actual surface-height bounds hide flat 100 m scenery beyond the two-height horizon",
      );
    await page.keyboard.press("f");
    const downloadPromise = page.waitForEvent("download", { timeout: 120000 });
    await page.locator("#export-scene").click();
    const download = await downloadPromise;
    const buffer = fs.readFileSync(await download.path());
    assert.equal(buffer.readUInt32LE(0), 0x46546c67);
    assert.equal(buffer.readUInt32LE(4), 2);
    assert.equal(buffer.readUInt32LE(8), buffer.length);
    const gltf = JSON.parse(buffer.subarray(20, 20 + buffer.readUInt32LE(12)).toString());
    const metadata = gltf.nodes.find((node) => node.extras?.units === "meters").extras;
    assert.equal(metadata.radiusMeters, 20000);
    const exportedTileCount = await page.evaluate(
      () =>
        [...walkInstance.renderer.patchTiles.values()].filter(
          (tile) => walkInstance.distance(tile.bounds).nearest <= TerrainHiGAN.globe.WALK_RADIUS,
        ).length,
    );
    assert.equal(gltf.meshes.length, exportedTileCount * 2);
    assert(gltf.images.length > 0, "GLB includes terrain textures");
    assert(
      gltf.accessors.some(
        (accessor) => accessor.type === "VEC3" && accessor.max?.some((value) => value > 19000),
      ),
      "GLB has kilometer-scale local meter coordinates",
    );
    const output = path.join(rootDir, "runs/globe-viewer");
    fs.mkdirSync(output, { recursive: true });
    await page.screenshot({ path: path.join(output, "walk.png") });
    await page.reload();
    await page.waitForFunction(
      () => document.getElementById("world").dataset.walkPrepared === "true",
      {},
      { timeout: 300000 },
    );
    assert.equal(await page.locator("#world").getAttribute("data-walk-paused"), "true");
    assert(await page.locator("#toggle-walk").isVisible());
    if (!process.env.REAL_TERRAIN) {
      await page.locator(".advanced > summary").click();
      await page.locator("#seed").fill("2");
      await page.locator("#seed").dispatchEvent("change");
      await page.waitForFunction(
        () =>
          document.getElementById("world").dataset.walkPrepared === "true" &&
          Number(document.getElementById("world").dataset.walkAltitude) < 0,
        {},
        { timeout: 60000 },
      );
      assert.equal(Number(await page.locator("#world").getAttribute("data-walk-clearance")), 2);
    }
    await page.locator("#toggle-walk").click();
    await page.waitForFunction(() => document.getElementById("world").dataset.walk === "false");
    assert(!(await minimap.isVisible()));
    assert(!(await page.locator("#exaggeration").isDisabled()));
    await page.keyboard.press("Home");
    await page.waitForFunction(() => document.getElementById("toggle-walk").disabled);
    assert.deepEqual(errors, []);
    console.log(
      "Walk browser: preparation gate, fixed scene on motion/rotation, ground pin, GLB download, paused URL restore, seed rebuild and orbit exit passed",
      {
        tiles: initial.walkSceneTiles,
        exportBytes: buffer.length,
        realTerrain: Boolean(process.env.REAL_TERRAIN),
      },
    );
  } finally {
    await browser.close();
  }
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
