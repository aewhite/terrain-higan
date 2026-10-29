const { demoRequire, rootDir } = require("../support/test-support.cjs");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { chromium } = demoRequire("playwright");
const { browserOptions } = demoRequire("./browser-options.cjs");
const baseUrl = process.env.DEMO_URL || "http://127.0.0.1:4173/src/terrain_higan/demos/web/";
const screenshots = path.join(rootDir, "runs/demo-inference-refactor");
fs.mkdirSync(screenshots, { recursive: true });

async function ready(page, timeout = 120000) {
  await page.waitForFunction(
    () => /Ready|Model failed/.test(document.querySelector("#status").textContent),
    {},
    { timeout },
  );
  assert.equal(await page.locator("#status").innerText(), "Ready");
}

async function inspectCanvas(page, name) {
  const result = await page.evaluate(() => {
    const canvas = document.querySelector("#world");
    const pixels = canvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height).data;
    const colors = new Set();
    for (let i = 0; i < pixels.length; i += 4)
      colors.add(pixels[i] * 65536 + pixels[i + 1] * 256 + pixels[i + 2]);
    return { colors: colors.size, width: canvas.width, height: canvas.height };
  });
  assert(result.colors > 20, "canvas must contain rendered terrain");
  await page.screenshot({ path: path.join(screenshots, name + ".png") });
  return result;
}

async function checkCoastlineSeam(browser) {
  const page = await browser.newPage();
  try {
    await page.route(baseUrl, async (route) => {
      const response = await route.fetch();
      await route.fulfill({ response, body: "<!doctype html><title>Terrain seam check</title>" });
    });
    await page.goto(baseUrl);
    const result = await page.evaluate(async () => {
      const makeGenerator = (workers) => {
        const worker = new Worker(`./model-worker.js?backend=wasm&threads=4&workers=${workers}`);
        let id = 0;
        const pending = new Map();
        worker.onmessage = ({ data }) => {
          const request = pending.get(data.id);
          if (!request || !["tile", "cancelled", "error"].includes(data.type)) return;
          pending.delete(data.id);
          clearTimeout(request.timeout);
          if (data.type === "error") request.reject(new Error(data.message));
          else
            request.resolve({
              ...data,
              values: data.elevations ? new Float32Array(data.elevations) : null,
              elapsedMs: performance.now() - request.started,
            });
        };
        worker.onerror = (event) => {
          for (const request of pending.values()) {
            clearTimeout(request.timeout);
            request.reject(new Error(event.message));
          }
          pending.clear();
        };
        return {
          generate(y, height, map = "realistic", zoom = 14, seed = "4829347") {
            return new Promise((resolve, reject) => {
              const key = String(++id);
              const timeout = setTimeout(
                () => reject(new Error("Block generation timed out")),
                180000,
              );
              pending.set(id, { resolve, reject, timeout, started: performance.now() });
              worker.postMessage({
                type: "generate",
                id,
                key,
                seed,
                zoom,
                x: 1576960,
                y,
                width: 256,
                height,
                map,
              });
            });
          },
          cancel() {
            worker.postMessage({ type: "cancel", id: ++id });
          },
          close() {
            worker.terminate();
          },
        };
      };
      const parallel = makeGenerator(4),
        serial = makeGenerator(1);
      try {
        // Compare both sides of the reported coastline break with one continuous request.
        const reference = await parallel.generate(1064960, 512);
        const upper = await parallel.generate(1064960, 256);
        const lower = await parallel.generate(1065216, 256);
        let maximumMeters = 0;
        for (let index = 0; index < reference.values.length; index++) {
          const value =
            index < upper.values.length
              ? upper.values[index]
              : lower.values[index - upper.values.length];
          maximumMeters = Math.max(
            maximumMeters,
            Math.abs(reference.values[index] - value) * 2439.687062251,
          );
        }
        const contour = await parallel.generate(1064960, 256, "contour");
        const baseline = await serial.generate(1064960, 512);
        let workerDifferences = 0,
          paletteDifferences = 0;
        for (let index = 0; index < reference.values.length; index++)
          if (reference.values[index] !== baseline.values[index]) workerDifferences++;
        for (let index = 0; index < upper.values.length; index++)
          if (upper.values[index] !== contour.values[index]) paletteDifferences++;
        const warmParallel = await parallel.generate(1064960, 512, "realistic", 14, "4829348");
        const warmSerial = await serial.generate(1064960, 512, "realistic", 14, "4829348");
        let warmWorkerDifferences = 0,
          seedDifferences = 0;
        for (let index = 0; index < reference.values.length; index++) {
          if (warmParallel.values[index] !== warmSerial.values[index]) warmWorkerDifferences++;
          if (warmParallel.values[index] !== reference.values[index]) seedDifferences++;
        }
        const abandoned = parallel.generate(1064960, 256, "realistic", 15);
        parallel.cancel();
        const cancelled = await abandoned;
        const revisit = await parallel.generate(1064960, 256);
        let revisitDifferences = 0;
        for (let index = 0; index < upper.values.length; index++)
          if (upper.values[index] !== revisit.values[index]) revisitDifferences++;
        return {
          maximumMeters,
          workerDifferences,
          paletteDifferences,
          warmWorkerDifferences,
          seedDifferences,
          revisitDifferences,
          cancelled: cancelled.type,
          cancellationMs: cancelled.elapsedMs,
          provider: reference.provider.label,
          metrics: reference.metrics,
          cropRuns: lower.metrics.modelRuns - reference.metrics.modelRuns,
          paletteRuns: contour.metrics.modelRuns - reference.metrics.modelRuns,
          serialRuns: baseline.metrics.modelRuns,
          parallelMs: reference.elapsedMs,
          serialMs: baseline.elapsedMs,
          warmParallelMs: warmParallel.elapsedMs,
          warmSerialMs: warmSerial.elapsedMs,
        };
      } finally {
        parallel.close();
        serial.close();
      }
    });
    assert.equal(result.provider, "CPU / WASM (4 workers)");
    assert.equal(result.metrics.inferenceWorkers, 4);
    assert.equal(result.metrics.maxParallelRuns, 4);
    assert(result.metrics.coalescedBlocks > 0);
    assert.equal(result.workerDifferences, 0, "worker count cannot change terrain");
    assert.equal(result.paletteDifferences, 0, "render halos cannot change terrain");
    assert.equal(result.warmWorkerDifferences, 0, "warm runtimes agree for another seed");
    assert(result.seedDifferences > 0, "seed cannot alias another world's cache");
    assert.equal(result.revisitDifferences, 0, "cancellation cannot change cached terrain");
    assert.equal(result.cropRuns, 0, "partial requests reuse complete blocks");
    assert.equal(result.paletteRuns, 0, "a new colormap reuses elevations");
    assert.equal(
      result.serialRuns,
      result.metrics.modelRuns,
      "worker count cannot duplicate block inference",
    );
    assert.equal(result.cancelled, "cancelled");
    assert(result.cancellationMs < 5000);
    assert(
      result.maximumMeters < 0.001,
      `Neighboring coastline tiles disagree by ${result.maximumMeters} meters`,
    );
    console.log(JSON.stringify({ coastlineSeam: result }));
  } finally {
    await page.close();
  }
}

(async () => {
  const browser = await chromium.launch(
    browserOptions({
      headless: true,
      args: ["--headless=new"],
    }),
  );
  try {
    if (process.env.SEAM_ONLY === "1") {
      await checkCoastlineSeam(browser);
      return;
    }
    const page = await browser.newPage({
      viewport: { width: 400, height: 320 },
      deviceScaleFactor: 2,
    });
    const errors = [];
    page.on("pageerror", (error) => errors.push(error.message));
    await page.goto(baseUrl + "?backend=wasm&threads=4#seed=4829347");
    await ready(page);
    assert.equal(new URLSearchParams(new URL(page.url()).hash.slice(1)).get("z"), "0");
    assert(await page.locator("#seed").isVisible());
    assert.equal(await page.locator("#seed").inputValue(), "4829347");
    await page.locator(".advanced summary").click();
    assert.equal(await page.locator("#model").innerText(), "z0");
    assert.match(await page.locator("#bundle").innerText(), /^[0-9a-f]{8}$/);
    assert.equal(await page.locator("#threads").innerText(), "4 (4 workers x 1)");
    assert.equal(await page.locator("#backend").innerText(), "CPU / WASM");
    assert.equal(await page.locator("#render-indicator").getAttribute("data-state"), "ready");
    assert.match(await page.locator("#visible-tiles").innerText(), /^\d+ \/ \d+$/);
    assert.equal(await page.locator("#zoom-in, #zoom-out").count(), 0);
    await page.locator("#toggle-shortcuts").click();
    assert(await page.locator("#shortcuts-dialog").isVisible());
    assert.equal(await page.locator(".shortcuts-table tbody tr").count(), 10);
    await page.keyboard.press("Escape");
    assert(!(await page.locator("#shortcuts-dialog").isVisible()));
    assert.equal(await page.locator("#world").getAttribute("data-max-parallel-runs"), "4");
    assert(await page.evaluate(() => crossOriginIsolated));
    assert.deepEqual(await page.evaluate(() => [world.width, world.height]), [800, 640]);
    await page.waitForFunction(() => / m$/.test(document.querySelector("#view-mean").textContent));
    await page.mouse.move(40, 160);
    assert.match(await page.locator("#elevation").innerText(), / m$/);
    for (const id of ["view-min", "view-max", "view-mean", "view-stddev"])
      assert.match(await page.locator(`#${id}`).innerText(), / m$/);
    await page.keyboard.press("f");
    assert(await page.locator("body").evaluate((body) => body.classList.contains("map-only")));
    assert.equal(
      await page.locator(".shell").evaluate((element) => getComputedStyle(element).display),
      "none",
    );
    assert(await page.locator(".scale").isVisible());
    await page.keyboard.press("Escape");
    assert(!(await page.locator("body").evaluate((body) => body.classList.contains("map-only"))));
    await page.locator('[data-map="contour"]').click();
    assert.equal(await page.locator('[data-map="contour"]').getAttribute("aria-pressed"), "true");
    await ready(page);
    const runs = Number(await page.locator("#world").getAttribute("data-model-runs"));
    const sourceColors = new Set(
      await page.evaluate(() => {
        const data = world.getContext("2d").getImageData(0, 0, world.width, world.height).data;
        const colors = new Set();
        for (let index = 0; index < data.length; index += 4)
          colors.add(data[index] * 65536 + data[index + 1] * 256 + data[index + 2]);
        return [...colors];
      }),
    );
    let release;
    const blocked = new Promise((resolve) => {
      release = resolve;
    });
    await page.route("**/assets/models/z1.onnx*", async (route) => {
      await blocked;
      await route.continue();
    });
    await page.keyboard.press("+");
    assert.equal(new URLSearchParams(new URL(page.url()).hash.slice(1)).get("z"), "1");
    assert.equal(await page.locator("#model").innerText(), "Loading");
    const preview = await page.evaluate(() => {
      const data = world.getContext("2d").getImageData(0, 0, world.width, world.height).data;
      const colors = new Set();
      for (let i = 0; i < data.length; i += 4)
        colors.add(data[i] * 65536 + data[i + 1] * 256 + data[i + 2]);
      return {
        colors: [...colors],
        smoothing: world.getContext("2d").imageSmoothingEnabled,
        runs: Number(world.dataset.modelRuns),
      };
    });
    assert(preview.colors.length > 1, "zoom should preserve visible terrain before z1 loads");
    assert.equal(preview.smoothing, false);
    assert(
      preview.colors.every((color) => sourceColors.has(color)),
      "zoom previews should preserve source colors without blending",
    );
    assert.equal(preview.runs, runs);
    await page.screenshot({ path: path.join(screenshots, "zoom-preview.png") });
    release();
    await ready(page);
    assert.equal(await page.locator("#model").innerText(), "z1");
    assert.equal(await page.locator("#status").innerText(), "Ready");
    const after = Number(await page.locator("#world").getAttribute("data-model-runs"));
    await page.keyboard.press("-");
    await ready(page);
    assert.equal(await page.locator("#model").innerText(), "z0");
    assert.equal(
      Number(await page.locator("#world").getAttribute("data-model-runs")),
      after,
      "cached zoom-out should perform no inference",
    );
    assert.deepEqual(errors, []);
    const bundlePage = await browser.newPage();
    let changedBundle = false;
    await bundlePage.route("**/assets/models/manifest.json*", async (route) => {
      const response = await route.fetch();
      const manifest = await response.json();
      if (changedBundle) manifest.bundle = "0000000000000000";
      await route.fulfill({ response, body: JSON.stringify(manifest) });
    });
    await bundlePage.goto(baseUrl + "?backend=wasm");
    await bundlePage.waitForFunction(() =>
      /^[0-9a-f]{8}$/.test(document.querySelector("#bundle").textContent),
    );
    changedBundle = true;
    await bundlePage.evaluate(() => dispatchEvent(new Event("focus")));
    await bundlePage.waitForFunction(
      () => document.querySelector("#bundle").textContent === "00000000",
      {},
      { timeout: 30000 },
    );
    await bundlePage.close();
    await page.setViewportSize({ width: 375, height: 812 });
    await page.waitForFunction(() => world.width === 750 && world.height === 1624);
    await ready(page);
    await inspectCanvas(page, "mobile");
    await page.keyboard.press("+");
    await page.keyboard.press("+");
    await page.keyboard.press("-");
    await page.keyboard.press("Home");
    await ready(page);
    assert.equal(await page.locator("#model").innerText(), "z0");
    assert.equal(await page.locator("#backend").innerText(), "CPU / WASM");
    await page.setViewportSize({ width: 1100, height: 760 });
    await page.waitForFunction(() => world.width === 2200 && world.height === 1520);
    await ready(page, 300000);
    await inspectCanvas(page, "desktop");
    await checkCoastlineSeam(browser);
    const failurePage = await browser.newPage({
      viewport: { width: 320, height: 400 },
      deviceScaleFactor: 1,
    });
    await failurePage.route("**/assets/models/z0.onnx*", (route) =>
      route.fulfill({ status: 503, body: "test failure" }),
    );
    await failurePage.goto(baseUrl + "?backend=wasm");
    await failurePage.waitForFunction(() =>
      document.querySelector("#status").textContent.startsWith("Model failed:"),
    );
    await assert.rejects(ready(failurePage));
    await failurePage.close();
    assert.deepEqual(errors, []);
    console.log(
      JSON.stringify({
        workers: 4,
        threadsPerWorker: 1,
        retina: [800, 640],
        previewColors: preview.colors,
        z0Runs: runs,
        z1AdditionalRuns: after - runs,
        cachedZoomOutRuns: 0,
      }),
    );
  } finally {
    await browser.close();
  }
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
