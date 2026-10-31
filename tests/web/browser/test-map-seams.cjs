const { demoRequire } = require("../support/test-support.cjs");
const assert = require("node:assert/strict");
const { chromium } = demoRequire("playwright");
const { browserOptions } = demoRequire("./browser-options.cjs");
const baseUrl = process.env.DEMO_URL || "http://127.0.0.1:4173/src/terrain_higan/demos/web/";
(async () => {
  const browser = await chromium.launch(browserOptions({ headless: true }));
  try {
    for (const deviceScaleFactor of [1, 2]) {
      const page = await browser.newPage({
        viewport: { width: 401, height: 359 },
        deviceScaleFactor,
      });
      const errors = [];
      page.on("pageerror", (e) => errors.push(e.message));
      await page.addInitScript(() => {
        // Continuous, non-flat colors expose any gaps or blending at tile boundaries.
        globalThis.terrainPixels = (x, y, width, height) => {
          const pixels = new Uint8ClampedArray(width * height * 4);
          for (let row = 0; row < height; row++)
            for (let col = 0; col < width; col++) {
              const i = (row * width + col) * 4;
              pixels[i] = 120 + 80 * Math.sin((x + col) / 13);
              pixels[i + 1] = 140 + 70 * Math.cos((y + row) / 17);
              pixels[i + 2] = 100 + 60 * Math.sin((x + col + y + row) / 23);
              pixels[i + 3] = 255;
            }
          return pixels;
        };
        globalThis.Worker = class {
          postMessage(r) {
            if (r.type !== "generate" || globalThis.holdTiles) return;
            const values = new Float32Array(r.width * r.height);
            setTimeout(
              () =>
                this.onmessage?.({
                  data: {
                    ...r,
                    type: "tile",
                    pixels: terrainPixels(r.x, r.y, r.width, r.height).buffer,
                    elevations: values.buffer,
                    model: "z" + r.zoom,
                    provider: { name: "wasm", label: "Test" },
                    elevationStats: { min: 0, max: 0, sum: 0, sumSquares: 0, count: values.length },
                    metrics: { modelRuns: 1, inferenceWorkers: 1 },
                  },
                }),
              0,
            );
          }
        };
      });
      for (const zoom of [1, 1.25, 1.75, 7.37]) {
        await page.goto(baseUrl + "?seam=" + zoom + "#seed=1&z=" + zoom + "&x=-12345&y=23456");
        await page.waitForFunction(() => document.querySelector("#status").textContent === "Ready");
        const report = await page.evaluate(() => {
          const params = new URLSearchParams(location.hash.slice(1));
          const z = Number(params.get("z")),
            model = Math.ceil(z),
            ratio = 2 ** (z - model);
          const mpp = TerrainHiGAN.metersPerPixel(model);
          const left = Number(params.get("x")) / mpp - world.width / (2 * ratio);
          const top = Number(params.get("y")) / mpp - world.height / (2 * ratio);
          const x = Math.floor(left) - 1,
            y = Math.floor(top) - 1;
          const width = Math.ceil(world.width / ratio) + 3,
            height = Math.ceil(world.height / ratio) + 3;
          const source = document.createElement("canvas");
          source.width = width;
          source.height = height;
          source
            .getContext("2d")
            .putImageData(new ImageData(terrainPixels(x, y, width, height), width, height), 0, 0);
          const reference = document.createElement("canvas");
          reference.width = world.width;
          reference.height = world.height;
          const ctx = reference.getContext("2d");
          ctx.imageSmoothingEnabled = ratio !== 1;
          ctx.drawImage(
            source,
            (x - left) * ratio,
            (y - top) * ratio,
            width * ratio,
            height * ratio,
          );
          const actual = world.getContext("2d").getImageData(0, 0, world.width, world.height).data;
          const expected = ctx.getImageData(0, 0, world.width, world.height).data;
          let maximum = 0,
            different = 0;
          // Exclude the viewport rim, where inference intentionally stops.
          for (let row = 3; row < world.height - 3; row++)
            for (let col = 3; col < world.width - 3; col++)
              for (let c = 0; c < 3; c++) {
                const i = (row * world.width + col) * 4 + c,
                  delta = Math.abs(actual[i] - expected[i]);
                maximum = Math.max(maximum, delta);
                if (delta > 1) different++;
              }
          return { maximum, different };
        });
        assert(
          report.maximum <= 1,
          `DPR ${deviceScaleFactor}, zoom ${zoom}: tiled viewport differs from continuous image: ${JSON.stringify(report)}`,
        );
      }
      await page.evaluate(() => (globalThis.holdTiles = true));
      const before = Number(await page.locator("#world").getAttribute("data-mosaic-builds"));
      const camera = await page.url();
      await page.keyboard.press("f");
      await page.mouse.move(200, 180);
      await page.mouse.down();
      await page.mouse.move(208, 180, { steps: 4 });
      await page.waitForFunction((old) => location.href !== old, camera);
      assert.equal(
        Number(await page.locator("#world").getAttribute("data-mosaic-builds")),
        before,
        "small pans reuse the native mosaic while tiles remain available",
      );
      await page.mouse.up();
      assert.deepEqual(errors, []);
      await page.close();
    }
    console.log(
      "2D tiled rendering matches a continuous image across fractional scales, subpixel pans, negative coordinates and DPR 1/2",
    );
  } finally {
    await browser.close();
  }
})().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});
