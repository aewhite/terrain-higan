const { demoRequire, rootDir } = require("../support/test-support.cjs");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { chromium } = demoRequire("playwright");
const { browserOptions } = demoRequire("./browser-options.cjs");
const baseUrl = process.env.DEMO_URL || "http://127.0.0.1:4173/src/terrain_higan/demos/web/";
const screenshots = path.join(rootDir, "runs/demo-measurement");

(async () => {
  fs.mkdirSync(screenshots, { recursive: true });
  const browser = await chromium.launch(
    browserOptions({
      headless: true,
      args: ["--headless=new"],
    }),
  );
  try {
    for (const deviceScaleFactor of [1, 2]) {
      const page = await browser.newPage({
        viewport: { width: 400, height: 320 },
        deviceScaleFactor,
        locale: "en-US",
      });
      const errors = [];
      page.on("pageerror", (error) => errors.push(error.message));
      await page.goto(baseUrl + "?backend=wasm#seed=4829347&z=0");
      await page.waitForFunction(
        () => Number(document.querySelector("#world").dataset.modelRuns) > 0,
        {},
        { timeout: 120000 },
      );
      await page.setViewportSize({ width: 900, height: 650 });
      await page.waitForFunction(
        () => document.querySelector("#world").height === 650 * devicePixelRatio,
      );
      assert.equal(await page.locator("[data-tool]").count(), 0);
      assert.equal(await page.locator("#zoom-value").count(), 0);
      await page.mouse.click(100, 400);
      await page.mouse.move(160, 480);
      const expected = `${(52428.8 * deviceScaleFactor).toLocaleString("en-US")} km`;
      assert.equal(await page.locator("#distance").innerText(), expected);
      await page.mouse.click(160, 480);
      await page.mouse.move(220, 520);
      assert.equal(await page.locator("#distance").innerText(), expected);
      const pixels = await page.locator("#world").evaluate((canvas) => {
        const context = canvas.getContext("2d");
        const colors = new Set();
        const data = context.getImageData(0, 0, canvas.width, canvas.height).data;
        for (let i = 0; i < data.length; i += 4)
          colors.add(data[i] * 65536 + data[i + 1] * 256 + data[i + 2]);
        return {
          colors: colors.size,
          endpoint: Array.from(
            context.getImageData(100 * devicePixelRatio, 400 * devicePixelRatio, 1, 1).data,
          ),
        };
      });
      assert(pixels.colors > 20, "terrain and overlay must render");
      assert.deepEqual(pixels.endpoint, [255, 224, 122, 255]);
      await page.screenshot({
        path: path.join(screenshots, `desktop-dpr${deviceScaleFactor}.png`),
      });
      await page.setViewportSize({ width: 375, height: 812 });
      await page.waitForFunction(
        () => document.querySelector("#world").height === 812 * devicePixelRatio,
      );
      await page.locator("#clear-measurement").click();
      await page.mouse.click(20, 620);
      await page.mouse.click(80, 700);
      assert.equal(await page.locator("#distance").innerText(), expected);
      await page.screenshot({ path: path.join(screenshots, `mobile-dpr${deviceScaleFactor}.png`) });
      await page.setViewportSize({ width: 900, height: 650 });
      await page.waitForFunction(
        () => document.querySelector("#world").height === 650 * devicePixelRatio,
      );

      assert.equal(await page.locator("#distance").innerText(), expected);
      await page.locator("#clear-measurement").click();
      await page.mouse.click(100, 400);
      await page.mouse.click(160, 480);

      const beforePan = new URLSearchParams(new URL(page.url()).hash.slice(1)).get("x");
      await page.mouse.move(250, 450);
      await page.mouse.down();
      await page.mouse.move(280, 450, { steps: 3 });
      await page.mouse.up();
      await page.waitForFunction(
        (before) => new URLSearchParams(location.hash.slice(1)).get("x") !== before,
        beforePan,
      );
      assert.equal(await page.locator("#distance").innerText(), expected);
      assert.deepEqual(
        await page
          .locator("#world")
          .evaluate((canvas) =>
            Array.from(
              canvas
                .getContext("2d")
                .getImageData(130 * devicePixelRatio, 400 * devicePixelRatio, 1, 1).data,
            ),
          ),
        [255, 224, 122, 255],
        "panning must move the anchored endpoint with terrain",
      );
      await page.keyboard.press("+");
      assert.equal(await page.locator("#distance").innerText(), expected);
      await page.locator('[data-map="contour"]').click();
      assert.equal(await page.locator("#distance").innerText(), expected);
      await page.locator("#clear-measurement").click();
      assert.equal(await page.locator("#distance").innerText(), "--");
      assert(await page.locator("#clear-measurement").isDisabled());
      await page.mouse.click(100, 400);
      await page.mouse.click(200, 400);
      await page.mouse.click(300, 400);
      assert.equal(await page.locator("#distance").innerText(), "0 m");
      await page.locator(".advanced summary").click();
      await page.locator("#randomize").click();
      assert.equal(await page.locator("#distance").innerText(), "--");
      assert.deepEqual(errors, []);
      await page.close();

      // Measurement uses geometry only; deep zoom checks need no model inference.
      const deep = await browser.newPage({
        viewport: { width: 375, height: 812 },
        deviceScaleFactor,
        hasTouch: true,
        locale: "en-US",
      });
      deep.on("pageerror", (error) => errors.push(error.message));
      await deep.route("**/model-worker.js?*", (route) =>
        route.fulfill({ contentType: "text/javascript", body: "onmessage = () => {};" }),
      );
      await deep.goto(baseUrl + "#z=20&x=-1234&y=5678");
      await deep.waitForFunction(() => document.querySelector("#world").width > 300);
      await deep.touchscreen.tap(20, 620);
      await deep.touchscreen.tap(80, 700);
      assert.equal(await deep.locator("#distance").innerText(), `${50 * deviceScaleFactor} m`);
      await deep.keyboard.press("+");
      await deep.waitForFunction(
        () => Number(document.getElementById("world").dataset.viewZoom) === 20,
      );
      assert.equal(await deep.locator("#distance").innerText(), `${50 * deviceScaleFactor} m`);
      await deep.keyboard.press("-");
      assert.equal(await deep.locator("#distance").innerText(), `${50 * deviceScaleFactor} m`);
      await deep.keyboard.press("Escape");
      assert.equal(await deep.locator("#distance").innerText(), "--");
      await deep.touchscreen.tap(20, 620);
      await deep.mouse.move(30, 620);
      await deep.mouse.down();
      await deep.mouse.move(40, 620);
      await deep.dispatchEvent("#world", "pointercancel", { pointerId: 1 });
      assert(
        !(await deep.locator("#world").evaluate((canvas) => canvas.classList.contains("dragging"))),
      );
      await deep.mouse.up();
      await deep.locator("#toggle-controls").click();
      await deep.locator("#clear-measurement").click();
      await deep.locator("#toggle-controls").click();
      await deep.keyboard.press("+");
      await deep.waitForFunction(
        () => Number(document.getElementById("world").dataset.viewZoom) === 20,
      );
      await deep.touchscreen.tap(20, 620);
      await deep.touchscreen.tap(21, 620);
      assert.equal(await deep.locator("#distance").innerText(), `${0.5 * deviceScaleFactor} m`);
      assert.deepEqual(errors, []);
      await deep.close();
    }
    console.log(
      "Measurement checks passed: scaled distances, Retina, touch, pan, zoom, clear, seeds",
    );
  } finally {
    await browser.close();
  }
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
