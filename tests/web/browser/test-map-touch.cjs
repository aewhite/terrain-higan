const { demoRequire, rootDir } = require("../support/test-support.cjs");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { chromium } = demoRequire("playwright");
const { browserOptions } = demoRequire("./browser-options.cjs");
const baseUrl = process.env.DEMO_URL || "http://127.0.0.1:4173/src/terrain_higan/demos/web/";
(async () => {
  const browser = await chromium.launch(browserOptions({ headless: true }));
  try {
    for (const deviceScaleFactor of [1, 2]) {
      const page = await browser.newPage({
        viewport: { width: 375, height: 812 },
        deviceScaleFactor,
        hasTouch: true,
      });
      const errors = [];
      page.on("pageerror", (e) => errors.push(e.message));
      // Input tests need the real DOM and camera, without running terrain inference.
      await page.addInitScript(() => {
        window.tileRequests = [];
        window.Worker = class {
          postMessage(message) {
            tileRequests.push(message);
          }
        };
      });
      await page.goto(baseUrl + "#seed=4829347");
      await page.waitForFunction(() => new URLSearchParams(location.hash.slice(1)).has("z"));
      const camera = () =>
        page.evaluate(() => {
          const p = new URLSearchParams(location.hash.slice(1));
          return { z: Number(p.get("z")), x: Number(p.get("x")), y: Number(p.get("y")) };
        });
      const initial = await camera();
      const area = 375 * 812 * deviceScaleFactor ** 2 * (524288 / 2 ** initial.z) ** 2;
      const ratio = area / (4 * Math.PI * 6371000 ** 2);
      assert(ratio >= 0.5 && ratio <= 2, "initial viewport approximates Earth surface area");
      assert(!(await page.locator("#terrain-controls").isVisible()));
      await page.locator("#toggle-controls").click();
      assert(await page.locator("#terrain-controls").isVisible());
      assert(!(await page.locator("#toggle-shortcuts").isVisible()));
      assert(!(await page.locator("#experimental").isVisible()));
      await page.locator("#toggle-controls").click();
      await page.keyboard.press("f");
      const session = await page.context().newCDPSession(page);
      const touch = async (type, points) => {
        await session.send("Input.dispatchTouchEvent", {
          type,
          touchPoints: points.map(([id, x, y]) => ({ id, x, y })),
        });
        await page.evaluate(() => new Promise(requestAnimationFrame));
      };
      await touch("touchStart", [
        [1, 80, 450],
        [2, 180, 450],
      ]);
      await touch("touchMove", [
        [1, 55, 450],
        [2, 205, 450],
      ]);
      const fractional = await camera();
      assert(
        Math.abs(fractional.z - initial.z - Math.log2(1.5)) < 1e-9,
        "pinch follows fractional zoom continuously",
      );
      assert(
        await page.evaluate(() =>
          tileRequests.filter((r) => r.type === "generate").every((r) => Number.isInteger(r.zoom)),
        ),
        "model inference always uses integer levels",
      );
      await touch("touchMove", [
        [1, 30, 450],
        [2, 230, 450],
      ]);
      const zoomed = await camera();
      assert.equal(zoomed.z, initial.z + 1);
      const before = (524288 / 2 ** initial.z) * deviceScaleFactor;
      const after = before / 2;
      assert(Math.abs(zoomed.x - (130 - 375 / 2) * (before - after)) <= 1);
      assert(Math.abs(zoomed.y - (450 - 812 / 2) * (before - after)) <= 1);
      await touch("touchMove", [
        [1, 80, 450],
        [2, 180, 450],
      ]);
      assert.deepEqual(await camera(), initial);
      await touch("touchEnd", [[2, 180, 450]]);
      await touch("touchMove", [[1, 100, 470]]);
      assert((await camera()).x < initial.x, "remaining finger pans");
      await touch("touchEnd", []);
      assert.equal(
        await page.locator("#distance").innerText(),
        "--",
        "pinch cannot place measurement",
      );
      await touch("touchStart", [
        [1, 80, 450],
        [2, 180, 450],
      ]);
      await touch("touchCancel", []);
      assert.equal(
        await page.locator("#world").evaluate((c) => c.classList.contains("dragging")),
        false,
      );
      await page.reload();
      await page.waitForFunction(() => world.width > 0 && world.style.width);
      assert.equal((await camera()).z, initial.z, "URL restores zoom");
      await page.keyboard.press("Home");
      assert.deepEqual(await camera(), initial, "Home restores Earth-scale view at the origin");
      await page.goto(baseUrl + "#seed=4829347&z=0");
      await page.reload();
      await page.waitForFunction(() => world.style.width);
      assert.equal((await camera()).z, 0, "explicit z0 URL overrides initial scale");
      await page.dispatchEvent("#world", "wheel", { deltaY: -10, clientX: 20, clientY: 620 });
      await page.waitForFunction(
        () => Number(new URLSearchParams(location.hash.slice(1)).get("z")) === 0.02,
      );
      assert(
        Math.abs((await camera()).z - 0.02) < 1e-9,
        "small wheel motion produces a small zoom change",
      );
      await page.reload();
      await page.waitForFunction(() => world.style.width);
      assert(Math.abs((await camera()).z - 0.02) < 1e-9, "URL restores fractional zoom");
      await page.evaluate(() => TerrainHiGAN.setRenderProgress(1, 4));
      assert.equal(await page.locator("#render-progress-value").evaluate((p) => p.value), 0.25);
      assert.match(await page.locator("#render-progress-label").innerText(), /25%/);
      const progressBounds = await page.locator("#render-progress").boundingBox();
      const scaleBounds = await page.locator(".scale").boundingBox();
      assert(
        progressBounds.y + progressBounds.height <= scaleBounds.y,
        "progress sits above the distance scale",
      );
      await page.locator("#toggle-fullscreen").click();
      await page.waitForFunction(() => document.fullscreenElement);
      assert.equal(await page.locator("#toggle-fullscreen").getAttribute("aria-pressed"), "true");
      await page.locator("#toggle-fullscreen").click();
      await page.waitForFunction(() => !document.fullscreenElement);
      await page.evaluate(() => {
        document.documentElement.requestFullscreen = () => Promise.reject(new Error("Unsupported"));
      });
      await page.locator("#toggle-fullscreen").click();
      assert(!(await page.locator("#toggle-fullscreen").isVisible()));
      assert.deepEqual(errors, []);
      await page.close();
    }
    const mobile = await browser.newPage({
      viewport: { width: 375, height: 812 },
      deviceScaleFactor: 2,
      hasTouch: true,
      userAgent:
        "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 Mobile/15E148 Safari/604.1",
    });
    const runtimes = [];
    await mobile.addInitScript(() => {
      const Worker = window.Worker;
      window.Worker = class extends Worker {
        postMessage(message, transfer) {
          if (message.type === "generate")
            document.querySelector("#world").dataset.requestTileSize = message.width;
          return super.postMessage(message, transfer);
        }
      };
    });
    mobile.on("request", (r) => {
      if (r.url().includes("ort.wasm.min.js")) runtimes.push(r.url());
    });
    await mobile.goto(baseUrl + "?backend=wasm#seed=4829347");
    await mobile.waitForFunction(
      () => /Ready|Model failed/.test(document.querySelector("#status").textContent),
      {},
      { timeout: 180000 },
    );
    assert.equal(await mobile.locator("#status").innerText(), "Ready");
    assert.equal(await mobile.locator("#world").getAttribute("data-inference-workers"), "1");
    assert.equal(await mobile.locator("#world").getAttribute("data-max-parallel-runs"), "1");
    assert.equal(await mobile.locator("#render-progress-value").evaluate((p) => p.value), 1);
    assert(
      !(await mobile.locator("#render-progress").isVisible()),
      "completed render hides progress",
    );
    assert.equal(await mobile.locator("#world").getAttribute("data-request-tile-size"), "128");
    assert(runtimes.length > 0, "mobile loads the smaller CPU runtime");
    const output = path.join(rootDir, "runs/demo-map-touch");
    fs.mkdirSync(output, { recursive: true });
    await mobile.screenshot({ path: path.join(output, "mobile-progress.png") });
    await mobile.close();
    console.log(
      "Earth-scale view, anchored pinch, pan continuation, cancellation, URL restoration, and real mobile single-worker inference passed",
    );
  } finally {
    await browser.close();
  }
})().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});
