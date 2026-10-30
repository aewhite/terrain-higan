const { demoRequire, rootDir } = require("../support/test-support.cjs");
const assert = require("node:assert/strict");
const path = require("node:path");
const fs = require("node:fs");
const { chromium } = demoRequire("playwright");
const { browserOptions } = demoRequire("./browser-options.cjs");
const baseUrl = process.env.DEMO_URL || "http://127.0.0.1:4173/src/terrain_higan/demos/web/";
(async () => {
  const browser = await chromium.launch(
    browserOptions({
      headless: true,
      args: ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
    }),
  );
  try {
    for (const mobile of [false, true]) {
      const page = await browser.newPage({
        viewport: { width: mobile ? 375 : 800, height: mobile ? 812 : 600 },
        hasTouch: mobile,
      });
      const errors = [];
      page.on("pageerror", (e) => errors.push(e.message));
      // Keep all inference pending, so a planet must be visible without any model results.
      await page.addInitScript(() => {
        globalThis.Worker = class {
          postMessage() {}
          terminate() {}
        };
      });
      await page.goto(baseUrl + "globe.html");
      await page.waitForFunction(
        () => world.dataset.loadingPlanet === "true" && world.dataset.walkAltitude,
      );
      assert.equal(
        await page.locator("#status").innerText(),
        "Loading",
        JSON.stringify({ errors, title: await page.locator("#status").getAttribute("title") }),
      );
      assert.equal(await page.locator("#render-progress-value").evaluate((p) => p.value), 0);
      assert(await page.locator("#toggle-fullscreen").isVisible());
      assert.equal(await page.locator("#terrain-controls").isVisible(), !mobile);
      assert.equal(await page.locator("#toggle-controls").isVisible(), mobile);
      if (mobile) {
        await page.locator("#toggle-controls").click();
        assert(await page.locator("#terrain-controls").isVisible());
        assert.equal(await page.locator("#toggle-controls").getAttribute("aria-expanded"), "true");
        assert(!(await page.locator("#toggle-shortcuts").isVisible()));
        assert(!(await page.locator("#experimental").isVisible()));
        await page.locator("#toggle-controls").click();
        const altitude = () =>
          page.locator("#world").getAttribute("data-walk-altitude").then(Number);
        const start = await altitude();
        const session = await page.context().newCDPSession(page);
        const touch = async (type, points) => {
          await session.send("Input.dispatchTouchEvent", {
            type,
            touchPoints: points.map(([id, x, y]) => ({ id, x, y })),
          });
          await page.evaluate(() => new Promise(requestAnimationFrame));
        };
        await touch("touchStart", [
          [1, 140, 400],
          [2, 230, 400],
        ]);
        await touch("touchMove", [
          [1, 95, 400],
          [2, 275, 400],
        ]);
        assert((await altitude()) < start * 0.8, "spreading fingers zooms toward planet");
        await touch("touchMove", [
          [1, 140, 400],
          [2, 230, 400],
        ]);
        assert((await altitude()) > start * 0.8, "closing fingers zooms out");
        await touch("touchEnd", [[2, 230, 400]]);
        await page.waitForTimeout(250); // Camera URLs are debounced by 200 ms.
        const position = new URLSearchParams(new URL(page.url()).hash.slice(1)).get("lat");
        await touch("touchMove", [[1, 165, 425]]);
        await page.waitForFunction(
          (latitude) => new URLSearchParams(location.hash.slice(1)).get("lat") !== latitude,
          position,
        );
        await touch("touchEnd", []);
        assert.equal(
          await page.locator("#distance").innerText(),
          "--",
          "pinch does not place measurement",
        );
        await touch("touchStart", [
          [1, 140, 400],
          [2, 230, 400],
        ]);
        await touch("touchCancel", []);
        assert(!(await page.locator("#world").evaluate((c) => c.classList.contains("dragging"))));
        await page.setViewportSize({ width: 812, height: 375 });
        assert(!(await page.locator("#terrain-controls").isVisible()));
        await page.locator("#toggle-controls").click();
        assert(await page.locator("#terrain-controls").isVisible());
        assert(!(await page.locator("#experimental").isVisible()));
        await page.locator("#toggle-controls").click();
      }
      const output = path.join(rootDir, "runs/globe-viewer");
      fs.mkdirSync(output, { recursive: true });
      await page.screenshot({
        path: path.join(output, mobile ? "loading-mobile.png" : "loading-desktop.png"),
      });
      assert.deepEqual(errors, []);
      await page.close();
    }
    const phone = await browser.newPage({
      viewport: { width: 375, height: 812 },
      deviceScaleFactor: 2,
      hasTouch: true,
      userAgent: "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) Mobile Safari",
    });
    const errors = [];
    phone.on("pageerror", (e) => errors.push(e.message));
    const cpuRuntime = [];
    phone.on("request", (r) => {
      if (r.url().includes("ort.wasm.min.js")) cpuRuntime.push(r.url());
    });
    await phone.goto(baseUrl + "globe.html?workers=1&backend=wasm#seed=4829347");
    await phone.waitForFunction(
      () => /Ready|Error/.test(document.getElementById("status").textContent),
      {},
      { timeout: 180000 },
    );
    assert.equal(await phone.locator("#status").innerText(), "Ready");
    assert(cpuRuntime.length > 0, "mobile globe uses compact runtime and small inference patches");
    assert(!(await phone.locator("#render-progress").isVisible()));
    assert.deepEqual(errors, []);
    await phone.close();
    console.log(
      "Globe pinch, continuation, cancellation, mobile controls, landscape layout and pending planet passed",
    );
  } finally {
    await browser.close();
  }
})().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});
