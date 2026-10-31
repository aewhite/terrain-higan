const { demoRequire } = require("../support/test-support.cjs");
const assert = require("node:assert/strict");
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
    for (const [view, mobile] of [
      ["", false],
      ["globe.html", false],
      ["globe.html", true],
    ]) {
      const page = await browser.newPage(
        mobile
          ? { viewport: { width: 375, height: 812 }, hasTouch: true, deviceScaleFactor: 2 }
          : { viewport: { width: 400, height: 320 } },
      );
      const errors = [];
      page.on("pageerror", (e) => errors.push(e.message));
      if (view) {
        await page.addInitScript(() => {
          const NativeWorker = Worker;
          let released = false;
          globalThis.pendingRefinements = [];
          globalThis.releaseRefinements = () => {
            released = true;
            for (const resume of pendingRefinements.splice(0)) resume();
          };
          globalThis.Worker = class extends NativeWorker {
            postMessage(message, ...transfer) {
              if (message.type === "tile" && message.depth > 0 && !released)
                pendingRefinements.push(() => super.postMessage(message, ...transfer));
              else super.postMessage(message, ...transfer);
            }
          };
        });
      }
      let releaseModel, releaseRuntime;
      const modelGate = new Promise((r) => (releaseModel = r)),
        runtimeGate = new Promise((r) => (releaseRuntime = r));
      await page.route("**/assets/models/z0.onnx*", async (route) => {
        await modelGate;
        await route.continue();
      });
      await page.route("**/assets/runtime/*.wasm", async (route) => {
        await runtimeGate;
        await route.continue();
      });
      await page.goto(baseUrl + view + "?workers=1&backend=wasm#seed=4829347&z=0");
      await page.waitForFunction(
        () => document.querySelector("#render-progress-label").textContent.includes("Download z0"),
        {},
        { timeout: 60000 },
      );
      assert(await page.locator("#render-progress").isVisible());
      assert.equal(
        await page
          .locator("#render-progress-label")
          .evaluate((e) => getComputedStyle(e).whiteSpace),
        "nowrap",
      );
      assert.equal(
        await page.locator("#render-progress-value").getAttribute("value"),
        null,
        "downloads use indeterminate progress",
      );
      releaseModel();
      await page.waitForFunction(
        () => document.querySelector("#render-progress-label").textContent.includes("Load z0"),
        {},
        { timeout: 60000 },
      );
      assert.equal(await page.locator("#render-progress-value").getAttribute("value"), null);
      releaseRuntime();
      if (view) {
        await page.waitForFunction(
          () => document.getElementById("world").dataset.previewReady === "true",
          {},
          { timeout: 120000 },
        );
        assert.equal(await page.locator("#world").getAttribute("data-loading-planet"), "false");
        assert.equal(await page.locator("#world").getAttribute("data-ready"), "false");
        assert.equal(await page.locator("#world").getAttribute("data-render-stage"), "refining");
        assert.equal(await page.locator("#world").getAttribute("data-max-model-level"), "2");
        assert(await page.locator("#render-progress").isVisible());
        assert.match(await page.locator("#render-progress-label").innerText(), /Refining/);
        assert.equal(await page.locator("#status").innerText(), "Loading");
        assert.equal(await page.evaluate(() => pendingRefinements.length), 1);
        // A held refinement cannot erase the preview or prematurely report Ready.
        await page.evaluate(
          () =>
            new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))),
        );
        assert.equal(await page.locator("#world").getAttribute("data-loading-planet"), "false");
        await page.evaluate(() => releaseRefinements());
      }
      await page.waitForFunction(
        () => document.querySelector("#status").textContent === "Ready",
        {},
        { timeout: 180000 },
      );
      assert(
        !(await page.locator("#render-progress").isVisible()),
        "finished render removes progress",
      );
      const widths = await page.evaluate(() => ({
        bar: document.querySelector(".scale i").getBoundingClientRect().width,
        panel: document.querySelector(".scale").getBoundingClientRect().width,
        metric: TerrainHiGAN.scalePixels,
      }));
      assert.equal(widths.bar, widths.panel - 20, "ruler fills panel content width");
      assert.equal(widths.metric, widths.bar, "distance label uses actual ruler width");
      await page.evaluate(() => {
        TerrainHiGAN.setRenderProgress(0, 5);
        TerrainHiGAN.setRenderPhase("Generating terrain");
      });
      assert(await page.locator("#render-progress").isVisible(), "new work restores progress");
      assert.equal(await page.locator("#render-progress-label").innerText(), "Generating terrain");
      await page.evaluate(() => TerrainHiGAN.setRenderProgress(5, 5));
      assert(!(await page.locator("#render-progress").isVisible()));
      assert.deepEqual(errors, []);
      await page.close();
    }
    for (const view of ["", "globe.html"]) {
      const page = await browser.newPage({ viewport: { width: 375, height: 812 }, hasTouch: true });
      const message = "WASM initialization failed: allocation error (test failure)";
      await page.addInitScript((message) => {
        window.Worker = class {
          postMessage(request) {
            if (!["tile", "generate"].includes(request.type)) return;
            setTimeout(
              () =>
                this.onmessage?.({
                  data: { type: "error", id: request.id, key: request.key, message },
                }),
              0,
            );
          }
          terminate() {}
        };
      }, message);
      await page.goto(baseUrl + view + "#seed=4829347&z=0");
      await page.waitForFunction(
        () => document.getElementById("render-progress").dataset.state === "failed",
      );
      assert(!(await page.locator("#terrain-controls").isVisible()));
      assert.equal(await page.locator("#render-progress-label").innerText(), "Out of memory");
      assert.equal(await page.locator("#render-progress").getAttribute("title"), message);
      assert.equal(await page.locator("#reload-viewer").innerText(), "Retry");
      assert(await page.locator("#reload-viewer").isVisible());
      let navigations = 0;
      page.on("request", (request) => {
        if (request.isNavigationRequest() && request.frame() === page.mainFrame()) navigations++;
      });
      await Promise.all([page.waitForEvent("load"), page.locator("#reload-viewer").click()]);
      await page.waitForFunction(
        () => document.getElementById("render-progress").dataset.state === "failed",
      );
      assert.equal(navigations, 1, "retry restarts the viewer only on user request");
      assert.equal(new URLSearchParams(new URL(page.url()).hash.slice(1)).get("seed"), "4829347");
      await page.close();
    }
    console.log(
      "Render feedback: download/runtime phases, coarse globe preview before fine tiles on desktop/mobile, completed/new work visibility, full-width scale, mobile errors and explicit retry",
    );
  } finally {
    await browser.close();
  }
})().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});
