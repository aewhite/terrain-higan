const { demoRequire, rootDir } = require("../support/test-support.cjs");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { chromium } = demoRequire("playwright");
const { browserOptions } = demoRequire("./browser-options.cjs");
const baseUrl = process.env.DEMO_URL || "http://127.0.0.1:4173/src/terrain_higan/demos/web/";
const output = path.join(rootDir, "runs/globe-viewer");
fs.mkdirSync(output, { recursive: true });

async function ready(page) {
  await page.waitForFunction(
    () => /Ready|Error/.test(document.getElementById("status").textContent),
    {},
    { timeout: 360000 },
  );
  assert.equal(await page.locator("#status").innerText(), "Ready");
  assert.equal(await page.locator("#world").getAttribute("data-loading-planet"), "false");
  assert.equal(await page.locator("#world").getAttribute("data-mesh-pending"), "false");
  assert.equal(await page.locator("#render-progress-value").evaluate((p) => p.value), 1);
  assert(!(await page.locator("#render-progress").isVisible()));
  await page.waitForFunction(() => Number(document.getElementById("world").dataset.samples) > 0);
}

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
  const report = {};
  try {
    const page = await browser.newPage({ viewport: { width: 640, height: 480 } });
    await page.addInitScript(() =>
      addEventListener("DOMContentLoaded", () => {
        globalThis.statusLabels = [];
        globalThis.meshStates = [];
        const status = document.getElementById("status");
        new MutationObserver(() => statusLabels.push(status.textContent)).observe(status, {
          childList: true,
          subtree: true,
        });
        const canvas = document.getElementById("world");
        new MutationObserver(() =>
          meshStates.push({ pending: canvas.dataset.meshPending, ready: canvas.dataset.ready }),
        ).observe(canvas, {
          attributes: true,
          attributeFilter: ["data-mesh-pending", "data-ready"],
        });
      }),
    );
    const errors = [];
    page.on("pageerror", (error) => {
      errors.push(error.message);
      console.error("Globe browser error:", error.message);
    });
    const started = Date.now();
    await page.goto(`${baseUrl}globe.html?workers=2&backend=wasm#seed=4829347`);
    await ready(page);
    report.overviewMs = Date.now() - started;
    assert(
      await page.evaluate(() => meshStates.some((state) => state.pending === "true")),
      "mesh work spans animation frames",
    );
    assert(
      await page.evaluate(() =>
        meshStates.every((state) => state.ready !== "true" || state.pending === "false"),
      ),
      "Ready waits for committed meshes",
    );
    assert(await page.locator("#seed").isVisible());
    console.log("Overview ready", report.overviewMs);
    report.overview = await page.locator("#world").evaluate((canvas) => ({ ...canvas.dataset }));
    assert(Number(report.overview.visibleTiles) > 0);
    assert.equal(await page.locator("#world").getAttribute("width"), "640");
    assert(
      Number(report.overview.samplePixels) <= 1.50001,
      "model samples stay near physical pixel resolution",
    );
    assert(
      await page.evaluate(() =>
        statusLabels.every((label) => ["Loading", "Ready"].includes(label)),
      ),
    );
    assert(
      await page.locator("#status").evaluate((status) => status.scrollWidth <= status.clientWidth),
    );
    await page.screenshot({ path: path.join(output, "overview.png") });
    await page.keyboard.press("f");
    await page.mouse.move(220, 220);
    assert.notEqual(await page.locator("#position").innerText(), "--");
    assert.notEqual(await page.locator("#elevation").innerText(), "--");
    const grabbed = await page.locator("#position").innerText();
    await page.mouse.down();
    await page.mouse.move(260, 240, { steps: 8 });
    await page.mouse.up();
    const moved = await page.locator("#position").innerText();
    const coordinates = (label) => label.match(/-?[\d.]+/g).map(Number);
    assert(
      coordinates(grabbed).every(
        (value, axis) => Math.abs(value - coordinates(moved)[axis]) < 0.05,
      ),
      "dragging carries the grabbed location with the pointer",
    );
    await page.mouse.click(240, 220);
    await page.mouse.click(300, 260);
    const distance = await page.locator("#distance").innerText();
    assert.match(distance, /km/);
    await page.keyboard.press("+");
    await ready(page);
    assert.equal(await page.locator("#distance").innerText(), distance);
    await page.keyboard.press("f");
    await page.getByRole("button", { name: "Elevation", exact: true }).click();
    await page.waitForFunction(
      () =>
        document.querySelector('[data-map="elevation"]').getAttribute("aria-pressed") === "true",
    );
    await page.screenshot({ path: path.join(output, "elevation-measurement.png") });
    await page.getByRole("button", { name: "Contours", exact: true }).click();
    await page.screenshot({ path: path.join(output, "contours.png") });
    await page.keyboard.press("f");
    assert.equal(await page.locator(".shell").isVisible(), false);
    assert.equal(await page.locator(".scale").isVisible(), true);
    await page.keyboard.press("Escape");
    await page.getByRole("button", { name: "Keyboard shortcuts", exact: true }).click();
    assert.equal(await page.locator("#shortcuts-dialog").isVisible(), true);
    await page.getByRole("button", { name: "Close shortcuts", exact: true }).click();
    await page.getByRole("button", { name: "Clear measurement", exact: true }).click();
    assert.equal(await page.locator("#distance").innerText(), "--");
    await page.getByRole("button", { name: "Terrain", exact: true }).click();
    const regionalStarted = Date.now();
    await page.keyboard.press("f");
    await page.mouse.move(320, 240);
    for (let index = 0; index < 30; index++) await page.mouse.wheel(0, -700);
    await page.waitForFunction(
      () => Number(document.getElementById("world").dataset.targetModelLevel) === 20,
    );
    await ready(page);
    report.regionalMs = Date.now() - regionalStarted;
    console.log("Regional view ready", report.regionalMs);
    report.regional = await page.locator("#world").evaluate((canvas) => ({ ...canvas.dataset }));
    assert(Number(report.regional.cachedTiles) <= Number(report.regional.tileCapacity));
    assert.equal(Number(report.regional.maxModelLevel), 20);
    assert(Number(report.regional.samplePixels) <= 1.02);
    await page.mouse.move(320, 240);
    const terrainHeight = Number(
      (await page.locator("#elevation").innerText()).replace(/[^\d.-]/g, ""),
    );
    await page.waitForFunction(
      (height) =>
        Number(new URLSearchParams(location.hash.slice(1)).get("altitude")) > Math.max(0, height),
      terrainHeight,
    );
    await page.screenshot({ path: path.join(output, "regional.png") });
    await page.mouse.move(320, 240);
    await page.mouse.down();
    await page.mouse.move(330, 250, { steps: 8 });
    await page.mouse.up();
    await ready(page);
    assert.equal(
      await page.locator("#distance").innerText(),
      "--",
      "dragging must not place endpoints",
    );
    await page.keyboard.press("Home");
    await ready(page);
    await page.keyboard.press("f");
    await page.getByText("Advanced", { exact: true }).click();
    await page.locator("#exaggeration").selectOption("5");
    await page.locator("#seed").fill("19");
    await page.locator("#seed").press("Enter");
    await ready(page);
    assert.equal(await page.locator("#seed").inputValue(), "19");
    await page.waitForFunction(() => location.hash.includes("seed=19"));
    const savedUrl = page.url();
    await page.reload();
    await ready(page);
    assert.equal(await page.locator("#seed").inputValue(), "19");
    assert.equal(await page.locator("#exaggeration").inputValue(), "5");
    const beforeReload = new URLSearchParams(new URL(savedUrl).hash.slice(1));
    const afterReload = new URLSearchParams(new URL(page.url()).hash.slice(1));
    for (const name of ["lat", "lon", "altitude"])
      assert(Math.abs(Number(beforeReload.get(name)) - Number(afterReload.get(name))) < 0.11);
    assert.match(await page.locator("#flat-link").getAttribute("href"), /seed=19/);
    assert.deepEqual(errors, []);
    await page.close();

    const retina = await browser.newPage({
      viewport: { width: 320, height: 240 },
      deviceScaleFactor: 2,
    });
    await retina.goto(`${baseUrl}globe.html?workers=1#lat=89.99&lon=179.99`);
    await ready(retina);
    assert.equal(await retina.locator("#world").getAttribute("width"), "640");
    assert(Number(await retina.locator("#world").getAttribute("data-sample-pixels")) <= 1.50001);
    await retina.screenshot({ path: path.join(output, "pole-retina.png") });
    await retina.close();

    const polar = await browser.newPage({ viewport: { width: 320, height: 240 } });
    await polar.addInitScript(() => {
      globalThis.globeURLWrites = 0;
      const replaceState = history.replaceState.bind(history);
      history.replaceState = (...args) => {
        globeURLWrites++;
        return replaceState(...args);
      };
    });
    for (const latitude of [-90, -89.9999, 89.9999, 90]) {
      await polar.goto(
        `${baseUrl}globe.html?workers=1&backend=wasm#lat=${latitude}&lon=37&altitude=1000`,
      );
      await polar.waitForFunction(() => globeURLWrites > 0);
      assert(Math.abs(Number(new URL(polar.url()).hash.match(/lat=([^&]+)/)[1]) - latitude) < 1e-6);
      await polar.reload();
      await polar.waitForFunction(() => globeURLWrites > 0);
      assert(
        Math.abs(
          Number(new URLSearchParams(new URL(polar.url()).hash.slice(1)).get("lat")) - latitude,
        ) < 1e-6,
      );
    }
    await polar.close();

    const failure = await browser.newPage();
    await failure.route("**/assets/models/*.onnx*", (route) => route.abort());
    await failure.goto(`${baseUrl}globe.html?workers=1`);
    await failure.waitForFunction(
      () => document.getElementById("status").textContent === "Error",
      {},
      { timeout: 60000 },
    );
    assert.equal(await failure.locator("#world").getAttribute("data-visible-tiles"), "0");
    await failure.close();
    report.browser = await browser.version();
    report.rendering = process.env.HEADED === "1" ? "desktop GPU" : "SwiftShader software WebGL2";
    fs.writeFileSync(
      path.join(output, "browser-checks.json"),
      JSON.stringify(report, null, 2) + "\n",
    );
    console.log("Globe browser checks passed", report);
  } finally {
    await browser.close();
  }
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
