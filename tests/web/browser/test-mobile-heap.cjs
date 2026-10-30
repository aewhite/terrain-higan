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
    for (const scenario of [
      { bounded: false, view: "" },
      { bounded: true, view: "" },
      { bounded: true, view: "globe.html" },
      { bounded: true, view: "globe.html", hash: "seed=4829347&altitude=1000&lat=20&lon=-25" },
    ]) {
      const page = await browser.newPage({
        viewport: { width: 320, height: 400 },
        hasTouch: true,
        userAgent: "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) Mobile Safari",
      });
      await page.addInitScript(() => {
        window.heapReservations = [];
        const NativeWorker = Worker;
        window.Worker = class extends NativeWorker {
          constructor(...args) {
            super(...args);
            this.addEventListener("message", ({ data }) => {
              if (data.type === "phase" && data.phase === "heap-test")
                heapReservations.push(data.maximum);
            });
          }
        };
      });
      await page.route("**/inference-worker.js?*", async (route) => {
        const response = await route.fetch();
        const source = await response.text();
        const guard = `
const TestNativeMemory = WebAssembly.Memory;
WebAssembly.Memory = class extends TestNativeMemory {
 constructor(options) {
  if (options.shared && options.maximum > 4096) throw new RangeError('Out of memory: test rejects shared memory reservations above 256 MiB');
  super(options);
  if (options.shared) postMessage({type:'phase',phase:'heap-test',message:'Runtime',maximum:options.maximum});
 }
};\n`;
        await route.fulfill({ response, body: guard + source });
      });
      if (!scenario.bounded)
        await page.route("**/ort-wasm-mobile.mjs*", async (route) => {
          const response = await route.fetch();
          const body = (await response.text()).replace(
            "new WebAssembly.Memory({initial:256,maximum:4096,shared:!0})",
            "new WebAssembly.Memory({initial:256,maximum:65536,shared:!0})",
          );
          await route.fulfill({ response, body });
        });
      await page.goto(
        baseUrl +
          scenario.view +
          "?workers=1&backend=wasm#" +
          (scenario.hash || "seed=4829347&z=0"),
      );
      await page.waitForFunction(
        () => /Ready|Error|Model failed/.test(document.querySelector("#status").textContent),
        {},
        { timeout: 180000 },
      );
      if (scenario.bounded) {
        assert.equal(await page.locator("#status").innerText(), "Ready");
        const reservations = await page.evaluate(() => heapReservations);
        assert(reservations.length > 0);
        if (scenario.hash)
          assert(
            Number(await page.locator("#world").getAttribute("data-max-model-level")) > 10,
            "near-ground test exercises the deeper model chain",
          );
        assert(reservations.every((pages) => pages <= 4096));
      } else {
        assert.match(await page.locator("#status").innerText(), /failed|Error/);
        assert.equal(
          await page.locator("#render-progress-label").innerText(),
          "Out of memory",
          "old reservation reproduces the startup failure under this memory policy",
        );
      }
      await page.close();
    }
    console.log(
      "Mobile heap regression: old 4 GiB reservation fails; 256 MiB loader completes real 2D, orbital and near-ground globe inference under the enforced reservation limit",
    );
  } finally {
    await browser.close();
  }
})().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});
