const { demoRequire, rootDir } = require("../support/test-support.cjs");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { chromium } = demoRequire("playwright");
const { browserOptions } = demoRequire("./browser-options.cjs");
const baseUrl = process.env.DEMO_URL || "http://127.0.0.1:4173/src/terrain_higan/demos/web/";

(async () => {
  const browser = await chromium.launch(
    browserOptions({ headless: false, args: ["--headless=new"] }),
  );
  const results = [];
  try {
    for (const view of ["", "globe.html"])
      for (const mode of ["auto", "missing", "rejected"]) {
        const page = await browser.newPage({
          viewport: { width: 320, height: 240 },
          hasTouch: true,
          userAgent: "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) Mobile Safari",
        });
        const errors = [];
        const runtimes = [];
        page.on("pageerror", (error) => errors.push(error.message));
        page.on("request", (request) => {
          if (/ort\.(webgpu|wasm)\.min\.js/.test(request.url())) runtimes.push(request.url());
        });
        await page.addInitScript(() => {
          globalThis.gpuSubmissions = 0;
          globalThis.workerCount = 0;
          const NativeWorker = Worker;
          globalThis.Worker = class extends NativeWorker {
            constructor(...args) {
              super(...args);
              workerCount++;
              globalThis.terrainWorker = this;
              this.addEventListener("message", ({ data }) => {
                if (data.type === "gpu-proof") gpuSubmissions++;
              });
            }
          };
        });
        await page.route(/\/(?:model-worker|globe-worker)\.js\?/, async (route) => {
          const response = await route.fetch();
          const source = await response.text();
          const override =
            mode === "missing"
              ? 'Object.defineProperty(navigator, "gpu", {value: undefined});'
              : mode === "rejected"
                ? 'Object.defineProperty(navigator, "gpu", {value: {requestAdapter: async () => null}});'
                : "";
          const guard = `${override}
          globalThis.Worker = class { constructor() { throw new Error("Nested inference workers are forbidden"); } };
          if (globalThis.GPUQueue) {
            const submit = GPUQueue.prototype.submit;
            let reported = false;
            GPUQueue.prototype.submit = function(...args) {
              const result = submit.apply(this, args);
              if (!reported) { reported = true; postMessage({type: "gpu-proof"}); }
              return result;
            };
          }
        `;
          await route.fulfill({ response, body: guard + source });
        });
        const started = Date.now();
        await page.goto(baseUrl + view + "#seed=4829347&z=0");
        await page.waitForFunction(
          () => /Ready|Error|Model failed/.test(document.getElementById("status").textContent),
          {},
          { timeout: 180000 },
        );
        assert.equal(
          await page.locator("#status").innerText(),
          "Ready",
          await page.locator("#status").getAttribute("title"),
        );
        const result = await page.evaluate(() => ({
          backend: document.getElementById("backend").textContent,
          detail: document.getElementById("backend").title,
          gpuSubmissions,
          workerCount,
          parallel: Number(document.getElementById("world").dataset.maxParallelRuns),
        }));
        assert.equal(result.workerCount, 1);
        assert.equal(result.parallel, 1);
        if (mode === "auto" && result.backend === "WebGPU")
          assert(result.gpuSubmissions > 0, "actual GPU work was submitted");
        else {
          assert.equal(result.backend, "CPU / WASM");
          assert.equal(result.gpuSubmissions, 0);
          assert(result.detail.length > 0, "CPU fallback explains its reason");
        }
        if (mode === "auto" && process.env.WEBGPU_REQUIRED === "1")
          assert.equal(result.backend, "WebGPU", result.detail);
        if (!view && mode === "auto" && result.backend === "WebGPU") {
          result.deep = await page.evaluate(async () => {
            let id = 10000;
            const generate = (worker, x, y, width, height) =>
              new Promise((resolve, reject) => {
                const requestId = ++id;
                const receive = ({ data }) => {
                  if (data.id !== requestId || !["tile", "error"].includes(data.type)) return;
                  worker.removeEventListener("message", receive);
                  data.type === "error" ? reject(new Error(data.message)) : resolve(data);
                };
                worker.addEventListener("message", receive);
                worker.postMessage({
                  type: "generate",
                  id: requestId,
                  key: String(requestId),
                  seed: "4829347",
                  zoom: 20,
                  x,
                  y,
                  width,
                  height,
                  map: "elevation",
                });
              });
            const reference = await generate(terrainWorker, 0, 0, 64, 64);
            terrainWorker.terminate();
            // A fresh runtime must agree with the same GPU-backed canonical
            // block after the first runtime has warmed all predecessor models.
            const fresh = new Worker(`./model-worker.js?mobile=1&bundle=${TerrainHiGAN.bundle}`);
            try {
              const crop = await generate(fresh, 8, 8, 8, 8);
              const full = new Float32Array(reference.elevations),
                values = new Float32Array(crop.elevations);
              if (reference.provider.name !== "webgpu" || crop.provider.name !== "webgpu")
                throw new Error("Deep model did not execute on WebGPU");
              for (let row = 0; row < 8; row++)
                for (let column = 0; column < 8; column++)
                  if (
                    !Number.isFinite(values[row * 8 + column]) ||
                    values[row * 8 + column] !== full[(row + 8) * 64 + column + 8]
                  )
                    throw new Error("GPU cache history or request shape changed canonical terrain");
              return {
                model: crop.model,
                provider: crop.provider.name,
                matchingPixels: values.length,
                maxParallelRuns: crop.metrics.maxParallelRuns,
              };
            } finally {
              fresh.terminate();
            }
          });
        }
        assert(
          runtimes.some((url) =>
            url.includes(mode === "missing" ? "ort.wasm.min.js" : "ort.webgpu.min.js"),
          ) ||
            (mode === "auto" &&
              result.backend === "CPU / WASM" &&
              runtimes.some((url) => url.includes("ort.wasm.min.js"))),
        );
        assert.deepEqual(errors, []);
        results.push({
          view: view || "map",
          mode,
          ...result,
          checkMs: Date.now() - started,
          browser: browser.version(),
        });
        console.log("WebGPU runtime", results.at(-1));
        await page.close();
      }
    const output = path.join(rootDir, "runs/webgpu");
    fs.mkdirSync(output, { recursive: true });
    fs.writeFileSync(
      path.join(output, `browser-checks-${browser.version()}.json`),
      JSON.stringify(results, null, 2),
    );
  } finally {
    await browser.close();
  }
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
