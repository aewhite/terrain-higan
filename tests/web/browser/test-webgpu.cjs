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
          viewport:
            !view && mode === "auto" ? { width: 375, height: 812 } : { width: 320, height: 240 },
          deviceScaleFactor: !view && mode === "auto" ? 2 : 1,
          hasTouch: true,
          userAgent: "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) Mobile Safari",
        });
        const errors = [];
        const runtimes = [];
        const modelRequests = [];
        page.on("pageerror", (error) => errors.push(error.message));
        page.on("request", (request) => {
          if (/ort\.(webgpu|wasm)\.min\.js/.test(request.url())) runtimes.push(request.url());
          if (/\/assets\/models\/[^/?]+\.onnx(?:\?|$)/.test(request.url()))
            modelRequests.push(request.url());
        });
        await page.addInitScript(() => {
          globalThis.gpuSubmissions = 0;
          globalThis.workerCount = 0;
          globalThis.gestureActive = false;
          globalThis.gestureInference = 0;
          globalThis.gesturePacing = [];
          globalThis.uploadFrames = [];
          globalThis.retainedMapFrames = 0;
          const nativeFrame = requestAnimationFrame;
          globalThis.requestAnimationFrame = (callback) =>
            nativeFrame((time) => {
              globalThis.currentTestFrame = time;
              callback(time);
            });
          const NativeWorker = Worker;
          globalThis.Worker = class extends NativeWorker {
            constructor(...args) {
              super(...args);
              workerCount++;
              globalThis.terrainWorker = this;
              this.addEventListener("message", ({ data }) => {
                if (data.type === "gpu-proof") gpuSubmissions++;
                if (gestureActive && data.type === "phase" && data.phase === "inference")
                  gestureInference++;
              });
            }
            postMessage(message, ...transfers) {
              if (gestureActive && message.type === "frame-budget")
                gesturePacing.push(message.delayMs);
              super.postMessage(message, ...transfers);
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
        if (mode === "auto" && result.backend === "WebGPU") {
          const before = await page.evaluate(() => ({
            hash: location.hash,
            zoom: world.dataset.viewZoom,
            altitude: world.dataset.walkAltitude,
          }));
          await page.evaluate(() => {
            const mapPrototype = TerrainHiGAN.MapRenderer?.prototype;
            if (mapPrototype) {
              const paint = mapPrototype.paint,
                draw = mapPrototype.draw;
              mapPrototype.paint = function (...args) {
                uploadFrames.push(currentTestFrame);
                return paint.apply(this, args);
              };
              mapPrototype.draw = function (...args) {
                if (this.job && this.front) retainedMapFrames++;
                return draw.apply(this, args);
              };
            }
            const prototype = TerrainHiGAN.globe.GlobeRenderer?.prototype;
            if (prototype) {
              const upload = prototype.uploadTile;
              prototype.uploadTile = function (tile) {
                if (![...this.patchTiles.values()].every((patch) => patch.mesh.visible))
                  throw new Error("Installed terrain disappeared before its replacement was ready");
                uploadFrames.push(currentTestFrame);
                return upload.call(this, tile);
              };
            }
            gestureActive = true;
          });
          const touch = await page.context().newCDPSession(page);
          const move = async (type, points) => {
            await touch.send("Input.dispatchTouchEvent", {
              type,
              touchPoints: points.map(([id, x, y]) => ({ id, x, y })),
            });
            await page.evaluate(() => new Promise(requestAnimationFrame));
          };
          await move("touchStart", [
            [1, 120, 110],
            [2, 200, 110],
          ]);
          for (let step = 0; step < 90; step++)
            await move("touchMove", [
              [1, 80 + (step % 2), 110],
              [2, 240 + (step % 2), 110],
            ]);
          const gesture = await page.evaluate(() => ({
            inference: gestureInference,
            zoom: world.dataset.viewZoom,
            retainedMapFrames,
            delays: gesturePacing,
            altitude: world.dataset.walkAltitude,
            hash: location.hash,
          }));
          assert(gesture.inference > 0, "actual inference continues throughout a sustained pinch");
          assert(
            gesture.delays.some((delay) => delay > 0 && delay <= 32),
            "gestures pace the worker without stopping it",
          );
          if (view)
            assert(
              Number(gesture.altitude) < Number(before.altitude),
              "the globe camera moves during refinement",
            );
          else {
            assert(
              Number(gesture.zoom) > Number(before.zoom),
              "the map camera moves during refinement",
            );
            assert(
              gesture.retainedMapFrames > 0,
              "the committed 2D image stays drawable while its replacement is prepared",
            );
          }
          await move("touchEnd", []);
          await page.evaluate(() => {
            gestureActive = false;
          });
          await page.waitForFunction(
            () => /Ready|Error|Model failed/.test(document.getElementById("status").textContent),
            {},
            { timeout: 180000 },
          );
          assert.equal(await page.locator("#status").innerText(), "Ready");
          const uploads = await page.evaluate(() => uploadFrames);
          assert.equal(
            new Set(uploads).size,
            uploads.length,
            "at most one replacement uploads per animation frame",
          );
          assert(uploads.length > 0, "the zoom exercised incremental replacement uploads");
          result.gesture = {
            inferencePhases: gesture.inference,
            pacedMessages: gesture.delays.length,
            replacementUploads: uploads.length,
          };
          await touch.detach();
        }
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
          assert.equal(
            modelRequests.length,
            21,
            "deep inference downloads each model only once despite session eviction",
          );
          const cachedRequests = modelRequests.length;
          // All weights are now present. A different viewer and a page reload
          // must work with model-file network requests explicitly blocked.
          await page.route("**/assets/models/*.onnx*", (route) => route.abort());
          await page.goto(baseUrl + "globe.html#seed=4829347");
          const ready = () =>
            page.waitForFunction(
              () => /Ready|Error/.test(document.getElementById("status").textContent),
              {},
              { timeout: 180000 },
            );
          await ready();
          assert.equal(
            await page.locator("#status").innerText(),
            "Ready",
            await page.locator("#status").getAttribute("title"),
          );
          await page.reload();
          await ready();
          assert.equal(
            await page.locator("#status").innerText(),
            "Ready",
            await page.locator("#status").getAttribute("title"),
          );
          assert.equal(
            modelRequests.length,
            cachedRequests,
            "globe navigation and reload use cached files with no model transfers",
          );
          result.cachedModels = await page.evaluate(
            async () => (await (await caches.open("terrain-higan-models-v1")).keys()).length,
          );
          assert.equal(result.cachedModels, 21);
        }
        assert.equal(
          new Set(modelRequests).size,
          modelRequests.length,
          "model reloads never redownload cached weights",
        );
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
