const { chromium } = require("playwright");
const fs = require("node:fs");
const path = require("node:path");
const { browserOptions } = require("./browser-options.cjs");
const { parseArgs } = require("node:util");
const { root, resolvePath } = require("./project-paths.cjs");

const { values } = parseArgs({ options: { "runs-dir": { type: "string" } } });

const baseUrl = process.env.DEMO_URL || "http://127.0.0.1:4173/src/terrain_higan/demos/web/";
const seed = "4829347";
const worldX = Number(process.env.WORLD_X ?? -339951344);
const worldY = Number(process.env.WORLD_Y ?? -31941824);
const size = 32;
const viewWidth = Number(process.env.VIEW_WIDTH ?? 1132);
const viewHeight = Number(process.env.VIEW_HEIGHT ?? 990);
const warmZoom = Number(process.env.WARM_ZOOM ?? 19);
const warmSize = Number(process.env.WARM_SIZE ?? 256);

(async () => {
  const browser = await chromium.launch(
    browserOptions({
      headless: false,
    }),
  );
  try {
    const page = await browser.newPage({
      ignoreHTTPSErrors: true,
      viewport: { width: 128, height: 128 },
    });
    page.on("console", (message) => {
      if (message.type() === "log") console.log(message.text());
    });
    page.on("pageerror", (error) => console.error(error));
    if (process.env.MODE === "ui") {
      await page.setViewportSize({ width: viewWidth, height: viewHeight });
      await page.goto(
        `${baseUrl}?backend=webgpu#seed=${seed}&z=20&x=${worldX}&y=${worldY}&map=realistic`,
      );
      await page.waitForFunction(
        () => document.querySelector("#status").textContent.includes("Model terrain ready"),
        null,
        { timeout: 600000 },
      );
      await page.mouse.move(Math.round(viewWidth / 2 + 52), Math.round(viewHeight / 2 - 52));
      console.log(
        JSON.stringify({
          comparison: "fresh-ui",
          status: await page.locator("#status").innerText(),
          elevation: await page.locator("#elevation").innerText(),
          bundle: await page.locator("#bundle").innerText(),
          modelRuns: await page.locator("#world").getAttribute("data-model-runs"),
          cacheHits: await page.locator("#world").getAttribute("data-cache-hits"),
        }),
      );
      const outputDir = path.join(
        resolvePath(values["runs-dir"] || path.join(root, "runs")),
        "diagnostics",
      );
      fs.mkdirSync(outputDir, { recursive: true });
      await page.screenshot({ path: path.join(outputDir, "terrain_higan-z20-fresh.png") });
      return;
    }
    await page.goto(`${baseUrl}?backend=wasm#seed=${seed}&z=0&x=0&y=0&map=elevation`);
    await page.evaluate(
      async ({ seed, worldX, worldY, size, mode, maxZoom, warmZoom, warmSize }) => {
        if (mode === "direct") {
          const script = document.createElement("script");
          script.src = "./assets/runtime/ort.webgpu.min.js";
          await new Promise((resolve, reject) => {
            script.onload = resolve;
            script.onerror = reject;
            document.head.append(script);
          });
          ort.env.wasm.wasmPaths = new URL("./assets/runtime/", location.href).href;
          const width = 32,
            height = 32;
          const parent = Float32Array.from(
            { length: width * height },
            (_, index) => 3.2 + 0.1 * Math.sin(index * 0.37),
          );
          const noise = Float32Array.from(
            { length: 4 * width * height },
            (_, index) => Math.sin(index * 0.47) + Math.cos(index * 0.19),
          );
          for (const name of ["z1", "z2", "z9", "z16s"]) {
            const bytes = new Uint8Array(
              await fetch(`./assets/models/${name}.onnx`, { cache: "no-store" }).then((response) =>
                response.arrayBuffer(),
              ),
            );
            const values = [];
            for (const backend of ["webgpu", "wasm"]) {
              const session = await ort.InferenceSession.create(bytes, {
                executionProviders: [backend],
              });
              const result = await session.run({
                input_map: new ort.Tensor("float32", parent, [1, 1, height, width]),
                noise: new ort.Tensor("float32", noise, [1, 4, height, width]),
              });
              values.push(result.elevation_normalized.data);
              await session.release();
            }
            let sum = 0,
              max = 0;
            for (let index = 0; index < values[0].length; index++) {
              const delta = Math.abs(values[0][index] - values[1][index]);
              sum += delta;
              max = Math.max(max, delta);
            }
            console.log(
              JSON.stringify({
                comparison: "direct-model",
                name,
                meanAbs: sum / values[0].length,
                maxAbs: max,
              }),
            );
          }
          return;
        }
        const manifest = await fetch("./assets/models/manifest.json", { cache: "no-store" }).then(
          (response) => response.json(),
        );
        const makeWorker = (backend) => {
          const worker = new Worker(
            `./model-worker.js?backend=${backend}&bundle=${manifest.bundle}`,
          );
          let actualBackend = "";
          let nextId = 0;
          const request = (zoom, x, y, width = size, height = size, map = "elevation") =>
            new Promise((resolve, reject) => {
              const id = ++nextId;
              const key = `${backend}-${id}`;
              const timeout = setTimeout(
                () => reject(new Error("Worker request timed out")),
                120000,
              );
              const fail = (message) => {
                clearTimeout(timeout);
                reject(new Error(message));
              };
              worker.onmessage = ({ data }) => {
                if (data.type === "bundle-changed") {
                  fail("Model bundle changed");
                  return;
                }
                if (data.id !== id || data.key !== key) return;
                if (data.type === "error") fail(data.message);
                else if (data.type === "cancelled") fail("Request cancelled");
                else if (data.type === "tile") {
                  clearTimeout(timeout);
                  actualBackend = data.provider.label;
                  resolve(new Float32Array(data.elevations));
                }
              };
              worker.onerror = (event) => fail(event.message);
              worker.postMessage({
                type: "generate",
                id,
                key,
                seed,
                zoom,
                x,
                y,
                width,
                height,
                map,
              });
            });
          return { request, backend: () => actualBackend, close: () => worker.terminate() };
        };
        const gpu = makeWorker("webgpu"),
          cpu = makeWorker("wasm");
        try {
          if (mode === "warm") {
            const zoom = 20,
              tileSize = 256;
            const x = Math.floor(worldX / 0.5 / tileSize) * tileSize;
            const y = Math.floor(worldY / 0.5 / tileSize) * tileSize;
            const warmMpp = 512 * 2 ** (10 - warmZoom);
            const parentX = Math.floor(worldX / warmMpp / warmSize) * warmSize;
            const parentY = Math.floor(worldY / warmMpp / warmSize) * warmSize;
            console.log(
              JSON.stringify({
                comparison: "warming-parent",
                warmZoom,
                warmSize,
                parentX,
                parentY,
                x,
                y,
              }),
            );
            await gpu.request(warmZoom, parentX, parentY, warmSize, warmSize, "realistic");
            const values = await gpu.request(zoom, x, y, tileSize, tileSize, "realistic");
            let min = Infinity,
              max = -Infinity,
              sum = 0,
              ocean = 0;
            for (const value of values) {
              const meters = value * 2439.687062251 - 2342.974207789;
              min = Math.min(min, meters);
              max = Math.max(max, meters);
              sum += meters;
              if (meters < 0) ocean++;
            }
            const fresh = makeWorker("webgpu");
            try {
              const reference = await fresh.request(zoom, x, y, tileSize, tileSize, "realistic");
              let differenceSum = 0,
                differenceMax = 0,
                differingPixels = 0;
              for (let index = 0; index < values.length; index++) {
                const difference = Math.abs(values[index] - reference[index]) * 2439.687062251;
                differenceSum += difference;
                differenceMax = Math.max(differenceMax, difference);
                if (difference) differingPixels++;
              }
              console.log(
                JSON.stringify({
                  comparison: "warmed-z20-tile",
                  backend: gpu.backend(),
                  min,
                  max,
                  mean: sum / values.length,
                  ocean: ocean / values.length,
                  meanAbsMeters: differenceSum / values.length,
                  maxAbsMeters: differenceMax,
                  differingPixels,
                }),
              );
              if (differingPixels) throw new Error("Cache history changed the z20 tile");
            } finally {
              fresh.close();
            }
            return;
          }
          if (mode === "spot") {
            const zoom = 9,
              x = -331989,
              y = -31199,
              width = 8,
              height = 8;
            const [gpuValues, cpuValues] = await Promise.all([
              gpu.request(zoom, x, y, width, height),
              cpu.request(zoom, x, y, width, height),
            ]);
            const meters = (values) =>
              Array.from({ length: height }, (_, row) =>
                Array.from({ length: width }, (_, column) =>
                  Math.round(values[row * width + column] * 2439.687062251 - 2342.974207789),
                ),
              );
            console.log(
              JSON.stringify({
                comparison: "z9-ancestor",
                gpu: gpu.backend(),
                cpu: cpu.backend(),
                gpuMeters: meters(gpuValues),
                cpuMeters: meters(cpuValues),
              }),
            );
          }
          for (let zoom = 0; mode === "trace" && zoom <= maxZoom; zoom++) {
            const mpp = 512 * 2 ** (10 - zoom);
            const x = Math.floor(worldX / mpp - size / 2),
              y = Math.floor(worldY / mpp - size / 2);
            const [gpuValues, cpuValues] = await Promise.all([
              gpu.request(zoom, x, y),
              cpu.request(zoom, x, y),
            ]);
            let sum = 0,
              max = 0,
              gpuMean = 0,
              cpuMean = 0;
            for (let index = 0; index < gpuValues.length; index++) {
              const delta = Math.abs(gpuValues[index] - cpuValues[index]);
              sum += delta;
              max = Math.max(max, delta);
              gpuMean += gpuValues[index];
              cpuMean += cpuValues[index];
            }
            console.log(
              JSON.stringify({
                zoom,
                gpu: gpu.backend(),
                cpu: cpu.backend(),
                meanAbsMeters: (sum / gpuValues.length) * 2439.687062251,
                maxAbsMeters: max * 2439.687062251,
                gpuMean: gpuMean / gpuValues.length,
                cpuMean: cpuMean / cpuValues.length,
              }),
            );
          }
        } finally {
          gpu.close();
          cpu.close();
        }
        if (mode !== "overlap") return;
        const zoom = 16,
          innerSize = 16,
          outerSize = 32,
          mpp = 512 * 2 ** (10 - zoom);
        const x = Math.floor(worldX / mpp - innerSize / 2),
          y = Math.floor(worldY / mpp - innerSize / 2);
        for (const backend of ["webgpu", "wasm"]) {
          const smallWorker = makeWorker(backend),
            largeWorker = makeWorker(backend);
          try {
            console.log(`Computing ${backend} inner region`);
            const small = await smallWorker.request(zoom, x, y, innerSize, innerSize);
            console.log(`Computing ${backend} outer region`);
            const large = await largeWorker.request(zoom, x - 8, y - 8, outerSize, outerSize);
            let sum = 0,
              max = 0;
            for (let row = 0; row < innerSize; row++)
              for (let column = 0; column < innerSize; column++) {
                const delta = Math.abs(
                  small[row * innerSize + column] - large[(row + 8) * outerSize + column + 8],
                );
                sum += delta;
                max = Math.max(max, delta);
              }
            console.log(
              JSON.stringify({
                comparison: "small-vs-large",
                backend: smallWorker.backend(),
                meanAbs: sum / small.length,
                maxAbs: max,
              }),
            );
          } finally {
            smallWorker.close();
            largeWorker.close();
          }
        }
      },
      {
        seed,
        worldX,
        worldY,
        size,
        warmZoom,
        warmSize,
        mode: process.env.MODE || "trace",
        maxZoom: Number(process.env.MAX_ZOOM ?? 16),
      },
    );
  } finally {
    await browser.close();
  }
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
