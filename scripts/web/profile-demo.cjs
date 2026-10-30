const fs = require("node:fs");
const path = require("node:path");
const net = require("node:net");
const os = require("node:os");
const { spawn } = require("node:child_process");
const { createRequire } = require("node:module");
const { parseArgs } = require("node:util");
const { performance } = require("node:perf_hooks");
const root = path.resolve(__dirname, "../..");
const demo = path.join(root, "src/terrain_higan/demos/web");
const demoRequire = createRequire(path.join(demo, "package.json"));
const { chromium } = demoRequire("playwright");
const bootstrap = fs.readFileSync(path.join(__dirname, "profile-instrumentation.js"), "utf8");
const { values } = parseArgs({
  options: {
    out: { type: "string" },
    only: { type: "string" },
    chrome: { type: "string" },
    deep: { type: "boolean", default: false },
    control: { type: "boolean", default: false },
  },
});
const output = path.resolve(
  values.out || path.join(root, "runs/web-profile", new Date().toISOString().replace(/[:.]/g, "-")),
);
fs.mkdirSync(output, { recursive: true });
const freePort = () =>
  new Promise((resolve) => {
    const s = net.createServer();
    s.listen(0, "127.0.0.1", () => {
      const port = s.address().port;
      s.close(() => resolve(port));
    });
  });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
class CDP {
  constructor(socket) {
    this.socket = socket;
    this.next = 0;
    this.pending = new Map();
    this.sessions = new Map();
    this.active = true;
    this.errors = [];
    socket.addEventListener("message", (event) => {
      const message = JSON.parse(event.data);
      if (message.id) {
        const p = this.pending.get(message.id);
        if (p) {
          this.pending.delete(message.id);
          message.error
            ? p.reject(new Error(p.method + ": " + message.error.message))
            : p.resolve(message.result);
        }
        return;
      }
      if (message.method === "Target.attachedToTarget")
        this.attach(message.params).catch((e) => this.errors.push(String(e)));
      if (message.method === "Tracing.tracingComplete" && this.traceResolve)
        this.traceResolve(message.params.stream);
      if (message.method === "Target.detachedFromTarget") {
        const s = this.sessions.get(message.params.sessionId);
        if (s) s.detached = true;
      }
      if (message.method === "Target.targetInfoChanged")
        for (const s of this.sessions.values())
          if (s.target.targetId === message.params.targetInfo.targetId)
            s.target = message.params.targetInfo;
    });
  }
  static async connect(port) {
    const version = await fetch(`http://127.0.0.1:${port}/json/version`).then((r) => r.json());
    const socket = new WebSocket(version.webSocketDebuggerUrl);
    await new Promise((resolve, reject) => {
      socket.addEventListener("open", resolve, { once: true });
      socket.addEventListener("error", reject, { once: true });
    });
    const c = new CDP(socket);
    await c.send("Target.setDiscoverTargets", { discover: true });
    await c.send("Target.setAutoAttach", {
      autoAttach: true,
      waitForDebuggerOnStart: true,
      flatten: true,
    });
    return c;
  }
  send(method, params = {}, sessionId) {
    const id = ++this.next;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject, method });
      this.socket.send(JSON.stringify({ id, method, params, ...(sessionId ? { sessionId } : {}) }));
    });
  }
  async attach({ sessionId, targetInfo, waitingForDebugger }) {
    if (!["page", "worker", "iframe"].includes(targetInfo.type)) {
      if (waitingForDebugger) await this.send("Runtime.runIfWaitingForDebugger", {}, sessionId);
      return;
    }
    const s = { id: sessionId, target: targetInfo, running: false, detached: false };
    this.sessions.set(sessionId, s);
    try {
      await this.send(
        "Target.setAutoAttach",
        { autoAttach: true, waitForDebuggerOnStart: true, flatten: true },
        sessionId,
      ).catch(() => {});
      await this.send("Profiler.enable", {}, sessionId);
      await this.send("Profiler.setSamplingInterval", { interval: 1000 }, sessionId);
      if (this.active) {
        await this.send("Profiler.start", {}, sessionId);
        s.running = true;
      }
    } finally {
      if (waitingForDebugger) await this.send("Runtime.runIfWaitingForDebugger", {}, sessionId);
    }
  }
  async reset() {
    this.active = true;
    for (const s of this.sessions.values())
      if (!s.detached) {
        if (!s.running) {
          await this.send("Profiler.start", {}, s.id);
          s.running = true;
        }
        await this.send(
          "Runtime.evaluate",
          { expression: "globalThis.__terrainProfile?.reset()" },
          s.id,
        ).catch(() => {});
      }
  }
  async snapshot(name) {
    this.active = false;
    const result = [];
    for (const s of this.sessions.values())
      if (!s.detached && s.running) {
        try {
          const { profile } = await this.send("Profiler.stop", {}, s.id);
          s.running = false;
          const snapshot = await this.send(
            "Runtime.evaluate",
            {
              expression: "JSON.stringify(globalThis.__terrainProfile?.snapshot() || null)",
              returnByValue: true,
            },
            s.id,
          );
          const instrumentation = snapshot.result.value ? JSON.parse(snapshot.result.value) : null;
          if (!instrumentation) continue;
          const role = s.target.type === "page" ? "page" : "terrain";
          const id = result.length;
          fs.writeFileSync(
            path.join(output, `${name}-${role}-${id}.cpuprofile`),
            JSON.stringify(profile),
          );
          result.push({ role, url: s.target.url, instrumentation, cpu: cpuSummary(profile) });
        } catch (e) {
          this.errors.push(name + ": " + e.message);
        }
      }
    return result;
  }
  async startTrace() {
    const available = await this.send("Tracing.getCategories");
    this.traceCategories = available.categories.filter((c) => /wasm|v8.compile/.test(c));
    await this.send("Tracing.start", {
      categories: ["devtools.timeline", "blink.user_timing", "v8", ...this.traceCategories].join(
        ",",
      ),
      transferMode: "ReturnAsStream",
      streamCompression: "gzip",
    });
  }
  async endTrace(name) {
    const streamPromise = new Promise((resolve) => (this.traceResolve = resolve));
    await this.send("Tracing.end");
    const handle = await streamPromise;
    const chunks = [];
    for (;;) {
      const part = await this.send("IO.read", { handle });
      chunks.push(Buffer.from(part.data, part.base64Encoded ? "base64" : "utf8"));
      if (part.eof) break;
    }
    await this.send("IO.close", { handle });
    const buffer = Buffer.concat(chunks);
    fs.writeFileSync(path.join(output, name + "-trace.json.gz"), buffer);
    const trace = JSON.parse(require("node:zlib").gunzipSync(buffer).toString());
    const interesting = {};
    for (const e of trace.traceEvents)
      if (e.ph === "X" && /wasm|compil/i.test(e.name)) {
        const row = (interesting[e.name] ||= { count: 0, ms: 0 });
        row.count++;
        row.ms += (e.dur || 0) / 1000;
      }
    return { categories: this.traceCategories, events: interesting };
  }
  close() {
    this.socket.close();
  }
}
function cpuSummary(profile) {
  const nodes = new Map(profile.nodes.map((n) => [n.id, n]));
  const parents = new Map();
  for (const n of profile.nodes) for (const child of n.children || []) parents.set(child, n.id);
  const self = new Map(),
    categories = {};
  let total = 0,
    idle = 0;
  for (let i = 0; i < (profile.samples || []).length; i++) {
    const n = nodes.get(profile.samples[i]),
      ms = Math.max(0, profile.timeDeltas?.[i] ?? 1000) / 1000;
    if (!n) continue;
    if (n.callFrame.functionName === "(idle)") {
      idle += ms;
      continue;
    }
    let chain = [],
      cursor = n;
    while (cursor) {
      chain.push(cursor.callFrame);
      cursor = nodes.get(parents.get(cursor.id));
    }
    const names = chain.map((f) => f.functionName).join(" "),
      urls = chain.map((f) => f.url).join(" ");
    let category = "Other JS/native";
    if (chain.some((f) => f.url.startsWith("wasm:") || f.functionName.includes("wasm-function")))
      category = "WASM";
    else if (
      names.includes("updateViewStats") ||
      names.includes("summarizeElevations") ||
      names.includes("updateStats")
    )
      category = "Statistics";
    else if (/surfacePick|intersectObject|intersectTriangle/.test(names))
      category = "Surface picking";
    else if (/createTile|meshSegments|meshVertex|stitchPatches|chartError|colors/.test(names))
      category = "Globe geometry/textures";
    else if (names.includes("drawTiles")) category = "2D compositing";
    else if (urls.includes("globe-renderer.js") && /draw|render/.test(names))
      category = "Globe draw submission";
    else if (urls.includes("globe-tile-controller.js")) category = "Globe LOD/selection";
    else if (urls.includes("noise.js") || names.includes("noise")) category = "Noise";
    else if (urls.includes("rendering.js")) category = "Terrain shading/colors";
    else if (urls.includes("globe-terrain.js") || urls.includes("/terrain.js"))
      category = "Terrain dependencies/cache";
    else if (urls.includes("model-runtime.js") || urls.includes("ort."))
      category = "Runtime JS/copies";
    else if (urls.includes("profile-instrumentation"))
      category = "Profiling instrumentation/other wrapped work";
    if (n.callFrame.functionName === "(garbage collector)") category = "GC";
    if (n.callFrame.functionName === "(program)") category = "Unattributed (program)";
    categories[category] = (categories[category] || 0) + ms;
    total += ms;
    const key = (n.callFrame.functionName || "(anonymous)") + " @ " + n.callFrame.url;
    self.set(key, (self.get(key) || 0) + ms);
  }
  const programSampleMs = categories["Unattributed (program)"] || 0;
  return {
    busySampleMs: total,
    attributedSampleMs: total - programSampleMs,
    programSampleMs,
    idleSampleMs: idle,
    categories,
    topSelf: [...self].sort((a, b) => b[1] - a[1]).slice(0, 30),
  };
}
function wrapArrow(source, name, next, label) {
  const start = source.indexOf(`  const ${name} = `),
    end = source.indexOf(next, start);
  if (start < 0 || end < 0) throw Error("Cannot instrument " + name);
  let block = source.slice(start, end);
  block = block.replace(
    `const ${name} = `,
    `const ${name} = __terrainProfile.measure(${JSON.stringify(label)}, `,
  );
  const close = block.lastIndexOf("};");
  if (close < 0) throw Error("Bad arrow " + name);
  block = block.slice(0, close) + "});" + block.slice(close + 2);
  return source.slice(0, start) + block + source.slice(end);
}
function instrument(file, source) {
  const h = "globalThis.__terrainProfile";
  source = bootstrap + "\n" + source;
  let tail = "";
  const wrap = (object, methods, prefix) =>
    (tail += `\n${h}.wrap(${object},${JSON.stringify(methods)},${JSON.stringify(prefix)});`);
  if (file === "terrain-common.js") wrap("TerrainHiGAN", ["summarizeElevations"], "terrain");
  if (file === "noise.js")
    wrap("TerrainHiGAN", ["normalField", "upscalerNoise", "noiseKey"], "noise");
  if (file === "rendering.js") wrap("TerrainHiGAN", ["renderRegion", "reliefShade"], "color");
  if (file === "terrain.js")
    wrap(
      "TerrainHiGAN.TerrainGenerator.prototype",
      ["generate", "generateUncached", "generateBlock"],
      "flat",
    );
  if (file === "globe-terrain.js")
    wrap(
      "TerrainHiGAN.globe.GlobeGenerator.prototype",
      ["tile", "noise", "gather", "rawBlock", "finalBlock", "vertices"],
      "globeTerrain",
    );
  if (file === "inference-runtime.js")
    wrap("TerrainHiGAN.InferenceRuntime.prototype", ["run"], "queue");
  if (file === "tile-controller.js")
    wrap(
      "TerrainHiGAN.TileController.prototype",
      ["setView", "dispatch", "receive"],
      "flatController",
    );
  if (file === "globe-tile-controller.js")
    wrap(
      "TerrainHiGAN.globe.GlobeTileController.prototype",
      ["update", "receive"],
      "globeController",
    );
  if (file === "globe-renderer.js") {
    wrap(
      "TerrainHiGAN.globe.GlobeRenderer.prototype",
      [
        "prepareView",
        "show",
        "createTile",
        "colors",
        "meshSegments",
        "draw",
        "surfacePick",
        "measure",
        "drag",
        "zoom",
      ],
      "globeRenderer",
    );
    tail += `\n{const original=TerrainHiGAN.globe.GlobeRenderer.prototype.setView;TerrainHiGAN.globe.GlobeRenderer.prototype.setView=function(...args){${h}.renderer=this;${h}.wrap(this.renderer,['render'],'three');return original.apply(this,args);};}`;
  }
  if (file === "model-runtime.js") {
    tail += `\n{const p=TerrainHiGAN.ModelRuntime.prototype;const original=p.loadSession;p.loadSession=${h}.measure(function(z){return 'model.load.'+TerrainHiGAN.modelName(z);},function(z){${h}.currentModel=TerrainHiGAN.modelName(z);return original.call(this,z);});`;
    tail += `const create=ort.InferenceSession.create;ort.InferenceSession.create=async function(...args){const name=${h}.currentModel||'unknown';const session=await ${h}.measure('onnx.init.'+name,create).apply(this,args);session.run=${h}.measure('onnx.run.'+name,session.run);session.release=${h}.measure('onnx.release.'+name,session.release);return session;};}`;
    wrap(
      "TerrainHiGAN.ModelRuntime.prototype",
      ["verifiedModelBytes", "run", "runSmallPatches"],
      "model",
    );
  }
  if (file === "app.js") {
    source = wrapArrow(source, "drawTiles", "\n  function receiveTile", "map.drawTiles");
    source = wrapArrow(source, "render", "\n  let renderFrame", "map.render");
    source = wrapArrow(source, "updateViewStats", "\n  const queueViewStats", "map.statistics");
  }
  if (file === "globe-app.js")
    source = wrapArrow(source, "updateView", "\n  const queueView", "globe.updateView");
  return source + tail;
}
const percentile = (values, q) => {
  const a = [...values].sort((a, b) => a - b);
  return a[Math.floor(a.length * q)] || 0;
};
async function state(page) {
  return page.evaluate(() => ({
    status: document.querySelector("#status").textContent,
    backend: document.querySelector("#backend").textContent,
    backendDetail: document.querySelector("#backend").title,
    canvas: { width: world.width, height: world.height, ...world.dataset },
    zoom: new URLSearchParams(location.hash.slice(1)).get("z"),
  }));
}
async function ready(page) {
  await page.waitForFunction(
    () => /Ready|Model failed|Error/.test(document.querySelector("#status").textContent),
    {},
    { timeout: 300000 },
  );
  const s = await state(page);
  if (s.status !== "Ready") throw Error(JSON.stringify(s));
  return s;
}
async function runControl(config, view, base) {
  const browser = await chromium.launch({
    executablePath:
      values.chrome ||
      process.env.CHROME_PATH ||
      "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    headless: false,
    args: ["--headless=new"],
  });
  try {
    const context = await browser.newContext({
      viewport: config.viewport,
      deviceScaleFactor: config.dpr,
      hasTouch: config.mobile,
      ...(config.mobile
        ? { userAgent: "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) Mobile Safari" }
        : {}),
    });
    const page = await context.newPage();
    const errors = [];
    page.on("pageerror", (e) => errors.push(e.message));
    const stages = [];
    const stage = async (name, action) => {
      const start = performance.now();
      await action();
      stages.push({ name, readyMs: performance.now() - start, state: await state(page) });
      await page.waitForTimeout(250);
    };
    await stage("startup", async () => {
      await page.goto(
        base + (view === "globe" ? "globe.html" : "") + `?workers=${config.workers}#seed=4829347`,
      );
      await ready(page);
    });
    await page.keyboard.press("f");
    const x = config.viewport.width / 2,
      y = config.viewport.height / 2;
    const drag = async (offset, steps = 16) => {
      await page.mouse.move(x, y);
      await page.mouse.down();
      for (let i = 1; i <= steps; i++) {
        await page.mouse.move(x + (offset * i) / steps, y);
        await page.waitForTimeout(16);
      }
      await page.mouse.up();
    };
    await drag(16);
    await ready(page);
    await drag(-16);
    await ready(page);
    await page.waitForTimeout(200);
    await stage("warm-pan", async () => {
      await drag(16, 24);
      await drag(-16, 24);
      await ready(page);
    });
    await stage("zoom-and-pan", async () => {
      if (view === "map") {
        const target = Number((await state(page)).zoom) + 1;
        await page.keyboard.press("+");
        await page.waitForFunction(
          (target) => Number(new URLSearchParams(location.hash.slice(1)).get("z")) === target,
          target,
        );
      } else await page.mouse.wheel(0, -500);
      await drag(32, 24);
      await ready(page);
    });
    const result = { config, view, browser: browser.version(), stages, errors };
    fs.writeFileSync(
      path.join(output, config.name + "-" + view + "-control.json"),
      JSON.stringify(result, null, 2),
    );
    console.log(
      JSON.stringify({
        control: config.name + "-" + view,
        stages: stages.map((s) => ({ name: s.name, readyMs: Math.round(s.readyMs) })),
        errors,
      }),
    );
    return result;
  } finally {
    await browser.close();
  }
}
async function runCase(config, view, base) {
  const label = config.name + "-" + view;
  console.log("PROFILE " + label);
  const port = await freePort();
  const chrome =
    values.chrome ||
    process.env.CHROME_PATH ||
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
  const browser = await chromium.launch({
    executablePath: chrome,
    headless: false,
    args: ["--headless=new", `--remote-debugging-port=${port}`],
  });
  const cdp = await CDP.connect(port);
  try {
    const context = await browser.newContext({
      viewport: config.viewport,
      deviceScaleFactor: config.dpr,
      hasTouch: config.mobile,
      ...(config.mobile
        ? { userAgent: "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) Mobile Safari" }
        : {}),
    });
    await context.addInitScript({ content: bootstrap });
    const page = await context.newPage();
    const errors = [];
    page.on("pageerror", (e) => {
      errors.push(e.message);
      console.error("PAGE", e.message);
    });
    page.on("requestfailed", (r) => console.error("REQUEST", r.url(), r.failure()));
    page.on("console", (m) => {
      if (m.type() === "error") console.error("CONSOLE", m.text());
    });
    await page.route("**/*", async (route) => {
      const url = new URL(route.request().url());
      const file = path.basename(url.pathname);
      if (
        url.origin === new URL(base).origin &&
        url.pathname.endsWith(".js") &&
        !url.pathname.includes("/assets/")
      ) {
        const response = await route.fetch();
        const body = instrument(file, await response.text());
        fs.writeFileSync(path.join(output, "instrumented-" + file), body);
        await route.fulfill({ response, body });
      } else await route.continue();
    });
    const stages = [];
    await cdp.startTrace();
    const stage = async (name, action) => {
      await cdp.reset();
      const before = await cdp.send("SystemInfo.getProcessInfo");
      await page.evaluate((name) => performance.mark("profile:" + name + ":start"), name);
      const started = performance.now();
      const info = await action();
      const readyMs = performance.now() - started;
      await page.waitForTimeout(250);
      await page.evaluate((name) => performance.mark("profile:" + name + ":end"), name);
      const after = await cdp.send("SystemInfo.getProcessInfo");
      const targets = await cdp.snapshot(label + "-" + name);
      const processCpu = {};
      for (const p of after.processInfo) {
        const old = before.processInfo.find((o) => o.id === p.id);
        processCpu[p.type] = (processCpu[p.type] || 0) + (p.cpuTime - (old?.cpuTime || 0)) * 1000;
      }
      const snapshot = await state(page);
      const frames =
        targets.find((t) => t.role === "page" && t.instrumentation)?.instrumentation.frames || [];
      const record = {
        name,
        readyMs,
        wallMs: performance.now() - started,
        state: snapshot,
        info,
        processCpu,
        frameIntervals: {
          count: frames.length,
          p50: percentile(frames, 0.5),
          p95: percentile(frames, 0.95),
          max: Math.max(0, ...frames),
          over50: frames.filter((x) => x > 50).length,
        },
        targets,
      };
      stages.push(record);
      console.log(
        JSON.stringify({
          case: label,
          stage: name,
          readyMs: Math.round(readyMs),
          frames: record.frameIntervals,
          modelRuns: snapshot.canvas.modelRuns,
        }),
      );
      fs.writeFileSync(
        path.join(output, label + ".json"),
        JSON.stringify({ config, view, stages, errors, cdpErrors: cdp.errors }, null, 2),
      );
    };
    const url =
      base + (view === "globe" ? "globe.html" : "") + `?workers=${config.workers}#seed=4829347`;
    await stage("startup", async () => {
      await page.goto(url);
      return ready(page);
    });
    await page.keyboard.press("f");
    const x = config.viewport.width / 2,
      y = config.viewport.height / 2;
    const drag = async (offset, steps = 16) => {
      await page.mouse.move(x, y);
      await page.mouse.down();
      for (let i = 1; i <= steps; i++) {
        await page.mouse.move(x + (offset * i) / steps, y);
        await page.waitForTimeout(16);
      }
      await page.mouse.up();
    };
    await drag(16);
    await ready(page);
    await drag(-16);
    await ready(page);
    await page.waitForTimeout(200);
    await stage("warm-pan", async () => {
      const before = await state(page);
      await drag(16, 24);
      await drag(-16, 24);
      await ready(page);
      return { before };
    });
    await stage("zoom-and-pan", async () => {
      const before = await state(page);
      if (view === "map") {
        const target = Number(before.zoom) + 1;
        await page.keyboard.press("+");
        await page.waitForFunction(
          (target) => Number(new URLSearchParams(location.hash.slice(1)).get("z")) === target,
          target,
        );
      } else await page.mouse.wheel(0, -500);
      await drag(32, 24);
      await ready(page);
      return { before };
    });
    if (values.deep && view === "globe")
      await stage("deep-view", async () => {
        if (view === "globe") {
          await page.evaluate(() => {
            __terrainProfile.renderer.setView(20, -25, 1000);
            __terrainProfile.renderer.onChange();
          });
          await page.waitForTimeout(100);
          return ready(page);
        }
        return { skipped: true };
      });
    const environment = await page.evaluate(() => {
      const r = __terrainProfile.renderer;
      let gpu = null;
      if (r) {
        const gl = r.renderer.getContext(),
          e = gl.getExtension("WEBGL_debug_renderer_info");
        if (e) gpu = gl.getParameter(e.UNMASKED_RENDERER_WEBGL);
      }
      return {
        userAgent: navigator.userAgent,
        gpu,
        hardwareConcurrency: navigator.hardwareConcurrency,
        devicePixelRatio,
        crossOriginIsolated,
      };
    });
    const trace = await cdp.endTrace(label);
    const result = {
      config,
      view,
      browser: browser.version(),
      environment,
      trace,
      stages,
      errors,
      cdpErrors: cdp.errors,
    };
    fs.writeFileSync(path.join(output, label + ".json"), JSON.stringify(result, null, 2));
    await page.screenshot({ path: path.join(output, label + ".png") });
    if (errors.length || cdp.errors.length)
      console.log(JSON.stringify({ case: label, errors, cdpErrors: cdp.errors }));
    return result;
  } finally {
    cdp.close();
    await browser.close();
  }
}
function report(results) {
  const lines = [
    "# Web demo profile",
    "",
    `Host: ${os.cpus()[0].model}, ${os.cpus().length} logical CPUs, ${(os.totalmem() / 2 ** 30).toFixed(0)} GiB RAM. Browser: ${results[0]?.browser}.`,
    "",
    "Local loopback serving; no network/CPU throttling. Mobile is a configuration emulation on this Mac, not physical phone timing. Browser processes are fresh per case; route interception disables HTTP caching. The browser launch is excluded. Timers and 1 ms CPU sampling add overhead. GPU renderer is reported per case.",
    "",
    "| Case | Stage | Ready wall (s) | Renderer process CPU (s) | GPU-process CPU (s) | rAF p95 (ms) | Frames >50ms |",
    "|---|---|---:|---:|---:|---:|---:|",
  ];
  for (const r of results)
    for (const s of r.stages)
      lines.push(
        `| ${r.config.name} ${r.view} | ${s.name} | ${(s.readyMs / 1000).toFixed(2)} | ${((s.processCpu.renderer || 0) / 1000).toFixed(2)} | ${((s.processCpu.GPU || s.processCpu.gpu || 0) / 1000).toFixed(2)} | ${s.frameIntervals.p95.toFixed(1)} | ${s.frameIntervals.over50} |`,
      );
  lines.push(
    "",
    "Component timers below are cumulative across workers and inclusive where named; they are not an additive wall-time breakdown. ONNX init includes runtime/WASM startup on first use. ONNX run measures session execution including bindings; mobile patch copying is additionally visible in model.runSmallPatches. Canvas/WebGL timers measure submission, not hardware GPU elapsed time.",
    "",
  );
  for (const r of results) {
    lines.push(
      `## ${r.config.name} ${r.view}`,
      "",
      `Viewport: ${r.config.viewport.width}×${r.config.viewport.height}, DPR ${r.config.dpr}, workers ${r.config.workers}. GPU: ${r.environment.gpu || "2D canvas"}.`,
      "",
    );
    for (const s of r.stages) {
      const timers = {},
        categories = {};
      let mainBusy = 0,
        workerBusy = 0,
        wasm = 0;
      for (const t of s.targets) {
        if (t.role === "page") mainBusy += t.cpu.busySampleMs;
        else workerBusy += t.cpu.busySampleMs;
        for (const [k, v] of Object.entries(t.cpu.categories))
          categories[k] = (categories[k] || 0) + v;
        for (const [k, v] of Object.entries(t.instrumentation?.stats || {})) {
          const x = (timers[k] ||= { count: 0, wallMs: 0, selfSyncMs: 0, maxMs: 0 });
          x.count += v.count;
          x.wallMs += v.wallMs;
          x.selfSyncMs += v.selfSyncMs;
          x.maxMs = Math.max(x.maxMs, v.maxMs);
        }
        wasm += (t.instrumentation?.memories || []).reduce((n, m) => n + m.bytes, 0);
      }
      lines.push(
        `### ${s.name}`,
        "",
        `Sampled non-idle time, including unattributed native/browser execution: main ${(mainBusy / 1000).toFixed(2)} s; workers ${(workerBusy / 1000).toFixed(2)} s. This is not a measurement of JavaScript CPU alone. Allocated WASM memory at end: ${(wasm / 2 ** 20).toFixed(1)} MiB. Model runs: ${s.state.canvas.modelRuns || "n/a"}.`,
        "",
        "| Component | Calls | Cumulative elapsed ms | Self synchronous ms | Max call ms |",
        "|---|---:|---:|---:|---:|",
      );
      for (const [k, v] of Object.entries(timers)
        .filter(([k]) =>
          /onnx\.|network\.|hash\.|map\.|globeRenderer\.(show|createTile|draw|surfacePick)|globeController.update|three.render|globeTerrain.noise/.test(
            k,
          ),
        )
        .sort((a, b) => b[1].wallMs - a[1].wallMs)
        .slice(0, 22))
        lines.push(
          `| ${k} | ${v.count} | ${v.wallMs.toFixed(1)} | ${v.selfSyncMs.toFixed(1)} | ${v.maxMs.toFixed(1)} |`,
        );
      lines.push(
        "",
        "Sampled CPU categories (summed contexts): " +
          Object.entries(categories)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 8)
            .map(([k, v]) => `${k}: ${(v / 1000).toFixed(2)} s`)
            .join("; ") +
          ".",
        "",
      );
    }
  }
  lines.push(
    "Raw per-stage V8 CPU profiles (`.cpuprofile`) can be opened in Chrome DevTools or an IDE CPU-profile viewer. JSON files retain per-target timers, frame intervals, resource sizes, CPU categories, and memory snapshots. rAF intervals measure scheduling responsiveness, not app draw rate.",
    "",
  );
  fs.writeFileSync(path.join(output, "report.md"), lines.join("\n"));
}
async function main() {
  demoRequire("./sync-models.cjs").checkModels();
  const port = await freePort();
  const server = spawn(process.execPath, [path.join(demo, "serve.cjs"), String(port)], {
    cwd: root,
    env: { ...process.env, HOST: "127.0.0.1" },
    stdio: ["ignore", "pipe", "inherit"],
  });
  await new Promise((resolve, reject) => {
    server.stdout.on("data", (data) => {
      if (String(data).includes("http://")) resolve();
    });
    server.once("exit", (code) => reject(Error("Server exited " + code)));
  });
  const results = [];
  try {
    const configurations = [
      {
        name: "desktop",
        mobile: false,
        viewport: { width: 1280, height: 800 },
        dpr: 1,
        workers: 1,
      },
      { name: "mobile", mobile: true, viewport: { width: 375, height: 812 }, dpr: 2, workers: 1 },
    ];
    for (const c of configurations)
      for (const view of ["map", "globe"]) {
        if (values.only && values.only !== c.name + "-" + view) continue;
        results.push(
          await (values.control ? runControl : runCase)(
            c,
            view,
            `http://127.0.0.1:${port}/src/terrain_higan/demos/web/`,
          ),
        );
        if (!values.control) report(results);
      }
    fs.writeFileSync(
      path.join(output, values.control ? "control-summary.json" : "summary.json"),
      JSON.stringify(
        {
          host: { cpu: os.cpus()[0].model, cpus: os.cpus().length, memory: os.totalmem() },
          results,
        },
        null,
        2,
      ),
    );
    console.log(
      "REPORT " + path.join(output, values.control ? "control-summary.json" : "report.md"),
    );
  } finally {
    server.kill();
  }
}
module.exports = { cpuSummary, report };
if (require.main === module)
  main().catch((e) => {
    console.error(e);
    process.exitCode = 1;
  });
