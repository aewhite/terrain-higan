const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const vm = require("node:vm");
const { demoDir, demoRequire, rootDir } = require("../support/test-support.cjs");
const { defaultWeightsDir, resolvePath } = demoRequire("./project-paths.cjs");
const { browserOptions } = demoRequire("./browser-options.cjs");

for (const version of ["16.2.0", "20.19.0"]) {
  const entryModule = {};
  const exitSignal = new Error("process.exit");
  let message;
  const requireEntry = () => assert.fail("Unsupported Node must exit before loading dependencies");
  requireEntry.main = entryModule;
  assert.throws(
    () =>
      vm.runInNewContext(fs.readFileSync(path.join(demoDir, "sync-models.cjs"), "utf8"), {
        module: entryModule,
        require: requireEntry,
        process: {
          versions: { node: version },
          version: `v${version}`,
          exit(code) {
            assert.equal(code, 1);
            throw exitSignal;
          },
        },
        console: {
          error(value) {
            message = value;
          },
        },
      }),
    (error) => error === exitSignal,
  );
  assert.match(message, /requires Node 22 or newer/);
  assert.ok(message.includes(`v${version}`));
}

const savedChrome = process.env.CHROME_PATH;
try {
  assert.equal(defaultWeightsDir, path.join(rootDir, ".local/weights"));
  assert.equal(resolvePath("custom-weights"), path.resolve("custom-weights"));
  assert.equal(resolvePath("~/weights"), path.join(os.homedir(), "weights"));
  delete process.env.CHROME_PATH;
  assert.equal(browserOptions().executablePath, undefined);
  process.env.CHROME_PATH = path.join(rootDir, "custom-browser");
  assert.equal(browserOptions({ headless: true }).executablePath, process.env.CHROME_PATH);
  assert.equal(browserOptions({ headless: true }).headless, true);
} finally {
  if (savedChrome === undefined) delete process.env.CHROME_PATH;
  else process.env.CHROME_PATH = savedChrome;
}

for (const [host, weights] of [
  [undefined, undefined],
  ["0.0.0.0", "custom-weights"],
]) {
  let handler;
  let binding;
  let checkedWeights;
  const context = {
    __dirname: demoDir,
    console: { log() {} },
    URL,
    process: {
      argv: ["node", "serve.cjs", "4199", ...(weights ? ["--weights-dir", weights] : [])],
      env: host ? { HOST: host } : {},
    },
    require(name) {
      if (name === "./sync-models.cjs")
        return {
          checkModels(directory) {
            checkedWeights = directory;
          },
        };
      if (name.startsWith("./")) return demoRequire(name);
      if (name === "node:http")
        return {
          createServer(callback) {
            handler = callback;
            return {
              listen(port, selectedHost) {
                binding = [port, selectedHost];
              },
            };
          },
        };
      return require(name);
    },
  };
  vm.runInNewContext(fs.readFileSync(path.join(demoDir, "serve.cjs"), "utf8"), context);
  assert.deepEqual(binding, [4199, host || "127.0.0.1"]);
  assert.equal(checkedWeights, weights ? path.resolve(weights) : defaultWeightsDir);
  for (const url of [
    "/.local/environment.sh",
    "/.idea/misc.xml",
    "/%2egit/config",
    "/.env",
    "/src/overrides.py",
  ]) {
    let status;
    let ended = false;
    const response = {
      writeHead(code) {
        status = code;
        return this;
      },
      end() {
        ended = true;
      },
    };
    handler({ url }, response);
    assert.equal(status, 403, url);
    assert.equal(ended, true);
  }
}

console.log("Environment paths, browser selection, local binding, and private-file checks passed");
