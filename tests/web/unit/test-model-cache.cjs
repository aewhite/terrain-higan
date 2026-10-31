const assert = require("node:assert/strict");
const { createHash } = require("node:crypto");
const { loadModules } = require("../support/test-support.cjs");

(async () => {
  const entries = new Map(),
    downloads = [],
    phases = [],
    released = [];
  const models = Object.fromEntries(
    Array.from({ length: 21 }, (_, zoom) => {
      const name = zoom > 10 ? `z${zoom}s` : `z${zoom}`;
      const bytes = Uint8Array.from([zoom, 1, 2, 3]);
      return [name, { bytes, hash: createHash("sha256").update(bytes).digest("hex") }];
    }),
  );
  let bundle = "1234567890abcdef",
    corruptDownload = false;
  const cache = {
    async match(key) {
      return entries.get(key)?.clone();
    },
    async put(key, response) {
      entries.set(key, response.clone());
    },
    async delete(key) {
      return entries.delete(key);
    },
  };
  const { api, context } = loadModules({
    Response,
    caches: {
      async open() {
        return cache;
      },
    },
    console: { ...console, warn() {} },
    async fetch(url, options) {
      if (url.endsWith("manifest.json")) {
        assert.equal(options.cache, "no-store", "the current bundle is always checked");
        return {
          ok: true,
          json: async () => ({
            bundle,
            models: Object.fromEntries(
              Object.entries(models).map(([name, model]) => [name, model.hash]),
            ),
          }),
        };
      }
      downloads.push({ url, options });
      const name = new URL(url).pathname.split("/").pop().replace(".onnx", "");
      return new Response(corruptDownload ? Uint8Array.from([99]) : models[name].bytes);
    },
  });
  const ort = {
    env: { wasm: { numThreads: 1 } },
    InferenceSession: {
      async create(bytes) {
        return {
          bytes,
          async release() {
            released.push(bytes);
          },
        };
      },
    },
  };
  const runtime = () =>
    new api.ModelRuntime(
      ort,
      new URLSearchParams(`bundle=${bundle}&mobile=1&backend=wasm`),
      (message) => phases.push(message),
    );
  const first = runtime();
  for (let zoom = 0; zoom <= 20; zoom++) await first.sessionFor(zoom);
  assert.equal(entries.size, 21, "all downloaded model levels stay on disk");
  assert.equal(downloads.length, 21);
  assert(released.length > 0, "loaded sessions are still evicted to bound RAM");
  const liveSessions = first.sessions.size;
  for (let zoom = 0; zoom <= 20; zoom++) await first.sessionFor(zoom);
  assert.equal(first.sessions.size, liveSessions);
  assert.equal(downloads.length, 21, "session eviction never redownloads a model file");
  assert(phases.some((message) => message.message === "Read z0"));

  // A new worker and a UI-only bundle update share the same immutable weights.
  bundle = "fedcba0987654321";
  const second = runtime();
  for (let zoom = 0; zoom <= 20; zoom++) await second.sessionFor(zoom);
  assert.equal(downloads.length, 21, "workers/reloads/code changes reuse downloaded weights");
  const name = "z20s";
  const key = [...entries.keys()].find((url) => url.includes(`${name}.onnx`));
  entries.set(key, new Response(Uint8Array.from([99])));
  const repaired = await runtime().verifiedModelBytes(20);
  assert.deepEqual(Array.from(repaired), Array.from(models[name].bytes));
  assert.equal(downloads.length, 22, "a damaged local entry is repaired with one download");
  assert.deepEqual(
    Array.from(new Uint8Array(await entries.get(key).clone().arrayBuffer())),
    Array.from(models[name].bytes),
  );

  const changed = Uint8Array.from([20, 4, 5, 6]);
  models[name] = { bytes: changed, hash: createHash("sha256").update(changed).digest("hex") };
  await runtime().verifiedModelBytes(20);
  assert.equal(downloads.length, 23, "changed weights get a new hash key");
  assert.equal(entries.size, 22, "other downloaded models are retained");
  const stale = runtime();
  bundle = "0000000000000000";
  await assert.rejects(stale.verifiedModelBytes(20), /Model bundle changed/);
  assert.equal(downloads.length, 23, "a stale page cannot bypass the bundle check via disk cache");

  entries.clear();
  corruptDownload = true;
  await assert.rejects(runtime().verifiedModelBytes(20), /does not match model bundle/);
  assert.equal(entries.size, 0, "unverified downloads are never persisted");
  corruptDownload = false;
  cache.put = async () => {
    throw new Error("QuotaExceededError");
  };
  assert.deepEqual(
    Array.from(await runtime().verifiedModelBytes(20)),
    Array.from(changed),
    "storage quota errors do not prevent rendering",
  );
  context.caches = undefined;
  await runtime().verifiedModelBytes(20);
  assert.equal(
    downloads.at(-1).options.cache,
    "force-cache",
    "HTTP caching remains available without Cache Storage",
  );
  context.caches = {
    async open() {
      throw new Error("SecurityError");
    },
  };
  await runtime().verifiedModelBytes(20);
  assert.equal(downloads.at(-1).options.cache, "force-cache");
  console.log(
    "All-level disk caching, bounded sessions, cross-worker/bundle reuse, integrity/repair and storage fallback passed",
  );
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
