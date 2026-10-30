const http = require("node:http");
const https = require("node:https");
const fs = require("node:fs");
const path = require("node:path");
const { checkModels } = require("./sync-models.cjs");
const { parseArgs } = require("node:util");
const { defaultWeightsDir, resolvePath } = require("./project-paths.cjs");

const root = path.resolve(__dirname, "../../../..");
const { values, positionals } = parseArgs({
  args: process.argv.slice(2),
  options: { "weights-dir": { type: "string" } },
  allowPositionals: true,
});
checkModels(resolvePath(values["weights-dir"] || defaultWeightsDir));
const port = Number(positionals[0] || 4173);
const host = process.env.HOST || "127.0.0.1";
const tls = process.env.TLS_CERT
  ? {
      cert: fs.readFileSync(process.env.TLS_CERT),
      key: fs.readFileSync(process.env.TLS_KEY),
    }
  : null;
const types = {
  ".html": "text/html",
  ".css": "text/css",
  ".js": "text/javascript",
  ".json": "application/json",
  ".webmanifest": "application/manifest+json",
  ".svg": "image/svg+xml",
  ".mjs": "text/javascript",
  ".wasm": "application/wasm",
  ".onnx": "application/octet-stream",
};
const handler = (request, response) => {
  let file = path.resolve(
    root,
    "." + decodeURIComponent(new URL(request.url, "http://localhost").pathname),
  );
  if (
    !file.startsWith(root + path.sep) ||
    path.basename(file) === "overrides.py" ||
    path
      .relative(root, file)
      .split(path.sep)
      .some((segment) => segment.startsWith("."))
  ) {
    response.writeHead(403).end();
    return;
  }
  if (file.endsWith(path.sep) || path.extname(file) === "") file = path.join(file, "index.html");
  fs.stat(file, (error, stat) => {
    if (error || !stat.isFile()) {
      response.writeHead(404).end();
      return;
    }
    response.writeHead(200, {
      "Content-Type": types[path.extname(file)] || "application/octet-stream",
      "Content-Length": stat.size,
      "Cache-Control": "no-cache",
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    });
    if (request.method === "HEAD") response.end();
    else fs.createReadStream(file).pipe(response);
  });
};
const server = tls ? https.createServer(tls, handler) : http.createServer(handler);
server.listen(port, host, () =>
  console.log(`${tls ? "https" : "http"}://${host}:${port}/src/terrain_higan/demos/web/`),
);
