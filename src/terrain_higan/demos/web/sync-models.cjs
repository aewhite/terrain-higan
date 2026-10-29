if (require.main === module && Number(process.versions.node.split(".")[0]) < 22) {
  console.error(
    `Weight syncing requires Node 22 or newer; found ${process.version}. Select Node 22+ on PATH and retry.`,
  );
  process.exit(1);
}

const fs = require("node:fs");
const path = require("node:path");
const crypto = require("node:crypto");
const { parseArgs } = require("node:util");
const { defaultWeightsDir, resolvePath } = require("./project-paths.cjs");

const schema = path.resolve(
  path.dirname(require.resolve("onnxruntime-web")),
  "../lib/onnxjs/ort-schema/protobuf/onnx.js",
);
const { onnx } = require(schema);
const modelNames = [
  "z0",
  ...Array.from({ length: 10 }, (_, index) => `z${index + 1}`),
  ...Array.from({ length: 10 }, (_, index) => `z${index + 11}s`),
];
const destinationDir = path.join(__dirname, "assets/models");
const manifestPath = path.join(destinationDir, "manifest.json");

function sourcePath(name, weightsDir) {
  return name === "z0"
    ? path.join(weightsDir, "z0_generator/latest/generator.onnx")
    : path.join(weightsDir, `zn_map_upscaler/${name}/latest/upscaler.onnx`);
}

function destinationPath(name) {
  return path.join(destinationDir, `${name}.onnx`);
}
function digest(file) {
  return crypto.createHash("sha256").update(fs.readFileSync(file)).digest("hex");
}
function bundleManifest(models, directory = __dirname) {
  // Include every extracted source and the pinned runtime identity, not just entry points.
  const files = fs.readdirSync(directory).filter((name) => /\.(html|js|css)$/.test(name));
  files.push("package-lock.json", "prepare-runtime.cjs");
  const runtimeDir = path.join(directory, "assets/runtime");
  if (fs.existsSync(runtimeDir))
    for (const file of ["three.module.js", "three.core.js", "OrbitControls.js", "GLTFExporter.js"])
      files.push(`assets/runtime/${file}`);
  const code = Object.fromEntries(
    files.sort().map((name) => [name, digest(path.join(directory, name))]),
  );
  const bundle = crypto
    .createHash("sha256")
    .update(JSON.stringify({ models, code }))
    .digest("hex")
    .slice(0, 16);
  return { bundle, models, code };
}

function checkInputs(name, file) {
  const model = onnx.ModelProto.decode(fs.readFileSync(file));
  const inputs = model.graph.input.map((input) => input.name);
  const expected = name === "z0" ? ["noise"] : ["input_map", "noise"];
  if (
    inputs.join(",") !== expected.join(",") ||
    model.graph.output.map((output) => output.name).join(",") !== "elevation_normalized"
  ) {
    throw new Error(`${name} has an incompatible ONNX input/output contract: ${inputs.join(", ")}`);
  }
}

function checkModels(weightsDir = defaultWeightsDir) {
  const stale = [];
  const models = {};
  for (const name of modelNames) {
    const source = sourcePath(name, weightsDir),
      destination = destinationPath(name);
    if (!fs.existsSync(source)) throw new Error(`Missing current model export: ${source}`);
    checkInputs(name, source);
    models[name] = digest(source);
    if (!fs.existsSync(destination) || models[name] !== digest(destination)) stale.push(name);
  }
  const expectedManifest = bundleManifest(models);
  if (
    !fs.existsSync(manifestPath) ||
    JSON.stringify(JSON.parse(fs.readFileSync(manifestPath, "utf8"))) !==
      JSON.stringify(expectedManifest)
  )
    stale.push("manifest.json");
  if (stale.length)
    throw new Error(
      `Demo models are missing or stale: ${stale.join(", ")}. Run npm run sync-models in ${__dirname}`,
    );
  console.log(`Verified ${modelNames.length} current demo models (${expectedManifest.bundle})`);
}

function syncModels(weightsDir = defaultWeightsDir) {
  fs.mkdirSync(destinationDir, { recursive: true });
  const models = {};
  for (const name of modelNames) {
    const source = sourcePath(name, weightsDir),
      destination = destinationPath(name);
    if (!fs.existsSync(source)) throw new Error(`Missing current model export: ${source}`);
    checkInputs(name, source);
    models[name] = digest(source);
    if (!fs.existsSync(destination) || models[name] !== digest(destination)) {
      const temporary = `${destination}.${process.pid}.tmp`;
      fs.copyFileSync(source, temporary);
      fs.renameSync(temporary, destination);
    }
  }
  const temporaryManifest = `${manifestPath}.${process.pid}.tmp`;
  fs.writeFileSync(temporaryManifest, JSON.stringify(bundleManifest(models)) + "\n");
  fs.renameSync(temporaryManifest, manifestPath);
  checkModels(weightsDir);
}

if (require.main === module) {
  const { values } = parseArgs({
    options: { check: { type: "boolean" }, "weights-dir": { type: "string" } },
  });
  const weightsDir = resolvePath(values["weights-dir"] || defaultWeightsDir);
  if (values.check) checkModels(weightsDir);
  else syncModels(weightsDir);
}

module.exports = { checkModels, bundleManifest };
