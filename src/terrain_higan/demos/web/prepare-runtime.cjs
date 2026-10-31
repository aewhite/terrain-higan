const fs = require("node:fs");
const path = require("node:path");

const runtimeFiles = [
  "ort.webgpu.min.js",
  "ort.wasm.min.js",
  "ort-wasm-simd-threaded.mjs",
  "ort-wasm-simd-threaded.wasm",
  "ort-wasm-simd-threaded.jsep.mjs",
  "ort-wasm-simd-threaded.jsep.wasm",
];
const sourceDir = path.dirname(require.resolve("onnxruntime-web"));
const destinationDir = path.join(__dirname, "assets/runtime");

fs.mkdirSync(destinationDir, { recursive: true });
for (const file of runtimeFiles)
  fs.copyFileSync(path.join(sourceDir, file), path.join(destinationDir, file));
// The pinned Emscripten loader reserves shared memory up to a 4 GiB maximum.
// Keep the mobile runtime's reservation bounded independently of model tile size.
const memoryDeclaration = "new WebAssembly.Memory({initial:256,maximum:65536,shared:!0})";
for (const suffix of ["", ".jsep"]) {
  const wasmLoader = fs.readFileSync(
    path.join(sourceDir, `ort-wasm-simd-threaded${suffix}.mjs`),
    "utf8",
  );
  if (wasmLoader.split(memoryDeclaration).length !== 2)
    throw new Error("ONNX Runtime memory declaration changed; review the mobile memory limit.");
  fs.writeFileSync(
    path.join(destinationDir, `ort-wasm-mobile${suffix}.mjs`),
    wasmLoader.replace(
      memoryDeclaration,
      "new WebAssembly.Memory({initial:256,maximum:4096,shared:!0})",
    ),
  );
}
fs.copyFileSync(require.resolve("js-sha256"), path.join(destinationDir, "sha256.js"));
const threeDir = path.resolve(path.dirname(require.resolve("three")), "..");
for (const file of ["three.module.js", "three.core.js"])
  fs.copyFileSync(path.join(threeDir, "build", file), path.join(destinationDir, file));
fs.copyFileSync(
  path.join(threeDir, "examples/jsm/controls/OrbitControls.js"),
  path.join(destinationDir, "OrbitControls.js"),
);
fs.copyFileSync(path.join(threeDir, "LICENSE"), path.join(destinationDir, "three-LICENSE"));
fs.copyFileSync(
  path.join(threeDir, "examples/jsm/exporters/GLTFExporter.js"),
  path.join(destinationDir, "GLTFExporter.js"),
);
console.log(`Copied ONNX Runtime browser assets from ${sourceDir}`);
