const fs = require("node:fs");
const path = require("node:path");
const { spawn, spawnSync } = require("node:child_process");
const { parseArgs } = require("node:util");
const { demoDir, demoRequire } = require("./support/test-support.cjs");

async function main() {
  if (Number(process.versions.node.split(".")[0]) < 22)
    throw new Error("Web tests require Node 22 or newer.");
  const { values, positionals } = parseArgs({
    options: { "weights-dir": { type: "string" } },
    allowPositionals: true,
  });
  const [group = "unit", ...selectedNames] = positionals;
  const { defaultWeightsDir, resolvePath } = demoRequire("./project-paths.cjs");
  const weightsDir = resolvePath(values["weights-dir"] || defaultWeightsDir);
  const groups = group === "all" ? ["unit", "integration", "models", "browser"] : [group];
  for (const name of groups)
    if (!["unit", "integration", "models", "browser"].includes(name))
      throw new Error(`Unknown test group: ${name}`);
  const files = groups
    .flatMap((name) =>
      fs
        .readdirSync(path.join(__dirname, name))
        .filter((file) => file.startsWith("test-") && file.endsWith(".cjs"))
        .sort()
        .map((file) => path.join(__dirname, name, file)),
    )
    .filter((file) => !selectedNames.length || selectedNames.includes(path.basename(file)));
  for (const name of selectedNames)
    if (!files.some((file) => path.basename(file) === name))
      throw new Error(`No ${group} test named ${name}`);
  if (groups.includes("models") || groups.includes("browser"))
    demoRequire("./sync-models.cjs").checkModels(weightsDir);

  let server;
  const env = { ...process.env };
  try {
    if (groups.includes("browser") && !env.DEMO_URL) {
      const port = env.TEST_PORT || "4189";
      env.HOST = "127.0.0.1";
      server = spawn(
        process.execPath,
        [path.join(demoDir, "serve.cjs"), port, "--weights-dir", weightsDir],
        {
          cwd: demoDir,
          env,
          stdio: ["ignore", "pipe", "inherit"],
        },
      );
      await new Promise((resolve, reject) => {
        server.stdout.on("data", (chunk) => {
          process.stdout.write(chunk);
          if (chunk.toString().includes("http://")) resolve();
        });
        server.once("error", reject);
        server.once("exit", (code) => reject(new Error(`Test server exited: ${code}`)));
      });
      env.DEMO_URL = `http://127.0.0.1:${port}/src/terrain_higan/demos/web/`;
    }
    const failures = [];
    for (const file of files) {
      console.log(`\nRunning ${path.relative(__dirname, file)}`);
      const result = spawnSync(process.execPath, [file], { cwd: demoDir, env, stdio: "inherit" });
      if (result.error) throw result.error;
      if (result.status !== 0) failures.push(path.relative(__dirname, file));
    }
    console.log(`\n${files.length - failures.length}/${files.length} web test scripts passed`);
    if (failures.length) {
      console.error(`Failed: ${failures.join(", ")}`);
      process.exitCode = 1;
    }
  } finally {
    server?.kill();
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
