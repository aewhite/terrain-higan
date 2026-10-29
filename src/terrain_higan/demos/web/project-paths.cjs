const path = require("node:path");
const os = require("node:os");

const root = path.resolve(__dirname, "../../../..");
const defaultWeightsDir = path.join(root, ".local/weights");

function resolvePath(value) {
  const expanded = value === "~" ? os.homedir() : value.replace(/^~\//, `${os.homedir()}/`);
  return path.resolve(expanded);
}

module.exports = { root, defaultWeightsDir, resolvePath };
