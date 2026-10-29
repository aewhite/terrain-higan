#!/usr/bin/env bash
set -euo pipefail

node -e 'if (Number(process.versions.node.split(".")[0]) < 22) { console.error(`Weight syncing requires Node 22 or newer; found ${process.version}. Select Node 22+ on PATH and retry.`); process.exit(1); }'

# Usage:
#   scripts/sync_latest_weights.sh [<models-source> [<weights-destination>]]
#
# <models-source> may be a local directory or an rsync remote, for example:
#   trainer:/path/to/project/data/models
# Also refreshes the web demo's ONNX models and bundle manifest.

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project_dir=$(cd -- "$script_dir/.." && pwd)
source_dir=${1:-${TERRAIN_HIGAN_WEIGHTS_SOURCE:-"$project_dir/data/models"}}
destination_dir=${2:-${TERRAIN_HIGAN_WEIGHTS_DIR:-"$project_dir/.local/weights"}}

mkdir -p "$destination_dir"
destination_dir=$(cd -- "$destination_dir" && pwd)

rsync \
  --archive \
  --copy-links \
  --prune-empty-dirs \
  --include='*/' \
  --include='latest/***' \
  --exclude='*' \
  "${source_dir%/}/" \
  "${destination_dir%/}/"

node "$project_dir/src/terrain_higan/demos/web/sync-models.cjs" --weights-dir "$destination_dir"
