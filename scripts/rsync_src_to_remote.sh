#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <src-dir> <remote-host> <remote-project-dir>" >&2
  echo "Received $# argument(s): $*" >&2
  exit 1
fi

SRC_DIR="${1%/}/"
REMOTE_HOST="$2"
REMOTE_ROOT="${3%/}"

# Ensure remote target exists before syncing.
ssh "${REMOTE_HOST}" "mkdir -p '${REMOTE_ROOT}/src'"

rsync -az --delete \
  --exclude "__pycache__/" \
  --exclude "*.pyc" \
  "${SRC_DIR}" \
  "${REMOTE_HOST}:${REMOTE_ROOT}/src/"
