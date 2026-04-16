#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/yassinebenacha/multilingual-noise-demo.git"
TARGET_DIR="${1:-multilingual-noise-demo}"

if [[ -e "$TARGET_DIR" ]]; then
  echo "Error: target path '$TARGET_DIR' already exists." >&2
  exit 1
fi

echo "Cloning $REPO_URL into $TARGET_DIR..."
git clone "$REPO_URL" "$TARGET_DIR"

echo "Done."
