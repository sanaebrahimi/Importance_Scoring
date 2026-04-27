#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

mkdir -p docs
python3 visualize_graph.py --load-mappings citation_mappings.json "$@" --output docs/index.html

rm -rf docs/lib
cp -R lib docs/lib

touch docs/.nojekyll

echo "GitHub Pages site rebuilt at docs/index.html"
