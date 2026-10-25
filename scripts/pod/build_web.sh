#!/usr/bin/env bash
# Build the board (web/dist, which the server serves). The runpod/pytorch image has no Node,
# so this installs Node 22 under /workspace/.node (survives pod restarts) if node is missing.
set -euo pipefail
cd "$(dirname "$0")/../.."

NODE_VERSION=${NODE_VERSION:-v22.11.0}
NODE_DIR=/workspace/.node/node-$NODE_VERSION-linux-x64
if ! command -v node >/dev/null; then
  if [ ! -x "$NODE_DIR/bin/node" ]; then
    mkdir -p /workspace/.node
    curl -fsSL "https://nodejs.org/dist/$NODE_VERSION/node-$NODE_VERSION-linux-x64.tar.xz" | tar -xJ -C /workspace/.node
  fi
  export PATH="$NODE_DIR/bin:$PATH"
  grep -qxF "export PATH=$NODE_DIR/bin:\$PATH" ~/.bashrc 2>/dev/null || echo "export PATH=$NODE_DIR/bin:\$PATH" >> ~/.bashrc
fi
echo "node $(node --version)"
cd web && npm ci --no-audit --no-fund && npm run build
