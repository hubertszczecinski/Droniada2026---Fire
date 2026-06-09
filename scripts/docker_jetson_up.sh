#!/usr/bin/env bash
# Build + start kontenera Droniada na Jetsonie (pełny panel WWW).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export WS_ROOT_PATH="${WS_ROOT_PATH:-$ROOT}"
HOST_PORT="${DRONIADA_HOST_PORT:-8088}"
if [[ "${DRONIADA_OFFLINE:-0}" == "1" ]]; then
  docker compose -f docker-compose.jetson.yml up -d
else
  docker compose -f docker-compose.jetson.yml up -d --build
fi
echo ""
echo "Panel: http://$(hostname -I 2>/dev/null | awk '{print $1}'):${HOST_PORT}/"
