#!/usr/bin/env bash
# Zatrzymaj kontener Droniada na Jetsonie.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
docker compose -f docker-compose.jetson.yml down
echo "Kontener droniada_vision zatrzymany."
