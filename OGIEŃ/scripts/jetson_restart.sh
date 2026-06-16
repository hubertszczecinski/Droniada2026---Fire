#!/usr/bin/env bash
# Szybki restart po sync kodu — bez przebudowy obrazu i bez ponownego skanowania kamery.
# Trwa ~20 s (graceful stop). Nie używaj gdy zmieniłeś kamerę / docker-compose devices.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export WS_ROOT_PATH="${WS_ROOT_PATH:-$ROOT}"

echo "=== Droniada — restart kontenera ==="
echo "(tylko kod z /ws — bez rebuildu; pełny start kamery: ./scripts/jetson_competition_start.sh)"
echo ""

if ! docker ps -a --filter name=droniada_vision --format '{{.Names}}' 2>/dev/null | grep -q droniada_vision; then
  echo "Kontener nie istnieje — uruchamiam pełny start…" >&2
  exec ./scripts/jetson_competition_start.sh
fi

echo "Restart… (poczekaj ~20 s, nie przerywaj Ctrl+C)"
docker compose -f docker-compose.jetson.yml restart
sleep 6

if docker ps --filter name=droniada_vision --format '{{.Status}}' | grep -qi up; then
  IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
  echo ""
  echo "OK — kontener działa."
  echo "Podgląd:    http://${IP:-<jetson-ip>}:${DRONIADA_HOST_PORT:-8088}/"
  echo "Sterowanie: http://${IP:-<jetson-ip>}:${DRONIADA_CONTROL_PORT:-8089}/"
else
  echo "Błąd restartu. Logi:" >&2
  docker logs droniada_vision 2>&1 | tail -15
  exit 1
fi
