#!/usr/bin/env bash
# Sprawdza pretrain Blender; po zakończeniu (epoka 100 + best.pt) uruchamia etap 2 raz.
set -euo pipefail
cd "$(dirname "$0")/.."

PRETRAIN_DIR="runs/pose/droniada_blender_pretrain"
RESULTS="${PRETRAIN_DIR}/results.csv"
BEST="${PRETRAIN_DIR}/weights/best.pt"
LOCK="logs/stage2_autostart.lock"
LOG="logs/watch_pretrain.log"

mkdir -p logs

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG"; }

if [[ -f "$LOCK" ]]; then
  if pgrep -f "droniada_real_finetune" >/dev/null 2>&1; then
    log "stage2: fine-tune w toku (lock=$LOCK)"
  else
    log "stage2: lock istnieje, fine-tune zakończony lub nie działa"
  fi
  exit 0
fi

if pgrep -f "droniada_real_finetune" >/dev/null 2>&1; then
  log "stage2: fine-tune już działa — pomijam"
  touch "$LOCK"
  exit 0
fi

last_epoch=""
if [[ -f "$RESULTS" ]]; then
  last_epoch="$(tail -1 "$RESULTS" | cut -d, -f1)"
fi

pretrain_running=0
if pgrep -f "droniada_blender_pretrain" >/dev/null 2>&1; then
  pretrain_running=1
fi

if [[ "$pretrain_running" -eq 1 ]]; then
  log "pretrain: działa, epoka ${last_epoch:-?}/100"
  exit 0
fi

if [[ ! -f "$BEST" ]]; then
  log "pretrain: brak procesu i brak $BEST — epoka ${last_epoch:-?}; czekam / sprawdź ręcznie"
  exit 1
fi

pretrain_done=0
if [[ -n "$last_epoch" ]] && [[ "$last_epoch" -ge 100 ]]; then
  pretrain_done=1
elif [[ -f logs/train_pretrain_resume.log ]] && grep -qE '[0-9]+ epochs completed' logs/train_pretrain_resume.log 2>/dev/null; then
  pretrain_done=1
elif [[ -f logs/train_two_stage_1_blender.log ]] && grep -q '100 epochs completed' logs/train_two_stage_1_blender.log 2>/dev/null; then
  pretrain_done=1
fi

if [[ "$pretrain_done" -eq 0 ]]; then
  log "pretrain: zatrzymany przy epoce ${last_epoch:-?}/100 (best.pt jest) — czekam na dokończenie"
  exit 1
fi

log "pretrain: zakończony (epoka $last_epoch). Startuję etap 2 (fine-tune real) w tle..."
touch "$LOCK"
nohup bash -c './scripts/train_yolo_two_stage.sh 2 2>&1 | tee logs/train_two_stage_2_real.log' \
  >> logs/stage2_nohup.log 2>&1 &
log "stage2: PID=$! (log: logs/train_two_stage_2_real.log)"
