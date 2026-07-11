#!/usr/bin/env bash
set -euo pipefail

CODE=${CODE:-/root/autodl-fs/projects/grt}
EVENT_DIR=${1:?Usage: autodl_export_tb_images.sh <event_dir> [out_dir]}
OUT_DIR=${2:-/root/autodl-fs/outputs/grt_visuals/$(basename "${EVENT_DIR}")_$(date +%Y%m%d_%H%M%S)}

export PYTHONPATH="${CODE}:${PYTHONPATH:-}"
mkdir -p "${OUT_DIR}"
cd "${CODE}"

python -u scripts/export_tb_images.py \
  --event-dir "${EVENT_DIR}" \
  --out-dir "${OUT_DIR}"

echo "DONE_EXPORT_TB ${OUT_DIR}"
