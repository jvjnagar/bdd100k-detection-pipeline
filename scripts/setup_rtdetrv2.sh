#!/usr/bin/env bash
# Fetch RT-DETRv2 source and pretrained weights from GitHub.
# Usage: scripts/setup_rtdetrv2.sh [rtdetrv2-s|rtdetrv2-m|rtdetrv2-l|rtdetrv2-x]
set -euo pipefail

REPO_URL="https://github.com/lyuwenyu/RT-DETR.git"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR_DIR="${PROJECT_ROOT}/third_party/RT-DETR"
WEIGHTS_DIR="${PROJECT_ROOT}/weights"

MODEL="${1:-rtdetrv2-s}"

case "$MODEL" in
  rtdetrv2-s)   SPEC="configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml|v0.2|rtdetrv2_r18vd_120e_coco_rerun_48.1.pth" ;;
  rtdetrv2-m-r34) SPEC="configs/rtdetrv2/rtdetrv2_r34vd_120e_coco.yml|v0.1|rtdetrv2_r34vd_120e_coco_ema.pth" ;;
  rtdetrv2-m)   SPEC="configs/rtdetrv2/rtdetrv2_r50vd_m_7x_coco.yml|v0.1|rtdetrv2_r50vd_m_7x_coco_ema.pth" ;;
  rtdetrv2-l)   SPEC="configs/rtdetrv2/rtdetrv2_r50vd_6x_coco.yml|v0.1|rtdetrv2_r50vd_6x_coco_ema.pth" ;;
  rtdetrv2-x)   SPEC="configs/rtdetrv2/rtdetrv2_r101vd_6x_coco.yml|v0.1|rtdetrv2_r101vd_6x_coco_from_paddle.pth" ;;
  *) echo "ERROR: unknown model '$MODEL'. Choose: rtdetrv2-s|rtdetrv2-m-r34|rtdetrv2-m|rtdetrv2-l|rtdetrv2-x" >&2; exit 2 ;;
esac

CONFIG_REL="${SPEC%%|*}"; REST="${SPEC#*|}"; TAG="${REST%%|*}"; WEIGHT_FILE="${REST##*|}"
WEIGHT_URL="https://github.com/lyuwenyu/storage/releases/download/${TAG}/${WEIGHT_FILE}"

echo "[setup] model: $MODEL  checkpoint: ${WEIGHT_FILE}"

# --- 1. Source ---
if [ -d "${VENDOR_DIR}/rtdetrv2_pytorch/src/core" ]; then
  echo "[setup] source already present, skipping."
elif command -v git &>/dev/null; then
  rm -rf "${VENDOR_DIR}"
  mkdir -p "$(dirname "${VENDOR_DIR}")"
  git clone --depth 1 --filter=blob:none --sparse "${REPO_URL}" "${VENDOR_DIR}"
  ( cd "${VENDOR_DIR}" && git sparse-checkout set rtdetrv2_pytorch )
  echo "[setup] source ready."
else
  echo "[setup] downloading source tarball via Python ..."
  TMP_TAR="/tmp/rtdetr_src.tar.gz"
  _URL="https://github.com/lyuwenyu/RT-DETR/archive/refs/heads/main.tar.gz" \
  _DEST="${TMP_TAR}" python3 -c "
import urllib.request, ssl, os
ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
urllib.request.urlretrieve(os.environ['_URL'], os.environ['_DEST'])
"
  rm -rf "${VENDOR_DIR}"; mkdir -p "${VENDOR_DIR}"
  tar xz -f "${TMP_TAR}" -C "${VENDOR_DIR}" --strip-components=1 --wildcards "RT-DETR-main/rtdetrv2_pytorch"
  rm -f "${TMP_TAR}"
  echo "[setup] source ready."
fi

if [ ! -f "${VENDOR_DIR}/rtdetrv2_pytorch/${CONFIG_REL}" ]; then
  echo "ERROR: config not found after setup: ${VENDOR_DIR}/rtdetrv2_pytorch/${CONFIG_REL}" >&2; exit 6
fi

# --- 2. Weights ---
mkdir -p "${WEIGHTS_DIR}"
WEIGHT_PATH="${WEIGHTS_DIR}/${WEIGHT_FILE}"
if [ -s "${WEIGHT_PATH}" ]; then
  echo "[setup] weights already present, skipping."
elif command -v curl &>/dev/null; then
  curl -fL --retry 3 -o "${WEIGHT_PATH}" "${WEIGHT_URL}"
  echo "[setup] weights saved ($(du -h "${WEIGHT_PATH}" | cut -f1))."
else
  echo "[setup] downloading weights via Python ..."
  _URL="${WEIGHT_URL}" _DEST="${WEIGHT_PATH}" python3 -c "
import urllib.request, ssl, os
ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
urllib.request.urlretrieve(os.environ['_URL'], os.environ['_DEST'])
"
  echo "[setup] weights saved ($(du -h "${WEIGHT_PATH}" | cut -f1))."
fi

echo "[setup] done."
