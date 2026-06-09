#!/usr/bin/env bash
#
#   analyze    Task 1 — Data analysis pipeline      -> python -m src.data.pipeline
#   dashboard  Task 1 — Interactive Dash dashboard  -> python -m src.data.dashboard  (port 8050)
#   visualize  Task 1 — Interesting-sample grids    -> python -m src.evaluation.visualize_samples
#   dataloader Task 2 — PyTorch DataLoader demo     -> python -m src.model.data_loader
#   detect     Task 2 — RT-DETRv2 inference         -> python -m src.model.detector
#   train      Task 2 — Fine-tune RT-DETRv2         -> python train.py
#   evaluate   Task 3 — Evaluate + failure analysis -> python -m src.evaluation.evaluate
#   compare    Task 3 — Multi-model comparison      -> python -m src.evaluation.compare_models
#   test       Run the unit-test suite               -> python -m pytest tests/
#   bash|sh    Drop into a shell

set -euo pipefail

DATA_DIR="${DATA_DIR:-/app/data}"
OUTPUT_DIR="${OUTPUT_DIR:-/app/output}"

_ensure_rtdetrv2_assets() {
    local vendor_dir="/app/third_party/RT-DETR/rtdetrv2_pytorch"
    local need_setup=false

    if [ ! -d "$vendor_dir" ]; then
        echo "  [setup] RT-DETRv2 source not found at third_party/"
        need_setup=true
    fi

    if ! ls /app/weights/*.pth 2>/dev/null | grep -q .; then
        echo "  [setup] No pretrained weights found in weights/"
        need_setup=true
    fi

    if [ "$need_setup" = "true" ]; then
        echo ""
        echo "======================================================="
        echo "  RT-DETRv2 assets missing — fetching from GitHub now."
        echo "  (Mount third_party/ and weights/ as host volumes so this"
        echo "   download only happens once.)"
        echo "============================================================"
        echo ""
        bash /app/scripts/setup_rtdetrv2.sh
        echo ""
    fi
}

CMD="${1:-analyze}"
shift || true

case "$CMD" in
  analyze)
    exec python -m src.data.pipeline "$DATA_DIR" "$OUTPUT_DIR" "$@"
    ;;
  dashboard)
    echo ""
    echo "======================================================"
    echo "  BDD100K Dataset Analysis Dashboard"
    echo "  Loading annotations ... (may take ~60s)"
    echo ""
    echo "  Open http://localhost:8050 in your browser."
    echo "============================================================"
    echo ""
    exec python -m src.data.dashboard "$DATA_DIR" "$@"
    ;;
  visualize)
    exec python -m src.evaluation.visualize_samples "$@"
    ;;
  dataloader)
    exec python -m src.model.data_loader "$@"
    ;;
  detect)
    _ensure_rtdetrv2_assets
    exec python -m src.model.detector "$@"
    ;;
  train)
    _ensure_rtdetrv2_assets
    echo ""
    echo "======================================================"
    echo "  RT-DETRv2 Fine-tuning on BDD100K"
    echo "  Smoke test: ... train --device cpu --epochs 1 --batch 2 --train-images 200"
    echo "  Full run:   ... train --device cuda --epochs 30 --batch 16"
    echo "  Checkpoints -> output/rtdetrv2_bdd100k/"
    echo "=========================================================="
    echo ""
    exec python train.py "$@"
    ;;
  evaluate)
    _ensure_rtdetrv2_assets
    echo ""
    echo "====================================================="
    echo "  RT-DETRv2 Evaluation on BDD100K Validation Set"
    echo "  Running inference on 500 images — results -> output/evaluation/"
    echo "============================================================"
    echo ""
    exec python -m src.evaluation.evaluate "$@"
    ;;
  compare)
    _ensure_rtdetrv2_assets
    echo ""
    echo "====================================================="
    echo "  Multi-Model Comparison: RT-DETRv2 vs YOLOv8"
    echo "  results -> output/comparison/"
    echo "========================================================="
    echo ""
    exec python -m src.evaluation.compare_models "$@"
    ;;
  test)
    exec python -m pytest tests/ "$@"
    ;;
  bash|sh)
    exec /bin/bash "$@"
    ;;
  *)
    exec "$CMD" "$@"
    ;;
esac
