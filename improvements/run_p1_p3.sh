#!/usr/bin/env bash
# Enchaîne P1 (AP par concept à périmètre cohérent) puis P3 (banc de vitesse).
# Séquentiel et non parallèle : la machine n'a que ~3 Go de RAM libres, et deux
# validations YOLOv8m simultanées la feraient tomber en OOM comme le
# ré-entraînement P7 avant elles.
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate

echo "===== P1 : AP par concept, perimetre coherent ====="
python improvements/p1_eval_par_concept.py --split val --plafond 600 --imgsz 640 \
  > reports/v2_results/p1_eval_par_concept.log 2>&1
echo "P1 termine (code $?)"

echo "===== P3 : banc vitesse + precision vs resolution ====="
python improvements/p3_benchmark_pipeline.py --images 15 --tailles 640,480 --valider-precision \
  > reports/v2_results/p3_benchmark.log 2>&1
echo "P3 termine (code $?)"

echo "===== TOUT TERMINE ====="
