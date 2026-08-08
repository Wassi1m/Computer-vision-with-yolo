#!/usr/bin/env python3
"""P7 — Ré-entraînement du détecteur de plaques sur la taxonomie unifiée (1 classe).

Part de `license_plate.pt` (yolo26n déjà entraîné sur ces images) : le backbone
connaît déjà le domaine, seule la tête de détection est réinitialisée pour
passer de 3 sorties redondantes à 1. Sur CPU c'est ce qui permet de converger
en quelques dizaines d'epochs plutôt qu'en centaines depuis les poids COCO.
"""

from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "surveillance_suite/data/dataset/license_plate_unified/data.yaml"
BASE = ROOT / "surveillance_suite/models/license_plate.pt"

if __name__ == "__main__":
    model = YOLO(str(BASE))
    model.train(
        data=str(DATA),
        epochs=80,
        imgsz=640,
        batch=16,
        device="cpu",
        workers=8,
        patience=20,
        project=str(ROOT / "reports/v2_results/runs"),
        name="p7_plate_unified",
        exist_ok=True,
        seed=0,
        plots=True,
        val=True,
    )
