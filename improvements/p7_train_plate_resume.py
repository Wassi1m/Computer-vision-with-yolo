#!/usr/bin/env python3
"""Reprise de l'entraînement P7 (license_plate unifié) depuis le dernier checkpoint."""

from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
LAST = ROOT / "reports/v2_results/runs/p7_plate_unified/weights/last.pt"

if __name__ == "__main__":
    model = YOLO(str(LAST))
    model.train(resume=True)
