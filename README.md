# computer-vision-with-yolo

Two YOLO-based computer vision projects.

## ppe_detection/

Personal protective equipment (PPE) detection.

- `models/` — trained weights (`best.pt`, `best_gloves.pt`, `yolov8n.pt`)
- `scripts/` — training/inference/validation scripts
- `docs/` — setup notes and model datasheet
- `data/` — local-only datasets (gitignored, not pushed)

Setup:
```
cd ppe_detection
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## surveillance_suite/

Multi-module surveillance system: fire/smoke detection, fall detection,
virtual line crossing, license plate recognition, and door state detection.

- `models/` — trained weights
- `modules/` — detection modules used by `main.py`
- `detectors/` — standalone detector entry points (crowd density, abandoned object)
- `evaluation/` — evaluation scripts
- `training/` — training scripts
- `legacy/` — superseded entry points, kept for reference
- `main.py` — current entry point (threaded, frame-skipping, optimized)
- `data/` — local-only dataset (gitignored, not pushed)

Setup:
```
cd surveillance_suite
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python main.py
```
