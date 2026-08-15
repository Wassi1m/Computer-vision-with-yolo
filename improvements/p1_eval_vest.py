#!/usr/bin/env python3
"""P1 — Mesure des classes gilet à périmètre identique.

`ppe_detector.pt` est évalué sur `ppe_vest_clean_14c` (les images du sous-ensemble
propre, labels 14 classes intacts) et le spécialiste sur `ppe_vest_clean_2c`
(les mêmes images, 2 classes). Les deux voient donc exactement les mêmes images
et la même vérité terrain pour les gilets : les AP@50 sont directement
comparables, ce qui n'était pas le cas d'une mesure sur le val complet.

Usage :
    p1_eval_vest.py baseline           # best.pt sur le sous-ensemble propre
    p1_eval_vest.py specialist <poids> # modèle 2 classes
"""

from pathlib import Path
import json
import sys
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/v2_results"


def run(weights: Path, data: Path, tag: str, split: str = "val", classes_of_interest=None):
    model = YOLO(str(weights))
    m = model.val(data=str(data), split=split, imgsz=640, device="cpu",
                  project=str(OUT / "runs"), name=f"eval_{tag}_{split}", exist_ok=True, plots=False)

    names = model.names
    per_class = {}
    for i, ci in enumerate(m.ap_class_index):
        per_class[names[int(ci)]] = {
            "AP50": round(float(m.box.ap50[i]), 4),
            "AP50_95": round(float(m.box.ap[i]), 4),
            "precision": round(float(m.box.p[i]), 4),
            "recall": round(float(m.box.r[i]), 4),
        }

    res = {
        "tag": tag,
        "weights": str(weights),
        "data": str(data),
        "split": split,
        "mAP50_global": round(float(m.box.map50), 4),
        "mAP50_95_global": round(float(m.box.map), 4),
        "per_class": per_class,
    }
    if classes_of_interest:
        res["classes_suivies"] = {c: per_class.get(c) for c in classes_of_interest}

    dest = OUT / f"p1_eval_{tag}_{split}.json"
    dest.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"\n-> {dest}")
    return res


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    split = sys.argv[3] if len(sys.argv) > 3 else "val"
    if mode == "baseline":
        run(ROOT / "ppe_detection/models/ppe_detector.pt",
            ROOT / "ppe_detection/data/extracted/ppe_vest_clean_14c/data.yaml",
            "baseline_best_sur_sousensemble_propre", split,
            classes_of_interest=["Safety Vest", "NO-Safety Vest"])
    else:
        run(Path(sys.argv[2]),
            ROOT / "ppe_detection/data/extracted/ppe_vest_clean_2c/data.yaml",
            "specialiste_gilet", split,
            classes_of_interest=["Safety Vest", "NO-Safety Vest"])
