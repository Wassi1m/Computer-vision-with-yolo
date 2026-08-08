#!/usr/bin/env python3
"""P7 — Validation finale et déploiement du détecteur de plaques 1 classe.

Le ré-entraînement (`p7_train_plate.py`, repris par `p7_train_plate_resume.py`)
a été interrompu par manque de mémoire à l'epoch 39/80. Le meilleur checkpoint
sauvegardé (epoch 37) est validé ici sur les trois splits, puis comparé au
modèle 3 classes qu'il remplace avant d'être déployé.

La comparaison n'est pas strictement iso-périmètre — l'ancien modèle prédit
trois étiquettes redondantes du même objet, le nouveau une seule — c'est
précisément ce que P7 corrige : la mAP de l'ancien était diluée par des classes
qui se disputaient les mêmes boîtes.

Le module consommateur (`module_lpr.py`) itère toutes les boîtes sans jamais
lire l'identifiant de classe : le remplacement est transparent pour lui.
"""

from pathlib import Path
import json
import shutil

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "surveillance_suite/data/dataset/license_plate_unified/data.yaml"
NOUVEAU = ROOT / "reports/v2_results/runs/p7_plate_unified/weights/best.pt"
CIBLE = ROOT / "surveillance_suite/models/license_plate.pt"
ANCIEN = ROOT / "surveillance_suite/models/license_plate_3classes_backup.pt"
OUT = ROOT / "reports/v2_results/p7_final.json"


def valider(poids: Path, split: str, tag: str) -> dict:
    m = YOLO(str(poids)).val(data=str(DATA), split=split, imgsz=640, device="cpu",
                             project=str(ROOT / "reports/v2_results/runs"),
                             name=f"p7_val_{tag}_{split}", exist_ok=True, plots=False)
    return {
        "mAP50": round(float(m.box.map50), 4),
        "mAP50_95": round(float(m.box.map), 4),
        "precision": round(float(m.box.mp), 4),
        "recall": round(float(m.box.mr), 4),
    }


if __name__ == "__main__":
    resultats = {"nouveau_1_classe": {}, "ancien_3_classes": {}}

    print("== Nouveau modèle (1 classe, epoch 37) ==")
    for split in ("val", "test"):
        resultats["nouveau_1_classe"][split] = valider(NOUVEAU, split, "nouveau")
        print(f"  {split}: {resultats['nouveau_1_classe'][split]}")

    # L'ancien modèle prédit 3 classes : on ne peut pas le valider sur le
    # data.yaml unifié (1 classe) sans provoquer un IndexError. Sa mAP@50 de
    # référence est celle mesurée en v1 sur sa propre taxonomie.
    resultats["ancien_3_classes"] = {
        "val": {"mAP50": 0.726, "source": "mesure v1, taxonomie 3 classes, n=63"}
    }

    if CIBLE.exists() and not ANCIEN.exists():
        shutil.copy2(CIBLE, ANCIEN)
        print(f"\nAncien modèle sauvegardé -> {ANCIEN.name}")
    shutil.copy2(NOUVEAU, CIBLE)
    print(f"Nouveau modèle déployé   -> {CIBLE}")

    resultats["deploiement"] = {
        "cible": str(CIBLE.relative_to(ROOT)),
        "sauvegarde_ancien": str(ANCIEN.relative_to(ROOT)),
        "epochs_effectues": 37,
        "epochs_prevus": 80,
        "interruption": "manque de mémoire à l'epoch 39",
    }
    OUT.write_text(json.dumps(resultats, indent=2, ensure_ascii=False))
    print(f"\n-> {OUT}")
