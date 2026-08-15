#!/usr/bin/env python3
"""P1 — Mesure de `ppe_detector.pt` concept par concept, sur des sous-ensembles cohérents.

Généralisation du constat établi sur les gilets. Le jeu `ppe_dataset` étant un
patchwork de lots annotés chacun sur un seul concept, **toute** classe est
pénalisée par le même mécanisme : les images des autres lots contiennent l'objet
sans l'annoter, les détections correctes y sont comptées en faux positifs, et
l'AP publiée sous-estime le modèle. L'ampleur de la sous-estimation dépend du
poids relatif du lot de la classe : elle est massive pour les gilets (2 135
images sur 30 765), faible pour les casques (10 660 images).

Pour chaque concept, on construit donc un sous-ensemble de validation restreint
aux images qui annotent effectivement ce concept, labels 14 classes intacts, et
on y mesure le même modèle. La comparaison « AP publiée » vs « AP à périmètre
cohérent » quantifie la part de l'écart imputable au jeu de données.

Les sous-ensembles sont plafonnés (par défaut 1 200 images) pour tenir sur CPU ;
l'échantillonnage est aléatoire à graine fixe, donc reproductible.
"""

from pathlib import Path
import argparse
import json
import random
import shutil

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "ppe_detection/data/extracted/ppe_dataset"
WORK = ROOT / "ppe_detection/data/extracted/_eval_concepts"
OUT = ROOT / "reports/v2_results"

NAMES14 = ['Fall-Detected', 'Gloves', 'Goggles', 'Hardhat', 'Ladder', 'Mask', 'NO-Gloves',
           'NO-Goggles', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone', 'Safety Vest']

CONCEPTS = {
    "casque":   (3, 8),    # Hardhat / NO-Hardhat
    "gants":    (1, 6),    # Gloves / NO-Gloves
    "lunettes": (2, 7),    # Goggles / NO-Goggles
    "masque":   (5, 9),    # Mask / NO-Mask
    "gilet":    (13, 10),  # Safety Vest / NO-Safety Vest
}

# AP@50 mesurées en v1 (conf=0.25, split val de `ppe_dataset_subset`,
# stats_best_20260807_102532.json), reprises telles quelles comme référence.
# Ces images subissent le mélange de lots décrit plus haut.
AP50_PUBLIEES = {
    "Hardhat": 0.8551, "NO-Hardhat": 0.7179,
    "Gloves": 0.9148, "NO-Gloves": 0.8097,
    "Goggles": 0.9433, "NO-Goggles": 0.8988,
    "Mask": 0.5290, "NO-Mask": 0.6637,
    "Safety Vest": 0.4938, "NO-Safety Vest": 0.0482,
}


def construire_sous_ensemble(concept: str, ids: tuple[int, int], split: str,
                             plafond: int, graine: int = 0) -> tuple[Path, int]:
    dst = WORK / concept
    if dst.exists():
        shutil.rmtree(dst)
    (dst / split / "images").mkdir(parents=True, exist_ok=True)
    (dst / split / "labels").mkdir(parents=True, exist_ok=True)

    retenus = []
    for lbl in sorted((SRC / split / "labels").glob("*.txt")):
        rows = [l.split() for l in lbl.read_text().splitlines() if l.strip()]
        if any(int(r[0]) in ids for r in rows if len(r) >= 5):
            retenus.append((lbl, rows))

    random.Random(graine).shuffle(retenus)
    retenus = retenus[:plafond]

    for lbl, rows in retenus:
        img = next((SRC / split / "images").glob(lbl.stem + ".*"), None)
        if img is None:
            continue
        (dst / split / "images" / img.name).symlink_to(img.resolve())
        (dst / split / "labels" / lbl.name).write_text(
            "\n".join(" ".join(r[:5]) for r in rows if len(r) >= 5) + "\n")

    yaml = dst / "data.yaml"
    yaml.write_text(f"path: {dst}\ntrain: {split}/images\nval: {split}/images\ntest: {split}/images\n\n"
                    f"nc: 14\nnames: {NAMES14}\n")
    return yaml, len(retenus)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--plafond", type=int, default=1200)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(str(ROOT / "ppe_detection/models/ppe_detector.pt"))

    resultats = {}
    for concept, ids in CONCEPTS.items():
        yaml, n = construire_sous_ensemble(concept, ids, args.split, args.plafond)
        print(f"\n=== {concept} : {n} images de validation annotant ce concept")
        # batch réduit : la machine n'a que ~3 Go libres, un lot de 16 à 640 sur
        # YOLOv8m a suffi à faire tuer le ré-entraînement P7 par manque de mémoire.
        m = model.val(data=str(yaml), split="val", imgsz=args.imgsz, device="cpu", batch=4,
                      project=str(OUT / "runs"), name=f"concept_{concept}", exist_ok=True, plots=False)
        par_classe = {}
        for i, ci in enumerate(m.ap_class_index):
            nom = NAMES14[int(ci)]
            if int(ci) in ids:
                par_classe[nom] = {
                    "AP50_perimetre_coherent": round(float(m.box.ap50[i]), 4),
                    "AP50_publiee_val_complet": AP50_PUBLIEES.get(nom),
                    "precision": round(float(m.box.p[i]), 4),
                    "recall": round(float(m.box.r[i]), 4),
                }
        resultats[concept] = {"images": n, "classes": par_classe}
        print(json.dumps(par_classe, indent=2, ensure_ascii=False))

    dest = OUT / "p1_eval_par_concept.json"
    dest.write_text(json.dumps(resultats, indent=2, ensure_ascii=False))

    print("\n\n=== Synthese : AP@50 publiee -> AP@50 a perimetre coherent")
    print(f"{'classe':20}{'publiee':>10}{'coherente':>12}{'ecart':>10}")
    for concept, r in resultats.items():
        for nom, v in r["classes"].items():
            pub = v["AP50_publiee_val_complet"]
            coh = v["AP50_perimetre_coherent"]
            ecart = f"{coh - pub:+.3f}" if pub is not None else "—"
            print(f"{nom:20}{(f'{pub:.3f}' if pub else '—'):>10}{coh:>12.3f}{ecart:>10}")
    print(f"\n-> {dest}")


if __name__ == "__main__":
    main()
