#!/usr/bin/env python3
"""P1 étape 2 — Construction d'un jeu de données « propre » pour les classes gilet.

Ce que l'audit a établi (voir `reports/v2_results/p1_audit_annotations_gilet.json`
et `p1_sources_dataset.json`) : le jeu `ppe_dataset` est un patchwork de lots
annotés chacun sur un seul concept. La matrice de co-occurrence au niveau image
est bloc-diagonale — `Hardhat` et `Safety Vest` n'apparaissent **jamais**
ensemble sur les 30 765 images d'entraînement, alors qu'un chantier réunit
évidemment les deux. Vérification visuelle : dans le lot « casques », des
ouvriers en gilet haute visibilité bien visible n'ont que leur casque annoté.

Conséquence pour l'apprentissage : environ 14 300 images enseignent au modèle
que la zone « gilet » est du fond, contre seulement ~2 800 qui lui enseignent
que c'est un gilet. Le rapport ~5:1 de négatifs erronés explique l'effondrement
du rappel sur `NO-Safety Vest` (AP@50 = 4.8 %) bien mieux qu'un problème de
seuil, et il n'est corrigeable ni par un abaissement de seuil ni par un simple
sur-échantillonnage : les négatifs erronés seraient sur-échantillonnés aussi.

Le correctif retenu est de restreindre l'apprentissage des classes gilet aux
seules images où le gilet est effectivement annoté. Deux sorties :

  - `ppe_vest_clean_2c` : 2 classes (`Safety Vest`, `NO-Safety Vest`), pour
    entraîner un modèle spécialiste.
  - `ppe_vest_clean_14c` : mêmes images, labels 14 classes d'origine intacts,
    pour mesurer `best.pt` sur exactement le même sous-ensemble et obtenir une
    comparaison à périmètre identique.
"""

from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "ppe_detection/data/extracted/ppe_dataset"
DST2 = ROOT / "ppe_detection/data/extracted/ppe_vest_clean_2c"
DST14 = ROOT / "ppe_detection/data/extracted/ppe_vest_clean_14c"

NAMES14 = ['Fall-Detected', 'Gloves', 'Goggles', 'Hardhat', 'Ladder', 'Mask', 'NO-Gloves',
           'NO-Goggles', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone', 'Safety Vest']
VEST, NOVEST = 13, 10
REMAP = {VEST: 0, NOVEST: 1}          # 0 = Safety Vest, 1 = NO-Safety Vest
SPLITS = {"train": "train", "val": "val", "test": "test"}


def build():
    for d in (DST2, DST14):
        if d.exists():
            shutil.rmtree(d)

    totals = {}
    for split in SPLITS:
        for d in (DST2, DST14):
            (d / split / "images").mkdir(parents=True, exist_ok=True)
            (d / split / "labels").mkdir(parents=True, exist_ok=True)

        n_img = n_vest = n_novest = 0
        for lbl in sorted((SRC / split / "labels").glob("*.txt")):
            rows = [l.split() for l in lbl.read_text().splitlines() if l.strip()]
            rows = [r for r in rows if len(r) >= 5]
            vest_rows = [r for r in rows if int(r[0]) in REMAP]
            if not vest_rows:
                continue

            img = next((SRC / split / "images").glob(lbl.stem + ".*"), None)
            if img is None:
                continue

            for d, keep in ((DST2, vest_rows), (DST14, rows)):
                (d / split / "images" / img.name).symlink_to(img.resolve())
                if d is DST2:
                    lines = [" ".join([str(REMAP[int(r[0])])] + r[1:5]) for r in keep]
                else:
                    lines = [" ".join(r[:5]) for r in keep]
                (d / split / "labels" / lbl.name).write_text("\n".join(lines) + "\n")

            n_img += 1
            n_vest += sum(1 for r in vest_rows if int(r[0]) == VEST)
            n_novest += sum(1 for r in vest_rows if int(r[0]) == NOVEST)

        totals[split] = (n_img, n_vest, n_novest)
        print(f"{split:6} images={n_img:5}  Safety Vest={n_vest:5}  NO-Safety Vest={n_novest:5}")

    (DST2 / "data.yaml").write_text(
        f"path: {DST2}\ntrain: train/images\nval: val/images\ntest: test/images\n\n"
        "nc: 2\nnames: ['Safety Vest', 'NO-Safety Vest']\n")
    (DST14 / "data.yaml").write_text(
        f"path: {DST14}\ntrain: train/images\nval: val/images\ntest: test/images\n\n"
        f"nc: 14\nnames: {NAMES14}\n")
    print(f"\n-> {DST2}\n-> {DST14}")
    return totals


if __name__ == "__main__":
    build()
