#!/usr/bin/env python3
"""P7 — Nettoyage de la taxonomie du dataset `License Plate Detector`.

Le dataset d'origine porte trois étiquettes pour un seul concept :
  - `licence`      (333 instances)  -> plaque
  - `number_plate` (312 instances)  -> plaque, strict synonyme de `licence`
  - `num_plate`    ( 18 instances)  -> plaque également

Vérification faite avant fusion (voir rapport v2) : les 18 boîtes `num_plate`
proviennent toutes de la série d'images « N*.jpeg », sont seules dans leur image
(aucun recouvrement avec une autre boîte, donc pas des doublons d'annotation) et
recadrées elles montrent bien des plaques d'immatriculation lisibles. Elles sont
donc **ré-attribuées** à la classe unifiée plutôt que supprimées.

Le script produit un dataset à 1 classe (`plate`) : les images sont liées en
symlink (pas de copie, ~7 Mo économisés et une seule source de vérité), seuls
les fichiers de labels sont réécrits.
"""

from pathlib import Path
import argparse
import shutil

SRC = Path(__file__).resolve().parents[1] / "surveillance_suite/data/dataset/License Plate Detector.v5i.yolo26_resplit"
DST = Path(__file__).resolve().parents[1] / "surveillance_suite/data/dataset/license_plate_unified"
SPLITS = ["train", "valid", "test"]


def build(src: Path = SRC, dst: Path = DST) -> Path:
    if dst.exists():
        shutil.rmtree(dst)

    stats = {}
    for split in SPLITS:
        (dst / split / "images").mkdir(parents=True, exist_ok=True)
        (dst / split / "labels").mkdir(parents=True, exist_ok=True)

        n_img = n_box = 0
        for img in sorted((src / split / "images").iterdir()):
            link = dst / split / "images" / img.name
            link.symlink_to(img.resolve())
            n_img += 1

        for lbl in sorted((src / split / "labels").glob("*.txt")):
            out_lines = []
            for line in lbl.read_text().splitlines():
                if not line.strip():
                    continue
                parts = line.split()
                # les 3 classes d'origine désignent le même objet -> classe 0
                out_lines.append(" ".join(["0"] + parts[1:5]))
                n_box += 1
            (dst / split / "labels" / lbl.name).write_text("\n".join(out_lines) + "\n")

        stats[split] = (n_img, n_box)

    yaml = dst / "data.yaml"
    yaml.write_text(
        f"path: {dst}\n"
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n\n"
        "nc: 1\n"
        "names: ['plate']\n"
    )

    for split, (i, b) in stats.items():
        print(f"{split:6} images={i:4}  boites={b:4}")
    print(f"\ndata.yaml -> {yaml}")
    return yaml


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--dst", type=Path, default=DST)
    build(ap.parse_args().src, ap.parse_args().dst)
