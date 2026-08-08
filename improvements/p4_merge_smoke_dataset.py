#!/usr/bin/env python3
"""P4 — Fusion du dataset fire/smoke avec le complément pyro-sdis.

`Fire -smoke Detection.v1i.yolo26` (12 127 images) a une classe `smoke` sous-
représentée : 1 947 instances train / 97 val contre 9 923 / 1 021 pour `fire`
(AP@50 27.6% vs 65.2%, cf. baseline_report.md). `p4_download_smoke.py` a déjà
récupéré 4 000 images train + 800 val de fumée depuis pyronear/pyro-sdis
(surveillance_suite/data/dataset/pyro_sdis_subset), annotées 1 seule classe
`smoke`.

Ce script fusionne les deux : labels `fire` (classe 0) intacts, labels `smoke`
(classe 1) réunis. Les images pyro-sdis sont renommées avec un préfixe pour
éviter toute collision de nom avec le dataset d'origine.
"""

from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "surveillance_suite/data/dataset/Fire -smoke Detection.v1i.yolo26"
PYRO = ROOT / "surveillance_suite/data/dataset/pyro_sdis_subset"
DST = ROOT / "surveillance_suite/data/dataset/fire_smoke_enriched"

# mapping split pyro -> split base (le dataset de base utilise "valid", pas "val")
SPLITS = {"train": "train", "valid": "valid"}
# id de classe smoke réellement utilisé dans les labels pyro-sdis (vérifié : uniquement "1"
# dans les fichiers .txt, malgré un data.yaml généré qui annonce nc=1 ['smoke'] à tort)
# -> id classe smoke dans le dataset fusionné (nc=2, ['fire','smoke'])
PYRO_SMOKE_ID, FUSION_SMOKE_ID = 1, 1


def copier_base(split_base: str):
    for sous in ("images", "labels"):
        src = BASE / split_base / sous
        dst = DST / split_base / sous
        dst.mkdir(parents=True, exist_ok=True)
        for f in src.iterdir():
            (dst / f.name).symlink_to(f.resolve())


def ajouter_pyro(split_pyro: str, split_base: str):
    n = 0
    for img in (PYRO / split_pyro / "images").iterdir():
        lbl = (PYRO / split_pyro / "labels" / img.with_suffix(".txt").name)
        if not lbl.exists():
            continue
        rows = [l.split() for l in lbl.read_text().splitlines() if l.strip()]
        remap = [" ".join([str(FUSION_SMOKE_ID)] + r[1:5]) for r in rows if int(r[0]) == PYRO_SMOKE_ID and len(r) >= 5]
        if not remap:
            continue
        nom = f"pyro_{img.stem}"
        (DST / split_base / "images" / f"{nom}{img.suffix}").symlink_to(img.resolve())
        (DST / split_base / "labels" / f"{nom}.txt").write_text("\n".join(remap) + "\n")
        n += 1
    return n


def main():
    if DST.exists():
        shutil.rmtree(DST)

    # test : conservé identique à l'origine, jamais enrichi (mesure comparable au baseline)
    for sous in ("images", "labels"):
        dst = DST / "test" / sous
        dst.mkdir(parents=True, exist_ok=True)
        for f in (BASE / "test" / sous).iterdir():
            (dst / f.name).symlink_to(f.resolve())

    for split_pyro, split_base in SPLITS.items():
        copier_base(split_base)
        n = ajouter_pyro(split_pyro, split_base)
        n_base = len(list((BASE / split_base / "images").iterdir()))
        print(f"{split_base:6} : {n_base} images base + {n} images pyro-sdis smoke ajoutées")

    (DST / "data.yaml").write_text(
        f"path: {DST}\ntrain: train/images\nval: valid/images\ntest: test/images\n\nnc: 2\nnames: ['fire', 'smoke']\n")
    print(f"\n-> {DST}")


if __name__ == "__main__":
    main()
