#!/usr/bin/env python3
"""P5 — Fusion du dataset chute local avec un complément téléchargé sur Hugging Face.

Source locale (`data_set.zip`, jamais extrait avant cette session) : 511 images
Roboflow (`fall detection 2.v2i.yolov8`), 2 classes `falling`/`stand`, déjà au
bon format (358 train / 102 val / 51 test).

Complément (`DeZan/fall-detection`, Hugging Face, licence MIT, téléchargement
libre sans clé) : 485 images (374 train / 111 val), mais avec 3 identifiants de
classe dans les labels (0/1/2) alors qu'aucune documentation n'accompagne le
dataset. Vérifié empiriquement (comptage par préfixe de nom de fichier) :
classe 0 domine très largement les fichiers `fall*.txt` (277/296 instances) et
les classes 1/2 dominent les fichiers `not fallen*.txt` (250/258 instances) --
0 = chute, {1, 2} = deux activités "pas chute" (vraisemblablement marche/assis,
cf. Kaggle "Fall Detected/Walking/Sitting" pour un dataset très similaire). On
remappe donc 0->0 (falling) et {1,2}->1 (stand), cohérent avec le schéma binaire
du dataset local.

Pas de split test dans DeZan : son train/val rejoint le train/val local, le
test local reste intact et non enrichi (mesure comparable au baseline).
"""

from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / "ppe_detection/data/extracted/fall_detection"
DEZAN = ROOT / "ppe_detection/data/extracted/dezan_fall_raw"
DST = ROOT / "ppe_detection/data/extracted/fall_detection_enriched"

SPLITS = {"train": "train", "val": "valid"}  # split DeZan -> split local
DEZAN_REMAP = {"0": "0", "1": "1", "2": "1"}  # falling reste 0, {marche,assis} -> stand (1)


def copier_local(split: str):
    for sous in ("images", "labels"):
        src = LOCAL / split / sous
        dst = DST / split / sous
        dst.mkdir(parents=True, exist_ok=True)
        for f in src.iterdir():
            (dst / f.name).symlink_to(f.resolve())


def ajouter_dezan(split_dezan: str, split_local: str) -> int:
    n = 0
    for img in (DEZAN / "images" / split_dezan).iterdir():
        lbl = DEZAN / "labels" / split_dezan / img.with_suffix(".txt").name
        if not lbl.exists():
            continue
        rows = [l.split() for l in lbl.read_text().splitlines() if l.strip()]
        remap = [" ".join([DEZAN_REMAP[r[0]]] + r[1:5]) for r in rows if r[0] in DEZAN_REMAP and len(r) >= 5]
        if not remap:
            continue
        nom = f"dezan_{img.stem}".replace(" ", "_")
        (DST / split_local / "images" / f"{nom}{img.suffix}").symlink_to(img.resolve())
        (DST / split_local / "labels" / f"{nom}.txt").write_text("\n".join(remap) + "\n")
        n += 1
    return n


def main():
    if DST.exists():
        shutil.rmtree(DST)

    for sous in ("images", "labels"):
        dst = DST / "test" / sous
        dst.mkdir(parents=True, exist_ok=True)
        for f in (LOCAL / "test" / sous).iterdir():
            (dst / f.name).symlink_to(f.resolve())

    for split_dezan, split_local in SPLITS.items():
        copier_local(split_local)
        n = ajouter_dezan(split_dezan, split_local)
        n_local = len(list((LOCAL / split_local / "images").iterdir()))
        print(f"{split_local:6} : {n_local} images locales + {n} images DeZan ajoutées")

    (DST / "data.yaml").write_text(
        f"path: {DST}\ntrain: train/images\nval: valid/images\ntest: test/images\n\nnc: 2\nnames: ['falling', 'stand']\n")
    print(f"\n-> {DST}")


if __name__ == "__main__":
    main()
