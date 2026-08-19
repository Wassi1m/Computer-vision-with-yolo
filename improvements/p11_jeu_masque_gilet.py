#!/usr/bin/env python3
"""Construit un jeu dédié Mask/NO-Mask/Safety Vest/NO-Safety Vest, sans le bruit du patchwork.

Pourquoi
--------
`p1_eval_par_concept.py` l'a déjà mesuré : sur un sous-ensemble qui annote
réellement le masque et le gilet, `ppe_detector.pt` atteint 0.96/0.92 d'AP50
sur Mask/NO-Mask et 0.89/0.67 sur Safety Vest/NO-Safety Vest -- contre
0.53/0.66 et 0.49/0.05 publiés sur `ppe_dataset` complet. L'écart n'est pas
une faiblesse du modèle : `ppe_dataset` est un patchwork où la plupart des
~29 000 autres images n'annotent ni le masque ni le gilet. Chaque détection
correcte du modèle y est comptée comme un faux positif, ce qui punit le
signal exactement là où on veut le renforcer.

La duplication tentée le 2026-08-16 (`entrainer_epi.py --dupliquer 4`) n'a pas
marché pour cette raison précise : même x4, les images concernées ne
représentaient que 27 % du jeu final -- la majorité contradictoire est restée
écrasante (voir `reports/v3_results/epi_14c_candidat_20260816.json`).

Ce script prend le problème à l'envers : au lieu de pousser le signal correct
au milieu du bruit, il RETIRE le bruit. Un modèle dédié à 4 classes, entraîné
uniquement sur les images qui annotent réellement masque et/ou gilet, n'a
techniquement AUCUNE image contradictoire à apprendre -- et rien d'autre à
oublier, contrairement au modèle à 14 classes.

Sécurité
--------
Ce jeu sert à entraîner un modèle SÉPARÉ (`masque_gilet.pt`), pas à modifier
`ppe_detector.pt`. Aucun fichier de production n'est touché par ce script.

Le split `test` de `ppe_dataset` n'est JAMAIS copié ici : il reste local et
intact, pour juger le candidat sans qu'il ait pu le voir -- même règle que
`p10_sous_ensemble_epi.py`.

    python improvements/p11_jeu_masque_gilet.py --simulation
    python improvements/p11_jeu_masque_gilet.py
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
SOURCE = RACINE / "ppe_detection/data/extracted/ppe_dataset"
DEST = RACINE / "ppe_detection/data/extracted/masque_gilet_coherent"

# classe ppe_dataset (14 classes) -> classe du nouveau jeu (4 classes)
REMAPPAGE = {5: 0, 9: 1, 13: 2, 10: 3}
NOMS4 = ["Mask", "NO-Mask", "Safety Vest", "NO-Safety Vest"]


def construire(split: str, simulation: bool) -> tuple[int, dict[int, int]]:
    labels_src = SOURCE / split / "labels"
    images_src = SOURCE / split / "images"
    instances = {i: 0 for i in REMAPPAGE.values()}
    n_images = 0

    if not simulation:
        (DEST / split / "images").mkdir(parents=True, exist_ok=True)
        (DEST / split / "labels").mkdir(parents=True, exist_ok=True)

    for lbl in sorted(labels_src.glob("*.txt")):
        lignes = [l.split() for l in lbl.read_text(encoding="utf-8").splitlines() if l.strip()]
        gardees = [(REMAPPAGE[int(r[0])], r[1:5]) for r in lignes
                   if len(r) >= 5 and int(r[0]) in REMAPPAGE]
        if not gardees:
            continue
        n_images += 1
        for c, _ in gardees:
            instances[c] += 1
        if simulation:
            continue

        img = next(images_src.glob(lbl.stem + ".*"), None)
        if img is None:
            continue
        shutil.copy2(img, DEST / split / "images" / img.name)
        (DEST / split / "labels" / f"{lbl.stem}.txt").write_text(
            "\n".join(f"{c} {' '.join(coords)}" for c, coords in gardees) + "\n",
            encoding="utf-8")

    return n_images, instances


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--simulation", action="store_true",
                    help="compte les images/instances sans rien copier")
    args = ap.parse_args()

    if not args.simulation:
        if DEST.exists():
            print(f"deja present : {DEST} -- suppression avant reconstruction")
            shutil.rmtree(DEST)

    total_images = 0
    for split in ("train", "val"):
        n, instances = construire(split, args.simulation)
        total_images += n
        print(f"{split:<6} : {n:>5} images  " +
              "  ".join(f"{NOMS4[c]}={v}" for c, v in sorted(instances.items())))

    # `test` n'est JAMAIS copié : mesuré seulement, pour juger le candidat sans
    # qu'il l'ait vu.
    n_test, instances_test = construire("test", simulation=True)
    print(f"test   : {n_test:>5} images (NON copiees, reste local pour le jugement) " +
          "  ".join(f"{NOMS4[c]}={v}" for c, v in sorted(instances_test.items())))

    if args.simulation:
        print(f"\nTotal train+val a copier : {total_images} images")
        print("Relancer sans --simulation pour construire le jeu.")
        return 0

    (DEST / "data.yaml").write_text(
        f"path: {DEST}\ntrain: train/images\nval: val/images\n\n"
        f"nc: {len(NOMS4)}\nnames: {NOMS4}\n", encoding="utf-8")

    print(f"\n-> {DEST}  ({total_images} images train+val)")
    print("Le split test reste local, non copie : voir la ligne 'test' ci-dessus.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
