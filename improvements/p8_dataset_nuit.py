#!/usr/bin/env python3
"""P8 — Construit un jeu d'entraînement enrichi de conditions dégradées.

Constat mesuré (`tests/mesure_robustesse.py`, résultats dans
`reports/v3_results/robustesse_conditions_reelles.json`) : tous les modèles
s'effondrent en faible luminosité — EPI 92.1 % → 15.4 % de mAP@50, chute
95.5 % → 49.2 %, feu/fumée 68.5 % → 34.1 %, plaques 86.4 % → 41.8 %. Les
plaques s'effondrent en plus sous flou de mouvement (−75 %), or lire la plaque
d'un véhicule *en mouvement* est le cas d'usage même de ce module.

La cause est simple : les jeux d'entraînement ne contiennent que des images
correctement exposées. Le modèle n'a jamais vu de scène nocturne, il ne peut
pas en traiter.

Ce script ajoute au jeu d'origine des copies dégradées des images
d'entraînement, avec les **mêmes annotations** — une boîte reste au même endroit
qu'on assombrisse l'image ou non. Le jeu de validation, lui, reste intact : le
mesurer sur des images dégradées fausserait la comparaison avec les métriques de
référence.

Les dégradations sont volontairement identiques à celles de
`tests/mesure_robustesse.py`. C'est ce qui rend le résultat interprétable : on
entraîne sur exactement ce que l'on mesure. Attention à la limite que cela pose
— une vraie scène de nuit n'est pas une image de jour assombrie (éclairage
artificiel ponctuel, bruit de capteur réel, balance des blancs différente). Ce
jeu améliore la tenue en basse lumière, il ne remplace pas des images
réellement nocturnes.

    python p8_dataset_nuit.py --source donnees/ppe_vest_clean_14c --proportion 0.6
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np


def faible_luminosite(img, rng):
    """Nuit / éclairage insuffisant : exposition fortement réduite + bruit capteur.

    L'intensité est tirée au hasard dans une plage plutôt que fixée : un modèle
    entraîné sur un unique niveau d'assombrissement apprendrait ce niveau, pas
    la pénombre en général.
    """
    alpha = rng.uniform(0.18, 0.40)
    sombre = cv2.convertScaleAbs(img, alpha=alpha, beta=rng.uniform(-15, 0))
    bruit = np.random.normal(0, rng.uniform(4, 12), sombre.shape)
    return np.clip(sombre.astype(np.int16) + bruit.astype(np.int16), 0, 255).astype(np.uint8)


def contre_jour(img, rng):
    """Sujet devant une source lumineuse : hautes lumières écrasées, sujet sombre."""
    h, w = img.shape[:2]
    halo = np.zeros((h, w), np.float32)
    cx = int(rng.uniform(0.2, 0.8) * w)
    cy = int(rng.uniform(0.0, 0.5) * h)
    cv2.circle(halo, (cx, cy), max(h, w) // 2, 1.0, -1)
    halo = cv2.GaussianBlur(halo, (0, 0), max(h, w) / 8)[..., None]
    return np.clip(img.astype(np.float32) * rng.uniform(0.45, 0.65)
                   + halo * rng.uniform(150, 220), 0, 255).astype(np.uint8)


def flou_mouvement(img, rng):
    """Sujet ou caméra en mouvement, temps de pose long (fréquent de nuit).

    Angle tiré au hasard : un flou toujours horizontal apprendrait au modèle une
    direction de mouvement particulière.
    """
    taille = int(rng.choice([9, 13, 17]))
    noyau = np.zeros((taille, taille), np.float32)
    noyau[taille // 2, :] = 1.0 / taille
    matrice = cv2.getRotationMatrix2D((taille / 2 - 0.5, taille / 2 - 0.5),
                                      rng.uniform(0, 180), 1.0)
    noyau = cv2.warpAffine(noyau, matrice, (taille, taille))
    somme = noyau.sum()
    return cv2.filter2D(img, -1, noyau / somme if somme > 0 else noyau)


def brouillard_pluie(img, rng):
    """Intempérie : contraste réduit, voile blanchâtre, léger flou."""
    voile = np.full_like(img, int(rng.uniform(180, 215)))
    poids = rng.uniform(0.30, 0.50)
    melange = cv2.addWeighted(img, 1 - poids, voile, poids, 0)
    return cv2.GaussianBlur(melange, (5, 5), 0)


DEGRADATIONS = {
    "nuit": faible_luminosite,
    "contrejour": contre_jour,
    "flou": flou_mouvement,
    "intemperie": brouillard_pluie,
}

# Repartition des degradations, proportionnelle a la perte mesuree : la faible
# luminosite est de loin le premier facteur d'echec, elle merite donc la plus
# grande part des images ajoutees.
POIDS_DEFAUT = {"nuit": 0.50, "contrejour": 0.20, "flou": 0.20, "intemperie": 0.10}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="jeu de donnees d'origine")
    ap.add_argument("--sortie", default=None, help="defaut : <source>_robuste")
    ap.add_argument("--proportion", type=float, default=0.6,
                    help="images degradees ajoutees, en part du train d'origine")
    ap.add_argument("--graine", type=int, default=0)
    args = ap.parse_args()

    source = Path(args.source).resolve()
    sortie = Path(args.sortie).resolve() if args.sortie else source.parent / f"{source.name}_robuste"
    if not source.is_dir():
        print(f"jeu introuvable : {source}", file=sys.stderr)
        return 1

    # Le nom du split d'entrainement varie selon l'origine du jeu (Roboflow
    # utilise "train", certains exports "images/train").
    dossier_train = next((source / n for n in ("train",) if (source / n / "images").is_dir()), None)
    if dossier_train is None:
        print(f"aucun split d'entrainement trouve dans {source}", file=sys.stderr)
        return 1

    if sortie.exists():
        shutil.rmtree(sortie)

    # Les splits autres que train sont repris a l'identique : degrader la
    # validation rendrait toute comparaison avec les metriques de reference
    # impossible.
    for split in ("train", "valid", "val", "test"):
        src = source / split
        if not (src / "images").is_dir():
            continue
        for sous in ("images", "labels"):
            (sortie / split / sous).mkdir(parents=True, exist_ok=True)
            for f in (src / sous).iterdir():
                if f.is_file() or f.is_symlink():
                    (sortie / split / sous / f.name).symlink_to(f.resolve())

    images = sorted(p for p in (dossier_train / "images").iterdir()
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not images:
        print("aucune image d'entrainement", file=sys.stderr)
        return 1

    rng = random.Random(args.graine)
    np.random.seed(args.graine)
    n_ajout = int(len(images) * args.proportion)
    print(f"{len(images)} images d'entrainement -> {n_ajout} copies degradees ajoutees")

    noms = list(DEGRADATIONS)
    poids = [POIDS_DEFAUT[n] for n in noms]
    compteur = {n: 0 for n in noms}
    ignorees = 0

    for i in range(n_ajout):
        chemin = images[rng.randrange(len(images))]
        etiquette = dossier_train / "labels" / f"{chemin.stem}.txt"
        if not etiquette.exists():
            ignorees += 1
            continue
        img = cv2.imread(str(chemin))
        if img is None:
            ignorees += 1
            continue

        nom_deg = rng.choices(noms, weights=poids, k=1)[0]
        degradee = DEGRADATIONS[nom_deg](img, rng)
        compteur[nom_deg] += 1

        base = f"{nom_deg}_{i:06d}_{chemin.stem}"
        cv2.imwrite(str(sortie / "train" / "images" / f"{base}.jpg"), degradee)
        # Les annotations sont recopiees telles quelles : une degradation
        # photometrique ou un flou ne deplacent aucune boite.
        shutil.copy2(etiquette, sortie / "train" / "labels" / f"{base}.txt")

        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{n_ajout}...", flush=True)

    yaml_src = (source / "data.yaml").read_text(encoding="utf-8")
    ligne = lambda p: next(l for l in yaml_src.splitlines() if l.startswith(p))  # noqa: E731
    split_val = "valid" if (sortie / "valid" / "images").is_dir() else "val"
    contenu = [f"path: {sortie}", "train: train/images", f"val: {split_val}/images"]
    if (sortie / "test" / "images").is_dir():
        contenu.append("test: test/images")
    contenu += ["", ligne("nc:"), ligne("names:")]
    (sortie / "data.yaml").write_text("\n".join(contenu) + "\n", encoding="utf-8")

    print("\nRepartition des degradations ajoutees :")
    for n, c in compteur.items():
        print(f"  {n:12} {c:6d}")
    if ignorees:
        print(f"  ({ignorees} images ignorees : annotation absente ou illisible)")
    total = len(images) + sum(compteur.values())
    print(f"\nTrain : {len(images)} -> {total} images")
    print(f"-> {sortie}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
