#!/usr/bin/env python3
"""Mesure ce qui compte en exploitation, et non la precision des boites.

Le projet pilote ses modeles sur la mAP@50. C'est la metrique standard en
detection, mais elle repond a la question « la boite est-elle au bon endroit ? »
alors que l'exploitation pose une tout autre question : « y a-t-il un depart de
feu devant cette camera, oui ou non ? ».

L'ecart entre les deux est enorme pour la fumee. Un panache n'a pas de contour
net -- deux annotateurs humains ne dessinent pas la meme boite autour du meme
nuage. La mAP@50 exige un recouvrement d'au moins 50 % avec une verite terrain
elle-meme incertaine : elle sanctionne donc lourdement une imprecision qui n'a
aucune consequence operationnelle. Pour un objet rigide (casque, plaque,
vehicule) ce probleme ne se pose pas, ce qui explique que ces modeles depassent
facilement 88 % la ou la fumee plafonne.

Ce script mesure donc, au niveau de l'image :

- **taux de detection** (rappel) : proportion de scenes contenant reellement
  l'objet ou le modele l'a signale. C'est le « on n'a pas rate le feu ».
- **taux de fausse alarme** : proportion de scenes sans objet ou le modele en
  signale un. C'est le « on ne reveille pas l'operateur pour rien ».

Ces deux chiffres, eux, se traduisent directement en exigence client, et se
lisent en fonction du seuil de confiance -- ce qui permet de choisir le
compromis plutot que de le subir.

    python tests/mesure_operationnelle.py --modele surveillance_suite/models/fire_smoke.pt \\
        --donnees surveillance_suite/data/dataset/fire_smoke_v9 --classes-fumee 1 2
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]


def verite_terrain(dossier: Path, classes: set[int]) -> dict[str, bool]:
    """Pour chaque image : la scene contient-elle au moins un objet vise ?"""
    presence = {}
    for lbl in (dossier / "labels").glob("*.txt"):
        trouve = False
        for ligne in lbl.read_text(encoding="utf-8").splitlines():
            parts = ligne.split()
            if parts and int(parts[0]) in classes:
                trouve = True
                break
        presence[lbl.stem] = trouve
    return presence


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modele", required=True)
    ap.add_argument("--donnees", required=True, help="racine du jeu (contient test/)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--classes-fumee", type=int, nargs="+", default=[1],
                    help="indices de classe consideres comme 'fumee' dans la verite terrain")
    ap.add_argument("--classes-modele", type=int, nargs="+", default=None,
                    help="indices cote modele (defaut : identiques a la verite terrain)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--degradation", choices=["aucune", "nuit"], default="aucune",
                    help="applique une degradation avant inference, pour mesurer "
                         "le seuil optimal dans cette condition")
    ap.add_argument("--images", type=int, default=0, help="0 = toutes")
    ap.add_argument("--graine", type=int, default=7)
    ap.add_argument("--sortie", default=None)
    args = ap.parse_args()

    from ultralytics import YOLO

    dossier = Path(args.donnees)
    if not (dossier / args.split).is_dir():
        print(f"split absent : {dossier / args.split}", file=sys.stderr)
        return 1

    cls_vt = set(args.classes_fumee)
    cls_modele = set(args.classes_modele if args.classes_modele else args.classes_fumee)

    presence = verite_terrain(dossier / args.split, cls_vt)
    images = sorted((dossier / args.split / "images").iterdir())
    if args.images and len(images) > args.images:
        import random
        images = random.Random(args.graine).sample(images, args.images)
        presence = {i.stem: presence[i.stem] for i in images}
    print(f"{len(images)} images, dont {sum(presence.values())} contenant l'objet vise")
    if args.degradation != "aucune":
        print(f"degradation appliquee : {args.degradation}")

    modele = YOLO(args.modele)
    # Une passe unique a seuil tres bas : on rejoue ensuite les seuils sur les
    # scores obtenus, plutot que de relancer une inference par seuil.
    meilleur_score: dict[str, float] = defaultdict(float)
    entrees: list = [str(p) for p in images]
    if args.degradation == "nuit":
        import cv2
        import numpy as np
        import random as _rd
        rng = _rd.Random(args.graine)
        np.random.seed(args.graine)
        entrees = []
        for p in images:
            img = cv2.imread(str(p))
            alpha = rng.uniform(0.18, 0.40)
            sombre = cv2.convertScaleAbs(img, alpha=alpha, beta=rng.uniform(-15, 0))
            bruit = np.random.normal(0, rng.uniform(4, 12), sombre.shape)
            entrees.append(np.clip(sombre.astype(np.int16) + bruit.astype(np.int16),
                                   0, 255).astype(np.uint8))

    for lot in range(0, len(images), 32):
        for img, res in zip(images[lot:lot + 32],
                            modele.predict(entrees[lot:lot + 32],
                                           imgsz=args.imgsz, conf=0.005, device="cpu",
                                           verbose=False)):
            for b in res.boxes:
                if int(b.cls) in cls_modele:
                    meilleur_score[img.stem] = max(meilleur_score[img.stem], float(b.conf))

    print(f"\n{'seuil':>7}{'detection':>12}{'fausse alarme':>16}{'manques':>10}")
    print("-" * 46)
    mesures = {}
    for seuil in (0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30, 0.50):
        vp = fp = fn = vn = 0
        for nom, reel in presence.items():
            predit = meilleur_score.get(nom, 0.0) >= seuil
            if reel and predit:      vp += 1
            elif reel and not predit: fn += 1
            elif not reel and predit: fp += 1
            else:                     vn += 1
        rappel = vp / (vp + fn) if vp + fn else 0.0
        fausse = fp / (fp + vn) if fp + vn else 0.0
        mesures[f"{seuil:.2f}"] = {"detection": round(rappel, 4),
                                   "fausse_alarme": round(fausse, 4),
                                   "manques": fn}
        print(f"{seuil:>7.2f}{rappel:>11.1%}{fausse:>16.1%}{fn:>10}")

    if args.sortie:
        Path(args.sortie).write_text(
            json.dumps({"modele": args.modele, "split": args.split,
                        "classes_fumee": args.classes_fumee, "seuils": mesures},
                       indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\n-> {args.sortie}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
