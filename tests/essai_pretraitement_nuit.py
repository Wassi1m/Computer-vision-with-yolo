#!/usr/bin/env python3
"""Le prétraitement d'image peut-il remplacer un réentraînement pour la nuit ?

Une image nocturne n'est pas « difficile » dans l'absolu : elle est **hors de la
distribution** sur laquelle le modèle a été entraîné. D'où l'hypothèse testée
ici : plutôt que de réapprendre au modèle à voir dans le noir (coûteux, et qui
a déjà dégradé la classe `smoke` lors du fine-tuning P8), ramener l'image dans
la plage de luminosité que le modèle sait déjà traiter.

Trois traitements sont comparés, du moins au plus agressif :

- **gamma** : remonte les tons sombres sans toucher aux hautes lumières. Sûr,
  mais amplifie le bruit du capteur en même temps que le signal.
- **CLAHE** : égalisation adaptative par tuiles, appliquée au seul canal de
  luminance (L de LAB) pour ne pas déformer les couleurs. Bien plus efficace
  sur les scènes à éclairage inégal, typiques de la surveillance nocturne.
- **CLAHE + débruitage** : le débruitage limite les faux détails créés par
  l'amplification, au prix d'un coût de calcul nettement supérieur.

Le résultat se lit en métriques d'exploitation (taux de détection et de fausse
alarme par scène), et non en mAP@50 : c'est la capacité à repérer un départ de
feu qui est en jeu, pas la précision du rectangle.

Attention à ce que ce script ne dit pas : les images nocturnes sont simulées
par assombrissement d'images de jour. Une vraie scène de nuit comporte un bruit
de capteur différent et des sources lumineuses ponctuelles. Les gains mesurés
ici sont donc indicatifs, à reconfirmer sur des séquences réelles.

    python tests/essai_pretraitement_nuit.py --images 400
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

RACINE = Path(__file__).resolve().parents[1]


def assombrir(img, rng):
    """Même dégradation que tests/mesure_robustesse.py, pour rester comparable."""
    alpha = rng.uniform(0.18, 0.40)
    sombre = cv2.convertScaleAbs(img, alpha=alpha, beta=rng.uniform(-15, 0))
    bruit = np.random.normal(0, rng.uniform(4, 12), sombre.shape)
    return np.clip(sombre.astype(np.int16) + bruit.astype(np.int16), 0, 255).astype(np.uint8)


def gamma(img, g=0.45):
    table = np.array([((i / 255.0) ** g) * 255 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(img, table)


def clahe(img):
    # Sur le canal L de LAB uniquement : egaliser les trois canaux RGB
    # deformerait la teinte, or la desaturation est justement un indice de fumee.
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def clahe_debruite(img):
    return cv2.fastNlMeansDenoisingColored(clahe(img), None, 6, 6, 7, 21)


TRAITEMENTS = {
    "aucun (nuit brute)": lambda i: i,
    "gamma 0.45": gamma,
    "CLAHE": clahe,
    "CLAHE + debruitage": clahe_debruite,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modele", default="surveillance_suite/models/fire_smoke.pt")
    ap.add_argument("--donnees", default="surveillance_suite/data/dataset/fire_smoke_v9")
    ap.add_argument("--split", default="test")
    ap.add_argument("--images", type=int, default=400)
    ap.add_argument("--seuil", type=float, default=0.15)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--classes-vt", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--classes-modele", type=int, nargs="+", default=[1])
    ap.add_argument("--graine", type=int, default=7)
    ap.add_argument("--sortie", default="reports/v3_results/pretraitement_nuit.json")
    args = ap.parse_args()

    from ultralytics import YOLO

    base = RACINE / args.donnees / args.split
    fichiers = sorted((base / "images").iterdir())
    rng_ech = random.Random(args.graine)
    if len(fichiers) > args.images:
        fichiers = rng_ech.sample(fichiers, args.images)

    cls_vt, cls_mod = set(args.classes_vt), set(args.classes_modele)
    presence = {}
    for f in fichiers:
        lbl = base / "labels" / (f.stem + ".txt")
        presence[f.stem] = any(
            p.split() and int(p.split()[0]) in cls_vt
            for p in lbl.read_text(encoding="utf-8").splitlines())

    positifs = sum(presence.values())
    print(f"{len(fichiers)} images ({positifs} avec fumee, {len(fichiers)-positifs} sans)")
    print(f"modele : {args.modele}   seuil : {args.seuil}\n")

    modele = YOLO(str(RACINE / args.modele))
    resultats = {}

    print(f"{'traitement':24}{'detection':>11}{'fausse alarme':>16}{'cout/image':>13}")
    print("-" * 64)
    for nom, fn in TRAITEMENTS.items():
        rng = random.Random(args.graine)          # meme nuit pour tous les traitements
        np.random.seed(args.graine)
        vp = fn_ = fp = vn = 0
        t0 = time.perf_counter()
        cout_pre = 0.0
        for f in fichiers:
            img = cv2.imread(str(f))
            nuit = assombrir(img, rng)
            t1 = time.perf_counter()
            traitee = fn(nuit)
            cout_pre += time.perf_counter() - t1
            r = modele.predict(traitee, imgsz=args.imgsz, conf=args.seuil,
                               device="cpu", verbose=False)[0]
            predit = any(int(b.cls) in cls_mod for b in r.boxes)
            reel = presence[f.stem]
            if reel and predit:        vp += 1
            elif reel and not predit:  fn_ += 1
            elif not reel and predit:  fp += 1
            else:                      vn += 1
        rappel = vp / (vp + fn_) if vp + fn_ else 0.0
        fausse = fp / (fp + vn) if fp + vn else 0.0
        ms = 1000 * cout_pre / len(fichiers)
        resultats[nom] = {"detection": round(rappel, 4), "fausse_alarme": round(fausse, 4),
                          "manques": fn_, "cout_pretraitement_ms": round(ms, 2)}
        print(f"{nom:24}{rappel:>10.1%}{fausse:>16.1%}{ms:>11.2f} ms")

    (RACINE / args.sortie).write_text(
        json.dumps({"modele": args.modele, "seuil": args.seuil,
                    "images": len(fichiers), "positifs": positifs,
                    "traitements": resultats}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    print(f"\n-> {args.sortie}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
