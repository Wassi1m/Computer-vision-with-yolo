#!/usr/bin/env python3
"""Mesure la tenue des modèles en conditions de prise de vue dégradées.

Le déploiement cible est une caméra de surveillance, souvent extérieure. Or
toutes les métriques du projet ont été obtenues sur des images correctement
exposées : rien ne dit ce que valent les modèles de nuit, à contre-jour, par
temps de pluie ou face à une caméra sale. Promettre un fonctionnement continu
sans cette mesure revient à découvrir l'écart en exploitation.

Ce script **ne ré-entraîne rien**. Il applique les modèles déjà déployés à des
versions dégradées des jeux de validation existants, et quantifie la perte.

Les dégradations sont simulées, ce qui est une approximation assumée : une vraie
scène de nuit n'est pas une image de jour assombrie (bruit du capteur, éclairage
artificiel ponctuel, mouvement flou). Les chiffres obtenus sont donc un
minorant de la difficulté réelle — s'ils sont déjà mauvais, le cas réel le sera
davantage.

    python tests/mesure_robustesse.py
    python tests/mesure_robustesse.py --modele fall_detector --images 100
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = Path(__file__).with_name("reference_modeles.json")
SORTIE = ROOT / "reports/v3_results"
TRAVAIL = ROOT / "ppe_detection/data/extracted/_robustesse"


# ── Dégradations ─────────────────────────────────────────────────────────────
# Chacune correspond à une condition d'exploitation identifiable, pas à un effet
# arbitraire : c'est ce qui rend le résultat interprétable.

def faible_luminosite(img):
    """Nuit / éclairage insuffisant : forte réduction d'exposition + bruit capteur."""
    sombre = cv2.convertScaleAbs(img, alpha=0.25, beta=-10)
    bruit = np.random.normal(0, 8, sombre.shape).astype(np.int16)
    return np.clip(sombre.astype(np.int16) + bruit, 0, 255).astype(np.uint8)


def contre_jour(img):
    """Sujet devant une source lumineuse : hautes lumières écrasées, sujet sous-exposé."""
    h, w = img.shape[:2]
    halo = np.zeros((h, w), np.float32)
    cv2.circle(halo, (w // 2, h // 4), max(h, w) // 2, 1.0, -1)
    halo = cv2.GaussianBlur(halo, (0, 0), max(h, w) / 8)[..., None]
    assombri = img.astype(np.float32) * 0.55
    return np.clip(assombri + halo * 190, 0, 255).astype(np.uint8)


def brouillard_pluie(img):
    """Intempérie : contraste réduit, voile blanchâtre, léger flou."""
    voile = np.full_like(img, 200)
    melange = cv2.addWeighted(img, 0.55, voile, 0.45, 0)
    return cv2.GaussianBlur(melange, (5, 5), 0)


def flou_mouvement(img):
    """Sujet ou caméra en mouvement, temps de pose long (fréquent de nuit)."""
    noyau = np.zeros((13, 13), np.float32)
    noyau[6, :] = 1.0 / 13
    return cv2.filter2D(img, -1, noyau)


def basse_resolution(img):
    """Sujet éloigné ou flux fortement compressé."""
    h, w = img.shape[:2]
    petit = cv2.resize(img, (max(w // 4, 1), max(h // 4, 1)), interpolation=cv2.INTER_AREA)
    return cv2.resize(petit, (w, h), interpolation=cv2.INTER_LINEAR)


DEGRADATIONS = {
    "reference": None,
    "faible_luminosite": faible_luminosite,
    "contre_jour": contre_jour,
    "brouillard_pluie": brouillard_pluie,
    "flou_mouvement": flou_mouvement,
    "basse_resolution": basse_resolution,
}


def construire_variante(spec: dict, nom_degradation: str, fn, limite: int) -> Path | None:
    """Copie un sous-ensemble du jeu de validation en y appliquant la dégradation."""
    data = ROOT / spec["data"]
    if not data.exists():
        return None

    base = data.parent
    split = spec["split"]
    src_img, src_lbl = base / split / "images", base / split / "labels"
    if not src_img.exists():  # certains jeux nomment le split "valid"
        for autre in ("valid", "val", "test"):
            if (base / autre / "images").exists():
                src_img, src_lbl = base / autre / "images", base / autre / "labels"
                break
    if not src_img.exists():
        return None

    dst = TRAVAIL / f"{spec['_nom']}_{nom_degradation}"
    if dst.exists():
        shutil.rmtree(dst)
    (dst / "val/images").mkdir(parents=True)
    (dst / "val/labels").mkdir(parents=True)

    for img_path in sorted(src_img.iterdir())[:limite]:
        lbl = src_lbl / img_path.with_suffix(".txt").name
        if not lbl.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        cv2.imwrite(str(dst / "val/images" / f"{img_path.stem}.jpg"),
                    img if fn is None else fn(img))
        shutil.copy2(lbl, dst / "val/labels" / lbl.name)

    # On reprend nc/names du jeu d'origine tels quels : la degradation ne
    # touche que les pixels, jamais la taxonomie ni les annotations.
    yaml_src = (base / "data.yaml").read_text(encoding="utf-8")
    ligne_names = next(l for l in yaml_src.splitlines() if l.startswith("names:"))
    ligne_nc = next(l for l in yaml_src.splitlines() if l.startswith("nc:"))
    (dst / "data.yaml").write_text(
        f"path: {dst}\ntrain: val/images\nval: val/images\n\n{ligne_nc}\n{ligne_names}\n")
    return dst / "data.yaml"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modele", help="ne mesurer qu'un modele")
    ap.add_argument("--images", type=int, default=120,
                    help="images par variante (defaut 120 : compromis CPU/represensativite)")
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    from ultralytics import YOLO

    ref = json.loads(REFERENCE.read_text(encoding="utf-8"))
    modeles = ref["modeles"]
    if args.modele:
        modeles = {args.modele: modeles[args.modele]}

    SORTIE.mkdir(parents=True, exist_ok=True)
    resultats = {}

    for nom, spec in modeles.items():
        spec = dict(spec, _nom=nom)
        poids = ROOT / spec["poids"]
        if not poids.exists():
            print(f"{nom} : modele absent, ignore")
            continue

        print(f"\n=== {nom} ===")
        modele = YOLO(str(poids))
        par_degradation = {}

        for nom_deg, fn in DEGRADATIONS.items():
            yaml = construire_variante(spec, nom_deg, fn, args.images)
            if yaml is None:
                print(f"  {nom_deg:20} jeu de donnees indisponible")
                continue
            r = modele.val(data=str(yaml), split="val", imgsz=args.imgsz,
                           device="cpu", batch=4, plots=False, verbose=False)
            par_degradation[nom_deg] = round(float(r.box.map50), 4)
            base = par_degradation.get("reference")
            ecart = "" if nom_deg == "reference" else f"  ({par_degradation[nom_deg] - base:+.4f})"
            print(f"  {nom_deg:20} mAP@50 = {par_degradation[nom_deg]:.4f}{ecart}")

        resultats[nom] = par_degradation

    dest = SORTIE / "robustesse_conditions_reelles.json"
    dest.write_text(json.dumps(resultats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("\n" + "=" * 66)
    print(f"{'modele':16}{'condition':22}{'mAP@50':>10}{'perte':>12}")
    for nom, d in resultats.items():
        base = d.get("reference")
        for cond, v in d.items():
            if cond == "reference" or base is None:
                continue
            perte = 100 * (v - base) / base if base else 0
            print(f"{nom:16}{cond:22}{v:>10.4f}{perte:>11.1f}%")
    print(f"\n-> {dest}")
    shutil.rmtree(TRAVAIL, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
