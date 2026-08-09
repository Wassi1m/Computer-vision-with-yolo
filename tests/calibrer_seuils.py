#!/usr/bin/env python3
"""Calibre les seuils de confiance à partir de mesures, pas d'intuition.

Les seuils décident directement de ce qui est signalé au client. Ils étaient
jusqu'ici choisis à la main et dispersés : `CONF_THRESHOLD = 0.4` global,
`conf=0.4` en dur dans quatre modules, `conf=0.5` dans un cinquième, et une
table par classe dans `ppe_taxonomy.py`. Aucun n'était justifié par une mesure,
et personne n'osait donc les modifier.

Ce script balaie l'axe de confiance et, pour chaque classe, retient le seuil qui
maximise un **F-bêta**. Le choix de bêta est la seule décision réellement
métier :

  bêta > 1  privilégie le rappel — manquer un évènement coûte plus cher qu'une
            fausse alerte. C'est le cas de la conformité EPI (un ouvrier sans
            casque non détecté est un risque corporel, une vérification inutile
            est une gêne) et de la sécurité (feu, chute).
  bêta = 1  équilibre les deux (F1). Convient à la lecture de plaques, où une
            fausse détection produit surtout du bruit.

Le balayage vient d'une seule passe de validation : Ultralytics renvoie les
courbes précision/rappel/F1 échantillonnées sur tout l'axe de confiance
(`box.p_curve`, `r_curve`, `f1_curve`, `px`). Inutile donc de revalider à
chaque seuil.

    python tests/calibrer_seuils.py
    python tests/calibrer_seuils.py --modele ppe_best --beta 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = Path(__file__).with_name("reference_modeles.json")
SORTIE = ROOT / "reports/v3_results"

# Bêta par modèle : traduit le coût relatif d'un manqué face à une fausse alerte.
BETA_PAR_MODELE = {
    "ppe_best": 2.0,        # conformite EPI : le manque d'EPI ne doit pas passer
    "fall_detector": 2.0,   # une chute non detectee est un risque corporel
    "fire_smoke": 2.0,      # un depart de feu non detecte est un risque majeur
    "license_plate": 1.0,   # une plaque manquee se rattrape a l'image suivante
}
# En deca de ce seuil, une recommandation n'est pas exploitable : le modele
# n'atteint jamais un compromis acceptable et c'est le modele qu'il faut
# reprendre, pas son seuil.
F_MINIMUM_UTILISABLE = 0.35


def f_beta(precision: np.ndarray, rappel: np.ndarray, beta: float) -> np.ndarray:
    b2 = beta ** 2
    denom = b2 * precision + rappel
    return np.where(denom > 0, (1 + b2) * precision * rappel / np.maximum(denom, 1e-9), 0.0)


def calibrer(nom: str, spec: dict, beta: float, imgsz: int) -> dict | None:
    from ultralytics import YOLO

    poids, data = ROOT / spec["poids"], ROOT / spec["data"]
    if not poids.exists() or not data.exists():
        print(f"  {nom} : modele ou donnees absents, ignore")
        return None

    modele = YOLO(str(poids))
    r = modele.val(data=str(data), split=spec["split"], imgsz=imgsz,
                   device="cpu", batch=4, plots=False, verbose=False)

    # `px` est l'axe de confiance ; les courbes sont indexees par classe.
    axe = np.asarray(r.box.px)
    p_courbes = np.asarray(r.box.p_curve)
    r_courbes = np.asarray(r.box.r_curve)

    resultat = {"beta": beta, "classes": {}}
    for i, ci in enumerate(r.box.ap_class_index):
        classe = modele.names.get(int(ci))
        if classe is None:
            continue
        p, rap = p_courbes[i], r_courbes[i]
        f = f_beta(p, rap, beta)
        k = int(np.argmax(f))
        resultat["classes"][classe] = {
            "seuil_recommande": round(float(axe[k]), 3),
            "precision": round(float(p[k]), 4),
            "rappel": round(float(rap[k]), 4),
            f"f{beta:g}": round(float(f[k]), 4),
            # Repere : ce que donnerait le seuil generique de 0.4 encore en place
            # dans plusieurs modules.
            "au_seuil_0.40": {
                "precision": round(float(p[int(np.argmin(np.abs(axe - 0.40)))]), 4),
                "rappel": round(float(rap[int(np.argmin(np.abs(axe - 0.40)))]), 4),
            },
        }
    return resultat


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modele")
    ap.add_argument("--beta", type=float, default=None,
                    help="force un beta pour tous les modeles (defaut : par modele)")
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    ref = json.loads(REFERENCE.read_text(encoding="utf-8"))
    modeles = ref["modeles"]
    if args.modele:
        modeles = {args.modele: modeles[args.modele]}

    SORTIE.mkdir(parents=True, exist_ok=True)
    tout = {}
    for nom, spec in modeles.items():
        beta = args.beta if args.beta is not None else BETA_PAR_MODELE.get(nom, 1.0)
        print(f"\n=== {nom} (beta={beta:g}) ===")
        res = calibrer(nom, spec, beta, args.imgsz)
        if res is None:
            continue
        tout[nom] = res
        for classe, v in res["classes"].items():
            marque = " (peu exploitable)" if v[f"f{beta:g}"] < F_MINIMUM_UTILISABLE else ""
            print(f"  {classe:18} seuil {v['seuil_recommande']:.3f}  "
                  f"P={v['precision']:.3f} R={v['rappel']:.3f}  "
                  f"F{beta:g}={v[f'f{beta:g}']:.3f}{marque}")
            a04 = v["au_seuil_0.40"]
            print(f"  {'':18} a 0.40 :      P={a04['precision']:.3f} R={a04['rappel']:.3f}")

    dest = SORTIE / "seuils_calibres.json"
    dest.write_text(json.dumps(tout, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n-> {dest}")
    print("\nCes valeurs sont une recommandation mesuree, pas une application "
          "automatique :\nles reporter dans improvements/ppe_taxonomy.py (table par classe) "
          "ou via\nMOTEUR_CONF apres relecture, en documentant la date de mesure.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
