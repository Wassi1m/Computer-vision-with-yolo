#!/usr/bin/env python3
"""Que vaut `door_classifier.pt`, le seul modèle en production jamais mesuré ?

L'audit du 2026-08-20 (point I3) l'a relevé : ce classifieur est chargé par le
moteur unifié, décide d'un évènement `porte` publié à la plateforme, et n'est
couvert par aucune mesure de référence ni par le harnais de non-régression. Le
registre lui attribuait « top-1 97,2 % » — un chiffre repris de la sortie
d'entraînement, jamais re-vérifié, et qui ne dit rien de la décision réellement
prise en exploitation.

Deux mesures, parce que deux questions différentes :

1. **Classification à trois classes** (`open` / `closed` / `semi`), telle que le
   modèle a été entraîné. C'est le chiffre comparable à celui du registre.

2. **Décision d'exploitation**, telle que `module_door_classifier.py` la produit
   réellement : `top_label == "open"`, donc **une porte entrebâillée compte comme
   fermée**, et une prédiction sous `min_confidence` (0.6 par défaut) est
   *ignorée* — le module conserve alors son état précédent, c'est-à-dire une
   lecture périmée. Ce sont ces deux comportements que la plateforme subit, pas
   la précision brute du modèle.

La mesure porte sur le split `test` du jeu Roboflow, jamais vu à l'entraînement.
Les images y sont des vignettes de porte déjà cadrées : elles jouent le rôle du
recadrage par ROI que le module applique en production.

    python tests/mesure_porte.py
    python tests/mesure_porte.py --min-confiance 0.6
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
SUITE = RACINE / "surveillance_suite"
MODELE = SUITE / "models/door_classifier.pt"
JEU = SUITE / "data/dataset/Door - Open - Closed -.v1i.folder/test"

CLASSES = ("Open", "Closed", "Semi")


def images_du_jeu(jeu: Path = JEU) -> list[tuple[Path, str]]:
    lot = []
    for classe in CLASSES:
        dossier = jeu / classe
        if not dossier.is_dir():
            continue
        for img in sorted(dossier.iterdir()):
            if img.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                lot.append((img, classe.lower()))
    return lot


def mesurer(modele_path: Path = MODELE, jeu: Path = JEU,
            min_confiance: float = 0.6, taille_lot: int = 32) -> dict:
    """Mesure le classifieur et renvoie le résultat, sans rien afficher.

    Séparée de l'affichage pour que `test_non_regression.py` s'en serve : sans
    cela, le garde-fou aurait sa propre copie de la mesure, et les deux
    finiraient par diverger.
    """
    if not modele_path.exists():
        raise FileNotFoundError(f"modèle absent : {modele_path}")
    lot = images_du_jeu(jeu)
    if not lot:
        raise FileNotFoundError(f"jeu de test absent : {jeu}")

    from ultralytics import YOLO
    modele = YOLO(str(modele_path))

    confusion: Counter[tuple[str, str]] = Counter()
    sous_seuil: Counter[str] = Counter()   # verite -> predictions ignorees
    par_classe: Counter[str] = Counter()

    for depart in range(0, len(lot), taille_lot):
        tranche = lot[depart:depart + taille_lot]
        resultats = modele.predict([str(c) for c, _ in tranche], verbose=False)
        for (_, verite), res in zip(tranche, resultats):
            predit = res.names[int(res.probs.top1)].lower()
            conf = float(res.probs.top1conf)
            confusion[(verite, predit)] += 1
            par_classe[verite] += 1
            if conf < min_confiance:
                sous_seuil[verite] += 1

    total = sum(par_classe.values())
    rappels = {c.lower(): round(confusion[(c.lower(), c.lower())] / par_classe[c.lower()], 4)
               for c in CLASSES if par_classe[c.lower()]}
    justes = sum(n for (v, p), n in confusion.items() if v == p)

    # Decision d'exploitation : le module ne publie pas trois etats mais
    # ouvert / ferme. `semi` y devient « fermee », convention qu'il faut mesurer
    # telle quelle -- c'est elle que la plateforme subit.
    vp = fp = vn = fn = 0
    for (verite, predit), n in confusion.items():
        reel, vu = verite == "open", predit == "open"
        if reel and vu:        vp += n
        elif reel and not vu:  fn += n
        elif not reel and vu:  fp += n
        else:                  vn += n

    return {
        "modele": str(modele_path),
        "jeu": str(jeu),
        "classes_du_modele": [modele.names[i] for i in sorted(modele.names)],
        "images": total,
        "par_classe": dict(par_classe),
        "trois_classes": {"exactitude": round(justes / max(total, 1), 4), "rappel": rappels},
        "confusion": confusion,
        "decision_publiee": {
            "rappel_ouvert": round(vp / max(vp + fn, 1), 4),
            "fausse_alarme": round(fp / max(fp + vn, 1), 4),
            "exactitude": round((vp + vn) / max(total, 1), 4),
            "vp": vp, "fp": fp, "vn": vn, "fn": fn,
        },
        "min_confiance": min_confiance,
        "predictions_ignorees": sum(sous_seuil.values()),
        "ignorees_par_classe": dict(sous_seuil),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-confiance", type=float, default=0.6,
                    help="seuil du module en production (min_confidence)")
    ap.add_argument("--lot", type=int, default=32, help="taille de lot d'inference")
    ap.add_argument("--sortie", default="reports/v3_results/porte_mesure.json")
    args = ap.parse_args()

    try:
        m = mesurer(MODELE, JEU, args.min_confiance, args.lot)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 2

    confusion, par_classe = m["confusion"], m["par_classe"]
    sous_seuil, rappels = m["ignorees_par_classe"], m["trois_classes"]["rappel"]
    exactitude, total = m["trois_classes"]["exactitude"], m["images"]
    print(f"{total} images de {JEU.name}, classes du modele : "
          f"{m['classes_du_modele']}\n")
    print(f"{'== 3 classes ==':<24}{'rappel':>10}{'images':>9}")
    for v, r in rappels.items():
        print(f"{v:<24}{r:>9.1%}{par_classe[v]:>9}")
    print(f"{'exactitude globale':<24}{exactitude:>9.1%}{total:>9}")

    print("\nmatrice de confusion (lignes = verite) :")
    entete = "".join(f"{c.lower():>9}" for c in CLASSES)
    print(f"{'':<10}{entete}")
    for c in CLASSES:
        v = c.lower()
        ligne = "".join(f"{confusion[(v, p.lower())]:>9}" for p in CLASSES)
        print(f"{v:<10}{ligne}")

    d = m["decision_publiee"]
    print(f"\n== decision publiee (ouvert / ferme, `semi` compte ferme) ==")
    print(f"portes ouvertes detectees : {d['rappel_ouvert']:.1%}  ({d['vp']}/{d['vp'] + d['fn']})")
    print(f"fausses ouvertures        : {d['fausse_alarme']:.1%}  ({d['fp']}/{d['fp'] + d['vn']})")
    print(f"exactitude                : {d['exactitude']:.1%}")

    ignorees = m["predictions_ignorees"]
    print(f"\n== effet du seuil min_confidence={args.min_confiance} ==")
    print(f"predictions ignorees : {ignorees}/{total} ({ignorees / max(total, 1):.1%})")
    if ignorees:
        detail = ", ".join(f"{v} {n}" for v, n in sorted(sous_seuil.items()))
        print(f"  par classe : {detail}")
    print("  une prediction ignoree laisse le module sur son etat precedent :")
    print("  la plateforme lit alors une porte dont l'etat n'a pas ete revu.")

    res = dict(m, date=str(date.today()),
               modele=str(MODELE.relative_to(RACINE)),
               jeu=str(JEU.relative_to(RACINE)),
               confusion={f"{v}->{p}": n for (v, p), n in sorted(confusion.items())})
    sortie = RACINE / args.sortie
    sortie.parent.mkdir(parents=True, exist_ok=True)
    sortie.write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n-> {args.sortie}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
