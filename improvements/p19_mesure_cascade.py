#!/usr/bin/env python3
"""P19 — Mesure la CASCADE COMPLÈTE, telle que le moteur la produit.

Pourquoi cette mesure n'existait pas, et pourquoi elle est le garde-fou
----------------------------------------------------------------------
Tout ce que le projet a mesuré jusqu'ici porte sur des modèles **isolés** :
`p14` juge un candidat seul, `test_non_regression.py` re-mesure chaque `.pt`
sur son jeu. Or ce que le client reçoit n'est aucun de ces modèles — c'est leur
**fusion**, avec ses seuils par classe, sa déduplication par concept et son
ordre de priorité (`ppe_taxonomy.fusionner`).

Rien ne garantit qu'une somme de bons modèles fasse une bonne cascade. Un seuil
mal choisi, une priorité inversée, une classe absente d'une table : chacun de
ces défauts laisse les modèles intacts et casse le résultat, sans qu'aucune
mesure existante ne le voie.

Le plan v8 §7 l'exigeait explicitement après chaque branchement — « mesurer les
14 classes de la cascade complète, pas seulement celles visées » — parce que la
mAP globale ne verrait pas un effondrement : elle est restée à 0,88 pendant que
douze classes étaient à zéro, en juillet 2026.

Ce que le script mesure
-----------------------
Pour chaque concept canonique et chaque polarité (porté / absent), la
proportion des scènes annotées où la cascade produit effectivement la bonne
sortie. C'est l'indicateur d'EXPLOITATION, celui qui décide — pas l'AP, qui ne
juge que la position du rectangle.

`--comparer-m1` refait la même mesure avec `ppe_detector.pt` SEUL, ce qui donne
la seule comparaison honnête : ce que la cascade apporte par rapport à l'état
d'avant les modèles dédiés.

    python improvements/p19_mesure_cascade.py --n 400
    python improvements/p19_mesure_cascade.py --n 400 --comparer-m1
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ppe_taxonomy as tax  # noqa: E402

RACINE = Path(__file__).resolve().parents[1]
MODELES = RACINE / "ppe_detection/models"
TEST = RACINE / "ppe_detection/data/extracted/ppe_dataset/test"

def verite_terrain() -> dict[int, tuple[str, bool]]:
    """{indice: (concept, porté)}, résolu contre le `data.yaml` du jeu.

    Jamais écrit en dur. La première version de ce script avait extrapolé ces
    numéros au lieu de les lire : `5` pris pour le gilet alors que c'est
    `Mask`, `Safety Vest` (13) purement oublié. La mesure publiait alors 0 %
    sur `gilet porté` pour un modèle à 0.93 d'AP, sans que rien ne l'indique --
    des numéros faux produisent des chiffres parfaitement plausibles.
    """
    yaml = (TEST.parent / "data.yaml").read_text(encoding="utf-8")
    ligne = next(l for l in yaml.splitlines() if l.startswith("names:"))
    return tax.indices_ppe_dataset(eval(ligne.split("names:", 1)[1].strip()))

CASCADE = [
    (tax.M1_NOM, "ppe_detector.pt"),
    (tax.M3_NOM, "masque_gilet.pt"),
    (tax.M4_NOM, "epi_casque.pt"),
    (tax.M5_NOM, "epi_gants_lunettes.pt"),
    (tax.M6_NOM, "epi_chaussures.pt"),
]


def charger(seulement_m1: bool):
    from ultralytics import YOLO
    lot = CASCADE[:1] if seulement_m1 else CASCADE
    charges = []
    for nom, fichier in lot:
        chemin = MODELES / fichier
        if not chemin.exists():
            print(f"  /!\\ absent, ignore : {fichier}")
            continue
        charges.append((nom, YOLO(str(chemin))))
    return charges


def mesurer(modeles, images: list[Path], imgsz: int, verite: dict) -> tuple[Counter, Counter]:
    """Renvoie (attendu, trouvé) indexés par (concept, porté)."""
    seuil_bas = min(c.conf_min for n, _ in modeles for c in tax.TABLES[n].values())
    attendu, trouve = Counter(), Counter()

    for i, img in enumerate(images):
        if i % 25 == 0:
            print(f"  {i}/{len(images)}", end="\r", flush=True)
        lbl = TEST / "labels" / f"{img.stem}.txt"
        if not lbl.is_file():
            continue
        cibles = set()
        for ligne in lbl.read_text(encoding="utf-8").splitlines():
            champs = ligne.split()
            if champs and int(champs[0]) in verite:
                cibles.add(verite[int(champs[0])])
        if not cibles:
            continue

        brutes = []
        for nom, mdl in modeles:
            for b in mdl.predict(str(img), conf=seuil_bas, imgsz=imgsz,
                                 verbose=False)[0].boxes:
                d = tax.traduire(nom, mdl.names[int(b.cls)], float(b.conf),
                                 tuple(map(int, b.xyxy[0])))
                if d:
                    brutes.append(d)
        # C'est bien la SORTIE FUSIONNEE qu'on evalue, pas l'union des modeles :
        # la fusion peut ecarter une detection juste, et c'est justement ce
        # qu'aucune mesure par modele ne pouvait detecter.
        vues = {(d.epi, d.porte) for d in tax.fusionner(brutes) if d.epi}
        for cible in cibles:
            attendu[cible] += 1
            if cible in vues:
                trouve[cible] += 1
    print(" " * 30, end="\r")
    return attendu, trouve


def afficher(titre: str, attendu: Counter, trouve: Counter) -> dict:
    print(f"\n=== {titre} ===")
    print(f"{'concept':<12} {'polarite':<10} {'scenes':>7} {'detecte':>9}")
    resultat = {}
    for concept in tax.EPI_CANONIQUES:
        for porte in (True, False):
            n = attendu[(concept, porte)]
            if not n:
                continue
            taux = trouve[(concept, porte)] / n
            resultat[f"{concept}_{'porte' if porte else 'absent'}"] = round(taux, 4)
            print(f"{concept:<12} {'porté' if porte else 'ABSENT':<10} "
                  f"{n:>7} {100*taux:>8.1f} %")
    return resultat


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--imgsz", type=int, default=480, help="celui du moteur")
    ap.add_argument("--comparer-m1", action="store_true",
                    help="refait la mesure avec ppe_detector.pt seul")
    ap.add_argument("--sortie", default="reports/v3_results/cascade_complete.json")
    args = ap.parse_args()

    images = sorted((TEST / "images").glob("*"))
    lot = random.Random(0).sample(images, min(args.n, len(images)))
    print(f"mesure sur {len(lot)} images de ppe_dataset/test (graine 0), imgsz={args.imgsz}\n")

    print("cascade complete :")
    modeles = charger(seulement_m1=False)
    print(f"  {len(modeles)} modeles : {', '.join(n for n, _ in modeles)}")
    verite = verite_terrain()
    a, t = mesurer(modeles, lot, args.imgsz, verite)
    res_cascade = afficher("CASCADE COMPLETE", a, t)

    res = {"date": str(date.today()),
           "mesure": f"{len(lot)} images de ppe_dataset/test (graine 0), imgsz={args.imgsz}, "
                     "sortie FUSIONNEE de la cascade",
           "modeles": [n for n, _ in modeles],
           "cascade": res_cascade}

    if args.comparer_m1:
        print("\nppe_detector.pt seul :")
        a1, t1 = mesurer(charger(seulement_m1=True), lot, args.imgsz, verite)
        res_m1 = afficher("ppe_detector.pt SEUL", a1, t1)
        res["ppe_detector_seul"] = res_m1
        print(f"\n=== APPORT DE LA CASCADE ===")
        print(f"{'concept':<12} {'polarite':<10} {'seul':>8} {'cascade':>9} {'ecart':>9}")
        ecarts = {}
        for cle, apres in res_cascade.items():
            avant = res_m1.get(cle)
            if avant is None:
                continue
            ecarts[cle] = round(apres - avant, 4)
            concept, pol = cle.rsplit("_", 1)
            fleche = "+" if apres > avant else ("=" if apres == avant else "")
            print(f"{concept:<12} {pol:<10} {100*avant:>7.1f}% {100*apres:>8.1f}% "
                  f"{fleche}{100*(apres-avant):>7.1f} pts")
        res["ecart"] = ecarts
        regressions = {k: v for k, v in ecarts.items() if v < -0.02}
        print()
        if regressions:
            for k, v in regressions.items():
                print(f"  /!\\ REGRESSION {k} : {100*v:+.1f} points")
        else:
            print("  Aucune regression : la cascade ne degrade aucun concept.")
        res["regressions"] = regressions

    sortie = RACINE / args.sortie
    sortie.write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n-> {sortie.relative_to(RACINE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
