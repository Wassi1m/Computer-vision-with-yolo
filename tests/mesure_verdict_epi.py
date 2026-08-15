#!/usr/bin/env python3
"""L'association par confinement change-t-elle le verdict de conformité EPI ?

`mesure_ancrage_epi.py` a montré que 25 % des EPI étaient imputés à une personne
qui ne les porte pas. Reste à vérifier ce qui compte vraiment : est-ce que la
correction améliore le **verdict** — « cette scène présente-t-elle une
non-conformité ? » — ou est-ce qu'elle se contente de déplacer l'erreur ?

La question n'est pas rhétorique. Rejeter un EPI non confiné supprime des
attributions fausses, mais si l'EPI appartenait en réalité à une personne que le
détecteur a manquée, on prive quelqu'un de son équipement et l'on fabrique une
**fausse violation** — l'erreur symétrique, aussi grave. Une mesure préalable a
d'ailleurs établi que le taux d'EPI « orphelins » tombe de 21 % à 6 % quand on
abaisse le seuil de détection des personnes : une bonne part des orphelins sont
donc des personnes ratées, pas de faux positifs.

D'où cette comparaison à verdict, au niveau de l'image :

- **vérité terrain** : l'image contient-elle une annotation `NO-Safety Vest` ?
- **ancienne logique** : IoU maximal, sans plancher — un EPI est toujours imputé.
- **nouvelle logique** : confinement avec rejet.

Le jeu `ppe_vest_clean_14c` n'annotant que le gilet dans son split de test, la
mesure porte sur ce seul équipement — suffisant pour trancher, puisque c'est la
logique d'association qui est en cause et non l'EPI concerné.

    python tests/mesure_verdict_epi.py --images 250
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE / "improvements"))

import qualification as qual  # noqa: E402
import ppe_taxonomy as tax  # noqa: E402

CLASSE_SANS_GILET = 10   # NO-Safety Vest


def verite_violation(lbl: Path) -> bool:
    """L'image contient-elle une personne sans gilet, d'après l'annotation ?"""
    for ligne in lbl.read_text(encoding="utf-8").splitlines():
        p = ligne.split()
        if p and int(p[0]) == CLASSE_SANS_GILET:
            return True
    return False


def verdict(dets, personnes, mode: str, seuil_confinement: float) -> bool:
    """Y a-t-il non-conformité gilet sur au moins une personne évaluée ?

    Reproduit la décision de `AnalyseurEPI` réduite à une seule image : le
    lissage temporel est neutralisé, sans quoi toute image isolée serait déclarée
    non conforme (aucun historique ne peut être « stable » sur une image).
    """
    if not personnes:
        return False
    porte = {i: set() for i in range(len(personnes))}
    absent = {i: set() for i in range(len(personnes))}
    for d in dets:
        if not d.epi:
            continue
        if mode == "ancien":
            # IoU maximal, sans plancher : l'EPI est attribué quoi qu'il arrive.
            i = max(range(len(personnes)), key=lambda k: tax._iou(d.box, personnes[k]))
        elif mode == "confinement":
            i = qual.associer_a_personne(d.box, personnes, seuil_confinement, epi=d.epi)
            if i is None:
                continue
        else:  # "mixte"
            # Le rejet n'a d'intérêt que s'il existe un risque de se tromper de
            # personne. Avec une seule personne évaluée, l'attribution est sans
            # ambiguïté : rejeter ne corrigerait rien et priverait cette personne
            # de son équipement, donc fabriquerait une fausse violation.
            if len(personnes) == 1:
                i = 0
            else:
                i = qual.associer_a_personne(d.box, personnes, seuil_confinement, epi=d.epi)
                if i is None:
                    continue
        (porte if d.porte else absent)[i].add(d.epi)
    # Non-conformité : soit un « NO-... » imputé, soit un EPI obligatoire absent
    # chez une personne pour laquelle on a bien détecté quelque chose.
    for i in range(len(personnes)):
        if absent[i]:
            return True
        if "gilet" not in porte[i]:
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modele", default="ppe_detection/models/ppe_detector.pt")
    ap.add_argument("--modele-personnes", default="surveillance_suite/models/yolo26n.pt")
    ap.add_argument("--donnees", default="ppe_detection/data/extracted/ppe_vest_clean_14c")
    ap.add_argument("--split", default="test")
    ap.add_argument("--images", type=int, default=250)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--conf-personne", type=float, default=0.40)
    ap.add_argument("--seuil-confinement", type=float, default=0.50)
    ap.add_argument("--graine", type=int, default=3)
    ap.add_argument("--sortie", default="reports/v3_results/verdict_epi.json")
    args = ap.parse_args()

    from ultralytics import YOLO
    import cv2

    base = RACINE / args.donnees / args.split
    fichiers = sorted((base / "images").iterdir())
    if len(fichiers) > args.images:
        fichiers = random.Random(args.graine).sample(fichiers, args.images)

    modele = YOLO(str(RACINE / args.modele))
    modele_pers = YOLO(str(RACINE / args.modele_personnes))

    MODES = ("ancien", "confinement", "mixte")
    resultats = {m: {"vp": 0, "fp": 0, "fn": 0, "vn": 0} for m in MODES}
    evalues = 0

    for f in fichiers:
        img = cv2.imread(str(f))
        if img is None:
            continue
        personnes = [tuple(map(int, b.xyxy[0]))
                     for b in modele_pers.predict(img, conf=args.conf_personne,
                                                  verbose=False)[0].boxes
                     if modele_pers.names[int(b.cls)] == "person"]
        if not personnes:
            continue
        dets = []
        for b in modele.predict(img, conf=args.conf, verbose=False)[0].boxes:
            d = tax.traduire("ppe_detector.pt", modele.names[int(b.cls)], float(b.conf),
                             tuple(map(int, b.xyxy[0])))
            if d and d.epi:
                dets.append(d)
        dets = tax.fusionner(dets)

        reel = verite_violation(base / "labels" / (f.stem + ".txt"))
        evalues += 1
        for mode in MODES:
            pred = verdict(dets, personnes, mode, args.seuil_confinement)
            r = resultats[mode]
            if reel and pred:        r["vp"] += 1
            elif reel and not pred:  r["fn"] += 1
            elif not reel and pred:  r["fp"] += 1
            else:                    r["vn"] += 1

    print(f"{evalues} images evaluees (avec au moins une personne detectee)\n")
    print(f"{'logique':14}{'detection':>12}{'fausse alarme':>16}{'exactitude':>13}")
    print("-" * 55)
    resume = {}
    for cle in MODES:
        r = resultats[cle]
        rappel = r["vp"] / max(r["vp"] + r["fn"], 1)
        fausse = r["fp"] / max(r["fp"] + r["vn"], 1)
        exact = (r["vp"] + r["vn"]) / max(sum(r.values()), 1)
        resume[cle] = {"detection": round(rappel, 4), "fausse_alarme": round(fausse, 4),
                       "exactitude": round(exact, 4), **r}
        print(f"{cle:14}{rappel:>11.1%}{fausse:>16.1%}{exact:>13.1%}")

    (RACINE / args.sortie).write_text(json.dumps(
        {"images": evalues, "conf": args.conf, "conf_personne": args.conf_personne,
         "seuil_confinement": args.seuil_confinement, "logiques": resume},
        indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n-> {args.sortie}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
