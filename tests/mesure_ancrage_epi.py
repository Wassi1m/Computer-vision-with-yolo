#!/usr/bin/env python3
"""Combien d'EPI sont attribués à une personne qui ne les porte pas ?

Le moteur associe chaque EPI détecté à la personne de meilleur IoU, sans
plancher (`unified_surveillance.py`, AnalyseurEPI.process). Deux défauts s'y
cumulent :

1. **L'IoU est la mauvaise mesure pour un rapport « partie de ».** Un casque
   occupe environ 3 % de la boîte d'une personne ; son IoU avec la personne qui
   le porte vaut donc ~0,03, indiscernable du bruit.
2. **Aucun rejet n'est possible.** `max()` attribue toujours l'EPI à quelqu'un,
   même lorsque le recouvrement est nul. Un casque détecté à tort ailleurs dans
   l'image est alors compté comme porté.

Le sens de l'erreur est le plus grave possible pour un système de sécurité : un
faux positif d'EPI **masque une infraction réelle**, puisque la personne à qui
il est attribué est déclarée conforme.

Ce script quantifie le phénomène sur le jeu de test, en comparant l'association
actuelle (IoU maximal) à l'association par confinement, avec rejet. Il ne
mesure pas le modèle mais la logique de décision : les détections sont les
mêmes dans les deux cas.

    python tests/mesure_ancrage_epi.py
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE / "improvements"))

import qualification as q  # noqa: E402
import ppe_taxonomy as tax  # noqa: E402

# Indices du jeu ppe_vest_clean_14c
CLASSE_PERSONNE = 11
EPI_PAR_CLASSE = {
    1: "gants", 2: "lunettes", 3: "casque", 5: "masque", 13: "gilet",
}


def charger_verite(lbl: Path, l: int, h: int):
    """Personnes et EPI annotés, quand ils le sont.

    Le jeu `ppe_vest_clean_14c` n'annote pas la classe `Person` dans son split
    de test : la vérité terrain ne peut donc pas servir de source de personnes.
    On se rabat sur les personnes **détectées**, ce qui correspond de toute
    façon au fonctionnement réel du moteur — en production, l'analyseur EPI
    reçoit les personnes de l'analyseur général, pas d'une annotation.
    """
    personnes, epis = [], []
    for ligne in lbl.read_text(encoding="utf-8").splitlines():
        p = ligne.split()
        if len(p) < 5:
            continue
        c = int(p[0])
        cx, cy, bw, bh = (float(v) for v in p[1:5])
        boite = (int((cx - bw / 2) * l), int((cy - bh / 2) * h),
                 int((cx + bw / 2) * l), int((cy + bh / 2) * h))
        if c == CLASSE_PERSONNE:
            personnes.append(boite)
        elif c in EPI_PAR_CLASSE:
            epis.append((EPI_PAR_CLASSE[c], boite))
    return personnes, epis


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modele", default="ppe_detection/models/best.pt")
    ap.add_argument("--modele-personnes", default="surveillance_suite/models/yolo26n.pt",
                    help="modele COCO fournissant les personnes, comme en production")
    ap.add_argument("--conf-personne", type=float, default=0.4)
    ap.add_argument("--donnees", default="ppe_detection/data/extracted/ppe_vest_clean_14c")
    ap.add_argument("--split", default="test")
    ap.add_argument("--images", type=int, default=350)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--seuil-confinement", type=float, default=0.50)
    ap.add_argument("--graine", type=int, default=3)
    ap.add_argument("--sortie", default="reports/v3_results/ancrage_epi.json")
    args = ap.parse_args()

    from ultralytics import YOLO
    import cv2

    base = RACINE / args.donnees / args.split
    fichiers = sorted((base / "images").iterdir())
    if len(fichiers) > args.images:
        fichiers = random.Random(args.graine).sample(fichiers, args.images)
    print(f"{len(fichiers)} images de {args.split}\n")

    modele = YOLO(str(RACINE / args.modele))
    modele_pers = YOLO(str(RACINE / args.modele_personnes))

    stats = Counter()
    confinements: list[float] = []
    ious: list[float] = []
    sans_personne = 0

    for f in fichiers:
        img = cv2.imread(str(f))
        if img is None:
            continue
        rp = modele_pers.predict(img, conf=args.conf_personne, verbose=False)[0]
        personnes_vt = [tuple(map(int, b.xyxy[0])) for b in rp.boxes
                        if modele_pers.names[int(b.cls)] == "person"]
        if not personnes_vt:
            sans_personne += 1
            continue

        res = modele.predict(img, conf=args.conf, verbose=False)[0]
        for b in res.boxes:
            nom = modele.names[int(b.cls)]
            d = tax.traduire("best.pt", nom, float(b.conf), tuple(map(int, b.xyxy[0])))
            if not d or not d.epi:
                continue
            boite = d.box
            stats["epi_detectes"] += 1

            # Comportement actuel : IoU maximal, sans plancher -> toujours attribue.
            i_iou = max(range(len(personnes_vt)),
                        key=lambda i: tax._iou(boite, personnes_vt[i]))
            ious.append(tax._iou(boite, personnes_vt[i_iou]))

            # Comportement propose : confinement, avec rejet possible.
            i_conf = q.associer_a_personne(boite, personnes_vt,
                                           args.seuil_confinement, epi=d.epi)
            confinements.append(max(q.confinement(boite, p) for p in personnes_vt))

            if i_conf is None:
                stats["rejetes_par_confinement"] += 1
                # L'ancien code attribuait pourtant cet EPI a quelqu'un.
                stats["attribues_a_tort"] += 1
            elif i_conf != i_iou:
                stats["attribues_a_la_mauvaise_personne"] += 1

    n = stats["epi_detectes"] or 1
    print(f"{'images sans personne detectee (ignorees)':44}{sans_personne:>7}")
    print(f"{'EPI detectes':44}{stats['epi_detectes']:>7}")
    print(f"{'  attribues a une personne a tort (actuel)':44}"
          f"{stats['attribues_a_tort']:>7}  {stats['attribues_a_tort']/n:>6.1%}")
    print(f"{'  attribues a la mauvaise personne':44}"
          f"{stats['attribues_a_la_mauvaise_personne']:>7}"
          f"  {stats['attribues_a_la_mauvaise_personne']/n:>6.1%}")
    total_faux = stats["attribues_a_tort"] + stats["attribues_a_la_mauvaise_personne"]
    print(f"{'  TOTAL mal attribues':44}{total_faux:>7}  {total_faux/n:>6.1%}")

    if ious and confinements:
        ious.sort(); confinements.sort()
        med = len(ious) // 2
        print(f"\nIoU median EPI<->personne          : {ious[med]:.3f}"
              f"   (inutilisable comme seuil absolu)")
        print(f"Confinement median EPI<->personne  : {confinements[med]:.3f}"
              f"   (separe nettement porte / non porte)")

    (RACINE / args.sortie).write_text(json.dumps({
        "modele": args.modele, "images": len(fichiers), "conf": args.conf,
        "seuil_confinement": args.seuil_confinement,
        "stats": dict(stats),
        "iou_median": round(ious[len(ious)//2], 4) if ious else None,
        "confinement_median": round(confinements[len(confinements)//2], 4) if confinements else None,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n-> {args.sortie}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
