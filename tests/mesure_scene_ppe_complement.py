#!/usr/bin/env python3
"""Détection de scène de `ppe_complement.pt`, classe par classe.

Outil unique de mesure pour les 5 classes mesurables de ce modèle (toutes
sauf `safety_shoe`) : 15 images de `ppe_dataset/test` tirées au sort (graine
7) par classe, seuil de confiance 0.25, et on regarde si le modèle signale
l'objet quelque part dans l'image (détection de scène, pas de recouvrement
de boîte). Reproduit la mesure du 2026-08-13 (helmet, Gloves, mask) et celle
du 2026-08-16 (Vest, goggles) -- désormais un seul script, pas deux mesures
séparées.

`ppe_complement.pt` n'a pas la même taxonomie que `ppe_dataset` : la
correspondance vient de `improvements/p2_table_correspondance_epi.py`.
`safety_shoe` n'a AUCUN équivalent dans `ppe_dataset` (aucune image annotée
« chaussure » n'existe sur cette machine) : il est donc structurellement
impossible de le mesurer sur ce jeu.

Sert de garde-fou de non-régression pour ce modèle en l'absence de tout jeu
de données local à lui (cf. `p2_table_correspondance_epi.py`) : à relancer
avant ET après tout fine-tuning, avec `--sortie` pour figer chaque mesure.

    python tests/mesure_scene_ppe_complement.py
    python tests/mesure_scene_ppe_complement.py --sortie reports/v3_results/ppe_complement_avant.json
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import date
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
MODELE = RACINE / "ppe_detection/models/ppe_complement.pt"
DONNEES = RACINE / "ppe_detection/data/extracted/ppe_dataset/test"
SEUIL = 0.25
IMAGES_PAR_CLASSE = 15
GRAINE = 7

# classe ppe_complement.pt (indice, nom) -> indice de la classe équivalente
# dans ppe_dataset (verite terrain). Reprend improvements/p2_table_correspondance_epi.py.
A_MESURER = {
    0: ("Gloves", 1),      # Gloves
    1: ("Vest", 13),       # Safety Vest
    2: ("goggles", 2),     # Goggles
    3: ("helmet", 3),      # Hardhat
    4: ("mask", 5),        # Mask
}
NON_MESURABLE = {5: "safety_shoe"}  # aucune image annotee "chaussure" disponible


def images_avec_classe(indice_gt: int) -> list[str]:
    trouvees = []
    for lbl in sorted((DONNEES / "labels").glob("*.txt")):
        for ligne in lbl.read_text(encoding="utf-8").splitlines():
            parts = ligne.split()
            if parts and int(parts[0]) == indice_gt:
                trouvees.append(lbl.stem)
                break
    return trouvees


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sortie", default=None, help="chemin JSON pour figer la mesure")
    args = ap.parse_args()

    from ultralytics import YOLO

    modele = YOLO(str(MODELE))
    rng = random.Random(GRAINE)

    print(f"{'classe ppe_complement':<22}{'equiv. ppe_dataset':<20}{'images':>8}{'detecte':>10}")
    print("-" * 62)
    resultats = {}
    for indice_modele, (nom, indice_gt) in A_MESURER.items():
        candidates = images_avec_classe(indice_gt)
        echantillon = rng.sample(candidates, min(IMAGES_PAR_CLASSE, len(candidates)))
        chemins = [DONNEES / "images" / f"{stem}.jpg" for stem in echantillon]
        chemins = [p for p in chemins if p.exists()] or \
            [DONNEES / "images" / f"{stem}.png" for stem in echantillon]

        detecte = 0
        for res in modele.predict([str(p) for p in chemins], conf=SEUIL, device="cpu", verbose=False):
            if any(int(b.cls) == indice_modele for b in res.boxes):
                detecte += 1

        taux = detecte / len(chemins) if chemins else 0.0
        resultats[nom] = round(taux, 4)
        print(f"{nom:<22}{indice_gt:<20}{len(chemins):>8}{taux:>10.0%}")

    print()
    for indice, nom in NON_MESURABLE.items():
        print(f"{nom} : non mesurable sur ppe_dataset (aucun equivalent, cf. p2_table_correspondance_epi.py)")

    if args.sortie:
        sortie = {
            "date": date.today().isoformat(),
            "modele": str(MODELE.relative_to(RACINE)),
            "jeu": "ppe_dataset/test (proxy, pas le jeu d'entrainement de ce modele)",
            "methode": f"detection de scene, {IMAGES_PAR_CLASSE} images/classe, graine {GRAINE}, seuil {SEUIL}",
            "par_classe": resultats,
            "non_mesurable": list(NON_MESURABLE.values()),
        }
        Path(args.sortie).write_text(json.dumps(sortie, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\n-> {args.sortie}")
    else:
        print("\nA reporter dans reports/v3_results/scores_par_classe.json puis :")
        print("    python improvements/generer_classes.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
