#!/usr/bin/env python3
"""Garde-fou de non-régression des modèles.

Re-mesure chaque modèle déployé sur son jeu de référence figé et compare aux
métriques enregistrées dans `reference_modeles.json`. Échoue si une classe perd
plus que la tolérance admise.

Raison d'être : une régression de modèle ne se voit pas. Un remplacement de
`.pt` qui dégrade une classe produit exactement la même exécution, les mêmes
logs et la même absence d'erreur — le défaut ne se manifeste que par des alertes
manquées chez le client. Ce script est le seul point du projet où cette
dégradation devient visible.

À lancer avant tout remplacement de modèle et avant toute livraison.

    python tests/test_non_regression.py                # tous les modèles
    python tests/test_non_regression.py --modele fall_detector
    python tests/test_non_regression.py --maj          # fige les valeurs mesurées

L'exécution est lente (validation CPU sur plusieurs centaines d'images) : c'est
une vérification de livraison, pas un test à lancer à chaque sauvegarde. Les
tests rapides sans modèle sont dans `test_logique_metier.py`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = Path(__file__).with_name("reference_modeles.json")


def mesurer(spec: dict, imgsz: int = 640) -> dict:
    """Re-mesure un modèle et renvoie {classe: AP@50, ...} + mAP50."""
    from ultralytics import YOLO

    poids = ROOT / spec["poids"]
    data = ROOT / spec["data"]
    if not poids.exists():
        raise FileNotFoundError(f"modèle absent : {poids}")
    if not data.exists():
        raise FileNotFoundError(f"jeu de données absent : {data}")

    modele = YOLO(str(poids))
    r = modele.val(data=str(data), split=spec["split"], imgsz=imgsz,
                   device="cpu", batch=4, plots=False, verbose=False)
    # `ap_class_index` référence les classes du *jeu de données*. Si le modèle
    # déployé a une autre taxonomie (mauvais fichier .pt, ré-entraînement sur
    # d'autres classes), certains index n'existent pas dans `modele.names` : on
    # les ignore ici, la comparaison signalera les classes attendues manquantes.
    # Sans cette précaution, le script planterait au lieu de rapporter le
    # problème — or c'est précisément un des cas qu'il doit détecter.
    par_classe = {}
    for i, ci in enumerate(r.ap_class_index):
        nom_classe = modele.names.get(int(ci))
        if nom_classe is not None:
            par_classe[nom_classe] = round(float(r.box.ap50[i]), 4)
    return {"classes": par_classe, "mAP50": round(float(r.box.map50), 4)}


def comparer(nom: str, spec: dict, mesure: dict, tolerance: float) -> list[str]:
    """Renvoie la liste des régressions constatées (vide si tout va bien)."""
    echecs = []
    for classe, attendu in spec["classes"].items():
        obtenu = mesure["classes"].get(classe)
        if obtenu is None:
            echecs.append(f"{nom}/{classe} : classe absente du modèle mesuré "
                          f"(taxonomie modifiée ?)")
            continue
        ecart = obtenu - attendu
        etat = "OK " if ecart >= -tolerance else "ECHEC"
        print(f"  {etat} {classe:18} attendu {attendu:.4f}  obtenu {obtenu:.4f}  ({ecart:+.4f})")
        if ecart < -tolerance:
            echecs.append(f"{nom}/{classe} : AP@50 {attendu:.4f} -> {obtenu:.4f} "
                          f"({ecart:+.4f}, tolérance {tolerance})")
    return echecs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modele", help="ne verifier qu'un modele (cle du fichier de reference)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--maj", action="store_true",
                    help="ecrit les valeurs mesurees comme nouvelle reference")
    args = ap.parse_args()

    ref = json.loads(REFERENCE.read_text(encoding="utf-8"))
    tolerance = ref["_documentation"]["tolerance_ap50"]
    modeles = ref["modeles"]
    if args.modele:
        if args.modele not in modeles:
            print(f"Modele inconnu : {args.modele}. Disponibles : {', '.join(modeles)}")
            return 2
        modeles = {args.modele: modeles[args.modele]}

    echecs_globaux, indisponibles = [], []
    for nom, spec in modeles.items():
        print(f"\n=== {nom} ({spec['poids']}, split {spec['split']}) ===")
        try:
            mesure = mesurer(spec, args.imgsz)
        except FileNotFoundError as e:
            # Un jeu de donnees absent n'est pas une regression : les datasets ne
            # sont pas versionnes (trop volumineux). On le signale sans echouer.
            print(f"  IGNORE : {e}")
            indisponibles.append(nom)
            continue

        if args.maj:
            spec["classes"] = {c: mesure["classes"][c] for c in spec["classes"]
                               if c in mesure["classes"]}
            spec["mAP50"] = mesure["mAP50"]
            print(f"  reference mise a jour : {spec['classes']}")
            continue

        echecs_globaux += comparer(nom, spec, mesure, tolerance)
        ecart_map = mesure["mAP50"] - spec["mAP50"]
        print(f"  mAP50 global : attendu {spec['mAP50']:.4f}  obtenu {mesure['mAP50']:.4f}  ({ecart_map:+.4f})")

    if args.maj:
        REFERENCE.write_text(json.dumps(ref, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\n-> {REFERENCE}")
        return 0

    print("\n" + "=" * 60)
    if indisponibles:
        print(f"Non verifies (donnees absentes) : {', '.join(indisponibles)}")
    if echecs_globaux:
        print(f"REGRESSION DETECTEE ({len(echecs_globaux)}) :")
        for e in echecs_globaux:
            print(f"  - {e}")
        return 1
    print("Aucune regression : tous les modeles verifies tiennent leur reference.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
