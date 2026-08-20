#!/usr/bin/env python3
"""Vérifie que les poids en place sont bien ceux qui ont été mesurés.

Le problème
-----------
Les `.pt` pèsent 142 Mo et sont exclus du dépôt — décision saine, git n'est pas
un stockage de modèles. Mais rien ne les remplaçait : ni registre, ni empreinte,
ni version. Il en découlait trois impossibilités, toutes constatées lors de
l'audit du 2026-08-20 :

- une image de conteneur ne peut pas être reconstruite à l'identique, le
  `Dockerfile` reconnaissant lui-même que le build « échoue sur un clone frais
  tant que les .pt n'ont pas été copiés manuellement » ;
- on ne peut pas prouver quel poids tourne en production, donc pas davantage
  affirmer que les chiffres publiés le concernent ;
- un retour arrière dépend d'une copie de sauvegarde que l'exploitant doit
  avoir pensé à faire.

Ce que fait ce script
---------------------
`--figer` enregistre l'empreinte SHA-256 de chaque modèle en place, avec sa
taille et ses classes lues dans le fichier lui-même. `--verifier` compare l'état
courant au manifeste et sort en code 1 à la moindre divergence : c'est ce que
doit exécuter la chaîne de déploiement avant de construire une image.

Le manifeste, lui, EST versionné : c'est un fichier texte de quelques lignes.
C'est lui qui fait le lien entre un poids binaire et les mesures publiées.

    python outils/verifier_modeles.py --verifier
    python outils/verifier_modeles.py --figer      # apres un remplacement assume
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
MANIFESTE = RACINE / "docs/manifeste_modeles.json"

# Modèles dont la présence est exigée pour que le moteur soit complet. Un
# modèle absent de cette liste mais présent sur disque est signalé, sans être
# une erreur : c'est le cas des poids de secours (`*_pre_*.pt`).
ATTENDUS = [
    "ppe_detection/models/ppe_detector.pt",
    "ppe_detection/models/masque_gilet.pt",
    "ppe_detection/models/epi_casque.pt",
    "ppe_detection/models/epi_gants_lunettes.pt",
    "ppe_detection/models/epi_chaussures.pt",
    "surveillance_suite/models/fire_smoke.pt",
    "surveillance_suite/models/fall_detector.pt",
    "surveillance_suite/models/license_plate.pt",
    # Chargé par le moteur unifié (`AnalyseurPorte`) et absent de cette liste
    # jusqu'au 2026-08-20 : il decidait d'un evenement publie a la plateforme
    # sans etre rattache a quoi que ce soit (audit, point I3).
    "surveillance_suite/models/door_classifier.pt",
]


def empreinte(chemin: Path) -> str:
    """SHA-256, lu par blocs : ces fichiers font 50 Mo pièce."""
    h = hashlib.sha256()
    with open(chemin, "rb") as f:
        for bloc in iter(lambda: f.read(1024 * 1024), b""):
            h.update(bloc)
    return h.hexdigest()


def classes_du_modele(chemin: Path) -> list[str] | None:
    """Noms de classes lus dans le poids lui-même, pas dans la documentation.

    Une taxonomie modifiée par un ré-entraînement est ainsi visible au
    déploiement, et non découverte en production.
    """
    try:
        from ultralytics import YOLO
        return list(YOLO(str(chemin)).names.values())
    except Exception:
        return None


def decrire(rel: str, avec_classes: bool) -> dict | None:
    chemin = RACINE / rel
    if not chemin.is_file():
        return None
    d = {"sha256": empreinte(chemin), "octets": chemin.stat().st_size}
    if avec_classes:
        noms = classes_du_modele(chemin)
        if noms is not None:
            d["classes"] = noms
    return d


def figer(avec_classes: bool) -> int:
    manifeste = {
        "_documentation": {
            "role": "Empreinte des poids en production. Fait le lien entre un "
                    "binaire et les mesures publiees dans models_calsse.txt.",
            "quand_mettre_a_jour": "Apres un remplacement de modele ASSUME et "
                                   "mesure. Jamais pour faire passer une "
                                   "verification en echec : c'est exactement la "
                                   "substitution que ce fichier existe pour "
                                   "detecter.",
            "verifier": "python outils/verifier_modeles.py --verifier",
        },
        "fige_le": str(date.today()),
        "modeles": {},
    }
    absents = []
    for rel in ATTENDUS:
        d = decrire(rel, avec_classes)
        if d is None:
            absents.append(rel)
            continue
        manifeste["modeles"][rel] = d
        print(f"  {rel:<52} {d['sha256'][:16]}…  {d['octets'] / 1e6:.1f} Mo")

    if absents:
        print("\nabsents, non figes :", file=sys.stderr)
        for rel in absents:
            print(f"  {rel}", file=sys.stderr)

    MANIFESTE.parent.mkdir(parents=True, exist_ok=True)
    MANIFESTE.write_text(json.dumps(manifeste, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    print(f"\n-> {MANIFESTE.relative_to(RACINE)}  ({len(manifeste['modeles'])} modeles)")
    return 0


def verifier(avec_classes: bool) -> int:
    if not MANIFESTE.is_file():
        print(f"manifeste absent : {MANIFESTE.relative_to(RACINE)}\n"
              "Le figer d'abord : python outils/verifier_modeles.py --figer",
              file=sys.stderr)
        return 1

    attendu = json.loads(MANIFESTE.read_text(encoding="utf-8"))["modeles"]
    problemes: list[str] = []

    for rel, ref in attendu.items():
        chemin = RACINE / rel
        if not chemin.is_file():
            problemes.append(f"{rel} : ABSENT")
            continue
        obtenu = decrire(rel, avec_classes)
        if obtenu["sha256"] != ref["sha256"]:
            problemes.append(
                f"{rel} : empreinte differente\n"
                f"    attendu {ref['sha256'][:24]}…  ({ref['octets'] / 1e6:.1f} Mo)\n"
                f"    obtenu  {obtenu['sha256'][:24]}…  ({obtenu['octets'] / 1e6:.1f} Mo)")
            continue
        if avec_classes and "classes" in ref and obtenu.get("classes") != ref["classes"]:
            problemes.append(f"{rel} : memes octets mais classes differentes -- "
                             "manifeste incoherent, le refiger")
            continue
        print(f"  OK  {rel}")

    for rel in ATTENDUS:
        if rel not in attendu:
            problemes.append(f"{rel} : attendu par le moteur, absent du manifeste")

    if problemes:
        print("\nVERIFICATION EN ECHEC :", file=sys.stderr)
        for p in problemes:
            print(f"  /!\\ {p}", file=sys.stderr)
        print("\nUn poids qui differe du manifeste n'est pas celui sur lequel les\n"
              "mesures publiees ont ete obtenues. Ne pas deployer avant d'avoir\n"
              "soit restaure le bon fichier, soit mesure et refige le nouveau.",
              file=sys.stderr)
        return 1

    print(f"\n{len(attendu)} modeles conformes au manifeste "
          f"(fige le {json.loads(MANIFESTE.read_text(encoding='utf-8'))['fige_le']}).")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--figer", action="store_true",
                   help="enregistre l'empreinte des poids en place")
    g.add_argument("--verifier", action="store_true",
                   help="compare les poids en place au manifeste ; code 1 si divergence")
    ap.add_argument("--sans-classes", action="store_true",
                    help="n'ouvre pas les modeles (plus rapide, mais ne detecte pas "
                         "un changement de taxonomie a empreinte identique)")
    a = ap.parse_args()
    return figer(not a.sans_classes) if a.figer else verifier(not a.sans_classes)


if __name__ == "__main__":
    raise SystemExit(main())
