#!/usr/bin/env python3
"""Régénère `models_calsse.txt` : les classes de chaque modèle et leurs scores.

Pourquoi un script plutôt qu'un fichier écrit à la main
-------------------------------------------------------
Les scores changent à chaque ré-entraînement — quatre en cinq jours pour le seul
modèle EPI. Un fichier tenu à la main serait périmé le lendemain et, pire,
donnerait l'illusion d'être à jour. C'est exactement ce qui est arrivé au
registre des modèles : il annonçait encore les métriques d'un entraînement
précédent des semaines après son remplacement.

Ici, deux sources font foi et rien d'autre :

- les **fichiers `.pt` eux-mêmes** pour les noms de classes, lus à chaque
  exécution. Une taxonomie modifiée par un ré-entraînement apparaît donc
  immédiatement, au lieu d'être découverte en production ;
- `reports/v3_results/scores_par_classe.json` pour les mesures, avec pour
  chacune le jeu et la date — un score sans sa provenance ne veut rien dire.

    python improvements/generer_classes.py

À relancer après chaque mesure. La commande est sans effet de bord : elle ne
fait que réécrire le fichier de sortie.
"""

from __future__ import annotations

import json
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
SORTIE = RACINE / "models_calsse.txt"
SCORES = RACINE / "reports/v3_results/scores_par_classe.json"

# Ordre d'affichage et description. Le dictionnaire porte l'intention
# éditoriale ; les chiffres et les noms de classes viennent des fichiers.
MODELES = [
    ("ppe_detection/models/ppe_detector.pt", "EPI — modèle principal",
     "C'est lui qui juge la conformité EPI. Les six classes `NO-` signalent une "
     "INFRACTION : ce sont les plus critiques, et les plus faibles du parc."),
    ("ppe_detection/models/masque_gilet.pt", "EPI — masque et gilet dédiés",
     "Troisième modèle de la cascade, prioritaire sur `ppe_detector.pt` pour "
     "CES 4 CLASSES UNIQUEMENT (`improvements/ppe_taxonomy.py`). "
     "`ppe_detector.pt` reste le filet de secours si ce modèle est absent."),
    ("ppe_detection/models/epi_casque.pt", "EPI — casque dédié",
     "Prioritaire sur `ppe_detector.pt` pour CES 2 CLASSES UNIQUEMENT "
     "(`improvements/ppe_taxonomy.py`, M4). Il corrige le plus gros déficit "
     "mesuré du parc : `ppe_detector.pt` seul ne voyait le casque que dans 65 % "
     "des scènes annotées, et son absence dans 72 %."),
    ("ppe_detection/models/epi_gants_lunettes.pt", "EPI — gants et lunettes dédiés",
     "Prioritaire sur `ppe_detector.pt` pour CES 4 CLASSES UNIQUEMENT "
     "(`improvements/ppe_taxonomy.py`, M5). Retenu parce qu'il bat "
     "`ppe_detector.pt` sur les quatre, qui y était déjà bon : un candidat "
     "antérieur perdant sur trois classes avait été rejeté."),
    ("ppe_detection/models/epi_chaussures.pt", "EPI — chaussures de sécurité dédiées",
     "Seul modèle du parc couvrant ce concept avec une classe NÉGATIVE, donc "
     "capable de signaler l'infraction et pas seulement de confirmer le port. "
     "Ses deux seuils sont volontairement inégaux (`improvements/ppe_taxonomy.py`) : "
     "affirmer à tort une conformité masque une infraction, alors qu'une fausse "
     "alerte ne coûte qu'une vérification. Il DÉPEND de l'ancrage à la personne."),
    ("ppe_detection/models/ppe_complement.pt", "EPI — modèle complémentaire",
     "Second modèle de la cascade. Il n'a AUCUNE classe négative : il ne peut "
     "donc jamais signaler une non-conformité, seulement confirmer un "
     "équipement présent. `safety_shoe` est son seul apport unique."),
    ("surveillance_suite/models/fire_smoke.pt", "Feu et fumée", ""),
    ("surveillance_suite/models/fall_detector.pt", "Chute", ""),
    ("surveillance_suite/models/license_plate.pt", "Plaques d'immatriculation",
     "Localisation seule. La lecture des caractères est faite ensuite par OCR "
     "(`surveillance_suite/modules/module_lpr.py`)."),
    ("surveillance_suite/models/door_classifier.pt", "État de porte",
     "Classifieur et non détecteur : il juge toute l'image, sans rectangle."),
]

TRADUCTIONS = {
    "Fall-Detected": "personne au sol", "Gloves": "gants portés",
    "Goggles": "lunettes portées", "Hardhat": "casque porté",
    "Ladder": "échelle", "Mask": "masque porté", "NO-Gloves": "SANS gants",
    "NO-Goggles": "SANS lunettes", "NO-Hardhat": "SANS casque",
    "NO-Mask": "SANS masque", "NO-Safety Vest": "SANS gilet",
    "Person": "personne", "Safety Cone": "cône de signalisation",
    "Safety Vest": "gilet porté", "Vest": "gilet", "goggles": "lunettes",
    "helmet": "casque", "mask": "masque", "safety_shoe": "chaussures de sécurité",
    "fire": "flammes visibles", "smoke": "fumée, panache",
    "falling": "personne au sol", "stand": "personne debout",
    "plate": "plaque d'immatriculation", "Closed": "fermée", "Open": "ouverte",
    "Semi": "entrouverte",
}

EVENEMENTS = [
    ("violation_epi", "un EPI obligatoire manque à une personne"),
    ("chute", "personne au sol"),
    ("fire", "départ de feu"),
    ("smoke", "fumée"),
    ("objet_abandonne", "bagage immobile, sans personne à proximité"),
    ("foule", "seuil d'effectif ou de densité franchi"),
    ("foule_terminee", "retour sous le seuil"),
    ("franchissement", "un objet suivi a franchi la ligne virtuelle"),
    ("plaque", "plaque lue"),
    ("porte", "changement d'état de la porte"),
    ("flux_perdu", "la caméra n'est plus lisible"),
    ("flux_repris", "la caméra est revenue"),
]


def classes_du_modele(chemin: Path) -> dict[int, str] | None:
    """Noms de classes lus dans le `.pt`, pas dans la documentation.

    `model` est parfois vide quand un entraînement a été tué avant sa
    finalisation ; les poids sont alors dans `ema`, ce qu'Ultralytics gère au
    chargement. On applique la même règle ici.
    """
    try:
        import torch
        d = torch.load(chemin, map_location="cpu", weights_only=False)
    except Exception:
        return None
    mod = (d.get("model") or d.get("ema")) if isinstance(d, dict) else d
    noms = getattr(mod, "names", None)
    if noms is None and isinstance(d, dict):
        noms = d.get("names")
    return noms


def colonne_score(mesures: dict) -> str:
    """Le score, ou une raison explicite de son absence."""
    if not mesures:
        return "non mesurée"
    bouts = []
    if "scene" in mesures:
        bouts.append(f"{mesures['scene'] * 100:.0f} % détecté")
    if "ap50" in mesures:
        bouts.append(f"AP {mesures['ap50']:.3f}")
    if "ap50_jeu_propre" in mesures:
        bouts.append(f"AP {mesures['ap50_jeu_propre']:.3f} (jeu propre)")
    return "  ·  ".join(bouts)


def main() -> int:
    scores = json.loads(SCORES.read_text(encoding="utf-8"))
    lignes = [
        "CLASSES DÉTECTÉES PAR LES MODÈLES",
        "=" * 74,
        "",
        "Fichier RÉGÉNÉRÉ — ne pas éditer à la main.",
        "    python improvements/generer_classes.py",
        "",
        "Les noms de classes sont lus dans les fichiers .pt eux-mêmes ; les scores",
        "viennent de reports/v3_results/scores_par_classe.json, avec leur date et",
        "le jeu sur lequel ils ont été mesurés.",
        "",
        "DEUX INDICATEURS, à ne jamais confondre :",
        "",
        "  « détecté »  proportion des scènes où le modèle voit l'objet.",
        "               C'est l'indicateur d'EXPLOITATION, celui qui décide.",
        "  « AP »       précision du rectangle (AP@50). Utile pour comparer deux",
        "               entraînements du même modèle, trompeur pour juger sa valeur.",
        "",
        "L'écart peut être énorme : la fumée sort à AP 0.407 et détecte pourtant",
        "96,7 % des scènes. Ne jamais juger un modèle sur son AP seule.",
        "",
    ]

    for rel, titre, note in MODELES:
        chemin = RACINE / rel
        noms = classes_du_modele(chemin)
        if noms is None:
            lignes += ["", "─" * 74, f"{titre}  —  {rel}", "", "  MODÈLE ABSENT OU ILLISIBLE", ""]
            continue
        info = scores.get(chemin.name, {})
        mesures = info.get("classes", {})

        lignes += ["", "─" * 74, f"{titre}  —  {len(noms)} classes", f"{rel}", ""]
        if info.get("date"):
            jeux = " / ".join(filter(None, [info.get("jeu_ap50"), info.get("jeu_scene")]))
            lignes.append(f"  mesuré le {info['date']} sur {jeux}")
        else:
            lignes.append("  AUCUNE mesure de référence")
        if info.get("mAP50") is not None:
            lignes.append(f"  mAP@50 global : {info['mAP50']}")
        lignes.append("")

        for i, nom in sorted(noms.items()):
            desc = TRADUCTIONS.get(nom, "")
            lignes.append(f"  {i:>2}  {nom:<16} {desc:<26} {colonne_score(mesures.get(nom, {}))}")

        if note:
            lignes += ["", "  " + note.replace(". ", ".\n  ")]
        if info.get("note"):
            lignes += ["", "  " + info["note"].replace(". ", ".\n  ")]
        lignes.append("")

    lignes += [
        "", "═" * 74,
        "ÉVÈNEMENTS PRODUITS PAR LE MOTEUR",
        "═" * 74,
        "",
        "Ce que la plateforme reçoit réellement. Ce ne sont PAS les classes des",
        "modèles : un évènement résulte d'une décision, souvent prise sur",
        "plusieurs images.",
        "",
    ]
    for nom, desc in EVENEMENTS:
        lignes.append(f"  {nom:<20} {desc}")
    lignes += [
        "",
        "Chaque évènement porte camera_id, site_id et event_id, ainsi que son état",
        "(suspicion / probable / confirmé) et ses éléments de preuve.",
        "Format complet : docs/contrat_api.md",
        "",
    ]

    SORTIE.write_text("\n".join(lignes), encoding="utf-8")
    print(f"écrit : {SORTIE.relative_to(RACINE)}  ({len(lignes)} lignes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
