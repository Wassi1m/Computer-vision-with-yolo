#!/usr/bin/env python3
"""Cascade EPI appliquée à une image isolée, pour les services à la demande.

**Pourquoi ce fichier.** `unified_surveillance.AnalyseurEPI` applique trois
règles avant de conclure quoi que ce soit sur un EPI : les seuils de confiance
propres à chaque classe de chaque modèle (`ppe_taxonomy`), l'ancrage de l'EPI
sur une personne par confinement (`qualification.associer_a_personne`), puis un
lissage temporel. Les services d'analyse à la demande — l'essai sur image et la
webcam du navigateur — n'appliquaient aucune des trois : ils faisaient tourner
chaque modèle en direct, à un seuil unique, et affichaient tout ce qui sortait.

Les conséquences étaient exactement celles que la documentation annonce :

- `epi_chaussures.pt` **dépend** de l'ancrage. Sans lui, il déclenche sur 59 %
  des images au lieu de 1 % (`reports/v3_results/chaussures_ancrage.json`) —
  d'où des « chaussures de sécurité » annoncées sur une chevelure ou un visage.
- Les seuils calibrés vont de 0,03 à 0,25 selon la classe et le modèle. Un seuil
  unique de 0,35 imposé par-dessus **supprime des classes entières** : `Gloves`
  est calibré à 0,08 sur le modèle principal et 0,05 sur le complément, donc
  jamais affiché.

Le lissage temporel, lui, n'a pas de sens sur une image isolée : il reste au
moteur de flux. Ce module porte les deux règles qui s'appliquent à une image.

Il ne détecte rien et ne charge aucun modèle : il **écarte** ou **conserve** des
détections déjà faites. Une détection écartée est renvoyée avec son motif, pour
qu'un écran puisse l'expliquer au lieu de la faire disparaître sans un mot.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ICI = Path(__file__).resolve().parent
if str(_ICI) not in sys.path:
    sys.path.insert(0, str(_ICI))

import ppe_taxonomy as tax  # noqa: E402
import qualification as qual  # noqa: E402

#: Fichiers de poids que la table de correspondance sait interpréter. Un modèle
#: absent de cette liste (feu, chute, plaque, porte) n'est pas concerné : il
#: garde le seuil demandé par l'appelant.
MODELES_EPI = frozenset(tax.TABLES)


def seuil_minimal() -> float:
    """Seuil auquel interroger les modèles EPI.

    On interroge bas et on filtre ensuite classe par classe, comme le fait le
    moteur de flux : c'est la table de correspondance qui porte la calibration,
    pas l'appelant. Interroger directement au seuil final ferait disparaître les
    classes calibrées plus bas que lui.
    """
    return min(c.conf_min for table in tax.TABLES.values() for c in table.values())


def _motif_de_rejet(boite, personnes) -> str:
    if not personnes:
        return "aucune personne détectée dans l'image"
    meilleur = max((qual.confinement(boite, p) for p in personnes), default=0.0)
    if meilleur < 0.50:
        return f"hors de toute personne (recouvrement {meilleur:.0%})"
    return "hauteur invraisemblable sur la personne"


#: Au-delà de cette fraction de l'image, une seule personne occupe pratiquement
#: tout le cadre : c'est un gros plan, pas une scène de surveillance.
FRACTION_GROS_PLAN = 0.80


def avertissement_cadrage(personnes, dimensions):
    """Prévient quand l'ancrage anatomique ne peut pas faire son travail.

    Les plages anatomiques rapportent la position d'un EPI à la **hauteur de la
    personne** : un casque se trouve dans le premier tiers, des chaussures dans
    le dernier. Cela suppose que la boîte de la personne cadre effectivement une
    personne. Sur un gros plan de webcam, cette boîte couvre toute l'image : le
    bas du cadre devient « les pieds » et le haut « la tête », alors qu'on n'y
    voit qu'un visage. La règle continue de s'appliquer, mais elle ne mesure
    plus rien d'utile — dans un sens comme dans l'autre.

    On ne modifie rien pour autant : on le dit. Filtrer davantage sur une
    géométrie qu'on sait fausse ne ferait qu'ajouter une erreur à une autre.
    Renvoie une phrase, ou une chaîne vide si le cadrage est normal.
    """
    if not personnes or not dimensions:
        return ''
    largeur, hauteur = dimensions
    aire_image = float(largeur) * float(hauteur)
    if aire_image <= 0:
        return ''
    plus_grande = max((abs(p[2] - p[0]) * abs(p[3] - p[1]) for p in personnes),
                      default=0.0)
    if plus_grande / aire_image < FRACTION_GROS_PLAN:
        return ''
    return ("Cadrage en gros plan : une personne occupe presque toute l'image. "
            "Les modèles EPI ont été mesurés sur des scènes de chantier cadrées "
            "en pied, et l'ancrage anatomique n'a plus de repère fiable ici. "
            "Les résultats EPI de cette image ne préjugent pas de ceux d'une "
            "caméra de surveillance.")


def qualifier(detections, personnes):
    """Sépare les détections retenues de celles qu'une règle écarte.

    `detections` : dictionnaires portant au moins `fichier` (nom du fichier de
    poids), `classe`, `confiance` et `box`. `personnes` : boîtes des personnes
    trouvées dans la même image, telles que l'analyseur général les fournit.

    Renvoie `(retenues, ecartees)`. Les détections qui ne viennent pas d'un
    modèle EPI traversent sans être jugées : ce module n'a rien à dire sur un
    départ de feu.
    """
    personnes = [tuple(p) for p in (personnes or [])]
    retenues, ecartees = [], []

    for detection in detections:
        fichier = detection.get('fichier') or ''
        if fichier not in MODELES_EPI:
            retenues.append(detection)
            continue

        boite = tuple(detection['box']) if detection.get('box') else None
        traduite = tax.traduire(fichier, detection['classe'],
                                float(detection['confiance']), boite or (0, 0, 0, 0))

        if traduite is None:
            ecartees.append({**detection,
                             'motif': "sous le seuil calibré de cette classe"})
            continue

        # `Person`, `Ladder`, `Safety Cone`, `Fall-Detected` : le modèle EPI les
        # produit mais elles ne décrivent pas un équipement porté. Rien à ancrer.
        if traduite.epi is None:
            retenues.append(detection)
            continue

        if boite is None:
            # Sans boîte, l'ancrage est impossible. Laisser passer plutôt que
            # rejeter : on n'écarte pas ce qu'on ne sait pas juger.
            retenues.append({**detection, 'epi': traduite.epi, 'porte': traduite.porte})
            continue

        indice = qual.associer_a_personne(boite, personnes, epi=traduite.epi)
        if indice is None:
            ecartees.append({**detection, 'epi': traduite.epi,
                             'motif': _motif_de_rejet(boite, personnes)})
            continue

        retenues.append({**detection, 'epi': traduite.epi,
                         'porte': traduite.porte, 'personne': indice})

    return retenues, ecartees
