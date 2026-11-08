"""Motif d'une violation EPI obligatoire : l'absence constatée prime (2026-09-27).

Défaut constaté : pour un EPI obligatoire (casque, gilet) jamais vu, le moteur
posait `jamais_confirme` et masquait `absence_detectee`, même quand une classe
négative (tête nue) était détectée avec une forte confiance. La plateforme
écarte `jamais_confirme` (`motif_minimum = absence_detectee`, posé par
`import_manifeste_modeles`) : sur flux continu, aucune alerte casque ni gilet
ne lui parvenait. Ces tests n'ouvrent aucun poids : la sortie des modèles est
simulée à l'endroit exact où `process` la lit.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "improvements"))
sys.path.insert(0, str(ROOT / "surveillance_suite"))

import numpy as np  # noqa: E402

import ppe_taxonomy as tax  # noqa: E402
from unified_surveillance import AnalyseurEPI  # noqa: E402

PERSONNE = (400, 200, 700, 800)
TETE = (500, 210, 600, 300)       # dans le premier cinquième de la personne
TORSE = (440, 320, 660, 520)


class _Boite:
    def __init__(self, cls, conf, box):
        self.cls, self.conf, self.xyxy = cls, conf, [box]


class _Modele:
    def __init__(self, names):
        self.names = names


CASQUE = _Modele({0: "Hardhat", 1: "NO-Hardhat"})
MASQUE_GILET = _Modele({0: "Mask", 1: "NO-Mask", 2: "Safety Vest", 3: "NO-Safety Vest"})


def _analyseur(sorties):
    """AnalyseurEPI sans poids : `_predire` rend, pour chaque modèle, les boîtes
    données par `sorties` (les mêmes à chaque image, comme une caméra fixe)."""
    a = AnalyseurEPI.__new__(AnalyseurEPI)
    a._cascade = [("epi_casque.pt", CASQUE), ("masque_gilet.pt", MASQUE_GILET)]
    a._pool_cascade = None
    a._seuil_bas = 0.01
    a.imgsz, a.every, a.ancrage = 480, 1, True
    a.rejetes_ancrage, a.couverture = 0, {}
    a.hist_n, a.hist_k = 12, 4                       # valeurs de production
    a._hist, a._vu_le, a.dets, a.violations = {}, {}, [], {}
    a._predire = lambda model, frame: sorties.get(model, [])
    return a


def _motifs(a, images=8, apres=0):
    """Motifs émis, image par image ; `apres` ignore les premières images
    (le temps que l'historique de `hist_k` images se remplisse)."""
    motifs = {}
    for i in range(images):
        evs = a.process(np.zeros((1000, 1000, 3), np.uint8),
                        {"t": i * 0.5, "frame": i, "personnes": [(7, PERSONNE)]})
        if i < apres:
            continue
        for ev in evs:
            motifs.setdefault(ev.extra["epi"], set()).add(ev.extra["motif"])
    return motifs


def test_une_tete_nue_detectee_en_continu_produit_absence_detectee():
    motifs = _motifs(_analyseur({CASQUE: [_Boite(1, 0.9, TETE)]}))
    assert motifs["casque"] == {"absence_detectee"}, motifs


def test_un_torse_sans_gilet_detecte_produit_absence_detectee():
    motifs = _motifs(_analyseur({CASQUE: [_Boite(0, 0.9, TETE)],
                                 MASQUE_GILET: [_Boite(3, 0.9, TORSE)]}))
    assert motifs["gilet"] == {"absence_detectee"}, motifs


def test_rien_vu_reste_jamais_confirme():
    """Sans classe négative, on n'a rien constaté : le motif faible est le bon."""
    motifs = _motifs(_analyseur({}))
    assert motifs["casque"] == {"jamais_confirme"}
    assert motifs["gilet"] == {"jamais_confirme"}


def test_un_casque_porte_ne_produit_plus_de_violation_une_fois_confirme():
    """Un casque porté doit être vu `hist_k` fois avant d'être confirmé : pendant
    ce temps `jamais_confirme` est le comportement voulu. Ensuite, plus rien."""
    a = _analyseur({CASQUE: [_Boite(0, 0.9, TETE)]})
    assert "casque" not in _motifs(a, images=10, apres=a.hist_k)


def test_les_concepts_restent_canoniques():
    assert {"casque", "gilet"} <= set(tax.EPI_CANONIQUES)
    assert tax.EPI_OBLIGATOIRES["casque"] and tax.EPI_OBLIGATOIRES["gilet"]
