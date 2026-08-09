#!/usr/bin/env python3
"""Tests de la logique métier du moteur — sans modèle, sans GPU, en quelques secondes.

Cette logique décide *ce qui est signalé* à partir des sorties brutes des
modèles : lissage temporel, association EPI ↔ personne, déduplication
inter-modèles, anti-répétition. Une erreur ici produit des alertes fausses ou
manquantes aussi sûrement qu'une erreur de modèle, et elle est bien plus facile
à introduire au fil des modifications.

Lancement : python -m pytest tests/ -q
"""

import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "improvements"))
sys.path.insert(0, str(ROOT / "surveillance_suite"))
sys.path.insert(0, str(ROOT / "surveillance_suite" / "modules"))

import ppe_taxonomy as tax  # noqa: E402
from unified_surveillance import (  # noqa: E402
    BusEvenements, ConfirmationTemporelle, Evenement, Sortie, CaptureRobuste,
)


class SortieMemoire(Sortie):
    """Sortie de test : conserve les évènements au lieu de les transmettre."""

    def __init__(self):
        self.recus = []

    def emettre(self, ev):
        self.recus.append(ev)


# ── Taxonomie EPI : correspondance et fusion inter-modèles ───────────────────

def test_traduction_respecte_le_seuil_propre_a_la_classe():
    """Chaque classe a son seuil : `Vest` (M2) accepte 0.06, `Safety Vest` (M1) non.

    C'est tout l'intérêt de la table : les deux modèles ne sont pas calibrés
    pareil, un seuil unique en ignorerait un.
    """
    assert tax.traduire("best_gloves.pt", "Vest", 0.06, (0, 0, 10, 10)) is not None
    assert tax.traduire("best.pt", "Safety Vest", 0.06, (0, 0, 10, 10)) is None
    assert tax.traduire("best.pt", "Safety Vest", 0.20, (0, 0, 10, 10)) is not None


def test_traduction_classe_inconnue_est_ignoree():
    assert tax.traduire("best.pt", "ClasseQuiNExistePas", 0.9, (0, 0, 10, 10)) is None


def test_fusion_supprime_le_doublon_inter_modeles():
    """Un même gilet vu par M1 et M2 ne doit produire qu'une détection, celle de M1."""
    box = (10, 10, 50, 50)
    m1 = tax.traduire("best.pt", "Safety Vest", 0.5, box)
    m2 = tax.traduire("best_gloves.pt", "Vest", 0.9, (12, 12, 52, 52))  # meme objet, decale
    fusion = tax.fusionner([m1, m2])
    assert len(fusion) == 1
    # M1 fait autorite meme avec une confiance plus faible : lui seul sait dire
    # l'absence, et ses scores sont calibres sur une plage utile.
    assert fusion[0].modele == "best.pt"


def test_fusion_conserve_les_chaussures_apport_unique_de_m2():
    """`safety_shoe` n'a pas d'équivalent dans M1 : jamais dédupliquée."""
    shoe = tax.traduire("best_gloves.pt", "safety_shoe", 0.5, (10, 10, 50, 50))
    casque = tax.traduire("best.pt", "Hardhat", 0.5, (10, 10, 50, 50))
    fusion = tax.fusionner([casque, shoe])
    assert {d.epi for d in fusion} == {"casque", "chaussures"}


def test_fusion_garde_deux_objets_distincts_du_meme_concept():
    """Deux casques éloignés sont deux casques, pas un doublon."""
    a = tax.traduire("best.pt", "Hardhat", 0.5, (0, 0, 20, 20))
    b = tax.traduire("best.pt", "Hardhat", 0.5, (200, 200, 220, 220))
    assert len(tax.fusionner([a, b])) == 2


def test_coherence_detecte_une_taxonomie_divergente():
    """Un modèle ré-entraîné avec d'autres classes doit être signalé, pas ignoré."""
    erreurs = tax.verifier_coherence({0: "ClasseInventee"}, {0: "helmet"})
    assert erreurs, "une divergence de taxonomie doit produire une erreur explicite"


# ── Confirmation temporelle (chute, feu) ─────────────────────────────────────

def test_confirmation_rejette_une_detection_isolee():
    """Le cas qui motive ce mécanisme : une image bruitée ne doit pas alerter."""
    c = ConfirmationTemporelle(fenetre=4, minimum=2)
    assert c.confirmer("zone", True) is False


def test_confirmation_accepte_une_detection_persistante():
    c = ConfirmationTemporelle(fenetre=4, minimum=2)
    c.confirmer("zone", True)
    assert c.confirmer("zone", True) is True


def test_confirmation_isole_les_zones():
    """Deux zones distinctes ne doivent pas s'additionner pour franchir le seuil."""
    c = ConfirmationTemporelle(fenetre=4, minimum=2)
    assert c.confirmer("zone_a", True) is False
    assert c.confirmer("zone_b", True) is False


def test_confirmation_oublie_au_dela_de_la_fenetre():
    """Deux détections trop espacées ne se cumulent pas."""
    c = ConfirmationTemporelle(fenetre=2, minimum=2)
    c.confirmer("z", True)
    c.confirmer("z", False)
    assert c.confirmer("z", False) is False


def test_purge_retire_les_cles_inactives():
    c = ConfirmationTemporelle(fenetre=2, minimum=2)
    c.confirmer("z", False)
    c.purger()
    assert "z" not in c._hist


# ── Bus d'évènements : anti-répétition et robustesse des sorties ─────────────

def test_anti_repetition_bloque_le_doublon_immediat():
    """Une condition qui persiste ne doit pas noyer le consommateur."""
    s = SortieMemoire()
    bus = BusEvenements([s], anti_repetition_s=3.0)
    t = time.time()
    assert bus.publier(Evenement(t, 1, "epi", "violation_epi", "P1 — SANS CASQUE !")) is True
    assert bus.publier(Evenement(t + 0.1, 2, "epi", "violation_epi", "P1 — SANS CASQUE !")) is False
    assert len(s.recus) == 1


def test_anti_repetition_laisse_passer_apres_le_delai():
    s = SortieMemoire()
    bus = BusEvenements([s], anti_repetition_s=1.0)
    t = time.time()
    bus.publier(Evenement(t, 1, "epi", "violation_epi", "P1 — SANS CASQUE !"))
    assert bus.publier(Evenement(t + 1.5, 9, "epi", "violation_epi", "P1 — SANS CASQUE !")) is True


def test_anti_repetition_distingue_les_personnes():
    """Deux personnes en infraction produisent bien deux évènements."""
    s = SortieMemoire()
    bus = BusEvenements([s], anti_repetition_s=3.0)
    t = time.time()
    bus.publier(Evenement(t, 1, "epi", "violation_epi", "Personne #1 — SANS CASQUE !"))
    bus.publier(Evenement(t, 1, "epi", "violation_epi", "Personne #2 — SANS CASQUE !"))
    assert len(s.recus) == 2


def test_une_sortie_defaillante_ne_bloque_pas_les_autres():
    """Une panne de transport ne doit jamais interrompre la détection."""
    class SortieCassee(Sortie):
        def emettre(self, ev):
            raise RuntimeError("consommateur injoignable")

    bonne = SortieMemoire()
    bus = BusEvenements([SortieCassee(), bonne], anti_repetition_s=0)
    assert bus.publier(Evenement(time.time(), 1, "feu", "fire", "FIRE détecté")) is True
    assert len(bonne.recus) == 1


# ── Capture résiliente ───────────────────────────────────────────────────────

class FauxCap:
    """VideoCapture minimal : échoue tant que `echecs_restants` n'est pas épuisé."""

    def __init__(self, echecs_restants=0):
        self.echecs_restants = echecs_restants

    def read(self):
        if self.echecs_restants > 0:
            self.echecs_restants -= 1
            return False, None
        return True, np.zeros((8, 8, 3), np.uint8)

    def isOpened(self):
        return True

    def release(self):
        pass

    def set(self, *a):
        pass


def _capture_test(bus):
    """CaptureRobuste sans aucun accès réseau ni fichier.

    `__new__` court-circuite le constructeur (qui ouvrirait une vraie source), et
    `_ouvrir` est remplacé par une réouverture simulée : sans cela, la boucle de
    reconnexion appellerait `cv2.VideoCapture("rtsp://test")` et bloquerait le
    test sur une résolution réseau.
    """
    cap = CaptureRobuste.__new__(CaptureRobuste)
    cap.source, cap.bus, cap.boucler = "rtsp://test", bus, False
    cap.echecs_avant_perte, cap.est_fichier = 2, False
    cap.connecte, cap._echecs, cap.reconnexions = True, 0, 0
    cap.DELAI_MIN_S = cap.DELAI_MAX_S = 0.01
    cap._delai = 0.01

    def _reouvrir_simule():
        cap._echecs = 0
        cap._delai = cap.DELAI_MIN_S
        return True

    cap._ouvrir = _reouvrir_simule
    return cap


def test_perte_et_reprise_du_flux_sont_signalees():
    """Sans ces deux évènements, une caméra tombée ressemble à un site calme."""
    s = SortieMemoire()
    bus = BusEvenements([s], anti_repetition_s=0)
    cap = _capture_test(bus)
    cap.cap = FauxCap(echecs_restants=3)
    assert cap.lire() is not None  # reconnexion puis reprise
    types = [e.type for e in s.recus]
    assert "flux_perdu" in types and "flux_repris" in types
    assert cap.reconnexions == 1


def test_flux_sain_ne_produit_aucun_evenement_technique():
    s = SortieMemoire()
    cap = _capture_test(BusEvenements([s], anti_repetition_s=0))
    cap.cap = FauxCap()
    assert cap.lire() is not None
    assert s.recus == []


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
