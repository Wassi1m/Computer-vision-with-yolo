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
    assert tax.traduire("ppe_complement.pt", "Vest", 0.06, (0, 0, 10, 10)) is not None
    assert tax.traduire("ppe_detector.pt", "Safety Vest", 0.06, (0, 0, 10, 10)) is None
    assert tax.traduire("ppe_detector.pt", "Safety Vest", 0.20, (0, 0, 10, 10)) is not None


def test_traduction_classe_inconnue_est_ignoree():
    assert tax.traduire("ppe_detector.pt", "ClasseQuiNExistePas", 0.9, (0, 0, 10, 10)) is None


def test_fusion_supprime_le_doublon_inter_modeles():
    """Un même gilet vu par M1 et M2 ne doit produire qu'une détection, celle de M1."""
    box = (10, 10, 50, 50)
    m1 = tax.traduire("ppe_detector.pt", "Safety Vest", 0.5, box)
    m2 = tax.traduire("ppe_complement.pt", "Vest", 0.9, (12, 12, 52, 52))  # meme objet, decale
    fusion = tax.fusionner([m1, m2])
    assert len(fusion) == 1
    # M1 fait autorite meme avec une confiance plus faible : lui seul sait dire
    # l'absence, et ses scores sont calibres sur une plage utile.
    assert fusion[0].modele == "ppe_detector.pt"


def test_fusion_conserve_les_chaussures_apport_unique_de_m2():
    """`safety_shoe` n'a pas d'équivalent dans M1 : jamais dédupliquée."""
    shoe = tax.traduire("ppe_complement.pt", "safety_shoe", 0.5, (10, 10, 50, 50))
    casque = tax.traduire("ppe_detector.pt", "Hardhat", 0.5, (10, 10, 50, 50))
    fusion = tax.fusionner([casque, shoe])
    assert {d.epi for d in fusion} == {"casque", "chaussures"}


def test_fusion_garde_deux_objets_distincts_du_meme_concept():
    """Deux casques éloignés sont deux casques, pas un doublon."""
    a = tax.traduire("ppe_detector.pt", "Hardhat", 0.5, (0, 0, 20, 20))
    b = tax.traduire("ppe_detector.pt", "Hardhat", 0.5, (200, 200, 220, 220))
    assert len(tax.fusionner([a, b])) == 2


def test_coherence_detecte_une_taxonomie_divergente():
    """Un modèle ré-entraîné avec d'autres classes doit être signalé, pas ignoré."""
    erreurs = tax.verifier_coherence({
        tax.M1_NOM: {0: "ClasseInventee"},
        tax.M2_NOM: {0: "helmet"},
    })
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


# ── Couche de qualification (plan v5) ────────────────────────────────────────

import qualification as q  # noqa: E402

PERSONNE = (100, 0, 200, 300)   # x1, y1, x2, y2 -- hauteur 300


def test_confinement_distingue_porte_de_flottant():
    """L'IoU ne sait pas juger un rapport « partie de » : un casque occupe ~3 %
    d'une personne, son IoU vaut donc ~0.03 qu'il soit porté ou non."""
    assert q.confinement((130, 10, 170, 40), PERSONNE) == pytest.approx(1.0)
    assert q.confinement((500, 500, 540, 530), PERSONNE) == 0.0


def test_epi_flottant_n_est_attribue_a_personne():
    """Le cas qui motive tout le module : sans rejet, un faux casque détecté
    n'importe où est attribué à la personne la moins éloignée et compte comme
    équipement porté — masquant une vraie infraction."""
    assert q.associer_a_personne((500, 500, 540, 530), [PERSONNE], epi="casque") is None


def test_casque_au_niveau_des_pieds_est_rejete():
    """Confinement parfait mais position aberrante : la géométrie seule ne suffit pas."""
    aux_pieds = (130, 260, 170, 290)
    assert q.confinement(aux_pieds, PERSONNE) == pytest.approx(1.0)
    assert q.associer_a_personne(aux_pieds, [PERSONNE], epi="casque") is None
    # le meme emplacement convient a des chaussures
    assert q.associer_a_personne(aux_pieds, [PERSONNE], epi="chaussures") == 0


def test_casque_sur_la_tete_est_accepte():
    assert q.associer_a_personne((130, 10, 170, 40), [PERSONNE], epi="casque") == 0


def test_epi_inconnu_n_est_jamais_rejete_par_l_anatomie():
    """Une règle ne doit pas rejeter ce qu'elle ne sait pas juger."""
    assert q.plausible_anatomiquement("brassard", (130, 260, 170, 290), PERSONNE)


def test_profil_ne_bascule_pas_sur_une_image_isolee():
    """Sans hystérésis, un phare ou un nuage reconfigurerait les seuils du moteur
    à chaque image, faisant apparaître et disparaître des évènements."""
    d = q.DetecteurProfil(stabilite=5)
    nuit = np.full((8, 8, 3), 30, dtype=np.uint8)
    for _ in range(4):
        assert d.mettre_a_jour(nuit).nom == "jour"
    assert d.mettre_a_jour(nuit).nom == "nuit"


def test_seuil_nocturne_est_plus_bas_et_confirmation_plus_stricte():
    """Le couple est indissociable : baisser le seuil sans durcir la
    confirmation noierait l'aval sous les fausses alertes."""
    jour, nuit = q.PROFILS["jour"], q.PROFILS["nuit"]
    assert nuit.seuil_detection < jour.seuil_detection
    assert nuit.minimum_confirmation > jour.minimum_confirmation


# ── Repli des labels feu/fumée ───────────────────────────────────────────────

def test_smoke_distant_est_replie_sur_l_evenement_smoke():
    """Un modèle à 3 classes émet `smoke_distant` ; sans ce repli, la fumée
    lointaine — celle qui permet la détection la plus précoce — ne déclencherait
    rien en aval. Le label d'origine reste exposé pour la couche v5."""
    import torch
    from module_fire_smoke import FireSmokeDetector

    class FauxBoites:
        cls = torch.tensor([0, 1, 2])
        xyxy = torch.tensor([[1, 2, 3, 4]] * 3)
        conf = torch.tensor([0.9, 0.9, 0.9])

        def __len__(self):
            return 3

    class FauxResultat:
        boxes = FauxBoites()

    detecteur = FireSmokeDetector.__new__(FireSmokeDetector)
    detecteur.conf = 0.4
    detecteur.model = type("M", (), {
        "names": {0: "fire", 1: "smoke", 2: "smoke_distant"},
        "predict": lambda self, *a, **k: [FauxResultat()],
    })()

    dets = detecteur.detect(None)
    assert [d["label"] for d in dets] == ["fire", "smoke", "smoke"]
    assert dets[2]["label_modele"] == "smoke_distant"


# ── Objet abandonné ──────────────────────────────────────────────────────────

from unified_surveillance import AnalyseurObjetAbandonne  # noqa: E402

SAC = (400, 400, 440, 440)          # diagonale ~57 px


def _ctx(t, objets, personnes=()):
    return {"t": t, "frame": int(t * 10), "objets": objets,
            "personnes": [(i, b) for i, b in enumerate(personnes)]}


def _sac(box=SAC, tid=1):
    return [{"box": box, "label": "backpack", "track_id": tid, "conf": 0.8}]


def test_objet_seul_declenche_apres_le_delai():
    """Le cœur du scénario : un sac posé, personne autour, 30 secondes."""
    a = AnalyseurObjetAbandonne(delai_s=30.0)
    assert a.process(None, _ctx(0.0, _sac())) == []      # premiere vue : ancrage
    assert a.process(None, _ctx(29.0, _sac())) == []     # pas encore
    evs = a.process(None, _ctx(30.5, _sac()))
    assert len(evs) == 1 and evs[0].type == "objet_abandonne"
    assert evs[0].extra["classe"] == "backpack"


def test_une_seule_alerte_par_objet():
    """Sans cela, un sac oublié produirait une alerte par image jusqu'à la fin
    du flux -- exactement le deluge que le plan v6 combat."""
    a = AnalyseurObjetAbandonne(delai_s=30.0)
    a.process(None, _ctx(0.0, _sac()))
    total = sum(len(a.process(None, _ctx(t, _sac()))) for t in (31.0, 32.0, 60.0, 120.0))
    assert total == 1


def test_personne_a_proximite_empeche_l_alerte():
    """Un sac au pied de son propriétaire n'est pas abandonné."""
    a = AnalyseurObjetAbandonne(delai_s=30.0)
    proprietaire = (420, 300, 460, 440)      # accole au sac
    for t in (0.0, 10.0, 20.0, 30.0, 40.0):
        evs = a.process(None, _ctx(t, _sac(), [proprietaire]))
    assert evs == []


def test_objet_deplace_n_est_pas_abandonne():
    """Un sac transporté bouge : le compteur doit repartir de zéro."""
    a = AnalyseurObjetAbandonne(delai_s=30.0)
    a.process(None, _ctx(0.0, _sac()))
    a.process(None, _ctx(20.0, _sac()))
    a.process(None, _ctx(25.0, _sac((600, 400, 640, 440))))   # deplace
    assert a.process(None, _ctx(40.0, _sac((600, 400, 640, 440)))) == []


def test_le_rayon_de_proximite_suit_la_taille_de_l_objet():
    """Seuils relatifs, pas en pixels absolus : à 50 m un sac fait quelques
    pixels, et un rayon fixe de 150 px y couvrirait la moitié de la scène."""
    a = AnalyseurObjetAbandonne(delai_s=30.0, proximite=2.0)
    petit, grand = (100, 100, 110, 110), (100, 100, 300, 300)
    personne = (140, 100, 160, 200)
    # meme personne, meme ecart : trop loin du petit objet, proche du grand
    assert a._distance_a_boite(a._centre(petit), personne) > 2.0 * a._diagonale(petit)
    assert a._distance_a_boite(a._centre(grand), personne) <= 2.0 * a._diagonale(grand)


def test_les_pistes_disparues_sont_oubliees():
    """Un flux de longue durée voit passer des milliers d'identifiants ; sans
    oubli, l'état croît indéfiniment (derive memoire, plan v6 §2.2)."""
    a = AnalyseurObjetAbandonne(delai_s=30.0, oubli_s=5.0)
    a.process(None, _ctx(0.0, _sac(tid=7)))
    assert 7 in a.etats
    a.process(None, _ctx(20.0, _sac(tid=8)))    # 7 a disparu depuis longtemps
    assert 7 not in a.etats


def test_classe_hors_perimetre_est_ignoree():
    """Restreindre le périmètre évite de noyer l'aval sous des objets sans
    enjeu de sécurité."""
    a = AnalyseurObjetAbandonne(delai_s=30.0)
    chaise = [{"box": SAC, "label": "chair", "track_id": 1, "conf": 0.9}]
    a.process(None, _ctx(0.0, chaise))
    assert a.process(None, _ctx(60.0, chaise)) == []
    assert a.etats == {}


# ── Calibration du sol et densité de foule ───────────────────────────────────

import calibration_sol as cal  # noqa: E402
from unified_surveillance import AnalyseurFoule  # noqa: E402

# Carre de 100 px representant 5 m de cote : echelle 0,05 m/px, aire 25 m2.
IMAGE_CARRE = [(0, 0), (100, 0), (100, 100), (0, 100)]
SOL_CARRE = [(0, 0), (5, 0), (5, 5), (0, 5)]


def test_projection_pixels_vers_metres():
    c = cal.CalibrationSol(IMAGE_CARRE, SOL_CARRE)
    x, y = c.vers_sol((50, 50))
    assert x == pytest.approx(2.5, abs=1e-3)
    assert y == pytest.approx(2.5, abs=1e-3)


def test_aire_reelle_d_une_zone():
    """L'aire en pixels n'a aucun rapport avec l'aire réelle : la perspective
    écrase le fond de la scène."""
    c = cal.CalibrationSol(IMAGE_CARRE, SOL_CARRE)
    assert c.aire_m2(IMAGE_CARRE) == pytest.approx(25.0, abs=1e-2)
    # moitie de l'image = moitie de la surface
    assert c.aire_m2([(0, 0), (50, 0), (50, 100), (0, 100)]) == pytest.approx(12.5, abs=1e-2)


def test_la_perspective_est_bien_corrigee():
    """Un trapèze à l'image correspond à un carré au sol : deux écarts de même
    longueur en pixels doivent donner des distances réelles différentes."""
    trapeze = [(20, 100), (80, 100), (100, 0), (0, 0)]   # bas etroit = premier plan
    c = cal.CalibrationSol(trapeze, SOL_CARRE)
    proche = c.vers_sol((50, 100))
    loin = c.vers_sol((50, 0))
    assert proche[1] != pytest.approx(loin[1], abs=0.5)


def test_calibration_refuse_des_points_degeneres():
    with pytest.raises(ValueError):
        cal.CalibrationSol([(0, 0), (1, 1)], [(0, 0), (1, 1)])


def test_point_au_sol_est_le_bas_de_la_boite():
    """Le seul point de la boîte qui appartienne vraiment au plan du sol."""
    assert cal.point_au_sol((10, 20, 30, 80)) == (20.0, 80.0)


def test_appartenance_a_la_zone():
    carre = [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert cal.dans_polygone((5, 5), carre)
    assert not cal.dans_polygone((15, 5), carre)


def _ctx_foule(t, boites):
    return {"t": t, "frame": int(t * 10), "objets": [],
            "personnes": [(i, b) for i, b in enumerate(boites)]}


ZONE_5M2 = [(0, 0), (20, 0), (20, 100), (0, 100)]   # 20x100 px -> 1 m x 5 m


def _gens(n):
    """n personnes, pieds repartis a l'interieur de ZONE_5M2 (x < 20)."""
    return [(2 + 1.5 * i - 2, 40, 2 + 1.5 * i + 2, 90) for i in range(n)]


def test_densite_declenche_au_seuil():
    """10 personnes sur 5 m² = 2 pers/m², le cas d'usage demandé."""
    c = cal.CalibrationSol(IMAGE_CARRE, SOL_CARRE)
    # zone de 5 m2 : moitie du carre de 25 m2 ne suffit pas, on prend 1/5
    zone = ZONE_5M2
    a = AnalyseurFoule(seuil_densite=2.0, zone=zone, calibration=c, fenetre=1, minimum=1)
    assert a.aire_m2 == pytest.approx(5.0, abs=1e-2)
    assert a.process(None, _ctx_foule(0.0, _gens(9))) == []       # 1,8 pers/m2
    evs = a.process(None, _ctx_foule(1.0, _gens(10)))             # 2,0 pers/m2
    assert len(evs) == 1 and evs[0].type == "foule"
    assert evs[0].extra["densite_pers_m2"] == pytest.approx(2.0, abs=1e-2)


def test_un_seul_evenement_tant_que_la_foule_dure():
    """Une foule stable ne doit pas produire un evenement par image."""
    c = cal.CalibrationSol(IMAGE_CARRE, SOL_CARRE)
    zone = ZONE_5M2
    a = AnalyseurFoule(seuil_densite=2.0, zone=zone, calibration=c, fenetre=1, minimum=1)
    total = sum(len(a.process(None, _ctx_foule(t, _gens(12)))) for t in range(10))
    assert total == 1


def test_fin_de_foule_est_signalee():
    """Sans le front descendant, la plateforme ne saurait pas que c'est fini."""
    c = cal.CalibrationSol(IMAGE_CARRE, SOL_CARRE)
    zone = ZONE_5M2
    a = AnalyseurFoule(seuil_densite=2.0, zone=zone, calibration=c, fenetre=1, minimum=1)
    a.process(None, _ctx_foule(0.0, _gens(12)))
    evs = a.process(None, _ctx_foule(1.0, _gens(2)))
    assert len(evs) == 1 and evs[0].type == "foule_terminee"


def test_sans_calibration_la_densite_est_nulle_pas_inventee():
    """Mieux vaut annoncer qu'on ne sait pas que publier une valeur fausse."""
    a = AnalyseurFoule(seuil_densite=2.0, seuil_effectif=5, fenetre=1, minimum=1)
    evs = a.process(None, _ctx_foule(0.0, _gens(6)))
    assert len(evs) == 1
    assert evs[0].extra["densite_pers_m2"] is None
    assert evs[0].extra["calibre"] is False
    assert evs[0].extra["effectif"] == 6


def test_personne_hors_zone_n_est_pas_comptee():
    c = cal.CalibrationSol(IMAGE_CARRE, SOL_CARRE)
    zone = ZONE_5M2
    a = AnalyseurFoule(seuil_densite=2.0, zone=zone, calibration=c, fenetre=1, minimum=1)
    dehors = [(500 + i, 40, 510 + i, 90) for i in range(20)]
    assert a.process(None, _ctx_foule(0.0, dehors)) == []
    assert a.dernier["effectif"] == 0


def test_la_confirmation_filtre_un_pic_isole():
    """Un groupe qui traverse le champ ne doit pas declencher une alerte."""
    c = cal.CalibrationSol(IMAGE_CARRE, SOL_CARRE)
    zone = ZONE_5M2
    a = AnalyseurFoule(seuil_densite=2.0, zone=zone, calibration=c, fenetre=5, minimum=3)
    assert a.process(None, _ctx_foule(0.0, _gens(12))) == []
    assert a.process(None, _ctx_foule(1.0, _gens(1))) == []
    assert a.process(None, _ctx_foule(2.0, _gens(1))) == []


# ── Identité des évènements (plan v6 §1.1, §1.3) ─────────────────────────────

def test_le_bus_appose_camera_id_et_site_id():
    """Sans identifiant de caméra, la plateforme reçoit des évènements sans
    savoir d'où ils viennent — inexploitable dès la deuxième caméra."""
    s = SortieMemoire()
    bus = BusEvenements([s], anti_repetition_s=0, camera_id="cam-quai-3", site_id="site-A")
    bus.publier(Evenement(time.time(), 1, "feu", "fire", "FIRE détecté"))
    assert s.recus[0].camera_id == "cam-quai-3"
    assert s.recus[0].site_id == "site-A"


def test_chaque_evenement_recoit_un_identifiant_unique():
    """`event_id` permet à la plateforme de dédupliquer après un rejeu réseau :
    sans lui, un webhook réémis crée un doublon indiscernable."""
    s = SortieMemoire()
    bus = BusEvenements([s], anti_repetition_s=0)
    for i in range(3):
        bus.publier(Evenement(time.time() + i, i, "feu", "fire", f"feu {i}"))
    ids = [e.event_id for e in s.recus]
    assert all(ids) and len(set(ids)) == 3


def test_un_identifiant_deja_pose_n_est_pas_ecrase():
    """Un analyseur qui met à jour un évènement existant doit pouvoir conserver
    son identité."""
    s = SortieMemoire()
    bus = BusEvenements([s], anti_repetition_s=0, camera_id="cam-1")
    ev = Evenement(time.time(), 1, "feu", "fire", "feu")
    ev.event_id, ev.camera_id = "fixe-123", "cam-9"
    bus.publier(ev)
    assert s.recus[0].event_id == "fixe-123"
    assert s.recus[0].camera_id == "cam-9"


# ── Livraison garantie (plan v6 §2.1) ────────────────────────────────────────

from unified_surveillance import SortieWebhook  # noqa: E402


def _webhook(tmp_path, reponses, **kw):
    """SortieWebhook dont l'envoi réseau est remplacé par une liste de verdicts."""
    w = SortieWebhook.__new__(SortieWebhook)
    w.url, w.timeout_s, w.jeton = "http://test", 0.01, ""
    w.tentatives_max = kw.get("tentatives_max", 3)
    w.attente_max_s = 0.0
    w.journal = tmp_path / "journal.jsonl"
    w.position = w.journal.with_suffix(".pos")
    w.echecs = w.abandons = w.livres = 0
    w._verrou = __import__("threading").Lock()
    w._reveil = __import__("threading").Event()
    w._stop = __import__("threading").Event()
    w._f = open(w.journal, "a", encoding="utf-8")
    w.envoyes = []

    def faux_poster(charge):
        ok = reponses.pop(0) if reponses else True
        if ok:
            w.envoyes.append(charge)
        else:
            w.echecs += 1
        return ok
    w._poster = faux_poster
    return w


def _ev(n=1):
    return Evenement(1000.0 + n, n, "feu", "fire", f"feu {n}", extra={"n": n})


def test_l_evenement_est_ecrit_sur_disque_avant_tout_envoi():
    """C'est ce qui garantit qu'un arrêt brutal ne perd rien."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        w = _webhook(Path(d), [])
        w.emettre(_ev())
        assert w.journal.exists() and w.journal.read_text().strip()


def test_un_echec_reseau_ne_perd_pas_l_evenement():
    """Le cas qui motive tout : un départ de feu détecté pendant un redémarrage
    de la plateforme ne doit pas disparaître."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        w = _webhook(Path(d), [False, False, True])
        w.emettre(_ev())
        ligne = w.journal.read_bytes().strip()
        assert w._livrer(ligne) is True        # reussit a la 3e tentative
        assert w.echecs == 2 and w.abandons == 0
        assert len(w.envoyes) == 1


def test_l_abandon_est_compte_jamais_silencieux():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        w = _webhook(Path(d), [False] * 3, tentatives_max=3)
        w.emettre(_ev())
        w._livrer(w.journal.read_bytes().strip())
        assert w.abandons == 1


def test_la_position_reprend_apres_redemarrage():
    """Position sur disque : au redémarrage, l'envoi repart où il s'était
    arrêté, sans rejouer ce qui est déjà parti ni perdre le reste."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        w = _webhook(Path(d), [])
        for i in (1, 2, 3):
            w.emettre(_ev(i))
        lignes = w.journal.read_bytes().splitlines(keepends=True)
        w._ecrire_position(len(lignes[0]))       # le premier est parti
        assert w._lire_position() == len(lignes[0])
        # ce qui reste a livrer = les deux suivants
        assert w.en_attente == len(lignes[1]) + len(lignes[2])


def test_position_illisible_rejoue_tout_plutot_que_de_perdre():
    """Si le fichier de position est corrompu, mieux vaut renvoyer des doublons
    — que `event_id` permet de filtrer — que perdre des évènements."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        w = _webhook(Path(d), [])
        w.emettre(_ev())
        w.position.write_text("pas un nombre")
        assert w._lire_position() == 0


def test_le_jeton_part_en_en_tete_authorization():
    """Sans authentification, quiconque connaît l'URL peut injecter de faux
    évènements dans la plateforme."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        w = _webhook(Path(d), [])
        w.jeton = "secret-123"
        entetes = {}

        def capture(url, data=None, headers=None):
            entetes.update(headers or {})
            raise RuntimeError("pas d'envoi reel")
        import urllib.request
        vrai = urllib.request.Request
        urllib.request.Request = capture
        try:
            SortieWebhook._poster(w, b"{}")
        finally:
            urllib.request.Request = vrai
        assert entetes.get("Authorization") == "Bearer secret-123"


# ── Machine à états des évènements gradués (plan v5 §1.3, plan v6 §1.2) ──────

def test_une_detection_isolee_reste_une_suspicion():
    """Le premier signalement n'est jamais une confirmation."""
    m = q.MachineEtats(seuil_probable=3, seuil_confirme=8)
    piste, etat = m.observer("k", 0.0, 0.5)
    assert etat == "suspicion" and piste.etat == "suspicion"


def test_la_piste_monte_en_grade_avec_la_persistance():
    m = q.MachineEtats(seuil_probable=3, seuil_confirme=5)
    etats = [m.observer("k", float(i), 0.5)[1] for i in range(6)]
    # suspicion a la 1re, probable a la 3e, confirme a la 5e, rien entre les deux
    assert etats == ["suspicion", None, "probable", None, "confirme", None]


def test_seuls_les_changements_d_etat_produisent_un_evenement():
    """C'est LA propriété qui tue le déluge : un feu observé mille fois ne
    produit pas mille évènements."""
    m = q.MachineEtats(seuil_probable=3, seuil_confirme=8)
    changements = [e for _, e in (m.observer("k", float(i), 0.5) for i in range(1000))
                   if e is not None]
    assert changements == ["suspicion", "probable", "confirme"]


def test_deux_zones_distinctes_sont_deux_faits():
    """Deux départs de feu éloignés ne doivent pas se confondre en un seul."""
    m = q.MachineEtats(resolution_px=160)
    a = m.cle_de("feu", "fire", (10, 10, 50, 50), "cam-1")
    b = m.cle_de("feu", "fire", (900, 900, 950, 950), "cam-1")
    assert a != b


def test_un_tremblement_de_boite_ne_cree_pas_une_piste_nouvelle():
    """Sans arrondi spatial, chaque oscillation de rectangle ouvrirait une
    piste et la machine ne servirait à rien."""
    m = q.MachineEtats(resolution_px=160)
    a = m.cle_de("feu", "fire", (100, 100, 200, 200), "cam-1")
    b = m.cle_de("feu", "fire", (103, 98, 204, 201), "cam-1")
    assert a == b


def test_deux_cameras_ne_se_melangent_pas():
    m = q.MachineEtats()
    a = m.cle_de("feu", "fire", (10, 10, 50, 50), "cam-1")
    b = m.cle_de("feu", "fire", (10, 10, 50, 50), "cam-2")
    assert a != b


def test_une_piste_non_alimentee_se_termine():
    """Sans signal de fin, la plateforme garderait une alerte ouverte
    indéfiniment."""
    m = q.MachineEtats(fin_apres_s=5.0)
    m.observer("k", 0.0, 0.5)
    assert m.expirer(3.0) == []
    finies = m.expirer(10.0)
    assert len(finies) == 1 and finies[0].cle == "k"
    assert "k" not in m.pistes          # la piste est bien liberee


def test_les_preuves_accompagnent_l_evenement():
    """Le moteur cesse de décider à la place de son intégrateur : la plateforme
    applique sa politique à partir de ces éléments."""
    m = q.MachineEtats(seuil_probable=2, seuil_confirme=3)
    for i in range(3):
        piste, _ = m.observer("k", float(i), 0.4 + i / 10)
    p = m.preuves(piste, 10.0)
    assert p["etat"] == "confirme"
    assert p["observations"] == 3
    assert p["duree_s"] == 10.0
    assert p["conf_max"] == pytest.approx(0.6)


# ── Intégration de la qualification dans le bus ──────────────────────────────

def test_le_bus_qualifie_reduit_le_flux():
    """Mille détections du même fait -> trois évènements, pas mille."""
    s = SortieMemoire()
    bus = BusEvenements([s], anti_repetition_s=0, camera_id="cam-1",
                        qualification=q.MachineEtats(seuil_probable=3, seuil_confirme=8,
                                                     fin_apres_s=1e9))
    for i in range(1000):
        bus.publier(Evenement(float(i) / 100, i, "feu", "fire", "FIRE",
                              conf=0.5, box=(100, 100, 200, 200)))
    assert [e.type for e in s.recus] == ["fire", "fire", "fire"]
    assert [e.extra["etat"] for e in s.recus] == ["suspicion", "probable", "confirme"]


def test_le_bus_signale_la_fin_de_l_evenement():
    s = SortieMemoire()
    bus = BusEvenements([s], anti_repetition_s=0, camera_id="cam-1",
                        qualification=q.MachineEtats(fin_apres_s=2.0))
    bus.publier(Evenement(0.0, 1, "feu", "fire", "FIRE", box=(10, 10, 50, 50)))
    # un autre fait, bien plus tard : il fait expirer le premier
    bus.publier(Evenement(60.0, 2, "feu", "fire", "FIRE", box=(900, 900, 950, 950)))
    assert any(e.type == "fire_termine" for e in s.recus)


def test_les_evenements_techniques_ne_sont_jamais_retardes():
    """Une caméra tombée doit se signaler immédiatement : la faire passer par
    la machine à états la mettrait en « suspicion »."""
    s = SortieMemoire()
    bus = BusEvenements([s], anti_repetition_s=0,
                        qualification=q.MachineEtats(seuil_probable=3))
    bus.publier(Evenement(0.0, -1, "capture", "flux_perdu", "Flux vidéo perdu"))
    assert len(s.recus) == 1
    assert "etat" not in s.recus[0].extra


def test_sans_qualification_le_comportement_d_origine_est_intact():
    """Garde-fou du plan v5 : la couche doit être débrayable pour comparer en
    parallèle et revenir en arrière sans redéploiement."""
    s = SortieMemoire()
    bus = BusEvenements([s], anti_repetition_s=0, qualification=None)
    for i in range(5):
        bus.publier(Evenement(float(i), i, "feu", "fire", f"FIRE {i}", box=(10, 10, 50, 50)))
    assert len(s.recus) == 5


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
