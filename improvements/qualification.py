#!/usr/bin/env python3
"""Couche de qualification des détections (plan v5).

Le moteur décide aujourd'hui image par image, à partir de la seule sortie des
modèles. Deux mesures ont montré que c'est là, et non dans la vision des
modèles, que se situe la marge de progrès :

- le détecteur feu/fumée repère **96,7 %** des scènes contenant de la fumée
  (`reports/v3_results/operationnel_fire_smoke_avant.json`), alors que sa
  mAP@50 de 40 % laissait croire l'inverse : ce qui était sanctionné, c'était la
  position du rectangle autour d'un objet sans contour, pas la détection ;
- en revanche 2 % de fausse alarme par image deviennent ~43 000 alertes par jour
  à 25 images/s, et la nuit le taux de détection tombe à 36 %.

Ce module regroupe les règles qui qualifient une détection à partir de son
**contexte** — géométrie de la scène, luminosité, cohérence anatomique — sans
rien changer aux modèles. Il ne crée jamais de détection : il ne peut que
rejeter, ou graduer.

Les règles temporelles (croissance d'un panache, transition d'une chute) ne sont
volontairement pas ici : elles exigent un corpus vidéo annoté pour être réglées,
qui n'existe pas encore. Ce module ne contient que ce qui est mesurable
aujourd'hui sur images fixes.
"""

from __future__ import annotations

from dataclasses import dataclass

Boite = tuple[int, int, int, int]


# ── Géométrie ────────────────────────────────────────────────────────────────

def confinement(interieur: Boite, exterieur: Boite) -> float:
    """Fraction de `interieur` qui tombe dans `exterieur`, entre 0 et 1.

    À ne pas confondre avec l'IoU, qui est la mauvaise mesure pour un rapport
    « partie de ». Un casque occupe environ 3 % de la boîte d'une personne :
    son IoU avec la bonne personne vaut donc ~0,03, une valeur indiscernable du
    bruit. Son confinement, lui, vaut ~1,0 s'il est réellement porté et ~0 s'il
    flotte ailleurs dans l'image — ce qui en fait un critère de rejet utilisable.
    """
    xa, ya = max(interieur[0], exterieur[0]), max(interieur[1], exterieur[1])
    xb, yb = min(interieur[2], exterieur[2]), min(interieur[3], exterieur[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    aire = (interieur[2] - interieur[0]) * (interieur[3] - interieur[1])
    return inter / aire if aire > 0 else 0.0


# Position verticale attendue de chaque EPI sur la personne, en fraction de sa
# hauteur (0 = sommet du crâne, 1 = pieds). Les plages sont larges à dessein :
# il s'agit d'écarter l'aberrant (un casque au niveau des pieds), pas d'imposer
# une morphologie. Une personne penchée, assise ou accroupie doit passer.
PLAGES_ANATOMIQUES: dict[str, tuple[float, float]] = {
    "casque":   (-0.10, 0.45),
    "lunettes": (-0.05, 0.45),
    "masque":   (0.00, 0.50),
    "gilet":    (0.10, 0.85),
    "gants":    (0.15, 1.00),
    "chaussures": (0.55, 1.10),
}


def plausible_anatomiquement(epi: str, boite_epi: Boite, boite_personne: Boite) -> bool:
    """L'EPI est-il à une hauteur cohérente sur cette personne ?

    Sert à écarter les confusions grossières — un « casque » détecté sur les
    chaussures, un « gilet » au-dessus de la tête. Renvoie True pour tout EPI
    inconnu de la table : la règle ne doit jamais rejeter ce qu'elle ne sait pas
    juger.
    """
    plage = PLAGES_ANATOMIQUES.get(epi)
    if plage is None:
        return True
    haut, bas = boite_personne[1], boite_personne[3]
    hauteur = bas - haut
    if hauteur <= 0:
        return True
    centre = (boite_epi[1] + boite_epi[3]) / 2
    position = (centre - haut) / hauteur
    return plage[0] <= position <= plage[1]


def associer_a_personne(boite_epi: Boite, personnes: list[Boite],
                        seuil_confinement: float = 0.50,
                        epi: str | None = None) -> int | None:
    """Indice de la personne qui porte cet EPI, ou None s'il n'est porté par
    personne.

    Renvoyer None est le comportement important, et celui qui manquait : sans
    lui, un EPI détecté à tort quelque part dans l'image est attribué d'office à
    la personne la « moins éloignée », et compte alors comme équipement porté.
    Un faux positif de casque peut ainsi masquer une vraie infraction — sur un
    système de sécurité, c'est le sens de l'erreur le plus grave.
    """
    if not personnes:
        return None
    scores = [confinement(boite_epi, p) for p in personnes]
    meilleur = max(range(len(personnes)), key=lambda i: scores[i])
    if scores[meilleur] < seuil_confinement:
        return None
    if epi and not plausible_anatomiquement(epi, boite_epi, personnes[meilleur]):
        return None
    return meilleur


# ── Profil de scène ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Profil:
    """Seuils de confiance associés à une condition de prise de vue."""
    nom: str
    seuil_detection: float
    fenetre_confirmation: int
    minimum_confirmation: int


# La nuit, le compromis change de nature. La courbe mesurée
# (`reports/v3_results/operationnel_nuit_seuils.json`) donne, sur le détecteur
# feu/fumée : 37 % de détection au seuil 0,15 contre 56 % à 0,03 — mais le taux
# de fausse alarme par image passe de 0,9 % à 5,4 %. Garder le seuil de jour
# revient donc à sacrifier un tiers des détections nocturnes ; le baisser sans
# rien d'autre revient à noyer l'aval sous les fausses alertes. D'où le couple :
# seuil plus bas ET confirmation plus exigeante, la précision perdue par image
# étant regagnée sur la séquence.
PROFILS: dict[str, Profil] = {
    "jour":    Profil("jour",    seuil_detection=0.10, fenetre_confirmation=4, minimum_confirmation=2),
    "degrade": Profil("degrade", seuil_detection=0.06, fenetre_confirmation=6, minimum_confirmation=3),
    "nuit":    Profil("nuit",    seuil_detection=0.03, fenetre_confirmation=8, minimum_confirmation=4),
}


# ── Machine à états des évènements gradués ───────────────────────────────────

ETATS = ("suspicion", "probable", "confirme")


@dataclass
class Piste:
    """Un fait suivi dans le temps, par opposition à une détection sur image."""
    cle: str
    debut: float
    derniere_vue: float
    observations: int = 1
    etat: str = "suspicion"
    conf_max: float = 0.0


class MachineEtats:
    """Transforme une suite de détections en **un évènement par fait**.

    C'est la brique qui répond au constat central du plan v6 : le moteur émet
    2 869 évènements en 30 minutes, soit ~137 600 par jour et par caméra, parce
    qu'il signale chaque *image* où quelque chose est vu au lieu de signaler le
    *fait* lui-même. Aucune plateforme ne consomme cela, et le problème n'est
    pas la vision des modèles — le détecteur de fumée repère 96,7 % des scènes.

    Une piste naît en `suspicion`, passe `probable` puis `confirme` à mesure
    qu'elle persiste, et se termine quand plus rien ne l'alimente. **Seuls les
    changements d'état produisent un évènement** : un feu observé pendant trois
    minutes en produit trois ou quatre, pas quatre mille.

    Les seuils sont volontairement des paramètres. Le plan v5 attribue à chaque
    scénario un budget de latence différent — 30 s pour un départ de feu, 3 à
    5 s pour une chute, où une personne au sol a besoin d'aide immédiatement.
    Une valeur unique serait donc fausse pour au moins un scénario.
    """

    def __init__(self, seuil_probable: int = 3, seuil_confirme: int = 8,
                 fin_apres_s: float = 5.0, resolution_px: int = 160):
        self.seuil_probable = seuil_probable
        self.seuil_confirme = seuil_confirme
        self.fin_apres_s = fin_apres_s
        self.resolution_px = resolution_px
        self.pistes: dict[str, Piste] = {}

    def cle_de(self, source: str, type_ev: str, boite=None, camera_id: str = "") -> str:
        """Identité d'un fait : sa nature et sa zone, pas sa position exacte.

        La position est arrondie à une grille grossière — deux détections du
        même panache à trois pixels d'écart sont le même fait. Sans cet
        arrondi, chaque tremblement de boîte créerait une piste nouvelle et la
        machine ne servirait à rien.
        """
        if not boite:
            return f"{camera_id}|{source}|{type_ev}"
        x = int((boite[0] + boite[2]) / 2 / self.resolution_px)
        y = int((boite[1] + boite[3]) / 2 / self.resolution_px)
        return f"{camera_id}|{source}|{type_ev}|{x}:{y}"

    def observer(self, cle: str, t: float, conf: float = 0.0) -> tuple[Piste, str | None]:
        """Enregistre une détection. Renvoie la piste et l'état si celui-ci
        vient de changer (`None` sinon — donc rien à émettre)."""
        p = self.pistes.get(cle)
        if p is None:
            p = self.pistes[cle] = Piste(cle, t, t, 1, "suspicion", conf)
            return p, "suspicion"

        p.derniere_vue = t
        p.observations += 1
        p.conf_max = max(p.conf_max, conf)
        avant = p.etat
        if p.observations >= self.seuil_confirme:
            p.etat = "confirme"
        elif p.observations >= self.seuil_probable:
            p.etat = "probable"
        return p, (p.etat if p.etat != avant else None)

    def expirer(self, t: float) -> list[Piste]:
        """Pistes que plus rien n'alimente : elles se terminent.

        Le signal de fin compte autant que celui de début — sans lui, la
        plateforme ne sait pas qu'un incident est clos et garde une alerte
        ouverte indéfiniment.
        """
        finies = [p for p in self.pistes.values() if t - p.derniere_vue > self.fin_apres_s]
        for p in finies:
            del self.pistes[p.cle]
        return finies

    def preuves(self, p: Piste, t: float) -> dict:
        """Éléments factuels accompagnant l'évènement.

        Le moteur cesse ainsi de décider à la place de son intégrateur : la
        plateforme applique sa propre politique — alerter dès `probable` sur un
        site sensible, attendre `confirme` ailleurs.
        """
        return {
            "etat": p.etat,
            "duree_s": round(t - p.debut, 2),
            "observations": p.observations,
            "conf_max": round(p.conf_max, 3),
        }


class DetecteurProfil:
    """Classe la scène en jour / dégradé / nuit d'après sa luminance.

    Le basculement est protégé par une hystérésis : au crépuscule, une scène
    oscille autour du seuil, et changer de profil à chaque image ferait varier
    les seuils de détection en permanence — donc apparaître et disparaître des
    évènements sans qu'il se passe quoi que ce soit dans la scène.
    """

    def __init__(self, seuil_nuit: float = 60.0, seuil_degrade: float = 100.0,
                 marge: float = 12.0, stabilite: int = 5):
        self.seuil_nuit, self.seuil_degrade = seuil_nuit, seuil_degrade
        self.marge, self.stabilite = marge, stabilite
        self.profil = "jour"
        self._candidat, self._compte = "jour", 0

    @staticmethod
    def luminance(frame) -> float:
        """Luminance médiane. La médiane, et non la moyenne : un projecteur ou
        un ciel surexposé suffirait à faire passer une scène nocturne pour une
        scène de jour."""
        import cv2
        import numpy as np
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        return float(np.median(gris))

    def _brut(self, lum: float) -> str:
        # Les bornes sont decalees selon le profil courant (hysteresis) : il faut
        # depasser franchement le seuil pour en sortir, pas seulement l'effleurer.
        m = self.marge
        if self.profil == "nuit":
            return "nuit" if lum < self.seuil_nuit + m else (
                "degrade" if lum < self.seuil_degrade + m else "jour")
        if self.profil == "jour":
            return "jour" if lum >= self.seuil_degrade - m else (
                "degrade" if lum >= self.seuil_nuit - m else "nuit")
        return "nuit" if lum < self.seuil_nuit else (
            "degrade" if lum < self.seuil_degrade else "jour")

    def mettre_a_jour(self, frame) -> Profil:
        candidat = self._brut(self.luminance(frame))
        if candidat == self._candidat:
            self._compte += 1
        else:
            self._candidat, self._compte = candidat, 1
        # Un profil ne bascule qu'apres plusieurs images concordantes : un nuage
        # qui passe ou un phare de voiture ne doit pas reconfigurer le moteur.
        if candidat != self.profil and self._compte >= self.stabilite:
            self.profil = candidat
        return PROFILS[self.profil]

    @property
    def courant(self) -> Profil:
        return PROFILS[self.profil]
