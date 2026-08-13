#!/usr/bin/env python3
"""Correspondance entre les pixels d'une image et les mètres au sol.

Pourquoi ce module existe
-------------------------
Compter des personnes est facile : le détecteur général le fait déjà. Exprimer
un résultat **par mètre carré** ne l'est pas, parce qu'une image ne contient
aucune information métrique. Deux personnes séparées de 50 pixels sont à 30 cm
l'une de l'autre au premier plan et à 5 m au fond de la scène.

Le détecteur historique (`surveillance_suite/detectors/crowd_density_detector_auto.py`)
contournait le problème en supposant que toute personne mesure 1,70 m : sa
hauteur en pixels donnait l'échelle locale. C'est ingénieux et sans réglage,
mais l'hypothèse s'effondre précisément en **vue plongeante** — l'angle habituel
d'une caméra de comptage de foule. Vu du dessus, une personne proche et une
personne lointaine ont presque la même hauteur apparente, et l'échelle estimée
devient fausse là où on en a le plus besoin.

Principe retenu
---------------
Une caméra de surveillance est **fixe**, et les personnes marchent sur un
**plan** (le sol). Ces deux faits suffisent : la transformation entre le plan du
sol et le plan image est une **homographie**, entièrement déterminée par quatre
points dont on connaît les coordonnées réelles.

C'est un réglage d'installation, fait une fois par caméra : repérer quatre
points au sol formant un quadrilatère (coins d'une dalle, marquage au sol,
angles d'un quai) et noter leurs positions réelles en mètres.

Ce que ce module ne fait pas
----------------------------
Il ne corrige pas les erreurs de détection. Si le détecteur rate des personnes
dans une foule dense — ce qui arrive dès que les occlusions deviennent fortes —
la densité calculée sera sous-estimée. La géométrie est exacte, le comptage
reste celui du détecteur.
"""

from __future__ import annotations

import json
from pathlib import Path


class CalibrationSol:
    """Projette un point de l'image vers le plan du sol, en mètres.

    `points_image` et `points_sol` doivent se correspondre un à un, dans le même
    ordre, et compter au moins quatre paires (une homographie a huit degrés de
    liberté). Au-delà de quatre, OpenCV ajuste au mieux, ce qui réduit l'effet
    d'un point mal pointé.
    """

    def __init__(self, points_image, points_sol):
        import numpy as np
        import cv2

        if len(points_image) != len(points_sol):
            raise ValueError("points_image et points_sol doivent avoir la meme longueur")
        if len(points_image) < 4:
            raise ValueError("au moins 4 points sont necessaires pour une homographie")

        src = np.asarray(points_image, dtype="float32").reshape(-1, 1, 2)
        dst = np.asarray(points_sol, dtype="float32").reshape(-1, 1, 2)
        H, _ = cv2.findHomography(src, dst)
        if H is None:
            raise ValueError("homographie impossible : les points sont probablement "
                             "alignes ou confondus")
        self.H = H
        self.points_image = [tuple(map(float, p)) for p in points_image]
        self.points_sol = [tuple(map(float, p)) for p in points_sol]

    # ── Chargement / sauvegarde ──────────────────────────────────────────────

    @classmethod
    def depuis_fichier(cls, chemin) -> "CalibrationSol":
        d = json.loads(Path(chemin).read_text(encoding="utf-8"))
        return cls(d["points_image"], d["points_sol"])

    def vers_fichier(self, chemin) -> None:
        Path(chemin).write_text(json.dumps({
            "points_image": self.points_image,
            "points_sol": self.points_sol,
            "_aide": "points_image en pixels, points_sol en metres, dans le meme ordre",
        }, indent=2) + "\n", encoding="utf-8")

    # ── Projection ───────────────────────────────────────────────────────────

    def vers_sol(self, point) -> tuple[float, float]:
        """Un point image (pixels) vers ses coordonnées au sol (mètres)."""
        return self.vers_sol_multiple([point])[0]

    def vers_sol_multiple(self, points) -> list[tuple[float, float]]:
        """Version vectorisée : une seule transformation pour toutes les personnes.

        Appeler `perspectiveTransform` par personne coûterait plus cher que la
        détection elle-même sur une scène chargée.
        """
        if not points:
            return []
        import numpy as np
        import cv2
        src = np.asarray(points, dtype="float32").reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(src, self.H).reshape(-1, 2)
        return [(float(x), float(y)) for x, y in dst]

    def aire_m2(self, polygone_image) -> float:
        """Aire réelle, en m², d'un polygone tracé sur l'image.

        Passe par la projection au sol avant de mesurer : l'aire en pixels n'a
        aucun rapport avec l'aire réelle, la perspective écrasant le fond de la
        scène. Formule du lacet (Gauss) sur les sommets projetés.
        """
        sommets = self.vers_sol_multiple(list(polygone_image))
        n = len(sommets)
        if n < 3:
            return 0.0
        somme = sum(sommets[i][0] * sommets[(i + 1) % n][1]
                    - sommets[(i + 1) % n][0] * sommets[i][1] for i in range(n))
        return abs(somme) / 2.0


def point_au_sol(boite) -> tuple[float, float]:
    """Point de contact d'une personne avec le sol : milieu du bord inférieur.

    C'est le seul point de la boîte qui appartienne réellement au plan du sol.
    Utiliser le centre projetterait la personne en arrière de sa position vraie,
    d'autant plus que la caméra est haute.
    """
    x1, _, x2, y2 = boite
    return ((x1 + x2) / 2.0, float(y2))


def dans_polygone(point, polygone) -> bool:
    """Test d'appartenance par lancer de rayon.

    Écrit ici plutôt qu'appelé à `cv2.pointPolygonTest` pour rester utilisable
    sans OpenCV — la logique de zone est testable sans dépendance graphique.
    """
    x, y = point
    dedans = False
    n = len(polygone)
    for i in range(n):
        x1, y1 = polygone[i]
        x2, y2 = polygone[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xi = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < xi:
                dedans = not dedans
    return dedans
