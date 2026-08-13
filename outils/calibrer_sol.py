#!/usr/bin/env python3
"""Calibration du sol d'une caméra, en quatre clics.

À faire **une fois par caméra**, au moment de l'installation. Sans cette
opération, le moteur sait compter les personnes mais pas exprimer une densité
en personnes par mètre carré — une image ne contient aucune information
métrique (voir `improvements/calibration_sol.py`).

Marche à suivre
---------------
1. Repérer dans le champ de la caméra un **rectangle au sol** dont on connaît
   les dimensions réelles : une dalle, un marquage, un quai, les lignes d'un
   parking. Il n'a pas besoin d'être grand, mais plus il couvre la zone
   surveillée, plus la calibration sera juste.
2. Lancer cet outil, cliquer ses **quatre coins dans le sens horaire**, en
   commençant par celui en haut à gauche *du rectangle réel* (pas de l'image).
3. Saisir sa largeur et sa longueur en mètres.

    python outils/calibrer_sol.py --source rtsp://... --sortie calibration.json
    python improvements/unified_surveillance.py --source rtsp://... \
        --calibration-sol calibration.json --zone-foule "..." --seuil-densite 2.0

Précision
---------
La qualité de la calibration dépend entièrement de celle des clics. Un coin
pointé à 10 pixels près sur un rectangle de 100 pixels introduit ~10 % d'erreur
sur les distances. Viser un rectangle large, et pointer les coins au sol — pas
en haut d'un obstacle.

Le rectangle doit être **posé au sol**. Utiliser le sommet d'un muret ou le
haut d'une table fausserait tout : le plan calibré ne serait pas celui où
marchent les personnes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "improvements"))

AIDE = [
    "Cliquer les 4 coins du rectangle au sol, dans le sens horaire.",
    "  c = effacer le dernier point     r = tout recommencer",
    "  entree = valider (4 points)      q / echap = abandonner",
]


def saisir_points(image) -> list[tuple[float, float]] | None:
    """Recueille quatre clics sur l'image. `None` si l'opérateur abandonne."""
    import cv2

    points: list[tuple[float, float]] = []
    fenetre = "Calibration du sol"

    def au_clic(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((float(x), float(y)))

    cv2.namedWindow(fenetre, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(fenetre, au_clic)

    while True:
        vue = image.copy()
        for i, (x, y) in enumerate(points):
            cv2.circle(vue, (int(x), int(y)), 6, (0, 255, 255), -1)
            cv2.putText(vue, str(i + 1), (int(x) + 10, int(y) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        if len(points) > 1:
            import numpy as np
            cv2.polylines(vue, [np.asarray(points, dtype="int32").reshape(-1, 1, 2)],
                          len(points) == 4, (0, 255, 255), 2)
        for i, ligne in enumerate(AIDE):
            cv2.putText(vue, ligne, (10, 25 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 255), 1)

        cv2.imshow(fenetre, vue)
        touche = cv2.waitKey(30) & 0xFF
        if touche in (ord("q"), 27):
            cv2.destroyWindow(fenetre)
            return None
        if touche == ord("c") and points:
            points.pop()
        if touche == ord("r"):
            points.clear()
        if touche in (13, 10) and len(points) == 4:
            cv2.destroyWindow(fenetre)
            return points


def premiere_image(source: str):
    import cv2
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise SystemExit(f"source illisible : {source}")
    # Quelques images sont lues avant d'en garder une : le premier decodage
    # d'un flux RTSP renvoie souvent une image grise ou incomplete.
    image = None
    for _ in range(10):
        ok, brut = cap.read()
        if ok:
            image = brut
    cap.release()
    if image is None:
        raise SystemExit(f"aucune image exploitable depuis : {source}")
    return image


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", required=True, help="flux, fichier video ou index de webcam")
    ap.add_argument("--sortie", default="calibration_sol.json")
    ap.add_argument("--largeur", type=float, default=None,
                    help="largeur reelle du rectangle en metres (sinon demandee)")
    ap.add_argument("--longueur", type=float, default=None,
                    help="longueur reelle du rectangle en metres (sinon demandee)")
    ap.add_argument("--points", default=None,
                    help="« x1,y1;x2,y2;x3,y3;x4,y4 » pour calibrer sans interface "
                         "graphique (machine sans ecran)")
    a = ap.parse_args()

    from calibration_sol import CalibrationSol

    if a.points:
        points = [tuple(float(v) for v in c.split(",")) for c in a.points.split(";")]
        if len(points) != 4:
            raise SystemExit("--points attend exactement 4 couples x,y")
    else:
        points = saisir_points(premiere_image(a.source))
        if points is None:
            print("abandon")
            return 1

    largeur = a.largeur if a.largeur is not None else float(
        input("Largeur reelle du rectangle (metres) : ").replace(",", "."))
    longueur = a.longueur if a.longueur is not None else float(
        input("Longueur reelle du rectangle (metres) : ").replace(",", "."))
    if largeur <= 0 or longueur <= 0:
        raise SystemExit("les dimensions doivent etre strictement positives")

    # Les clics sont dans le sens horaire depuis le coin haut-gauche du
    # rectangle reel : ils correspondent donc a ces quatre coins-la.
    sol = [(0.0, 0.0), (largeur, 0.0), (largeur, longueur), (0.0, longueur)]
    calib = CalibrationSol(points, sol)
    calib.vers_fichier(a.sortie)

    aire = calib.aire_m2(points)
    print(f"\nCalibration ecrite dans {a.sortie}")
    print(f"  rectangle de reference : {largeur} x {longueur} m = {largeur * longueur:.2f} m2")
    print(f"  aire re-mesuree apres projection : {aire:.2f} m2")
    ecart = abs(aire - largeur * longueur) / (largeur * longueur)
    if ecart > 0.01:
        print(f"  ATTENTION : ecart de {ecart:.1%} -- points probablement mal pointes")
    else:
        print("  coherent.")
    print("\nA utiliser avec :")
    print(f"  --calibration-sol {a.sortie} --seuil-densite 2.0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
