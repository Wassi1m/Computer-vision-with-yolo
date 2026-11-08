#!/usr/bin/env python3
"""P20 — Capture annotée de visages en gros plan, pour la classe « lunettes ».

Pourquoi ce script existe
-------------------------
`epi_gants_lunettes.pt` (M5) retrouve `NO-Goggles` sur 95 % des scènes de son
jeu de validation, mais **rien du tout** sur un gros plan de webcam : mesuré le
2026-09-07 à un seuil de 0,03, en 480/640/960/1280, sur fond gris comme sur fond
de chantier, la sortie est vide. Le casque et le masque, eux, généralisent à ce
cadrage — les lunettes non.

La cause est mesurable et tient en un chiffre : sur 401 boîtes `NO-Goggles` du
jeu de validation, l'aire médiane vaut **0,23 % de l'image** (maximum 0,77 %).
Le modèle n'a jamais vu un visage occupant plus d'un centième du cadre. Sur la
webcam le visage en occupe le cinquième, soit vingt à cent fois plus. Ce n'est
pas un défaut de seuil ni d'ancrage : il n'y a rien à seuiller, le modèle ne
propose aucune boîte.

Il faut donc lui donner des exemples de ce cadrage. Ce script les produit.

Ce qu'annote ce script, et pourquoi le visage entier
----------------------------------------------------
L'inspection des boîtes `NO-Goggles` du jeu existant montre qu'elles couvrent
**le visage entier**, du haut du front au menton, oreilles comprises — pas la
seule zone des yeux. Leur rapport largeur/hauteur médian est de 1,27
(quartiles 1,14 et 1,42) : une boîte plus large que haute, centrée sur les yeux.

Annoter la zone des yeux produirait un jeu incohérent avec les 28 000 images
déjà en place, et le modèle apprendrait deux conventions contradictoires. La
boîte écrite ici reproduit donc la convention existante, dérivée du cadre que
`DetecteurVisage` renvoie déjà.

Pourquoi réutiliser le détecteur de la plateforme
-------------------------------------------------
`surveillance_suite/modules/module_visage.DetecteurVisage` est le composant qui
tourne réellement en production, et la reconnaissance faciale est toujours
active. Cadrer avec lui garantit que le jeu d'entraînement décrit la géométrie
que le moteur rencontrera. Un second détecteur, même meilleur, introduirait un
écart entre ce qu'on annote et ce qu'on verra.

Aucune annotation manuelle
--------------------------
L'étiquette vient de la **session**, pas d'un jugement image par image :
`--port avec` écrit `Goggles`, `--port sans` écrit `NO-Goggles`. On enregistre
une série en portant les lunettes de sécurité, une autre sans. C'est à
l'opérateur de ne pas se tromper de session — le script le rappelle au
démarrage et exige une confirmation.

⚠️ Les lunettes de VUE comptent pour `NO-Goggles`
-------------------------------------------------
Une paire de lunettes correctrices n'est pas un équipement de protection. Une
session `--port sans` filmée avec des lunettes de vue est donc non seulement
permise mais **souhaitable** : sans ces exemples, le modèle apprendrait que
toute monture vaut conformité, ce qui est précisément le défaut de sécurité
qu'on cherche à éviter.

Une seule personne par image, par défaut
----------------------------------------
Un visage présent mais non annoté enseigne au modèle que ce visage est du fond,
et supprime la détection au lieu de l'améliorer. Le script refuse donc les
images où il voit plusieurs visages, sauf `--tous` — à n'utiliser que si
**toutes** les personnes du cadre sont dans le même état vis-à-vis des lunettes.

Usage
-----
    # série sans lunettes de sécurité (dont, idéalement, des lunettes de vue)
    python improvements/p20_capture_lunettes.py --port sans --images 150

    # série avec les lunettes de sécurité du site
    python improvements/p20_capture_lunettes.py --port avec --images 150

    # planche de contrôle : vérifier les boîtes AVANT d'entraîner
    python improvements/p20_capture_lunettes.py --verifier
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

_ICI = Path(__file__).resolve().parent
_RACINE = _ICI.parent
sys.path.insert(0, str(_RACINE / "surveillance_suite" / "modules"))

#: Indices de classe du jeu `epi_gants_lunettes`, dans l'ordre de son data.yaml
#: (`['Gloves', 'NO-Gloves', 'Goggles', 'NO-Goggles']`). Écrits en dur ici
#: serait une erreur si l'ordre changeait : ils sont relus du data.yaml quand il
#: est présent, et ces valeurs ne servent que de repli documenté.
CLASSES = {"avec": 2, "sans": 3}
NOMS = {"avec": "Goggles", "sans": "NO-Goggles"}

#: Rapport largeur/hauteur des boîtes du jeu existant (médiane de 401 boîtes
#: `NO-Goggles`). Voir l'en-tête : s'en écarter produirait un jeu incohérent.
RATIO = 1.27

#: La boîte du détecteur serre le visage ; celles du jeu existant incluent le
#: front et les oreilles. Ce facteur les rapproche.
MARGE = 1.15

DESTINATION = _RACINE / "ppe_detection" / "data" / "extracted" / "lunettes_gros_plan"


def indices_depuis_yaml() -> dict[str, int]:
    """Relit l'ordre des classes du jeu existant plutôt que de le supposer."""
    yaml = (_RACINE / "ppe_detection" / "data" / "extracted"
            / "epi_gants_lunettes" / "data.yaml")
    if not yaml.exists():
        return CLASSES
    for ligne in yaml.read_text(encoding="utf-8").splitlines():
        if ligne.startswith("names:"):
            noms = [n.strip(" '\"") for n in
                    ligne.split("[", 1)[-1].rstrip("]").split(",")]
            if "Goggles" in noms and "NO-Goggles" in noms:
                return {"avec": noms.index("Goggles"),
                        "sans": noms.index("NO-Goggles")}
    return CLASSES


def boite_annotation(boite_visage, forme) -> tuple[float, float, float, float]:
    """Boîte YOLO normalisée (cx, cy, w, h), à la convention du jeu existant."""
    x1, y1, x2, y2 = boite_visage
    hauteur_image, largeur_image = forme[:2]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    largeur = (x2 - x1) * MARGE
    hauteur = largeur / RATIO
    return (cx / largeur_image, cy / hauteur_image,
            largeur / largeur_image, hauteur / hauteur_image)


def capturer(port: str, cible: int, intervalle: float, camera: int,
             tous: bool, sortie: Path) -> int:
    from module_visage import DetecteurVisage

    indices = indices_depuis_yaml()
    classe = indices[port]
    print(f"Session « {port} » -> classe {classe} ({NOMS[port]})")
    if port == "sans":
        print("Rappel : les lunettes de VUE relèvent de cette session, pas de l'autre.")
    else:
        print("Rappel : uniquement de vraies lunettes de SÉCURITÉ.")
    if input("Confirmer (o/N) ? ").strip().lower() not in ("o", "oui", "y"):
        print("Annulé."); return 1

    detecteur = DetecteurVisage(gpu=False)
    if not detecteur.disponible:
        print("Reconnaissance faciale indisponible : impossible d'annoter.",
              file=sys.stderr)
        return 1

    images = sortie / "images"
    labels = sortie / "labels"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)

    flux = cv2.VideoCapture(camera)
    if not flux.isOpened():
        print(f"Caméra {camera} inaccessible.", file=sys.stderr)
        return 1

    print(f"\n{cible} images, une toutes les {intervalle} s. Ctrl-C pour arrêter.")
    print("Variez sans cesse : distance, angle de tête, éclairage, position "
          "dans le cadre. Un jeu tourné immobile n'apprend qu'une pose.\n")

    horodatage = time.strftime("%Y%m%d_%H%M%S")
    ecrites = ignorees = 0
    try:
        while ecrites < cible:
            ok, image = flux.read()
            if not ok:
                print("lecture impossible", file=sys.stderr); break

            visages = detecteur.detecter(image)
            if not visages:
                ignorees += 1
                print(f"\r{ecrites}/{cible} écrites — aucun visage ({ignorees} ignorées)",
                      end="", flush=True)
                time.sleep(intervalle); continue
            if len(visages) > 1 and not tous:
                ignorees += 1
                print(f"\r{ecrites}/{cible} écrites — {len(visages)} visages, "
                      f"image ignorée ({ignorees} ignorées)", end="", flush=True)
                time.sleep(intervalle); continue

            nom = f"{horodatage}_{port}_{ecrites:04d}"
            cv2.imwrite(str(images / f"{nom}.jpg"), image)
            (labels / f"{nom}.txt").write_text(
                "\n".join(
                    "%d %.6f %.6f %.6f %.6f" % ((classe,) + boite_annotation(v["box"], image.shape))
                    for v in visages) + "\n",
                encoding="utf-8")
            ecrites += 1
            print(f"\r{ecrites}/{cible} écrites ({ignorees} ignorées)",
                  end="", flush=True)
            time.sleep(intervalle)
    except KeyboardInterrupt:
        print("\ninterrompu")
    finally:
        flux.release()

    print(f"\n{ecrites} images dans {images}")
    print("Contrôlez les boîtes avant d'entraîner :\n"
          "    python improvements/p20_capture_lunettes.py --verifier")
    return 0


def verifier(sortie: Path, combien: int) -> int:
    """Planche de contact : les boîtes dessinées, pour les juger à l'œil."""
    images = sorted((sortie / "images").glob("*.jpg"))
    if not images:
        print(f"Rien à vérifier dans {sortie}.", file=sys.stderr); return 1

    pas = max(1, len(images) // combien)
    vignettes = []
    compte: dict[str, int] = {}
    for chemin in images[::pas][:combien]:
        image = cv2.imread(str(chemin))
        hauteur, largeur = image.shape[:2]
        etiquette = (sortie / "labels" / (chemin.stem + ".txt"))
        for ligne in etiquette.read_text(encoding="utf-8").splitlines():
            if not ligne.strip():
                continue
            classe, cx, cy, w, h = ligne.split()
            compte[classe] = compte.get(classe, 0) + 1
            x1 = int((float(cx) - float(w) / 2) * largeur)
            y1 = int((float(cy) - float(h) / 2) * hauteur)
            x2 = int((float(cx) + float(w) / 2) * largeur)
            y2 = int((float(cy) + float(h) / 2) * hauteur)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        vignettes.append(cv2.resize(image, (240, int(240 * hauteur / largeur))))

    hauteur_max = max(v.shape[0] for v in vignettes)
    planche = cv2.hconcat([
        cv2.copyMakeBorder(v, 0, hauteur_max - v.shape[0], 0, 4,
                           cv2.BORDER_CONSTANT, value=(255, 255, 255))
        for v in vignettes])
    destination = sortie / "controle_boites.png"
    cv2.imwrite(str(destination), planche)

    total = sum(1 for _ in (sortie / "labels").glob("*.txt"))
    print(f"{total} images annotées ; instances par classe : {compte}")
    print(f"Planche de contrôle : {destination}")
    print("La boîte doit couvrir le visage entier, front et menton compris.")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", choices=("avec", "sans"),
                    help="l'opérateur porte-t-il des lunettes de SÉCURITÉ ?")
    ap.add_argument("--images", type=int, default=150)
    ap.add_argument("--intervalle", type=float, default=0.7,
                    help="secondes entre deux prises ; laisse le temps de bouger")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--tous", action="store_true",
                    help="annote TOUS les visages du cadre. À n'utiliser que si "
                         "toutes les personnes présentes sont dans le même état")
    ap.add_argument("--sortie", type=Path, default=DESTINATION)
    ap.add_argument("--verifier", action="store_true",
                    help="dessine les boîtes déjà écrites et s'arrête")
    ap.add_argument("--vignettes", type=int, default=8)
    a = ap.parse_args()

    if a.verifier:
        sys.exit(verifier(a.sortie, a.vignettes))
    if not a.port:
        ap.error("--port est requis (ou --verifier)")
    sys.exit(capturer(a.port, a.images, a.intervalle, a.camera, a.tous, a.sortie))
