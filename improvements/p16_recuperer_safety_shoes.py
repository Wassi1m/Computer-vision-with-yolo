#!/usr/bin/env python3
"""P16 — Reconstruit `ahmed-alqulayti/safety-shoes-dataset` image par image.

Pourquoi ce script existe
-------------------------
Ce jeu est le seul apport de données possible pour `safety_shoe`, la seule
classe du parc qu'aucun modèle ne couvre. Il est public et compte 1 089 images
pour 2 184 instances -- plus du double de ce dont on dispose aujourd'hui (823).

Mais il est **inexportable par la voie normale** : son propriétaire n'a jamais
généré de version (`versions: []` renvoyé par l'API), et `roboflow.download()`
comme l'interface web ne savent exporter qu'une version. Le SDK échoue par
« Version number 1 is not found. »

Les données sont pourtant toutes accessibles, par trois points d'entrée que
l'API expose indépendamment des versions :

    POST /{workspace}/{projet}/search      -> enumere les identifiants d'images
    GET  /{workspace}/{projet}/images/{id} -> boites, labels, split, urls
    GET  <url de l'image>                  -> le fichier lui-meme

Ce script les recompose en un jeu YOLO standard. Il est reprenable : une image
déjà écrite n'est pas retéléchargée, une interruption ne coûte rien.

Format des boîtes
-----------------
L'API donne des pixels, centre + dimensions (`x`, `y`, `width`, `height`), avec
la taille de l'image dans `annotation.width/height`. YOLO attend les mêmes
grandeurs normalisées par la taille de l'image -- la conversion est donc une
simple division, sans changement de convention.

    python improvements/p16_recuperer_safety_shoes.py --cle <API_KEY>
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
SOURCES = RACINE / "ppe_detection/data/extracted/sources_roboflow"

# Projets sans version exportable rencontres sur ce projet. Le cas n'est pas
# marginal : deux des jeux les plus utiles pour `safety_shoe` sont dans ce cas,
# et ce sont justement ceux que personne n'a pris la peine de versionner.
#
# `classes` fixe AUSSI l'ordre des indices YOLO du jeu reconstruit : toute
# classe absente de cette liste est ignoree a l'ecriture, jamais devinee.
PROJETS = {
    "alqulayti": {
        "espace": "ahmed-alqulayti", "projet": "safety-shoes-dataset",
        "dest": "safety_shoes_alqulayti",
        # `person` est conservee ici pour des raisons historiques : p13 ne
        # retient que l'indice 0 de ce jeu.
        "classes": ["safety_shoe", "person"],
    },
    # Ajoute le 2026-08-19. C'est LE jeu qui manquait : 712 instances
    # `no_safety-shoe` contre 380 dans tout le corpus precedent. Le modele
    # rejete le 19 aout confondait basket de ville et chaussure de securite
    # faute d'avoir jamais vu les deux opposees ; ce jeu les oppose.
    "vertical_farming": {
        "espace": "vertical-farming-tvyjl", "projet": "safety_shoe-eimup-mn0qq",
        "dest": "safety_shoe_vertical_farming",
        "classes": ["safety-shoe", "no_safety-shoe"],
    },
}

# Renseignes par main() a partir du projet choisi.
DEST: Path = SOURCES
ESPACE = PROJET = ""
CLASSES: list[str] = []


def _json(url: str, payload: dict | None = None, essais: int = 3) -> dict:
    for tentative in range(essais):
        try:
            req = urllib.request.Request(
                url,
                data=(json.dumps(payload).encode() if payload else None),
                headers={"Content-Type": "application/json", "User-Agent": "curl/8"})
            return json.load(urllib.request.urlopen(req, timeout=60))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            if tentative == essais - 1:
                raise
    return {}


def enumerer(cle: str) -> list[str]:
    """Liste les identifiants de toutes les images du projet, par pages."""
    url = f"https://api.roboflow.com/{ESPACE}/{PROJET}/search?api_key={cle}"
    ids, offset = [], 0
    while True:
        d = _json(url, {"limit": 100, "offset": offset, "fields": ["id"]})
        lot = d.get("results", [])
        if not lot:
            break
        ids += [r["id"] for r in lot]
        offset += len(lot)
        print(f"  {len(ids)}/{d.get('total', '?')} identifiants", end="\r", flush=True)
        if offset >= d.get("total", 0):
            break
    print(" " * 40, end="\r")
    return ids


def recuperer_une(iid: str, cle: str) -> str:
    """Télécharge une image et écrit son label YOLO. Rend un code de résultat."""
    d = _json(f"https://api.roboflow.com/{ESPACE}/{PROJET}/images/{iid}?api_key={cle}")
    im = d.get("image", {})
    ann = im.get("annotation", {})
    boites, L, H = ann.get("boxes", []), ann.get("width"), ann.get("height")
    if not boites or not L or not H:
        return "sans_annotation"
    # L'API n'est PAS homogene d'un projet a l'autre : `ahmed-alqulayti` rend
    # des nombres, `vertical-farming` des chaines ("2088.50"). Sans cette
    # conversion, la division leve TypeError -- 793 images sur 1310 perdues au
    # premier passage du 2026-08-19, sans que le compteur d'erreurs dise
    # pourquoi. On convertit donc systematiquement plutot que de faire
    # confiance au type recu.
    L, H = float(L), float(H)

    # `valid` est le nom Roboflow, conserve tel quel : p13 le lit sous ce nom.
    split = im.get("split", "train")
    dossier_img = DEST / split / "images"
    dossier_lbl = DEST / split / "labels"
    cible = dossier_img / f"{iid}.jpg"
    if cible.exists() and (dossier_lbl / f"{iid}.txt").exists():
        return "deja"

    lignes = []
    for b in boites:
        if b["label"] not in CLASSES:
            continue
        # Pixels centre+dimensions -> YOLO normalise. Bornage a [0,1] : quelques
        # boites Roboflow debordent legerement du cadre et Ultralytics rejette
        # le fichier entier si une valeur sort de l'intervalle.
        bx, by = float(b["x"]), float(b["y"])
        bw, bh = float(b["width"]), float(b["height"])
        x, y = min(max(bx / L, 0.0), 1.0), min(max(by / H, 0.0), 1.0)
        w, h = min(bw / L, 1.0), min(bh / H, 1.0)
        if w <= 0 or h <= 0:
            continue
        lignes.append(f"{CLASSES.index(b['label'])} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
    if not lignes:
        return "sans_classe_utile"

    url_img = (im.get("urls", {}) or {}).get("original") \
        or f"https://source.roboflow.com/{im.get('owner', '')}/{iid}/original.jpg"
    try:
        req = urllib.request.Request(url_img, headers={"User-Agent": "curl/8"})
        donnees = urllib.request.urlopen(req, timeout=90).read()
    except Exception:
        return "image_indisponible"
    if len(donnees) < 1024:
        return "image_tronquee"

    dossier_img.mkdir(parents=True, exist_ok=True)
    dossier_lbl.mkdir(parents=True, exist_ok=True)
    cible.write_bytes(donnees)
    (dossier_lbl / f"{iid}.txt").write_text("\n".join(lignes) + "\n", encoding="utf-8")
    return "ok"


def main() -> int:
    global DEST, ESPACE, PROJET, CLASSES
    ap = argparse.ArgumentParser()
    ap.add_argument("--cle", required=True)
    ap.add_argument("--projet", choices=list(PROJETS), default="alqulayti",
                    help="jeu a reconstruire (voir PROJETS en tete de fichier)")
    ap.add_argument("--fils", type=int, default=8,
                    help="telechargements simultanes ; au-dela l'API limite")
    args = ap.parse_args()

    spec = PROJETS[args.projet]
    ESPACE, PROJET = spec["espace"], spec["projet"]
    CLASSES = spec["classes"]
    DEST = SOURCES / spec["dest"]

    print(f"enumeration de {ESPACE}/{PROJET} ...")
    ids = enumerer(args.cle)
    print(f"{len(ids)} images a recuperer\n")

    compte: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=args.fils) as pool:
        futurs = {pool.submit(recuperer_une, i, args.cle): i for i in ids}
        for n, f in enumerate(as_completed(futurs), 1):
            try:
                code = f.result()
            except Exception as e:
                code = f"erreur:{type(e).__name__}"
            compte[code] = compte.get(code, 0) + 1
            if n % 25 == 0:
                print(f"  {n}/{len(ids)}  {compte}", end="\r", flush=True)
    print(" " * 90, end="\r")

    print("\nresultat :")
    for k, v in sorted(compte.items(), key=lambda x: -x[1]):
        print(f"  {k:<20} {v}")

    if compte.get("ok", 0) or compte.get("deja", 0):
        (DEST / "data.yaml").write_text(
            f"path: {DEST}\ntrain: train/images\nval: valid/images\n\n"
            f"nc: {len(CLASSES)}\nnames: {CLASSES}\n", encoding="utf-8")
        for split in ("train", "valid", "test"):
            d = DEST / split / "images"
            if d.is_dir():
                print(f"  {split:<6} {len(list(d.glob('*')))} images")
        print(f"\n-> {DEST}")
        return 0
    print("rien recupere", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
