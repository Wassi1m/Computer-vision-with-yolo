#!/usr/bin/env python3
"""P9 — Reconstruit le jeu feu/fumée pour lever le blocage de la classe `smoke`.

Constat mesuré (analyse de `fire_smoke_enriched`, août 2026) : `fire` plafonne à
90 % de mAP@50 quand `smoke` stagne à 40 %, et le fine-tuning P8 a dégradé
`smoke` au lieu de l'améliorer. La cause n'est ni le modèle ni le manque de
données — c'est le jeu lui-même, sur trois points :

1. **Une classe pour deux objets.** `smoke` fusionne deux sources aux
   conventions incompatibles : les panaches lointains de pyro-sdis (médiane
   39x32 px sur du 1280x720, soit 19x16 px une fois ramenés à imgsz=640) et la
   fumée rapprochée de Roboflow (médiane 307x364 px sur du 640x640). Un facteur
   19 en taille linéaire, avec une distribution franchement bimodale — le creux
   de l'histogramme se situe entre 1 % et 10 % de l'aire de l'image. Un
   détecteur assigne ces deux populations à des têtes de résolution
   différentes ; les réunir sous un même label revient à lui demander deux
   choses contradictoires.

2. **Le test ne mesure pas le cas d'usage.** Il ne contient aucune image
   pyro : 100 % du test `smoke` est de la fumée rapprochée, alors que 70 % de
   l'entraînement est de la fumée lointaine. Or c'est précisément la fumée
   lointaine qui correspond au déploiement visé (caméra de surveillance qui
   doit repérer un départ de feu à distance). On optimisait et on validait sur
   deux distributions différentes.

3. **La métrique est trop bruitée pour arbitrer.** L'AP `smoke` du test porte
   sur 99 instances. Le rejet du modèle P8 s'est joué sur 3,3 points d'écart —
   soit du bruit d'échantillonnage, pas une régression démontrée.

S'y ajoutent 28 images sources communes entre train et test (fuite), et des
doublons d'augmentation Roboflow qui gonflent le train (14 589 fichiers pour
7 438 images sources distinctes).

Ce script reconstruit donc le jeu :

- **Séparation de `smoke` en deux classes** selon l'aire relative de la boîte
  (seuil 2 %, placé dans le creux de l'histogramme) : `smoke` pour la fumée
  rapprochée, `smoke_distant` pour le panache lointain. Le pipeline d'inférence
  remappe les deux vers un même évènement `smoke` : le contrat d'API ne change
  pas pour la plateforme aval.
- **Découpage par image source**, doublons d'augmentation regroupés du même
  côté — plus aucune fuite entre les splits.
- **Test et validation stratifiés** sur les deux domaines, dimensionnés pour
  que chaque classe dispose d'assez d'instances pour une AP interprétable.
- **Pas de doublon d'augmentation en évaluation** : une seule copie par image
  source, sinon la métrique récompense la redondance.

    python improvements/p9_dataset_fumee.py
    python improvements/p9_dataset_fumee.py --seuil-distant 0.02 --test 0.12
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
SOURCE = RACINE / "surveillance_suite/data/dataset/fire_smoke_enriched"
DESTINATION = RACINE / "surveillance_suite/data/dataset/fire_smoke_v9"

CLASSES = ["fire", "smoke", "smoke_distant"]
FIRE, SMOKE, SMOKE_DISTANT = 0, 1, 2


def image_source(nom: str) -> str:
    """Identifiant de l'image d'origine, doublons d'augmentation confondus.

    Roboflow nomme ses variantes `<origine>_jpg.rf.<hash>.jpg` : plusieurs
    fichiers correspondent alors à une seule photo. Les regrouper est ce qui
    empêche qu'une version augmentée d'une image de test se retrouve dans le
    train.
    """
    return nom.split("_jpg.rf.")[0] if "_jpg.rf." in nom else nom.rsplit(".", 1)[0]


def lire_annotations(chemin: Path) -> list[tuple[int, float, float, float, float]]:
    boites = []
    for ligne in chemin.read_text(encoding="utf-8").splitlines():
        parts = ligne.split()
        if len(parts) < 5:
            continue
        boites.append((int(parts[0]), *(float(v) for v in parts[1:5])))
    return boites


def reclasser(boites, seuil: float):
    """Sépare `smoke` en fumée rapprochée / lointaine selon l'aire de la boîte.

    Le critère est l'aire et non la source : c'est la taille apparente qui
    détermine à quelle échelle le détecteur doit travailler, et une image pyro
    peut contenir un panache proche comme une image Roboflow une fumée
    lointaine.
    """
    sortie = []
    for cls, x, y, w, h in boites:
        if cls == 1:
            cls = SMOKE if w * h >= seuil else SMOKE_DISTANT
        sortie.append((cls, x, y, w, h))
    return sortie


def signature(boites) -> tuple:
    """Classes présentes dans l'image — clé de stratification du découpage."""
    return tuple(sorted({c for c, *_ in boites}))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=SOURCE)
    ap.add_argument("--destination", type=Path, default=DESTINATION)
    ap.add_argument("--seuil-distant", type=float, default=0.02,
                    help="aire relative sous laquelle une fumee est dite lointaine")
    ap.add_argument("--valid", type=float, default=0.10)
    ap.add_argument("--test", type=float, default=0.12)
    ap.add_argument("--graine", type=int, default=1234)
    args = ap.parse_args()

    if not args.source.is_dir():
        print(f"jeu source absent : {args.source}", file=sys.stderr)
        return 1

    # ── Regroupement de tous les fichiers par image source ───────────────────
    groupes: dict[str, list[tuple[Path, Path]]] = defaultdict(list)
    for split in ("train", "valid", "test"):
        dossier_img = args.source / split / "images"
        if not dossier_img.is_dir():
            continue
        for img in dossier_img.iterdir():
            lbl = args.source / split / "labels" / (img.stem + ".txt")
            if lbl.exists():
                groupes[image_source(img.name)].append((img, lbl))

    print(f"{sum(len(v) for v in groupes.values())} fichiers -> "
          f"{len(groupes)} images sources distinctes")

    # ── Stratification : signature de classes x domaine ──────────────────────
    strates: dict[tuple, list[str]] = defaultdict(list)
    annotations: dict[str, list] = {}
    for cle, fichiers in groupes.items():
        # le premier fichier fait foi : les variantes d'augmentation partagent
        # les memes objets, seule leur apparence differe
        boites = reclasser(lire_annotations(fichiers[0][1]), args.seuil_distant)
        annotations[cle] = boites
        strates[signature(boites)].append(cle)

    rng = random.Random(args.graine)
    repartition: dict[str, str] = {}
    for sig, cles in strates.items():
        rng.shuffle(cles)
        n = len(cles)
        n_test = round(n * args.test)
        n_valid = round(n * args.valid)
        for i, cle in enumerate(cles):
            repartition[cle] = ("test" if i < n_test
                                else "valid" if i < n_test + n_valid
                                else "train")

    # ── Écriture ─────────────────────────────────────────────────────────────
    if args.destination.exists():
        shutil.rmtree(args.destination)
    for split in ("train", "valid", "test"):
        for sous in ("images", "labels"):
            (args.destination / split / sous).mkdir(parents=True, exist_ok=True)

    compte_fichiers = Counter()
    compte_instances: dict[str, Counter] = defaultdict(Counter)
    compte_images: dict[str, Counter] = defaultdict(Counter)

    for cle, fichiers in groupes.items():
        split = repartition[cle]
        # En évaluation, une seule copie par image source : garder les variantes
        # d'augmentation y ferait compter plusieurs fois la même scène.
        retenus = fichiers if split == "train" else fichiers[:1]
        for img, lbl in retenus:
            boites = reclasser(lire_annotations(lbl), args.seuil_distant)
            if not boites:
                continue
            dest_img = args.destination / split / "images" / img.name
            dest_lbl = args.destination / split / "labels" / (img.stem + ".txt")
            shutil.copy2(img, dest_img)
            dest_lbl.write_text(
                "".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for c, x, y, w, h in boites),
                encoding="utf-8")
            compte_fichiers[split] += 1
            for c, *_ in boites:
                compte_instances[split][c] += 1
            for c in {c for c, *_ in boites}:
                compte_images[split][c] += 1

    (args.destination / "data.yaml").write_text(
        f"path: {args.destination}\n"
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n\n"
        f"nc: {len(CLASSES)}\n"
        f"names: {CLASSES}\n", encoding="utf-8")

    # ── Rapport ──────────────────────────────────────────────────────────────
    print(f"\nseuil fumee lointaine : aire < {args.seuil_distant}")
    print(f"{'split':8}{'images':>9}" + "".join(f"{n:>16}" for n in CLASSES))
    for split in ("train", "valid", "test"):
        ligne = f"{split:8}{compte_fichiers[split]:>9}"
        for c in range(len(CLASSES)):
            ligne += f"{compte_instances[split][c]:>9} inst"
        print(ligne)

    resume = {
        "seuil_distant": args.seuil_distant,
        "classes": CLASSES,
        "images_sources": len(groupes),
        "splits": {s: {"fichiers": compte_fichiers[s],
                       "instances": {CLASSES[c]: compte_instances[s][c] for c in range(len(CLASSES))},
                       "images_par_classe": {CLASSES[c]: compte_images[s][c] for c in range(len(CLASSES))}}
                   for s in ("train", "valid", "test")},
    }
    rapport = RACINE / "reports/v3_results/p9_dataset_fumee.json"
    rapport.parent.mkdir(parents=True, exist_ok=True)
    rapport.write_text(json.dumps(resume, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n-> {args.destination}")
    print(f"-> {rapport}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
