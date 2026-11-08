#!/usr/bin/env python3
"""P21 — Ajoute des gros plans au jeu `epi_gants_lunettes`, sans le déséquilibrer.

Pourquoi ce script
------------------
Mesuré le 2026-09-07 : sur le cadrage webcam, `epi_gants_lunettes.pt` (M5) ne
propose **aucune boîte**, ni pour les gants ni pour les lunettes, même à un seuil
de 0,03. Ce n'est pas un défaut de seuil : c'est un écart d'échelle. Dans le jeu
actuel, les boîtes plafonnent à **4,6 %** de l'image pour les gants et **1,1 %**
pour les lunettes. Sur une webcam, un visage en occupe 20 % et une main 30 %.

Le casque et le masque, eux, généralisent à ce cadrage — parce que leurs corpus
contiennent des gros plans (jusqu'à 46 % et 54 %). C'est la seule différence.

Ce script fabrique donc les gros plans manquants et les fond dans le jeu
existant, en respectant trois règles apprises sur ce projet.

Les trois règles, et ce qu'il en coûte de les ignorer
-----------------------------------------------------
1. **Le domaine chantier reste majoritaire.** C'est celui de la production, et
   M5 y vaut 95 % de rappel. Verser 76 000 gros plans contre 28 000 images de
   chantier retournerait la balance et détruirait ce qui marche. Le plan v7
   plafonne une source mono-domaine à un tiers du total (déjà appliqué à
   `safety_gloves`, cf. `p13_jeux_roboflow.py:159`) ; `--plafond-visages` le fait
   respecter ici.

2. **Les lunettes de vue relèvent de `NO-Goggles`.** Le jeu de visages sépare
   `with_glasses` de `without_glasses`, mais l'inspection visuelle montre des
   lunettes de vue et de soleil — jamais de lunettes de sécurité. Une monture
   correctrice n'est pas un EPI : les **deux** dossiers alimentent donc la classe
   négative. Étiqueter `with_glasses` en `Goggles` apprendrait au modèle que
   toute paire de lunettes vaut conformité.

3. **Une image versée doit être annotée en entier.** Un visage ou une main
   présents mais non annotés enseignent au modèle que ce sont du fond, et
   suppriment la détection au lieu de l'améliorer.

Pourquoi les visages sont recollés sur un fond
-----------------------------------------------
Les images du jeu de visages sont des recadrages serrés de 256×256 : le visage
occupe **toute** l'image. Versées telles quelles, elles apprendraient « boîte =
image entière », ce qui est encore un autre cadrage que celui de la webcam, où
le visage occupe un cinquième. On recolle donc chaque visage sur un fond plus
grand, à une échelle tirée au sort dans la plage réellement observée.

Le fond est fabriqué à partir du visage lui-même, agrandi et fortement flouté.
C'est laid mais c'est le seul choix sûr : prendre une vraie photo de scène comme
fond y ferait entrer des personnes non annotées, ce qui viole la règle 3.

⚠️ La classe `Goggles` en gros plan
------------------------------------
Aucun jeu public ne fournit de vraies lunettes de sécurité vues de près. Sans
elles, ce script ne verse QUE des négatifs en gros plan, et le modèle apprendra
« grand visage ⇒ pas de lunettes » — il annoncerait « Sans lunettes » même sur
un porteur en règle. `--captures` verse les prises de `p20_capture_lunettes.py`,
qui fournissent cette classe. Le script REFUSE de s'exécuter sans elles, sauf
`--sans-positifs-assume` : ce défaut doit être un choix explicite, jamais un
oubli.

Usage
-----
    python improvements/p21_jeu_gros_plan.py \
        --base   ppe_detection/data/extracted/epi_gants_lunettes \
        --gants  /chemin/glove-and-no-gloves-2-3k \
        --visages /chemin/facecropped-glasses-vs-noglasses \
        --captures ppe_detection/data/extracted/lunettes_gros_plan \
        --sortie ppe_detection/data/extracted/epi_gants_lunettes_gp
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

#: Ordre des classes du jeu `epi_gants_lunettes`. Relu du data.yaml de la base
#: plutôt que supposé : un décalage d'indice ici empoisonnerait tout le jeu.
NOMS_ATTENDUS = ["Gloves", "NO-Gloves", "Goggles", "NO-Goggles"]

#: Correspondance des classes du jeu de gants téléchargé.
MAP_GANTS = {"gloves": "Gloves", "no-gloves": "NO-Gloves"}

#: Rapport largeur/hauteur des boîtes du jeu existant (médiane de 401 boîtes
#: `NO-Goggles` mesurées le 2026-09-07). Voir `p20_capture_lunettes.py`.
RATIO = 1.27

#: Fraction de l'image que le visage recollé doit occuper. La borne basse
#: recouvre le haut de la plage du jeu existant (1,1 %), la borne haute dépasse
#: le cas webcam mesuré (~20 %) pour ne pas apprendre une seule échelle.
AIRE_MIN, AIRE_MAX = 0.02, 0.35

COTE_SORTIE = 640


def lire_noms(base: Path) -> list[str]:
    for ligne in (base / "data.yaml").read_text(encoding="utf-8").splitlines():
        if ligne.startswith("names:"):
            return [n.strip(" '\"") for n in
                    ligne.split("[", 1)[-1].rstrip("]").split(",")]
    raise SystemExit(f"pas de `names:` dans {base / 'data.yaml'}")


def _ecrire(dest: Path, prefixe: str, compteur: int, image_src: Path | np.ndarray,
            lignes: list[str]) -> None:
    nom = f"{prefixe}_{compteur:06d}"
    if isinstance(image_src, Path):
        shutil.copy2(image_src, dest / "images" / f"{nom}{image_src.suffix}")
    else:
        cv2.imwrite(str(dest / "images" / f"{nom}.jpg"), image_src)
    (dest / "labels" / f"{nom}.txt").write_text("\n".join(lignes) + "\n",
                                                encoding="utf-8")


def verser_base(base: Path, sortie: Path) -> dict[str, int]:
    """Recopie le jeu existant tel quel. Il est la référence, on n'y touche pas."""
    comptes = {"train": 0, "val": 0}
    for split in ("train", "val"):
        src_i, src_l = base / split / "images", base / split / "labels"
        if not src_i.is_dir():
            continue
        for lbl in sorted(src_l.glob("*.txt")):
            img = next(src_i.glob(lbl.stem + ".*"), None)
            if img is None:
                continue
            shutil.copy2(img, sortie / split / "images" / img.name)
            shutil.copy2(lbl, sortie / split / "labels" / lbl.name)
            comptes[split] += 1
    return comptes


def verser_gants(source: Path, sortie: Path, noms: list[str],
                 part_val: float, alea: random.Random) -> dict[str, int]:
    """Verse le jeu de gants en gros plan, en remappant ses deux classes."""
    src_noms = lire_noms(source)
    correspondance = {}
    for i, n in enumerate(src_noms):
        cible = MAP_GANTS.get(n.strip().lower())
        if cible is None:
            print(f"  classe « {n} » ignorée (hors gants)", file=sys.stderr)
            continue
        correspondance[i] = noms.index(cible)

    comptes = {"train": 0, "val": 0}
    for split_src in ("train", "valid", "test"):
        dossier = source / split_src
        if not (dossier / "labels").is_dir():
            continue
        for lbl in sorted((dossier / "labels").glob("*.txt")):
            gardees = []
            for ligne in lbl.read_text(encoding="utf-8").splitlines():
                champs = ligne.split()
                if len(champs) >= 5 and int(champs[0]) in correspondance:
                    gardees.append(
                        f"{correspondance[int(champs[0])]} " + " ".join(champs[1:5]))
            if not gardees:
                continue
            img = next((dossier / "images").glob(lbl.stem + ".*"), None)
            if img is None:
                continue
            split = "val" if alea.random() < part_val else "train"
            _ecrire(sortie / split, "gantsgp", comptes[split], img, gardees)
            comptes[split] += 1
    return comptes


def _recoller(visage: np.ndarray, alea: random.Random) -> tuple[np.ndarray, str]:
    """Pose un recadrage de visage sur un fond plus grand, à une échelle tirée.

    Renvoie l'image composée et la ligne YOLO (sans l'indice de classe).
    """
    aire = alea.uniform(AIRE_MIN, AIRE_MAX)
    cote = int(round((aire * COTE_SORTIE * COTE_SORTIE) ** 0.5))
    cote = max(24, min(cote, COTE_SORTIE - 8))

    # Fond tiré des couleurs du visage, mais réduit à 6×6 avant d'être réétiré :
    # il ne subsiste que des aplats, aucune structure de visage. Un simple flou
    # gaussien ne suffisait pas — le visage restait reconnaissable, et le modèle
    # aurait appris qu'un grand visage flou est du FOND, exactement l'inverse de
    # ce qu'on veut lui enseigner. Constaté en relisant les images produites.
    fond = cv2.resize(cv2.resize(visage, (6, 6), interpolation=cv2.INTER_AREA),
                      (COTE_SORTIE, COTE_SORTIE), interpolation=cv2.INTER_LINEAR)
    fond = cv2.GaussianBlur(fond, (0, 0), sigmaX=COTE_SORTIE / 24)
    petit = cv2.resize(visage, (cote, cote), interpolation=cv2.INTER_AREA)
    x = alea.randint(0, COTE_SORTIE - cote)
    y = alea.randint(0, COTE_SORTIE - cote)
    fond[y:y + cote, x:x + cote] = petit

    # Boîte à la convention du jeu existant : centrée sur le visage, plus large
    # que haute (rapport 1,27), et non le carré du recadrage.
    largeur = cote * 0.95
    hauteur = largeur / RATIO
    cx, cy = x + cote / 2, y + cote / 2
    return fond, (f"{cx / COTE_SORTIE:.6f} {cy / COTE_SORTIE:.6f} "
                  f"{largeur / COTE_SORTIE:.6f} {hauteur / COTE_SORTIE:.6f}")


def verser_visages(source: Path, sortie: Path, noms: list[str], plafond: int,
                   part_val: float, alea: random.Random) -> dict[str, int]:
    """Verse des visages en gros plan, TOUS en `NO-Goggles` (voir règle 2)."""
    indice = noms.index("NO-Goggles")
    fichiers = [p for p in source.rglob("*")
                if p.suffix.lower() in (".jpg", ".jpeg", ".png") and p.is_file()]
    alea.shuffle(fichiers)
    fichiers = fichiers[:plafond]

    comptes = {"train": 0, "val": 0}
    for chemin in fichiers:
        visage = cv2.imread(str(chemin))
        if visage is None:
            continue
        image, boite = _recoller(visage, alea)
        split = "val" if alea.random() < part_val else "train"
        _ecrire(sortie / split, "visagegp", comptes[split], image,
                [f"{indice} {boite}"])
        comptes[split] += 1
    return comptes


def verser_captures(source: Path, sortie: Path, part_val: float,
                    alea: random.Random) -> dict[str, int]:
    """Verse les prises de `p20`, déjà annotées aux bons indices."""
    comptes = {"train": 0, "val": 0}
    labels = source / "labels"
    if not labels.is_dir():
        return comptes
    for lbl in sorted(labels.glob("*.txt")):
        img = next((source / "images").glob(lbl.stem + ".*"), None)
        if img is None:
            continue
        lignes = [l for l in lbl.read_text(encoding="utf-8").splitlines() if l.strip()]
        if not lignes:
            continue
        split = "val" if alea.random() < part_val else "train"
        _ecrire(sortie / split, "capturegp", comptes[split], img, lignes)
        comptes[split] += 1
    return comptes


def compter_classes(dossier: Path, noms: list[str]) -> dict[str, int]:
    compte = {n: 0 for n in noms}
    for lbl in (dossier / "labels").glob("*.txt"):
        for ligne in lbl.read_text(encoding="utf-8").splitlines():
            champs = ligne.split()
            if champs:
                compte[noms[int(champs[0])]] += 1
    return compte


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--gants", type=Path, required=True)
    ap.add_argument("--visages", type=Path, required=True)
    ap.add_argument("--captures", type=Path,
                    help="prises de p20_capture_lunettes.py (classe Goggles en gros plan)")
    ap.add_argument("--sortie", type=Path, required=True)
    ap.add_argument("--plafond-visages", type=int, default=8000,
                    help="au-delà, les gros plans écrasent le domaine chantier")
    ap.add_argument("--part-val", type=float, default=0.15)
    ap.add_argument("--graine", type=int, default=0)
    ap.add_argument("--sans-positifs-assume", action="store_true",
                    help="construit sans aucun Goggles en gros plan, en acceptant "
                         "que le modele annonce « Sans lunettes » sur un porteur en regle")
    a = ap.parse_args()

    noms = lire_noms(a.base)
    if noms != NOMS_ATTENDUS:
        raise SystemExit(f"classes inattendues dans la base : {noms}")

    captures_presentes = a.captures and (a.captures / "labels").is_dir() and \
        any((a.captures / "labels").glob("*.txt"))

    # Ce qui est dangereux n'est pas l'absence de captures : c'est de verser des
    # `NO-Goggles` en gros plan sans un seul `Goggles` à la même échelle. Sans
    # contre-exemple, le modèle apprend « grand visage ⇒ pas de lunettes » sans
    # exception. Avec `--plafond-visages 0` aucun gros plan de visage n'entre :
    # les lunettes restent exactement ce qu'elles sont aujourd'hui, et il n'y a
    # rien à déséquilibrer — c'est la voie choisie quand seuls les gants sont
    # traités.
    if a.plafond_visages > 0 and not captures_presentes and not a.sans_positifs_assume:
        print("REFUS : des NO-Goggles en gros plan seraient verses sans aucun\n"
              "Goggles a la meme echelle. Le modele apprendrait « grand visage =>\n"
              "pas de lunettes » sans exception, et annoncerait « Sans lunettes »\n"
              "meme sur un porteur en regle.\n\n"
              "Trois issues :\n"
              "  1. fournir les positifs :\n"
              "     python improvements/p20_capture_lunettes.py --port avec --images 150\n"
              "  2. ne traiter que les gants : --plafond-visages 0\n"
              "  3. forcer en connaissance de cause : --sans-positifs-assume",
              file=sys.stderr)
        return 1

    alea = random.Random(a.graine)
    for split in ("train", "val"):
        for quoi in ("images", "labels"):
            (a.sortie / split / quoi).mkdir(parents=True, exist_ok=True)

    print("1/4 jeu existant (domaine chantier, il reste la reference)")
    print("   ", verser_base(a.base, a.sortie))
    print("2/4 gants en gros plan")
    print("   ", verser_gants(a.gants, a.sortie, noms, a.part_val, alea))
    print(f"3/4 visages en gros plan (plafond {a.plafond_visages}, tous NO-Goggles)")
    print("   ", verser_visages(a.visages, a.sortie, noms, a.plafond_visages,
                                a.part_val, alea))
    if captures_presentes:
        print("4/4 captures webcam (Goggles en gros plan)")
        print("   ", verser_captures(a.captures, a.sortie, a.part_val, alea))
    else:
        print("4/4 AUCUNE capture — construit sans positifs en gros plan, assume")

    (a.sortie / "data.yaml").write_text(
        f"path: {a.sortie.resolve()}\ntrain: train/images\nval: val/images\n\n"
        f"nc: {len(noms)}\nnames: {noms}\n", encoding="utf-8")

    print("\nInstances par classe :")
    for split in ("train", "val"):
        print(f"  {split:5s}", compter_classes(a.sortie / split, noms))
    print(f"\nJeu ecrit dans {a.sortie}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
