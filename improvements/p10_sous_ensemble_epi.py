#!/usr/bin/env python3
"""Sous-ensemble équilibré de `ppe_dataset`, pour un envoi possible sur Kaggle.

Pourquoi
--------
Le modèle EPI déployé (`ppe_detection/models/ppe_detector.pt`) ne détecte plus que deux
de ses quatorze classes. Mesuré le 2026-08-13, avec cas témoin : `Safety Vest`
95 % et `NO-Safety Vest` 85 % de détection de scène, **les douze autres à 0 %**,
et rien d'autre ne sort même en abaissant le seuil à 0,01.

La cause est un **oubli catastrophique** : le fine-tuning a tourné sur
`ppe_vest_clean_14c`, qui est exactement le sous-ensemble des images de
`ppe_dataset` contenant du gilet — 2 728 images où seules les classes 10 et 13
apparaissent. Sur de nombreuses époques, le réseau a réaffecté sa capacité aux
seules classes vues et effacé les autres.

Le correctif est donc de ré-entraîner sur `ppe_dataset` complet, qui est un
**sur-ensemble strict** : il contient les mêmes 4 499 `Safety Vest` et 1 435
`NO-Safety Vest`, plus les douze autres classes. Rien n'est perdu côté gilet.

Reste un obstacle matériel : `ppe_dataset` pèse 2,7 Go, soit ~14 h d'envoi sur
la liaison disponible (54 Ko/s), qui a déjà cassé sur 774 Mo. D'où ce script.

Méthode
-------
Sélection gloutonne guidée par la classe la moins couverte. À chaque tour, on
prend l'image qui apporte le plus d'instances des classes encore sous leur
quota. C'est ce qui évite le piège du tirage aléatoire : `Hardhat` compte 28 996
instances contre 734 pour `Ladder`, et un échantillon uniforme reproduirait ce
déséquilibre au lieu de le corriger.

Les images sont déjà en 640 px (57 Ko en moyenne), donc redimensionner
n'apporterait rien : seul le nombre d'images fait la taille.

    python improvements/p10_sous_ensemble_epi.py --quota 3000 --simulation
    python improvements/p10_sous_ensemble_epi.py --quota 3000
"""

from __future__ import annotations

import argparse
import shutil
from collections import Counter, defaultdict
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
SOURCE = RACINE / "ppe_detection/data/extracted/ppe_dataset"
NOMS = ['Fall-Detected', 'Gloves', 'Goggles', 'Hardhat', 'Ladder', 'Mask',
        'NO-Gloves', 'NO-Goggles', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest',
        'Person', 'Safety Cone', 'Safety Vest']


def lire_split(split: str) -> dict[str, Counter]:
    """{stem: Counter(classe -> instances)} pour un split."""
    dossier = SOURCE / split / "labels"
    contenu = {}
    for p in dossier.glob("*.txt"):
        c = Counter()
        for ligne in p.read_text().splitlines():
            if ligne.strip():
                c[int(ligne.split()[0])] += 1
        contenu[p.stem] = c
    return contenu


def selectionner(contenu: dict[str, Counter], quota: int,
                 completes: set[int] = frozenset()) -> list[str]:
    """Sélection par classes rares d'abord, jusqu'au quota de chacune.

    Une sélection gloutonne réévaluant toutes les images à chaque tour donnerait
    un résultat marginalement meilleur pour un coût quadratique — mesuré à plus
    de dix minutes sans terminer sur 30 765 images. Traiter les classes de la
    plus rare à la plus fréquente atteint le même objectif en temps linéaire :
    les images d'une classe rare sont prises en premier, et comme elles
    apportent aussi des instances des classes fréquentes, celles-ci atteignent
    leur quota sans effort.

    Le quota est plafonné par ce qui existe : viser 3 000 `Ladder` quand le jeu
    n'en contient que 734 ne servirait à rien.
    """
    total = Counter()
    for c in contenu.values():
        total.update(c)

    acquis = Counter()
    choisis: list[str] = []
    pris = set()

    # Classes prises INTEGRALEMENT, avant tout quota. Le gilet en fait partie :
    # c'est la seule paire de classes que le modele actuel detecte encore
    # (0.9173 / 0.8534), et en perdre un tiers par plafonnement serait
    # exactement la regression que ce re-entrainement doit eviter.
    for stem, c in contenu.items():
        if any(i in c for i in completes):
            choisis.append(stem)
            pris.add(stem)
            acquis.update(c)

    # Puis, de la classe la plus rare a la plus frequente.
    for classe in sorted((i for i in range(14) if total[i]), key=lambda i: total[i]):
        cible = min(quota, total[classe])
        if acquis[classe] >= cible:
            continue
        # Parmi les images contenant cette classe, prendre d'abord celles qui en
        # contiennent le plus : moins d'images pour la meme couverture.
        candidats = sorted(((s, c) for s, c in contenu.items()
                            if classe in c and s not in pris),
                           key=lambda sc: -sc[1][classe])
        for stem, c in candidats:
            if acquis[classe] >= cible:
                break
            choisis.append(stem)
            pris.add(stem)
            acquis.update(c)
    return choisis


def index_images(split: str) -> dict[str, Path]:
    """{stem: chemin} construit en un seul parcours du dossier.

    Indispensable : chercher chaque image par `glob(f"{stem}.*")` relance un
    balayage des 30 765 fichiers du dossier à chaque appel, ce qui rend le
    script quadratique et de fait interminable.
    """
    return {p.stem: p for p in (SOURCE / split / "images").iterdir() if p.is_file()}


def copier(split: str, stems: list[str], dest: Path, index: dict[str, Path]) -> int:
    octets = 0
    for sous in ("images", "labels"):
        (dest / split / sous).mkdir(parents=True, exist_ok=True)
    for stem in stems:
        src_img = index.get(stem)
        if src_img is None:
            continue
        src_lbl = SOURCE / split / "labels" / f"{stem}.txt"
        shutil.copy2(src_lbl, dest / split / "labels" / src_lbl.name)
        shutil.copy2(src_img, dest / split / "images" / src_img.name)
        octets += src_img.stat().st_size
    return octets


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quota", type=int, default=3000,
                    help="instances visees par classe dans le split train")
    ap.add_argument("--quota-val", type=int, default=400,
                    help="instances visees par classe dans le split val envoye")
    ap.add_argument("--sans-test", action="store_true",
                    help="ne pas copier le split test : il reste local, c'est lui "
                         "qui jugera le candidat")
    ap.add_argument("--sortie", default=str(RACINE / "ppe_detection/data/extracted/ppe_14c_equilibre"))
    ap.add_argument("--simulation", action="store_true",
                    help="affiche la composition sans rien copier")
    a = ap.parse_args()

    dest = Path(a.sortie)
    total_octets = 0
    for split in ("train", "val", "test"):
        contenu = lire_split(split)
        if not contenu:
            continue
        if split == "train":
            stems = selectionner(contenu, a.quota, completes={10, 13})
        elif split == "val":
            # Le `val` envoye ne sert qu'a suivre l'entrainement : un
            # echantillon equilibre suffit, et le reduire divise le temps
            # d'envoi. Le jugement, lui, se fera LOCALEMENT sur le `test`
            # complet et intact -- qui ne quitte donc pas la machine.
            stems = selectionner(contenu, a.quota_val, completes=set())
        else:
            stems = list(contenu)
            if a.sans_test:
                continue

        acquis = Counter()
        for s in stems:
            acquis.update(contenu[s])
        index = index_images(split)
        octets = (sum(index[s].stat().st_size for s in stems if s in index)
                  if a.simulation else copier(split, stems, dest, index))
        total_octets += octets

        print(f"\n=== {split} : {len(stems)} images ({octets/1e6:.0f} Mo) ===")
        for i in range(14):
            if acquis[i]:
                print(f"  {NOMS[i]:16} {acquis[i]:>6}")

    print(f"\nTOTAL : {total_octets/1e6:.0f} Mo"
          f"  -> {total_octets/54_000/3600:.1f} h d'envoi a 54 Ko/s")

    if not a.simulation:
        # `path: .` et non un chemin absolu : le jeu sera monte ailleurs sur
        # Kaggle, et un chemin de la machine locale y serait introuvable.
        lignes = ["path: .", "train: train/images", "val: val/images"]
        if (dest / "test" / "images").is_dir():
            lignes.append("test: test/images")
        lignes += ["", "nc: 14", f"names: {NOMS}"]
        (dest / "data.yaml").write_text("\n".join(lignes) + "\n", encoding="utf-8")
        print(f"\nEcrit dans {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
