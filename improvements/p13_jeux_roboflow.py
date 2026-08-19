#!/usr/bin/env python3
"""P13 — Construit les jeux fusionnés `epi_casque` et `epi_gants_lunettes`.

Suite de `p12_audit_sources_roboflow.py`, dont les verdicts fixent le contenu
des tables ci-dessous. Rien n'est deviné sur les noms de classe : chaque
correspondance est écrite, et toute classe absente d'une table est ignorée.

Ce que l'audit du 2026-08-17 a changé par rapport au plan v8
------------------------------------------------------------
- **Le modèle chaussures est abandonné.** `PPEs` annonce `shoes`/`no_shoes`
  mais ne les porte que sur 51 images (21 + 30) ; avec les 273 images de
  `safety_shoes_detection`, le total plafonne à ~294 images pour la classe
  positive et 42 pour la négative. `safety_shoe` reste donc le seul vrai trou
  du parc, et c'est une donnée à acquérir, pas à entraîner.
- **`PPEs` est en réalité un jeu gants + lunettes.** Ses classes `helmet`,
  `mask`, `no_helmet` et `no_mask` sont totalement vides (0 instance), malgré
  la version « allclasses » de son intitulé.
- **`head` de Hard Hat Universe désigne bien une tête nue** : sur 4 781 boîtes,
  3 seulement recouvrent un casque (IoU >= 0.5). Ce sont des erreurs
  d'annotation, filtrées ici plutôt que d'écarter le jeu entier.
- **`construction_helmet_detection` est écarté** : une seule classe `head`,
  présente sur la totalité de ses 4 916 images, sans aucune classe casque pour
  situer ce que `head` désigne. Contrairement à Hard Hat Universe, rien ne
  permet d'y vérifier que ce sont des têtes nues.

Sécurité
--------
Aucun modèle en place n'est touché : ces jeux servent à entraîner des modèles
NEUFS (`epi_casque.pt`, `epi_gants_lunettes.pt`) qui se brancheront en cascade,
comme `masque_gilet.pt` depuis le 2026-08-17. Un modèle neuf ne peut pas
oublier ce qu'il n'a jamais appris.

Le split `test` de `ppe_dataset` n'est JAMAIS copié : il reste local et intact
pour juger les candidats sans qu'ils aient pu le voir -- même règle que
`p10_sous_ensemble_epi.py` et `p11_jeu_masque_gilet.py`.

    python improvements/p13_jeux_roboflow.py --simulation
    python improvements/p13_jeux_roboflow.py --jeu casque
    python improvements/p13_jeux_roboflow.py --jeu gants
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
EXTRAIT = RACINE / "ppe_detection/data/extracted"
SOURCES = EXTRAIT / "sources_roboflow"

# Plafond de la contribution de `ppe_dataset` au jeu casque. L'objectif est la
# DIVERSITE (le plan v7 l'a établi : 28 996 exemples locaux n'ont pas suffi,
# ils viennent des mêmes chantiers sous les mêmes angles). Laisser les 13 561
# images locales noyer les 5 840 images externes reproduirait exactement le
# déséquilibre qui a fait plafonner `Hardhat` à 65 %.
PLAFOND_LOCAL = 6000

# --- Images de fond -----------------------------------------------------
# La campagne du 2026-08-17 a entraîné les trois modèles avec `0 backgrounds`,
# faute d'avoir prévu le cas : `collecter()` ne retient une image que si elle
# porte au moins une annotation de la classe cible. Un détecteur entraîné ainsi
# n'apprend pas « voici l'objet », il apprend « il y en a toujours un quelque
# part, trouve-le ». Le modèle chaussures l'a démontré en dessinant des boîtes à
# 0.94 de confiance sur de la neige, un visage et un torse.
#
# Le jeu `portes` sert de source : ce sont des scènes intérieures sans aucune
# personne ni EPI (vérifié à l'œil le 2026-08-18). On ne peut PAS utiliser
# `ppe_dataset` pour cela : c'est un patchwork où l'absence d'annotation ne
# garantit pas l'absence de l'objet, et lui apprendre « casque = fond » est
# exactement le mécanisme qui a effacé douze classes en juillet.
FOND_PORTES = (RACINE /
               "surveillance_suite/data/dataset/Door - Open - Closed -.v1i.folder")
PROPORTION_FOND = 0.25  # part d'images de fond dans le jeu final

JEUX = {
    "casque": {
        "dest": "epi_casque",
        "noms": ["Hardhat", "NO-Hardhat"],
        "sources": [
            # Hard Hat Universe : `hi-viz helmet` et `hi-viz vest` sont vides
            # (audit P12), mais la correspondance est écrite quand même -- une
            # version future du jeu pourrait les peupler.
            {"jeu": "hard_hat_universe", "racine": SOURCES,
             "map": {1: 0, 2: 0, 0: 1}, "splits": ("train", "valid"),
             "filtre": "tete_nue"},
            {"jeu": "construction_ppe", "racine": SOURCES,
             "map": {0: 0, 1: 1}, "splits": ("train", "valid")},
            {"jeu": "ppe_dataset", "racine": EXTRAIT,
             "map": {3: 0, 8: 1}, "splits": ("train", "val"),
             "plafond": PLAFOND_LOCAL},
        ],
    },
    # Les splits d'origine sont INUTILISABLES pour ce concept : le `train` de
    # PPEs n'annote la chaussure que sur 21 images quand son `valid` en compte
    # 348. Entraîner sur ces splits tels quels reviendrait à ne rien apprendre
    # puis à valider sur l'essentiel des données. On met donc tout en commun et
    # on redécoupe soi-même, à graine fixe.
    "chaussures": {
        "dest": "epi_chaussures",
        "noms": ["safety_shoe", "NO-safety_shoe"],
        "repartition": 0.8,
        "sources": [
            {"jeu": "ppes", "racine": SOURCES,
             "map": {10: 0, 9: 1}, "splits": ("train", "valid", "test")},
            # `Safety-shoes` (0) et `safety_shoe` (2) désignent le même concept :
            # deux classes pour une seule idée, artefact d'annotation constaté
            # par l'audit P12. Elles sont fusionnées plutôt qu'une des deux
            # ignorée -- 81 images valent d'être gardées sur un jeu si maigre.
            {"jeu": "safety_shoes_detection", "racine": SOURCES,
             "map": {2: 0, 0: 0, 1: 1}, "splits": ("train", "valid", "test")},
            # AJOUT du 2026-08-18. Ce jeu triple le corpus (2 184 instances
            # `safety_shoe` contre 1 699 pour les deux autres sources reunies).
            # Il n'etait pas telechargeable par la voie normale -- son
            # proprietaire n'a jamais genere de version Roboflow -- et a du etre
            # reconstruit image par image via l'API
            # (improvements/p16_recuperer_safety_shoes.py).
            # `person` (1) est ignoree : ce modele ne detecte que la chaussure,
            # la personne etant deja servie par ppe_detector.pt dans la cascade.
            {"jeu": "safety_shoes_alqulayti", "racine": SOURCES,
             "map": {0: 0}, "splits": ("train", "valid", "test")},
            # AJOUTS du 2026-08-19, apres le rejet du candidat du 19.
            # Le diagnostic n'etait plus le volume mais l'ANNOTATION : aucune
            # source n'opposait chaussure de securite et chaussure ordinaire, si
            # bien que le modele avait appris a localiser des pieds chausses. Il
            # donnait des baskets de ville pour des chaussures de securite a
            # 0.86, et des cones de signalisation avec.
            #
            # `xszud` est la piece maitresse : 465 images `no_safety-shoe`
            # contre 380 dans tout le corpus precedent. `Helmet` (0) est
            # ignoree, servie ailleurs dans la cascade par epi_casque.pt.
            {"jeu": "safety_shoe_xszud", "racine": SOURCES,
             "map": {2: 0, 1: 1}, "splits": ("train", "valid", "test")},
            # `nedrick` n'apporte presque aucun negatif (46 images) mais 1 436
            # images positives d'un domaine nouveau. C'est la version etendue
            # du `safety_shoes_detection` deja present ; les doublons eventuels
            # sont sans gravite, le prefixe de `_ecrire` empechant tout
            # ecrasement silencieux.
            {"jeu": "safety_shoes_nedrick", "racine": SOURCES,
             "map": {1: 0, 0: 1}, "splits": ("train", "valid", "test")},
            # `vertical_farming` n'a, lui non plus, aucune version exportable :
            # reconstruit image par image par p16 (--projet vertical_farming).
            # 712 instances negatives a la source, sans augmentation.
            {"jeu": "safety_shoe_vertical_farming", "racine": SOURCES,
             "map": {0: 0, 1: 1}, "splits": ("train", "valid", "test")},
        ],
    },
    "gants": {
        "dest": "epi_gants_lunettes",
        "noms": ["Gloves", "NO-Gloves", "Goggles", "NO-Goggles"],
        "sources": [
            # PPEs annote gants ET lunettes sur les mêmes images : c'est le
            # socle multiclasse qui limite le piège du fond.
            {"jeu": "ppes", "racine": SOURCES,
             "map": {0: 0, 5: 1, 1: 2, 6: 3}, "splits": ("train", "valid")},
            # Safety Gloves est mono-concept : sur ses images, la zone des
            # lunettes devient du fond. Le plan v7 plafonne ce type de source à
            # un tiers du total ; ici 9 015 / 28 431 = 31.7 %, sous le plafond.
            {"jeu": "safety_gloves", "racine": SOURCES,
             "map": {0: 0, 1: 1}, "splits": ("train", "valid")},
            # AJOUT du 2026-08-18. Son absence explique le rejet du candidat du
            # 17 : `epi_casque` avait 6 000 images de `ppe_dataset` et a battu
            # `ppe_detector.pt` (+0.095 / +0.203) ; `epi_gants_lunettes` n'en
            # avait AUCUNE et a fait jeu égal en perdant 0.020 sur NO-Gloves.
            # Le modèle avait appris un seul domaine, puis a été jugé dans un
            # autre. Ces images lui donnent les deux.
            {"jeu": "ppe_dataset", "racine": EXTRAIT,
             "map": {1: 0, 6: 1, 2: 2, 7: 3}, "splits": ("train", "val"),
             "plafond": PLAFOND_LOCAL},
        ],
    },
}


def iou_xywh(a, b) -> float:
    def coins(c):
        cx, cy, w, h = c
        return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    ax1, ay1, ax2, ay2 = coins(a)
    bx1, by1, bx2, by2 = coins(b)
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def filtrer_tete_nue(gardees: list[tuple[int, list[str]]]) -> list[tuple[int, list[str]]]:
    """Retire les boîtes `NO-Hardhat` qui recouvrent un `Hardhat`.

    L'audit P12 en a trouvé 3 sur 4 781 dans Hard Hat Universe -- du bruit
    d'annotation, mais sur la classe la plus critique du parc : une tête casquée
    étiquetée `NO-Hardhat` apprendrait au modèle qu'un ouvrier en règle est une
    infraction.
    """
    casques = [tuple(float(v) for v in c) for cl, c in gardees if cl == 0]
    return [(cl, c) for cl, c in gardees
            if cl != 1 or not any(
                iou_xywh(tuple(float(v) for v in c), k) >= 0.5 for k in casques)]


def collecter(src: dict, split_src: str) -> list[tuple[Path, list[tuple[int, list[str]]]]]:
    """Lit un split d'une source et rend les (image, boîtes remappées) retenues."""
    base = src["racine"] / src["jeu"] / split_src
    labels, images = base / "labels", base / "images"
    if not labels.is_dir():
        return []

    retenues = []
    for lbl in sorted(labels.glob("*.txt")):
        gardees = []
        for ligne in lbl.read_text(encoding="utf-8").splitlines():
            champs = ligne.split()
            if len(champs) >= 5 and int(champs[0]) in src["map"]:
                gardees.append((src["map"][int(champs[0])], champs[1:5]))
        if not gardees:
            continue
        if src.get("filtre") == "tete_nue":
            gardees = filtrer_tete_nue(gardees)
            if not gardees:
                continue
        img = next(images.glob(lbl.stem + ".*"), None)
        if img is not None:
            retenues.append((img, gardees))
    return retenues


def _ecrire(dest: Path, split: str, jeu: str, img: Path,
            boites: list[tuple[int, list[str]]]) -> None:
    """Copie une image et son label remappé. Le préfixe évite qu'un même nom de
    fichier présent dans deux sources en écrase silencieusement un autre."""
    (dest / split / "images").mkdir(parents=True, exist_ok=True)
    (dest / split / "labels").mkdir(parents=True, exist_ok=True)
    tige = f"{jeu}__{img.stem}"
    shutil.copy2(img, dest / split / "images" / f"{tige}{img.suffix}")
    (dest / split / "labels" / f"{tige}.txt").write_text(
        "\n".join(f"{c} {' '.join(co)}" for c, co in boites) + "\n",
        encoding="utf-8")


def ajouter_fonds(dest: Path, total: dict, rng: random.Random,
                  simulation: bool) -> dict:
    """Ajoute des images SANS objet, avec un fichier label vide.

    C'est la correction du défaut central de la campagne du 2026-08-17 (voir le
    commentaire de `FOND_PORTES`). Un fichier label vide est la convention
    Ultralytics pour « cette image ne contient rien » : elle apprend au réseau à
    ne rien produire, ce qu'aucune image annotée ne peut lui enseigner.

    Effet attendu sur les métriques : l'AP peut BAISSER de quelques points, le
    modèle devenant plus prudent. C'est le compromis assumé -- un modèle qui
    détecte un peu moins mais n'hallucine pas est exploitable, l'inverse non.
    """
    if not FOND_PORTES.is_dir():
        print(f"  /!\\ source de fond absente ({FOND_PORTES}) -- aucun fond ajoute")
        return {"train": 0, "val": 0}

    disponibles = sorted(p for p in FOND_PORTES.rglob("*")
                         if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    rng.shuffle(disponibles)

    # fond / (fond + annotees) = PROPORTION_FOND, pour chaque split
    vises = {s: int(total[s] * PROPORTION_FOND / (1 - PROPORTION_FOND))
             for s in ("train", "val")}
    # Si le stock ne suffit pas, les deux splits sont réduits dans la même
    # proportion. Servir `train` en premier laisserait `val` à zéro -- et une
    # validation sans image de fond ne mesure justement pas ce qu'on cherche à
    # contrôler ici : les fausses détections sur scène vide.
    demande = sum(vises.values())
    if demande > len(disponibles):
        facteur = len(disponibles) / demande
        vises = {s: int(v * facteur) for s, v in vises.items()}
        print(f"  /!\\ stock de fonds insuffisant ({len(disponibles)} pour "
              f"{demande} demandees) -- les deux splits sont reduits d'autant")

    ajoutes = {}
    curseur = 0
    for split in ("train", "val"):
        vise = vises[split]
        lot = disponibles[curseur:curseur + vise]
        curseur += len(lot)
        ajoutes[split] = len(lot)
        print(f"  fonds ({split:<5}) {len(lot):>6} images sans objet"
              + ("" if len(lot) == vise else f"  /!\\ {vise} demandees, stock epuise"))
        if simulation:
            continue
        (dest / split / "images").mkdir(parents=True, exist_ok=True)
        (dest / split / "labels").mkdir(parents=True, exist_ok=True)
        for img in lot:
            tige = f"fond__{img.stem}"
            shutil.copy2(img, dest / split / "images" / f"{tige}{img.suffix}")
            # Label VIDE : la convention Ultralytics pour une image de fond.
            (dest / split / "labels" / f"{tige}.txt").write_text("", encoding="utf-8")
    return ajoutes


def grouper_quasi_doublons(entrees: list, seuil: int = 5) -> list[int]:
    """Regroupe les images quasi identiques, quelle que soit leur source.

    Pourquoi c'est indispensable ici, et pourquoi le nom de fichier ne suffit
    pas (constat du 2026-08-19)
    ---------------------------------------------------------------------
    Deux mecanismes distincts produisent des images quasi identiques dans ce
    corpus, et AUCUN ne se voit dans les noms de fichiers :

    1. **L'augmentation Roboflow.** Une meme photo ressort en 2 a 3 variantes,
       sous des empreintes differentes.
    2. **La video.** Plusieurs sources sont des extractions d'images de
       videosurveillance. `ppes` et `safety_shoe_vertical_farming` filment la
       meme installation : 3 124 paires quasi identiques entre elles, parfois a
       une seconde d'intervalle, sous des identifiants sans rapport.

    Repartir ces images au hasard en met de part et d'autre du decoupage, et la
    mAP de validation mesure alors de la MEMORISATION. C'est precisement ce qui
    a produit le 0.912 du candidat rejete le 19 aout, qui dessinait pourtant des
    boites sur des cones de signalisation.

    On calcule donc une empreinte perceptuelle (dHash 8x8) et on relie toutes
    les images a distance de Hamming <= `seuil`. Chaque groupe est ensuite
    affecte ENTIER a un seul split.
    """
    import cv2
    import numpy as np

    empreintes, valides = [], []
    for idx, (_, img, _) in enumerate(entrees):
        im = cv2.imread(str(img), cv2.IMREAD_GRAYSCALE)
        if im is None:
            continue
        im = cv2.resize(im, (9, 8))
        empreintes.append((im[:, 1:] > im[:, :-1]).flatten())
        valides.append(idx)

    X = np.packbits(np.array(empreintes, dtype=bool), axis=1)
    pere = list(range(len(entrees)))

    def racine(i: int) -> int:
        while pere[i] != i:
            pere[i] = pere[pere[i]]
            i = pere[i]
        return i

    popcount = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(1)
    for debut in range(0, len(X), 512):  # par blocs : la matrice complete ne tient pas
        bloc = X[debut:debut + 512]
        d = popcount[(bloc[:, None, :] ^ X[None, :, :])].sum(2)
        for local, global_ in enumerate(range(debut, min(debut + 512, len(X)))):
            for autre in np.where(d[local] <= seuil)[0]:
                if autre > global_:
                    ra, rb = racine(valides[global_]), racine(valides[int(autre)])
                    if ra != rb:
                        pere[rb] = ra
    return [racine(i) for i in range(len(entrees))]


def _construire_repartie(spec: dict, dest: Path, noms: list[str],
                         rng: random.Random, simulation: bool,
                         total: dict, instances: dict) -> int:
    """Met toutes les sources en commun, puis redécoupe train/val/test soi-même.

    Réservé aux concepts dont les splits d'origine sont inexploitables (voir le
    commentaire de l'entrée `chaussures`).

    Deux garde-fous, tous deux payés par une campagne perdue :

    - **Le découpage porte sur des GROUPES de quasi-doublons**, jamais sur des
      images isolées (voir `grouper_quasi_doublons`). Sans cela la validation
      fuit et son chiffre ne veut rien dire.
    - **Un split `test` est réservé d'emblée.** Les deux campagnes précédentes
      n'en avaient aucun : faute d'images jamais vues, le verdict reposait sur
      un proxy — le taux de déclenchement sur un jeu qui n'annote même pas le
      concept. Un jeu de test se prévoit avant d'entraîner, pas après.

    La stratification se fait sur la classe la plus rare présente dans le
    groupe : un tirage naïf peut vider le test de sa classe négative et rendre
    la mesure ininterprétable.
    """
    entrees = []
    for src in spec["sources"]:
        for split_src in src["splits"]:
            for img, boites in collecter(src, split_src):
                entrees.append((src["jeu"], img, boites))

    print(f"  regroupement des quasi-doublons sur {len(entrees)} images ...")
    groupes = grouper_quasi_doublons(entrees)
    par_groupe: dict[int, list] = {}
    for entree, g in zip(entrees, groupes):
        par_groupe.setdefault(g, []).append(entree)
    print(f"  {len(par_groupe)} groupes independants "
          f"({len(entrees) - len(par_groupe)} images redondantes absorbees)")

    # Strate d'un groupe : la classe la plus rare qu'il contient, pour que les
    # classes minoritaires soient reparties dans les trois splits.
    strates: dict[int, list] = {i: [] for i in range(len(noms))}
    for g, lot in par_groupe.items():
        presentes = {c for _, _, boites in lot for c, _ in boites}
        strates[max(presentes)].append(lot)

    part_train = spec["repartition"]
    part_test = spec.get("repartition_test", 0.15)
    for strate, groupes_strate in strates.items():
        rng.shuffle(groupes_strate)
        n_images = sum(len(lot) for lot in groupes_strate)

        # Repartition par VOLUME D'IMAGES, pas par nombre de groupes. Les
        # groupes sont de tailles tres inegales -- une camera de surveillance
        # peut en fournir un de plusieurs centaines d'images quasi identiques,
        # quand une photo isolee en fait un a lui seul. Compter les groupes
        # donnait 490 images d'entrainement pour 759 de validation sur la classe
        # negative (constate le 2026-08-19) : un seul gros groupe suffisait a
        # renverser la proportion.
        # Les groupes sont donc servis du plus gros au plus petit, chacun au
        # split le plus en retard sur son quota. Le groupe reste entier -- c'est
        # la regle qu'on ne negocie pas -- mais les proportions sont tenues.
        quotas = {"test": n_images * part_test}
        quotas["train"] = (n_images - quotas["test"]) * part_train
        quotas["val"] = n_images - quotas["test"] - quotas["train"]
        compte = {"train": 0, "val": 0, "test": 0}
        for lot in sorted(groupes_strate, key=len, reverse=True):
            split = min(compte, key=lambda s: (compte[s] - quotas[s]) / max(quotas[s], 1))
            for jeu, img, boites in lot:
                compte[split] += 1
                total[split] += 1
                for c, _ in boites:
                    instances[c] += 1
                if not simulation:
                    _ecrire(dest, split, jeu, img, boites)
        print(f"  strate {noms[strate]:<16} {len(groupes_strate):>5} groupes / "
              f"{n_images:>5} images -> train {compte['train']}, "
              f"val {compte['val']}, test {compte['test']}")

    fonds = ajouter_fonds(dest, total, rng, simulation)
    print(f"\n  train {total['train'] + fonds['train']:>6}   "
          f"val {total['val'] + fonds['val']:>6}   "
          f"test {total['test']:>6}   "
          f"(dont {fonds['train'] + fonds['val']} images de fond)")
    print("  " + "   ".join(f"{noms[c]}={v}" for c, v in sorted(instances.items())))
    if simulation:
        return 0
    # `test` est declare mais n'est PAS utilise a l'entrainement : il ne sert
    # qu'au jugement, sur des images qu'aucun gradient n'a vues.
    (dest / "data.yaml").write_text(
        f"path: {dest}\ntrain: train/images\nval: val/images\ntest: test/images\n\n"
        f"nc: {len(noms)}\nnames: {noms}\n", encoding="utf-8")
    print(f"\n  -> {dest}")
    return 0


def construire(cle: str, simulation: bool) -> int:
    spec = JEUX[cle]
    dest = EXTRAIT / spec["dest"]
    noms = spec["noms"]
    rng = random.Random(0)  # tirage reproductible du plafond local

    if not simulation and dest.exists():
        print(f"deja present : {dest} -- suppression avant reconstruction")
        shutil.rmtree(dest)

    total = {"train": 0, "val": 0, "test": 0}
    instances = {i: 0 for i in range(len(noms))}

    if spec.get("repartition"):
        return _construire_repartie(spec, dest, noms, rng, simulation,
                                    total, instances)

    for src in spec["sources"]:
        split_train, split_val = src["splits"]
        for split_src, split_dest in ((split_train, "train"), (split_val, "val")):
            retenues = collecter(src, split_src)

            # Le plafond vaut pour les deux splits : une validation gonflée par
            # 3 909 images locales coûte du temps GPU à chaque époque sans rien
            # mesurer de plus, et biaise le critère vers le domaine local.
            plafond = src.get("plafond")
            if plafond and split_dest == "val":
                plafond //= 3
            if plafond and len(retenues) > plafond:
                print(f"  {src['jeu']:<24} {len(retenues):>6} -> {plafond} images (plafond)")
                retenues = rng.sample(retenues, plafond)
            else:
                print(f"  {src['jeu']:<24} {len(retenues):>6} images ({split_dest})")

            if not simulation and retenues:
                (dest / split_dest / "images").mkdir(parents=True, exist_ok=True)
                (dest / split_dest / "labels").mkdir(parents=True, exist_ok=True)

            for img, boites in retenues:
                total[split_dest] += 1
                for c, _ in boites:
                    instances[c] += 1
                if simulation:
                    continue
                # Préfixe la source : deux jeux peuvent contenir le même nom de
                # fichier, et une collision écraserait silencieusement une image.
                tige = f"{src['jeu']}__{img.stem}"
                shutil.copy2(img, dest / split_dest / "images" / f"{tige}{img.suffix}")
                (dest / split_dest / "labels" / f"{tige}.txt").write_text(
                    "\n".join(f"{c} {' '.join(co)}" for c, co in boites) + "\n",
                    encoding="utf-8")

    fonds = ajouter_fonds(dest, total, rng, simulation)
    print(f"\n  train {total['train'] + fonds['train']:>6}   "
          f"val {total['val'] + fonds['val']:>6}   "
          f"(dont {fonds['train'] + fonds['val']} images de fond)")
    print("  " + "   ".join(f"{noms[c]}={v}" for c, v in sorted(instances.items())))

    if simulation:
        return 0

    (dest / "data.yaml").write_text(
        f"path: {dest}\ntrain: train/images\nval: val/images\n\n"
        f"nc: {len(noms)}\nnames: {noms}\n", encoding="utf-8")
    print(f"\n  -> {dest}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jeu", choices=list(JEUX) + ["tous"], default="tous")
    ap.add_argument("--simulation", action="store_true",
                    help="compte les images sans rien copier")
    args = ap.parse_args()

    if not SOURCES.is_dir():
        print(f"introuvable : {SOURCES}", file=sys.stderr)
        return 1

    for cle in (list(JEUX) if args.jeu == "tous" else [args.jeu]):
        print(f"\n=== {cle} -> {JEUX[cle]['dest']} ===")
        construire(cle, args.simulation)

    if args.simulation:
        print("\nRelancer sans --simulation pour construire.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
