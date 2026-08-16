#!/usr/bin/env python3
"""Ré-entraînement du modèle EPI 14 classes sur Kaggle.

Pourquoi ce ré-entraînement
---------------------------
Le modèle déployé `ppe_detection/models/ppe_detector.pt` ne détecte plus que deux de
ses quatorze classes. Mesuré le 2026-08-13 sur les 4 423 images du split test,
avec cas témoin :

    Safety Vest      AP@50 0.5354      (detection de scene 95 %)
    NO-Safety Vest   AP@50 0.0621      (detection de scene 85 %)
    les 12 autres    AP@50 0.0000      (detection de scene  0 %)
    mAP@50 global    0.0427

Rien d'autre ne sort même en abaissant le seuil de confiance à 0,01 : les
classes ne sont pas faibles, elles ont disparu.

La cause est un **oubli catastrophique**. Le fine-tuning avait tourné sur
`ppe_vest_clean_14c`, qui est exactement le sous-ensemble des images de
`ppe_dataset` contenant du gilet — 2 728 images où seules les classes 10 et 13
apparaissent. En s'entraînant longtemps sur elles seules, le réseau a réaffecté
sa capacité au gilet et effacé le reste.

Ce qui protège contre la répétition
-----------------------------------
1. **Un jeu équilibré**, construit par `improvements/p10_sous_ensemble_epi.py` :
   ~2 500 instances par classe, et surtout **l'intégralité** des instances de
   gilet (4 499 + 1 435, les totaux exacts du jeu complet). Le gilet est la
   seule paire de classes qui fonctionne encore ; en perdre une partie serait
   la régression à éviter.
2. **`optimizer="SGD"` avec `lr0=0.001`.** Les deux ensemble, et c'est le
   point à ne pas manquer : `lr0` seul est **sans aucun effet**. Avec le
   réglage par défaut `optimizer="auto"`, Ultralytics recalcule le taux
   d'apprentissage et jette celui qu'on lui a donné, en le disant dans son
   journal — « optimizer='auto' found, ignoring 'lr0=0.001' ... MuSGD(lr=0.01) ».
   C'est ce qui a détruit le modèle feu/fumée le 2026-08-12 (fitness effondrée
   de 0.61 à 0.43 en quatre époques, puis 22 époques à remonter sans y
   parvenir), et ce qui a failli recommencer le 2026-08-15 : `lr0` avait alors
   été corrigé, mais sans nommer l'optimiseur — donc en pure perte.
3. **Départ depuis `ppe_detector.pt`** et non depuis un modèle COCO neutre : son
   ossature est déjà adaptée à l'imagerie de chantier. Seule la tête est à
   réapprendre, ce que le jeu équilibré permet sans sacrifier le gilet.

Usage dans une cellule Kaggle :

    !pip install -q ultralytics
    !python "$(find /kaggle/input -name entrainer_epi.py | head -1)" --epochs 80
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from entrainer_kaggle import ENTREE, TRAVAIL, trouver, preparer_reprise, publier  # noqa: E402

NOM_RUN = "epi_14c"

# Classes a renforcer, choisies sur la DETECTION REELLE mesuree le 2026-08-16 et
# non sur l'AP : les deux divergent fortement ici, l'AP etant ecrasee par les
# annotations incompletes de `ppe_dataset` (`Mask` sort a 0.55 d'AP mais 95 % de
# detection ; `Hardhat` a 0.83 d'AP mais seulement 65 %).
#
#   9   NO-Mask         78 %  (1 577 exemples) -- a recule de 95 % a 78 %
#   10  NO-Safety Vest  70 %  (1 435 exemples) -- a recule deux fois de suite
#
# Le critere est la conjonction FAIBLE **et** RARE : dupliquer ne repare que ce
# que la rarete a cause. Deux classes faibles en sont donc exclues a dessein :
#
#   NO-Hardhat (72 %, 9 705 exemples) -- aussi nombreux que Safety Cone qui
#     sort a 98 %. Sa difficulte tient a la nature de la tache (reconnaitre une
#     ABSENCE), pas au manque d'exemples. Le dupliquer x4 le porterait a 38 820
#     instances vues, presque quatre fois Hardhat lui-meme, pour ~14 h
#     d'entrainement : la coupure serait garantie et le gain nul.
#   Hardhat (65 %, 28 996 exemples) -- la classe la MIEUX pourvue du jeu.
#
# Les deux classes retenues sont des negatives, celles qui signalent
# l'infraction. Elles sont systematiquement moins bien apprises que leurs
# positives : c'est le defaut structurel de ce jeu de donnees.
CLASSES_RARES = {9, 10}


def trouver_jeu_epi() -> Path | None:
    """Localise le jeu 14 classes parmi les entrées attachées.

    On l'identifie par sa structure et son nombre de classes plutôt que par son
    nom : le nom de montage d'un jeu Kaggle dépend du titre donné à l'upload,
    qu'on ne maîtrise pas depuis le script.
    """
    if not ENTREE.is_dir():
        return None
    for profondeur in ("*/", "*/*/", "*/*/*/"):
        for yaml in sorted(ENTREE.glob(f"{profondeur}data.yaml")):
            texte = yaml.read_text(encoding="utf-8")
            if "nc: 14" in texte and (yaml.parent / "train" / "images").is_dir():
                return yaml.parent
    print("introuvable : un jeu a 14 classes avec train/images", file=sys.stderr)
    return None


def dupliquer_sur_place(source: Path, classes: set[int], facteur: int) -> Path:
    """Renforce des classes rares en répétant leurs images, sans rien renvoyer.

    Le jeu monté sur Kaggle contient déjà **toutes** les images des classes
    rares — leur inclusion intégrale est forcée à la construction. Rééquilibrer
    ne demande donc aucun nouveau téléversement : il suffit de montrer ces
    images plus souvent, ce qui se fait ici au démarrage de la session.

    Les copies sont des **liens symboliques** : `/kaggle/input` est en lecture
    seule mais reste lisible, et `/kaggle/working` est limité (~20 Go). Répéter
    quatre fois deux mille images coûterait plusieurs gigaoctets en copies
    réelles, contre quelques kilo-octets en liens.

    Pourquoi c'est nécessaire : `NO-Safety Vest` ne compte que 1 435 instances
    contre 28 996 pour `Hardhat`. Même en prenant toutes ses images, elle reste
    vingt fois moins vue — et elle a reculé à chaque rééquilibrage (0.8534 puis
    0.7786 puis 0.6839). L'entraîner seule effacerait les treize autres classes,
    exactement le desastre du 2026-08-13 : il faut donc la répéter *dans* le jeu
    complet, jamais l'isoler.
    """
    dest = TRAVAIL / "donnees" / "epi_renforce"
    if (dest / "data.yaml").exists():
        print(f"Jeu renforce deja present : {dest}")
        return dest

    for split in ("train", "val"):
        for sous in ("images", "labels"):
            (dest / split / sous).mkdir(parents=True, exist_ok=True)

    # `val` est repris tel quel : le dupliquer fausserait le suivi
    # d'entrainement sans rien apporter.
    for sous in ("images", "labels"):
        for f in (source / "val" / sous).iterdir():
            (dest / "val" / sous / f.name).symlink_to(f)

    concernees = ajouts = total = 0
    for lbl in (source / "train" / "labels").iterdir():
        img = next((source / "train" / "images").glob(lbl.stem + ".*"), None)
        if img is None:
            continue
        total += 1
        (dest / "train" / "labels" / lbl.name).symlink_to(lbl)
        (dest / "train" / "images" / img.name).symlink_to(img)

        vise = any(int(l.split()[0]) in classes
                   for l in lbl.read_text().splitlines() if l.strip())
        if not vise:
            continue
        concernees += 1
        for k in range(2, facteur + 1):
            base = f"{lbl.stem}__x{k}"
            (dest / "train" / "labels" / f"{base}.txt").symlink_to(lbl)
            (dest / "train" / "images" / f"{base}{img.suffix}").symlink_to(img)
            ajouts += 1

    src = (source / "data.yaml").read_text(encoding="utf-8").splitlines()
    (dest / "data.yaml").write_text("\n".join([
        f"path: {dest}", "train: train/images", "val: val/images", "",
        next(l for l in src if l.startswith("nc:")),
        next(l for l in src if l.startswith("names:")),
    ]) + "\n", encoding="utf-8")

    print(f"Renforcement x{facteur} des classes {sorted(classes)} : "
          f"{concernees} images concernees sur {total}, {ajouts} repetitions "
          f"-> {total + ajouts} images d'entrainement")
    return dest


def yaml_absolu(source: Path) -> Path:
    """Réécrit le data.yaml avec le chemin de montage réel.

    Le fichier livré porte `path: .`, qu'Ultralytics résout par rapport à son
    propre répertoire de jeux de données et non par rapport au fichier — ce qui
    le ferait chercher les images au mauvais endroit sur Kaggle.
    """
    lignes = [f"path: {source}", "train: train/images", "val: val/images"]
    src = (source / "data.yaml").read_text(encoding="utf-8").splitlines()
    lignes += ["", next(l for l in src if l.startswith("nc:")),
               next(l for l in src if l.startswith("names:"))]
    dest = TRAVAIL / "data_epi.yaml"
    dest.write_text("\n".join(lignes) + "\n", encoding="utf-8")
    return dest


def entrainer(epochs: int = 60, imgsz: int = 640, batch: int = 16,
              patience: int = 20, workers: int = 2, lr0: float = 0.001,
              heures: float = 10.5, dupliquer: int = 1) -> int:
    from ultralytics import YOLO
    import torch

    if not torch.cuda.is_available():
        print("Aucun GPU : activer l'accelerateur dans les parametres du notebook.",
              file=sys.stderr)
        return 1

    # Utiliser TOUTES les cartes disponibles. Le run du 2026-08-15 a tourne sur
    # une seule alors que la session en offrait deux, et s'est fait couper a
    # l'epoque 79 sur 80 par la limite de 12 h de Kaggle.
    n = torch.cuda.device_count()
    device = list(range(n)) if n > 1 else 0
    for i in range(n):
        print(f"GPU {i} : {torch.cuda.get_device_name(i)}")
    print(f"-> entrainement sur {n} carte(s)")

    sorties = TRAVAIL / "sorties"
    dossier_run = sorties / NOM_RUN
    sorties.mkdir(parents=True, exist_ok=True)

    if preparer_reprise_epi(dossier_run):
        print("Etat restaure, reprise la ou l'entrainement s'etait arrete.")
        YOLO(str(dossier_run / "weights" / "last.pt")).train(resume=True)
        return publier(dossier_run)

    source = trouver_jeu_epi()
    if source is None:
        print("Attacher le jeu 14 classes a la session (voir README).", file=sys.stderr)
        return 1
    poids = trouver("ppe_detector.pt", "poids EPI de depart") or "yolov8m.pt"
    print(f"Jeu   : {source}")
    print(f"Poids : {poids}")

    # Renforcement des classes rares sans nouveau televersement : le jeu monte
    # contient deja toutes leurs images, il suffit de les repeter par liens.
    if dupliquer > 1:
        source = dupliquer_sur_place(source, CLASSES_RARES, dupliquer)

    modele = YOLO(str(poids))
    modele.train(
        data=str(source / "data.yaml" if (source / "data.yaml").is_relative_to(TRAVAIL)
                 else yaml_absolu(source)),
        epochs=epochs, imgsz=imgsz, batch=batch, workers=workers,
        device=device, patience=patience,
        # Budget de temps, et c'est le garde-fou le plus important de ce script.
        # Kaggle tue la session a 12 h : le run du 2026-08-15 s'est fait couper a
        # l'epoque 79 sur 80, laissant un best.pt jamais finalise (155 Mo avec
        # l'optimiseur, `model` vide, poids seulement dans `ema`). Avec `time`,
        # Ultralytics s'arrete de lui-meme avant la fin du budget et termine
        # proprement : validation finale, strip_optimizer, resume ecrit.
        time=heures,
        project=str(sorties), name=NOM_RUN, exist_ok=True, seed=0, plots=True,
        # `optimizer` DOIT etre nomme explicitement. Avec le defaut `auto`,
        # Ultralytics recalcule le taux d'apprentissage et ecrase celui qu'on
        # lui donne -- son propre journal l'annonce :
        #   « optimizer='auto' found, ignoring 'lr0=0.001' ... MuSGD(lr=0.01) »
        # C'est ainsi que le run du 2026-08-15 est reparti a 0.01, la valeur
        # meme qui avait detruit le modele feu/fumee trois jours plus tot.
        # Passer lr0 sans fixer l'optimiseur ne sert donc a RIEN.
        optimizer="SGD",
        lr0=lr0,
        save_period=5,
    )
    return publier(dossier_run)


def preparer_reprise_epi(dossier_run: Path) -> bool:
    """`preparer_reprise` de `entrainer_kaggle`, mais pour le run EPI."""
    import entrainer_kaggle as ek
    ancien, ek.NOM_RUN = ek.NOM_RUN, NOM_RUN
    try:
        return preparer_reprise(dossier_run)
    finally:
        ek.NOM_RUN = ancien


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--lr0", type=float, default=0.001)
    ap.add_argument("--heures", type=float, default=10.5,
                    help="budget de temps ; Kaggle tue la session a 12 h")
    ap.add_argument("--dupliquer", type=int, default=1,
                    help="repete les images des classes rares (NO-Safety Vest). "
                         "4 rapproche leur frequence des classes majoritaires. "
                         "Fait sur place par liens : aucun televersement")
    a = ap.parse_args()
    sys.exit(entrainer(a.epochs, a.imgsz, a.batch, a.patience, a.workers,
                       a.lr0, a.heures, a.dupliquer))
