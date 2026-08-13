#!/usr/bin/env python3
"""Ré-entraînement du modèle EPI 14 classes sur Kaggle.

Pourquoi ce ré-entraînement
---------------------------
Le modèle déployé `ppe_detection/models/best.pt` ne détecte plus que deux de
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
2. **`lr0=0.001`**, taux de fine-tuning. Le run feu/fumée du 2026-08-12 avait
   laissé le défaut d'Ultralytics (0.01), prévu pour un entraînement depuis
   zéro : il avait effondré la fitness de 0.61 à 0.43 en quatre époques avant
   de passer 22 époques à remonter sans y parvenir.
3. **Départ depuis `best.pt`** et non depuis un modèle COCO neutre : son
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


def entrainer(epochs: int = 80, imgsz: int = 640, batch: int = 16,
              patience: int = 30, workers: int = 2, lr0: float = 0.001) -> int:
    from ultralytics import YOLO
    import torch

    if not torch.cuda.is_available():
        print("Aucun GPU : activer l'accelerateur dans les parametres du notebook.",
              file=sys.stderr)
        return 1
    print(f"GPU : {torch.cuda.get_device_name(0)}")

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
    poids = trouver("best.pt", "poids EPI de depart") or "yolov8m.pt"
    print(f"Jeu   : {source}")
    print(f"Poids : {poids}")

    modele = YOLO(str(poids))
    modele.train(
        data=str(yaml_absolu(source)),
        epochs=epochs, imgsz=imgsz, batch=batch, workers=workers,
        device=0, patience=patience,
        project=str(sorties), name=NOM_RUN, exist_ok=True, seed=0, plots=True,
        # Voir l'en-tete : c'est ce reglage qui avait sabote le run du 12 aout.
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
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--lr0", type=float, default=0.001)
    a = ap.parse_args()
    sys.exit(entrainer(a.epochs, a.imgsz, a.batch, a.patience, a.workers, a.lr0))
