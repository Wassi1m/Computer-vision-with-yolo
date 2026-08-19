#!/usr/bin/env python3
"""Entraînement d'un modèle dédié Mask/NO-Mask/Safety Vest/NO-Safety Vest.

Pourquoi un modèle séparé, et pas un réglage de plus sur `ppe_detector.pt`
---------------------------------------------------------------------------
Le 2026-08-16, un ré-entraînement de `ppe_detector.pt` visant `NO-Mask` et
`NO-Safety Vest` par duplication (`entrainer_epi.py --dupliquer 4`) a été
REJETÉ : `NO-Mask` a reculé de 4 points, `NO-Safety Vest` a stagné, et
plusieurs autres classes (`Hardhat`, `NO-Hardhat`, `Fall-Detected`) ont
régressé au passage. Voir `reports/v3_results/epi_14c_candidat_20260816.json`.

Cause diagnostiquée par `improvements/p1_eval_par_concept.py` (mesure déjà
faite, pas une hypothèse) : sur un sous-ensemble qui annote réellement le
masque et le gilet, le même modèle atteint 0.96/0.92 d'AP50 sur Mask/NO-Mask
et 0.89/0.67 sur Safety Vest/NO-Safety Vest -- très loin des 0.53/0.66 et
0.49/0.05 publiés sur `ppe_dataset` complet. Le modèle sait déjà faire ;
`ppe_dataset` est un patchwork où la plupart des images n'annotent ni l'un ni
l'autre concept, et chaque détection correcte y est comptée comme un faux
positif. Dupliquer les images correctement annotées ne change rien tant que
la majorité contradictoire domine le jeu (27 % du jeu même après x4).

Ce script entraîne donc un PETIT modèle séparé, seulement sur les images qui
annotent réellement l'un des deux concepts (`improvements/p11_jeu_masque_gilet.py`,
4 698 images train + 1 361 val, 0 image contradictoire). Comme ce modèle n'a
que 4 classes et ne remplace rien, l'oubli catastrophique des 12 autres
classes de `ppe_detector.pt` est structurellement IMPOSSIBLE : ce script ne
touche JAMAIS `ppe_detector.pt`, il écrit un fichier séparé (`masque_gilet.pt`)
qui se branchera en cascade, comme `ppe_complement.pt` aujourd'hui.

Mêmes garde-fous que `entrainer_epi.py`, déjà payés cher sur ce projet :
- `optimizer="SGD"` nommé explicitement -- `lr0` seul est sans effet avec
  l'optimiseur `auto` par défaut (leçon du 2026-08-12).
- `lr0=0.001`, taux de fine-tuning et non d'entraînement depuis zéro.
- Départ depuis `ppe_detector.pt` (ossature déjà adaptée à l'imagerie de
  chantier), pas un modèle COCO neutre -- Ultralytics réinitialise seule la
  tête de détection pour les 4 classes, le reste du réseau est repris.
- `time=` en heures : Kaggle tue la session a 12 h, ce garde-fou arrête
  proprement avant (validation finale, strip_optimizer, resume écrit).
- Reprise automatique si la session est coupée.

Le split `test` (655 images) n'est PAS dans ce jeu : il reste local, jamais
téléversé, pour juger le résultat sans qu'il ait pu le voir.

Usage dans une cellule Kaggle :

    !pip install -q ultralytics
    !python "$(find /kaggle/input -name entrainer_masque_gilet.py | head -1)"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from entrainer_kaggle import ENTREE, TRAVAIL, trouver, preparer_reprise, publier  # noqa: E402

NOM_RUN = "masque_gilet"


def trouver_jeu() -> Path | None:
    """Localise le jeu 4 classes parmi les entrées attachées.

    Identifié par sa structure ET son contenu (`nc: 4` + `NO-Safety Vest`
    parmi les noms), pas par le nom de montage qui dépend du titre donné à
    l'upload.
    """
    if not ENTREE.is_dir():
        return None
    for profondeur in ("*/", "*/*/", "*/*/*/"):
        for yaml in sorted(ENTREE.glob(f"{profondeur}data.yaml")):
            texte = yaml.read_text(encoding="utf-8")
            if "nc: 4" in texte and "NO-Safety Vest" in texte \
                    and (yaml.parent / "train" / "images").is_dir():
                return yaml.parent
    print("introuvable : un jeu a 4 classes (Mask/NO-Mask/Safety Vest/NO-Safety Vest)",
          file=sys.stderr)
    return None


def yaml_absolu(source: Path) -> Path:
    """Réécrit le data.yaml avec le chemin de montage réel (même raison que entrainer_epi.py)."""
    lignes = [f"path: {source}", "train: train/images", "val: val/images"]
    src = (source / "data.yaml").read_text(encoding="utf-8").splitlines()
    lignes += ["", next(l for l in src if l.startswith("nc:")),
               next(l for l in src if l.startswith("names:"))]
    dest = TRAVAIL / "data_masque_gilet.yaml"
    dest.write_text("\n".join(lignes) + "\n", encoding="utf-8")
    return dest


def preparer_reprise_mg(dossier_run: Path) -> bool:
    import entrainer_kaggle as ek
    ancien, ek.NOM_RUN = ek.NOM_RUN, NOM_RUN
    try:
        return preparer_reprise(dossier_run)
    finally:
        ek.NOM_RUN = ancien


def entrainer(epochs: int = 60, imgsz: int = 640, batch: int = 16,
              patience: int = 20, workers: int = 2, lr0: float = 0.001,
              heures: float = 6.0) -> int:
    from ultralytics import YOLO
    import torch

    if not torch.cuda.is_available():
        print("Aucun GPU : activer l'accelerateur dans les parametres du notebook.",
              file=sys.stderr)
        return 1

    n = torch.cuda.device_count()
    device = list(range(n)) if n > 1 else 0
    for i in range(n):
        print(f"GPU {i} : {torch.cuda.get_device_name(i)}")
    print(f"-> entrainement sur {n} carte(s)")

    sorties = TRAVAIL / "sorties"
    dossier_run = sorties / NOM_RUN
    sorties.mkdir(parents=True, exist_ok=True)

    if preparer_reprise_mg(dossier_run):
        print("Etat restaure, reprise la ou l'entrainement s'etait arrete.")
        YOLO(str(dossier_run / "weights" / "last.pt")).train(resume=True)
        return publier(dossier_run)

    source = trouver_jeu()
    if source is None:
        print("Attacher le jeu masque_gilet_coherent a la session.", file=sys.stderr)
        return 1
    poids = trouver("ppe_detector.pt", "poids EPI de depart") or "yolo26n.pt"
    print(f"Jeu   : {source}")
    print(f"Poids de depart : {poids} (transfert -- la tete sera reinitialisee pour 4 classes)")

    modele = YOLO(str(poids))
    modele.train(
        data=str(yaml_absolu(source)),
        epochs=epochs, imgsz=imgsz, batch=batch, workers=workers,
        device=device, patience=patience,
        time=heures,
        project=str(sorties), name=NOM_RUN, exist_ok=True, seed=0, plots=True,
        # Voir l'entete du fichier : sans nommer l'optimiseur, `lr0` est ignore
        # (constate deux fois sur ce projet, 2026-08-12 et 2026-08-15).
        optimizer="SGD",
        lr0=lr0,
        save_period=5,
    )
    return publier(dossier_run)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--lr0", type=float, default=0.001)
    ap.add_argument("--heures", type=float, default=6.0,
                    help="budget de temps ; jeu 4x plus petit que epi_14c, "
                         "converge normalement bien avant la limite Kaggle de 12 h")
    a = ap.parse_args()
    sys.exit(entrainer(a.epochs, a.imgsz, a.batch, a.patience, a.workers,
                       a.lr0, a.heures))
