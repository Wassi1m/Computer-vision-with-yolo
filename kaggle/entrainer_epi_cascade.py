#!/usr/bin/env python3
"""Entraîne un modèle EPI dédié destiné à la cascade — casque ou gants/lunettes.

Un seul script pour les deux modèles : ils ne diffèrent que par leur jeu et
leur nombre de classes, et dupliquer le fichier dupliquerait aussi les
garde-fous qu'il a fallu apprendre à la dure.

Pourquoi des modèles séparés plutôt qu'un fine-tuning de `ppe_detector.pt`
--------------------------------------------------------------------------
Parce que le fine-tuning global a déjà détruit ce modèle une fois : en juillet
2026, un entraînement sur un jeu ne contenant que du gilet a effacé douze de
ses quatorze classes, et il a fallu cinq jours et quatre ré-entraînements pour
les récupérer. La duplication ciblée tentée le 2026-08-16
(`entrainer_epi.py --dupliquer 4`) a elle aussi été rejetée : `NO-Mask` avait
reculé de 4 points et trois autres classes avaient régressé au passage.

La voie qui a fonctionné est celle de `masque_gilet.pt` (2026-08-17, mAP
0.9098) : un petit modèle dédié, entraîné sur un jeu propre, branché
prioritairement en cascade pour SES classes seulement. `ppe_detector.pt` n'est
jamais réécrit, donc l'oubli catastrophique de ses autres classes est
structurellement impossible -- et un candidat raté se jette sans rien casser,
la cascade retombant sur le modèle en place.

Les deux modèles produits ici
-----------------------------
`casque` : Hardhat / NO-Hardhat. Les deux classes les plus faibles du parc
    (65 % et 72 % de détection) et, d'après le plan v7, les seules dont le
    plafond vient d'un manque de DIVERSITE et non de volume -- 28 996 exemples
    locaux issus des mêmes chantiers n'ont pas suffi. Le jeu mêle donc Hard Hat
    Universe et Construction PPE (chantiers étrangers au corpus local) à un
    échantillon plafonné de `ppe_dataset`.

`gants`  : Gloves / NO-Gloves / Goggles / NO-Goggles. `ppe_detector.pt` y est
    déjà bon en AP (0.93 / 0.91 / 0.96 / 0.96) mais son taux de détection de
    scène n'a jamais été mesuré sur ces classes, et les deux classes négatives
    signalent une infraction. Les 28 000 images de PPEs + Safety Gloves
    apportent une diversité de domaine qu'aucun jeu local n'a.
    Ce candidat n'est à brancher QUE s'il bat `ppe_detector.pt` -- voir le
    critère de rejet dans `reports/plan_amelioration/v8_campagne_epi.md` §7.

Garde-fous, tous payés cher sur ce projet
-----------------------------------------
- `optimizer="SGD"` nommé explicitement : avec l'optimiseur `auto` par défaut,
  `lr0` est purement et simplement ignoré (constaté le 2026-08-12 puis le
  2026-08-15).
- `lr0=0.001`, taux de fine-tuning. Le défaut d'Ultralytics (0.01) est celui
  d'un entraînement depuis zéro : laissé tel quel le 12 août, il a écrasé les
  poids acquis et coûté un run entier.
- Départ depuis `ppe_detector.pt`, ossature déjà adaptée à l'imagerie de
  chantier. Ultralytics réinitialise seule la tête de détection.
- `time=` en heures : Kaggle tue la session à 12 h, ce garde-fou arrête
  proprement avant (validation finale, strip_optimizer, resume écrit).
- `save_period=5` et reprise automatique : une coupure ne coûte jamais plus de
  cinq époques.

Usage dans une cellule Kaggle :

    !pip install -q ultralytics
    !python "$(find /kaggle/input -name entrainer_epi_cascade.py | head -1)" --modele casque
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from entrainer_kaggle import ENTREE, TRAVAIL, trouver, preparer_reprise, publier  # noqa: E402

# Chaque modèle est identifié par le CONTENU de son data.yaml, pas par le nom de
# montage : celui-ci dépend du titre donné à l'upload, et un chemin écrit en dur
# a déjà fait échouer un lancement sur ce projet.
MODELES = {
    "casque": {
        "run": "epi_casque",
        "nc": 2,
        "marqueur": "NO-Hardhat",
        "heures": 5.0,
        "epochs": 80,
    },
    "gants": {
        "run": "epi_gants_lunettes",
        "nc": 4,
        "marqueur": "NO-Goggles",
        "heures": 5.0,
        "epochs": 60,
    },
    # Jeu volontairement petit (938 images) : c'est tout ce qui existe pour ce
    # concept, aucun jeu local n'annotant la moindre chaussure. On compense par
    # plus d'époques -- chacune est brève -- et par une patience resserrée, le
    # sur-apprentissage étant ici le risque principal. Le critère qui décide
    # n'est pas l'AP mais le taux de FAUX POSITIFS : une chaussure hallucinée
    # vaut moins que pas de classe du tout.
    # Budget revu le 2026-08-19 : le jeu est passe de 938 a 5 142 images
    # d'entrainement (trois sources ajoutees, cf. son PROVENANCE.txt). A 150
    # epoques il demanderait une douzaine d'heures et serait coupe par la limite
    # Kaggle ; 60 epoques suffisent pour un fine-tuning partant de
    # `ppe_detector.pt`, et `time=` reste le garde-fou.
    "chaussures": {
        "run": "epi_chaussures",
        "nc": 2,
        "marqueur": "NO-safety_shoe",
        "heures": 5.0,
        "epochs": 60,
    },
}


def trouver_jeu(spec: dict) -> Path | None:
    """Localise le jeu attendu parmi les entrées attachées, par son contenu."""
    if not ENTREE.is_dir():
        return None
    for profondeur in ("*/", "*/*/", "*/*/*/"):
        for yaml in sorted(ENTREE.glob(f"{profondeur}data.yaml")):
            texte = yaml.read_text(encoding="utf-8")
            if f"nc: {spec['nc']}" in texte and spec["marqueur"] in texte \
                    and (yaml.parent / "train" / "images").is_dir():
                return yaml.parent
    print(f"introuvable : un jeu a {spec['nc']} classes contenant {spec['marqueur']!r}",
          file=sys.stderr)
    return None


def yaml_absolu(source: Path, run: str) -> Path:
    """Réécrit data.yaml avec le chemin de montage réel.

    Le `path:` écrit à la construction pointe vers la machine locale ; sur
    Kaggle le jeu est monté ailleurs, en lecture seule.
    """
    src = (source / "data.yaml").read_text(encoding="utf-8").splitlines()
    lignes = [f"path: {source}", "train: train/images", "val: val/images", "",
              next(l for l in src if l.startswith("nc:")),
              next(l for l in src if l.startswith("names:"))]
    dest = TRAVAIL / f"data_{run}.yaml"
    dest.write_text("\n".join(lignes) + "\n", encoding="utf-8")
    return dest


def preparer_reprise_run(dossier_run: Path, run: str) -> bool:
    """`preparer_reprise` lit le NOM_RUN global du module ; on le prête le temps de l'appel."""
    import entrainer_kaggle as ek
    ancien, ek.NOM_RUN = ek.NOM_RUN, run
    try:
        return preparer_reprise(dossier_run)
    finally:
        ek.NOM_RUN = ancien


def entrainer(modele: str, imgsz: int, batch: int, patience: int,
              workers: int, lr0: float, epochs: int | None,
              heures: float | None) -> int:
    from ultralytics import YOLO
    import torch

    spec = MODELES[modele]
    run = spec["run"]
    epochs = epochs or spec["epochs"]
    heures = heures or spec["heures"]

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
    dossier_run = sorties / run
    sorties.mkdir(parents=True, exist_ok=True)

    if preparer_reprise_run(dossier_run, run):
        print("Etat restaure, reprise la ou l'entrainement s'etait arrete.")
        YOLO(str(dossier_run / "weights" / "last.pt")).train(resume=True)
        return publier(dossier_run)

    source = trouver_jeu(spec)
    if source is None:
        print(f"Attacher le jeu {run} a la session.", file=sys.stderr)
        return 1
    poids = trouver("ppe_detector.pt", "poids EPI de depart") or "yolo26n.pt"
    print(f"Modele : {modele}  ({spec['nc']} classes, run {run})")
    print(f"Jeu    : {source}")
    print(f"Poids de depart : {poids} (transfert -- tete reinitialisee)")

    YOLO(str(poids)).train(
        data=str(yaml_absolu(source, run)),
        epochs=epochs, imgsz=imgsz, batch=batch, workers=workers,
        device=device, patience=patience,
        time=heures,
        project=str(sorties), name=run, exist_ok=True, seed=0, plots=True,
        # Voir l'entete : sans nommer l'optimiseur, `lr0` est ignore.
        optimizer="SGD",
        lr0=lr0,
        save_period=5,
    )
    return publier(dossier_run)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--modele", choices=list(MODELES), required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--lr0", type=float, default=0.001)
    ap.add_argument("--epochs", type=int, default=None,
                    help="par defaut : la valeur propre au modele choisi")
    ap.add_argument("--heures", type=float, default=None,
                    help="budget de temps ; garde-fou avant la limite Kaggle de 12 h")
    a = ap.parse_args()
    sys.exit(entrainer(a.modele, a.imgsz, a.batch, a.patience, a.workers,
                       a.lr0, a.epochs, a.heures))
