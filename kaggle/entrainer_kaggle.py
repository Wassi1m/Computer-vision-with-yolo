#!/usr/bin/env python3
"""Entraînement feu/fumée sur Kaggle, conçu pour survivre à une coupure de session.

Pourquoi ce script existe séparément des scripts `deploy/`
----------------------------------------------------------
L'entraînement se faisait jusqu'ici sur une VM GPU Google Cloud, pilotée par
`deploy/00_tout_entrainer.sh`. Les crédits GCP étant épuisés, la plateforme
devient Kaggle — qui impose trois contraintes que la VM n'avait pas :

1. **La session est limitée** (une douzaine d'heures) et peut être interrompue
   sans préavis. Un entraînement de 60 époques ne tient pas forcément d'un seul
   tenant : il faut pouvoir **reprendre**, pas recommencer.
2. **`/kaggle/input` est en lecture seule.** Le jeu dégradé ne peut donc pas
   être écrit à côté du jeu source comme sur la VM ; il va dans
   `/kaggle/working`. Les liens symboliques de `p8_dataset_nuit.py` pointant
   vers l'entrée restent lisibles, donc seules les images dégradées occupent
   réellement de la place.
3. **Seul `/kaggle/working` est conservé** en sortie de session, et il est
   limité (~20 Go). Tout ce qui doit survivre y est écrit.

Reprise après interruption
--------------------------
Ultralytics sait reprendre depuis `last.pt`, à condition de retrouver le dossier
du run. Sur Kaggle chaque session repart d'un disque vierge : la marche à suivre
est donc d'attacher la sortie de la session précédente comme jeu de données
d'entrée, ce script la détecte et recopie l'état avant de relancer. Voir le
`README.md` de ce dossier pour les manipulations exactes.

Usage dans une cellule de notebook Kaggle :

    !pip install -q ultralytics
    !python /kaggle/input/ppe-scripts/entrainer_kaggle.py --epochs 60 --imgsz 896

ou, plus simplement, en important la fonction :

    from entrainer_kaggle import entrainer
    entrainer(epochs=60, imgsz=896)
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ENTREE = Path("/kaggle/input")
TRAVAIL = Path("/kaggle/working")
NOM_RUN = "fumee_robuste"


def trouver(motif: str, description: str, racine: Path | None = None) -> Path | None:
    """Cherche un fichier ou dossier dans les jeux attachés à la session.

    Les jeux Kaggle sont montés sous un nom choisi à l'upload, que l'on ne
    connaît pas forcément d'avance : on cherche donc par motif plutôt que par
    chemin en dur, pour que le script fonctionne quel que soit le nom donné.

    `racine` vaut `ENTREE` par défaut, mais résolu à l'appel et non à la
    définition : une valeur par défaut figée rendrait la fonction intestable
    hors de Kaggle, où `/kaggle/input` n'existe pas.
    """
    racine = racine or ENTREE
    if not racine.is_dir():
        return None
    for profondeur in ("*/", "*/*/", "*/*/*/"):
        for candidat in sorted(racine.glob(f"{profondeur}{motif}")):
            return candidat
    print(f"  introuvable : {description} (motif '{motif}')")
    return None


def trouver_jeu(racine: Path | None = None) -> Path | None:
    """Localise le jeu de données parmi les entrées attachées.

    On ne cherche pas un dossier nommé `fire_smoke_v9` : `kaggle datasets
    create -p fire_smoke_v9` téléverse le **contenu** du dossier, pas le dossier
    lui-même. Le jeu se retrouve donc directement sous
    `/kaggle/input/<titre-donne-a-l-upload>/`, sous un nom que l'on ne maîtrise
    pas. On identifie plutôt le jeu par sa structure : un `data.yaml`
    accompagné d'un `train/images`.
    """
    racine = racine or ENTREE
    if not racine.is_dir():
        return None
    for profondeur in ("*/", "*/*/", "*/*/*/"):
        for yaml in sorted(racine.glob(f"{profondeur}data.yaml")):
            if (yaml.parent / "train" / "images").is_dir():
                return yaml.parent
    print("  introuvable : un jeu contenant data.yaml et train/images")
    return None


def preparer_reprise(dossier_run: Path) -> bool:
    """Recopie l'état d'une session précédente pour permettre `resume=True`.

    Renvoie True si un entraînement interrompu a été retrouvé et restauré.
    """
    precedent = trouver(f"{NOM_RUN}/weights/last.pt", "poids d'une session precedente")
    if precedent is None:
        return False
    source_run = precedent.parent.parent
    print(f"Reprise detectee : {source_run}")
    dossier_run.mkdir(parents=True, exist_ok=True)
    for element in source_run.iterdir():
        cible = dossier_run / element.name
        if cible.exists():
            continue
        if element.is_dir():
            shutil.copytree(element, cible)
        else:
            shutil.copy2(element, cible)
    return (dossier_run / "weights" / "last.pt").exists()


def entrainer(epochs: int = 60, imgsz: int = 896, batch: int = 16,
              proportion: float = 0.35, patience: int = 25,
              workers: int = 2, lr0: float = 0.001) -> int:
    from ultralytics import YOLO
    import torch

    if not torch.cuda.is_available():
        print("Aucun GPU : activer l'accelerateur GPU dans les parametres du notebook.",
              file=sys.stderr)
        return 1
    print(f"GPU : {torch.cuda.get_device_name(0)}")

    sorties = TRAVAIL / "sorties"
    dossier_run = sorties / NOM_RUN
    sorties.mkdir(parents=True, exist_ok=True)

    # ── Reprise éventuelle ───────────────────────────────────────────────────
    reprise = preparer_reprise(dossier_run)
    if reprise:
        print("Etat restaure, reprise de l'entrainement la ou il s'etait arrete.")
        modele = YOLO(str(dossier_run / "weights" / "last.pt"))
        modele.train(resume=True)
        return publier(dossier_run)

    # ── Première session : construire le jeu dégradé ─────────────────────────
    source = trouver_jeu()
    if source is None:
        print("Attacher le jeu de donnees a la session (voir README).", file=sys.stderr)
        return 1
    poids = trouver("fire_smoke.pt", "poids de depart") or "yolo26n.pt"
    script_nuit = trouver("p8_dataset_nuit.py", "script de degradation")

    jeu = TRAVAIL / "donnees" / "fire_smoke_v9_robuste"
    if not (jeu / "data.yaml").exists():
        if script_nuit is None:
            print("script p8_dataset_nuit.py absent : entrainement sans images degradees")
            data_yaml = ecrire_yaml_direct(source)
        else:
            print(f"Construction du jeu degrade (proportion {proportion})...")
            jeu.parent.mkdir(parents=True, exist_ok=True)
            code = subprocess.call([sys.executable, str(script_nuit),
                                    "--source", str(source), "--sortie", str(jeu),
                                    "--proportion", str(proportion)])
            if code != 0:
                print("construction echouee", file=sys.stderr)
                return 1
            data_yaml = jeu / "data.yaml"
    else:
        print("Jeu degrade deja present, reutilise.")
        data_yaml = jeu / "data.yaml"

    print(f"Entrainement : {epochs} epoques, imgsz={imgsz}, batch={batch}")
    modele = YOLO(str(poids))
    modele.train(
        data=str(data_yaml),
        epochs=epochs, imgsz=imgsz, batch=batch, workers=workers,
        device=0, patience=patience,
        # Taux d'apprentissage de FINE-TUNING, dix fois plus bas que le defaut
        # d'Ultralytics (0.01), prevu lui pour un entrainement depuis zero.
        # Le run du 2026-08-12 a paye cette confusion : parti de fire_smoke.pt,
        # il a effondre sa fitness de 0.61 (epoque 1) a 0.43 (epoque 4) en
        # ecrasant les poids acquis, puis a passe 22 epoques a remonter sans
        # jamais retrouver son point de depart -- 2 h 54 de GPU pour rien, et un
        # arret par patience declenche precisement parce que l'epoque 1 n'a
        # jamais ete battue.
        lr0=lr0,

        project=str(sorties), name=NOM_RUN, exist_ok=True, seed=0, plots=True,
        # Augmentation photometrique renforcee, comme sur les runs precedents :
        # c'est `hsv_v` qui porte l'essentiel de l'effet recherche pour la nuit.
        hsv_h=0.015, hsv_s=0.8, hsv_v=0.9,
        # Sauvegarde reguliere : sans cela, une session interrompue entre deux
        # sauvegardes par defaut perd plusieurs epoques de calcul.
        save_period=5,
    )
    return publier(dossier_run)


def ecrire_yaml_direct(source: Path) -> Path:
    """data.yaml pointant vers le jeu d'entrée, sans dégradation.

    `/kaggle/input` étant en lecture seule, le fichier est écrit dans
    `/kaggle/working` et référence la source par chemin absolu.
    """
    src = (source / "data.yaml").read_text(encoding="utf-8")
    ligne = lambda p: next(l for l in src.splitlines() if l.startswith(p))  # noqa: E731
    val = "valid" if (source / "valid" / "images").is_dir() else "val"
    dest = TRAVAIL / "data_direct.yaml"
    contenu = [f"path: {source}", "train: train/images", f"val: {val}/images"]
    if (source / "test" / "images").is_dir():
        contenu.append("test: test/images")
    contenu += ["", ligne("nc:"), ligne("names:")]
    dest.write_text("\n".join(contenu) + "\n", encoding="utf-8")
    return dest


def publier(dossier_run: Path) -> int:
    """Résume l'entraînement et vérifie que les poids sont bien dans la sortie.

    Seul `/kaggle/working` est conservé quand la session se termine : un modèle
    écrit ailleurs serait perdu sans avertissement.
    """
    meilleur = dossier_run / "weights" / "best.pt"
    if not meilleur.exists():
        print("Aucun best.pt produit -- l'entrainement n'a probablement pas abouti.",
              file=sys.stderr)
        return 1

    resume = {"poids": str(meilleur), "taille_octets": meilleur.stat().st_size}
    csv = dossier_run / "results.csv"
    if csv.exists():
        lignes = csv.read_text(encoding="utf-8").strip().splitlines()
        entetes = [c.strip() for c in lignes[0].split(",")]
        col_map50 = next((i for i, c in enumerate(entetes)
                          if "mAP50" in c and "95" not in c), None)
        col_map5095 = next((i for i, c in enumerate(entetes) if "mAP50-95" in c), None)
        valeurs = [l.split(",") for l in lignes[1:]]
        resume["epoques"] = len(valeurs)
        if col_map50 is not None and valeurs:
            # Ultralytics choisit `best.pt` sur la *fitness*, pas sur la mAP50 :
            # classer ici par mAP50 seule ferait designer une epoque qui n'est
            # pas celle enregistree dans best.pt -- deja observe, et trompeur.
            def fitness(v: list[str]) -> float:
                m50 = float(v[col_map50])
                if col_map5095 is None:
                    return m50
                return 0.1 * m50 + 0.9 * float(v[col_map5095])

            meilleure = max(valeurs, key=fitness)
            resume["meilleure_epoque"] = meilleure[0].strip()
            resume["meilleur_mAP50"] = round(float(meilleure[col_map50]), 4)
            if col_map5095 is not None:
                resume["meilleur_mAP50_95"] = round(float(meilleure[col_map5095]), 4)
                resume["fitness"] = round(fitness(meilleure), 4)
            resume["mAP50_max_toutes_epoques"] = round(
                max(float(v[col_map50]) for v in valeurs), 4)

    (TRAVAIL / "resume_entrainement.json").write_text(
        json.dumps(resume, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("\n=== Termine ===")
    for k, v in resume.items():
        print(f"  {k} : {v}")
    print(f"\nPoids a telecharger : {meilleur}")
    print("Ils sont dans /kaggle/working, donc conserves en sortie de session.")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=896)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--proportion", type=float, default=0.35)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    sys.exit(entrainer(a.epochs, a.imgsz, a.batch, a.proportion, a.patience, a.workers))
