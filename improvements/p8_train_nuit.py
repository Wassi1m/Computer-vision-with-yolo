#!/usr/bin/env python3
"""P8 — Fine-tuning pour la robustesse aux conditions dégradées, sur GPU.

Se lance sur la VM GCP (NVIDIA L4), sur le jeu produit par `p8_dataset_nuit.py`.

Deux leviers agissent ensemble :

1. **Les images dégradées ajoutées au jeu** (assombrissement, contre-jour, flou
   de mouvement, intempérie) — c'est le levier principal, il donne au modèle des
   exemples de ce qu'il ne savait pas traiter.
2. **L'augmentation photométrique d'Ultralytics**, poussée au-delà des valeurs
   par défaut (`hsv_v` 0.4 → 0.9) : elle fait varier la luminosité à chaque
   époque, ce qui multiplie la diversité vue au-delà des copies figées.

Le taux d'apprentissage reste bas (1e-3) et le nombre d'époques modéré : on
corrige un angle mort sans réapprendre la tâche. Un `lr` élevé ferait perdre au
modèle ce qu'il sait déjà faire de jour — c'est le risque principal ici, et la
mesure « avant / après en conditions normales » ci-dessous est là pour le
détecter.

    python p8_train_nuit.py --donnees donnees/ppe_vest_clean_14c_robuste \\
                            --poids modeles/best.pt --nom epi_robuste --epochs 60
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def evaluer(modele, data_yaml: str, split: str, etiquette: str, device) -> dict:
    r = modele.val(data=data_yaml, split=split, imgsz=640, device=device,
                   plots=False, verbose=False)
    par_classe = {}
    for i, ci in enumerate(r.box.ap_class_index):
        nom = modele.names.get(int(ci))
        if nom is not None:
            par_classe[nom] = round(float(r.box.ap50[i]), 4)
    mesure = {"mAP50": round(float(r.box.map50), 4),
              "mAP50_95": round(float(r.box.map), 4),
              "classes": par_classe}
    print(f"  [{etiquette}] mAP@50 = {mesure['mAP50']:.4f}  {par_classe}", flush=True)
    return mesure


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--donnees", required=True, help="data.yaml du jeu enrichi")
    ap.add_argument("--poids", required=True, help="poids de depart (fine-tuning)")
    ap.add_argument("--nom", required=True, help="nom du run")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lr0", type=float, default=0.001)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--sorties", default="sorties")
    ap.add_argument("--reference", default=None,
                    help="data.yaml d'origine, pour mesurer la tenue en conditions normales")
    ap.add_argument("--reprendre", action="store_true",
                    help="reprend un entrainement interrompu depuis last.pt")
    ap.add_argument("--device", default="0",
                    help="'0' = premier GPU (defaut), 'cpu' = CPU (lent : a reserver aux "
                         "petits jeux ou en depannage sans GPU disponible)")
    args = ap.parse_args()

    from ultralytics import YOLO
    import torch

    if args.device == "cpu":
        print("ATTENTION : entrainement sur CPU (pas de GPU disponible) -- tres lent.",
              file=sys.stderr, flush=True)
    elif not torch.cuda.is_available():
        print("CUDA indisponible : l'entrainement tournerait sur CPU, abandon. "
              "Utiliser --device cpu pour forcer.", file=sys.stderr)
        return 1
    else:
        print(f"GPU : {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} Go)", flush=True)

    data_yaml = str(Path(args.donnees).resolve())
    if Path(data_yaml).is_dir():
        data_yaml = str(Path(data_yaml) / "data.yaml")
    sorties = Path(args.sorties).resolve()
    dernier = sorties / args.nom / "weights" / "last.pt"

    resultats: dict = {"nom": args.nom, "donnees": data_yaml, "poids_depart": args.poids}

    if args.reprendre and dernier.exists():
        print(f"Reprise depuis {dernier}", flush=True)
        YOLO(str(dernier)).train(resume=True)
    else:
        # Mesure « avant », sur les donnees d'origine si fournies : c'est le seul
        # moyen de verifier apres coup qu'on n'a pas degrade le cas nominal en
        # corrigeant le cas degrade.
        if args.reference:
            print("\n== Avant fine-tuning ==", flush=True)
            # Cette mesure est informative : elle sert a comparer apres coup.
            # Elle ne doit en aucun cas interrompre un entrainement de plusieurs
            # heures -- ce qui arriverait par exemple si les poids de depart et
            # le jeu n'ont pas le meme nombre de classes, cas normal quand on
            # scinde une classe (P9 : `smoke` -> `smoke` + `smoke_distant`).
            try:
                modele = YOLO(args.poids)
                resultats["avant"] = {
                    "conditions_normales": evaluer(modele, args.reference, "val", "normales", args.device),
                    "jeu_enrichi": evaluer(modele, data_yaml, "val", "enrichi", args.device),
                }
            except Exception as e:
                print(f"  mesure 'avant' impossible ({type(e).__name__}: {e}) -- "
                      f"on poursuit l'entrainement", file=sys.stderr, flush=True)
                resultats["avant"] = {"erreur": f"{type(e).__name__}: {e}"}

        print("\n== Fine-tuning ==", flush=True)
        modele = YOLO(args.poids)
        modele.train(
            data=data_yaml,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            lr0=args.lr0,
            patience=args.patience,
            project=str(sorties),
            name=args.nom,
            exist_ok=True,
            seed=0,
            plots=True,
            # Augmentation photometrique renforcee : c'est `hsv_v` (luminosite)
            # qui porte l'essentiel de l'effet recherche ici.
            hsv_h=0.015, hsv_s=0.8, hsv_v=0.9,
            # Le reste laisse aux valeurs par defaut : modifier la geometrie
            # (rotation, perspective) n'a rien a voir avec le probleme traite et
            # ajouterait du bruit a l'apprentissage.
        )

    meilleur = sorties / args.nom / "weights" / "best.pt"
    if not meilleur.exists():
        print(f"poids finaux introuvables : {meilleur}", file=sys.stderr)
        return 1

    print("\n== Apres fine-tuning ==", flush=True)
    modele = YOLO(str(meilleur))
    resultats["apres"] = {"jeu_enrichi": evaluer(modele, data_yaml, "val", "enrichi", args.device)}
    if args.reference:
        resultats["apres"]["conditions_normales"] = evaluer(
            modele, args.reference, "val", "normales", args.device)

    dest = sorties / args.nom / "resultats_p8.json"
    dest.write_text(json.dumps(resultats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    avant = resultats.get("avant", {}).get("conditions_normales", {}).get("mAP50")
    apres = resultats.get("apres", {}).get("conditions_normales", {}).get("mAP50")
    if avant is not None and apres is not None:
        ecart = apres - avant
        print(f"\nConditions normales : {avant:.4f} -> {apres:.4f} ({ecart:+.4f})")
        if ecart < -0.02:
            print("  ATTENTION : le cas nominal s'est degrade de plus de 2 points.\n"
                  "  Reduire --lr0 ou --proportion (moins d'images degradees) et relancer.")

    print(f"\nPoids  : {meilleur}")
    print(f"Mesures: {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
