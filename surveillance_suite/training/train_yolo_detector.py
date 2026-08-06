"""
Entraîne un modèle YOLO de détection sur un dataset au format Roboflow/YOLO
(data.yaml + dossiers train/valid/test). Réutilisable pour n'importe quel
dataset de détection téléchargé depuis Roboflow Universe.

Usage :
    # Fumée / feu
    python train_yolo_detector.py --data datasets/fire_smoke/data.yaml \
        --output models/fire_smoke.pt --epochs 50

    # Plaques d'immatriculation
    python train_yolo_detector.py --data datasets/license_plate/data.yaml \
        --output models/license_plate.pt --epochs 50

Options utiles :
    --imgsz 640       taille d'image (640 par défaut, diminue à 416 si trop lent sur CPU)
    --batch 8          diminue si erreur mémoire (--batch 4)
    --model yolo26n.pt  poids de base pour le transfer learning (nano = plus rapide sur CPU)
"""

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="chemin vers data.yaml du dataset")
    parser.add_argument("--output", required=True, help="ou sauvegarder le modele final (.pt)")
    parser.add_argument("--model", default="yolo26n.pt", help="poids de base (transfer learning)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--project", default="runs_custom_training")
    args = parser.parse_args()

    data_yaml = Path(args.data)
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"{data_yaml} introuvable. Verifie que tu as bien dezippe le dataset "
            f"et que le chemin --data pointe vers son fichier data.yaml"
        )

    print(f"Entrainement sur {data_yaml} ({args.epochs} epochs, imgsz={args.imgsz})...")
    model = YOLO(args.model)
    results = model.train(
        data=str(data_yaml.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=data_yaml.parent.name,
        patience=30,      # 15 -> 30 : laisse plus de temps sur des taches difficiles
        degrees=10,        # rotation legere -> plus de robustesse/variete
        mixup=0.1,          # melange d'images -> aide sur petits datasets
        copy_paste=0.1,     # colle des objets d'une image sur une autre
    )

    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best_weights, output_path)

    print("\n" + "=" * 60)
    print("ENTRAINEMENT TERMINE")
    print("=" * 60)
    print(f"Modele sauvegarde : {output_path.resolve()}")
    print(f"mAP50 (validation) : voir {results.save_dir}/results.csv ou results.png")
    print("Ce fichier est deja au bon endroit pour config.py -> relance main.py normalement.")
    print("=" * 60)


if __name__ == "__main__":
    main()
