#!/usr/bin/env python3
"""P4 — Téléchargement de données `smoke` complémentaires.

Constat de départ : dans `Fire -smoke Detection.v1i.yolo26`, la classe `smoke`
compte 1 947 instances d'entraînement contre 9 923 pour `fire`, et surtout
seulement 97 instances de validation — l'AP@50 de 27.6 % est donc à la fois bas
*et* mesuré sur trop peu d'exemples pour être stable.

Source retenue : **pyronear/pyro-sdis** (Hugging Face, Apache-2.0, public, non
gated). 33 636 images de fumée annotées au format YOLO, produites par les
caméras de détection de l'association Pyronear avec les SDIS français. C'est la
seule source trouvée qui soit simultanément : librement téléchargeable sans
clé d'API, déjà au format YOLO, et suffisamment volumineuse.

Réserve importante, à garder en tête pour l'interprétation des résultats :
pyro-sdis est de la fumée de *feu de forêt vue de loin depuis une tour de
guet* — panaches diffus sur fond de paysage. Le jeu existant contient de la
fumée plus proche et plus variée. On n'importe donc qu'un sous-ensemble
(par défaut ~4 000 images) plutôt que les 33 000 : l'objectif est de rééquilibrer
`smoke` face à `fire`, pas de transformer le modèle en détecteur de feux de
forêt. L'évaluation reste faite sur le jeu de validation d'origine pour que la
comparaison avec les 27.6 % actuels garde un sens.

Les images sont stockées en binaire dans les parquet, les annotations YOLO dans
une colonne texte ; on ne télécharge qu'un shard d'entraînement sur six.
"""

from pathlib import Path
import argparse
import io

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "surveillance_suite/data/dataset/pyro_sdis_subset"
REPO = "pyronear/pyro-sdis"


def extraire(parquet_path: Path, split_out: str, limite: int, dest: Path) -> int:
    import pyarrow.parquet as pq
    from PIL import Image

    (dest / split_out / "images").mkdir(parents=True, exist_ok=True)
    (dest / split_out / "labels").mkdir(parents=True, exist_ok=True)

    n = 0
    pf = pq.ParquetFile(parquet_path)
    for batch in pf.iter_batches(batch_size=64, columns=["image", "annotations", "image_name"]):
        d = batch.to_pydict()
        for img_struct, ann, name in zip(d["image"], d["annotations"], d["image_name"]):
            if n >= limite:
                return n
            lignes = [l.strip() for l in (ann or "").splitlines() if l.strip()]
            if not lignes:
                continue  # on ne garde que les images effectivement annotées fumée
            stem = Path(name).stem
            raw = img_struct["bytes"] if isinstance(img_struct, dict) else img_struct
            try:
                Image.open(io.BytesIO(raw)).verify()
            except Exception:
                continue
            (dest / split_out / "images" / f"{stem}.jpg").write_bytes(raw)
            (dest / split_out / "labels" / f"{stem}.txt").write_text("\n".join(lignes) + "\n")
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=4000, help="nb d'images d'entraînement à extraire")
    ap.add_argument("--val", type=int, default=800, help="nb d'images de validation à extraire")
    ap.add_argument("--dest", type=Path, default=DEST)
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download

    cache = ROOT / ".cache_hf"
    cache.mkdir(exist_ok=True)

    print("Téléchargement du shard d'entraînement (1/6) ...")
    p_train = Path(hf_hub_download(REPO, "data/train-00000-of-00006.parquet",
                                   repo_type="dataset", cache_dir=str(cache)))
    print("Téléchargement du shard de validation ...")
    p_val = Path(hf_hub_download(REPO, "data/val-00000-of-00001.parquet",
                                 repo_type="dataset", cache_dir=str(cache)))

    n_tr = extraire(p_train, "train", args.train, args.dest)
    n_va = extraire(p_val, "valid", args.val, args.dest)

    (args.dest / "data.yaml").write_text(
        f"path: {args.dest}\ntrain: train/images\nval: valid/images\n\nnc: 1\nnames: ['smoke']\n")
    print(f"\ntrain={n_tr} images, valid={n_va} images -> {args.dest}")


if __name__ == "__main__":
    main()
