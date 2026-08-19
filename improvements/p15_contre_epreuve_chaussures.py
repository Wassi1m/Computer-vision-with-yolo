#!/usr/bin/env python3
"""P15 — Contre-épreuve faux positifs pour `epi_chaussures.pt`.

Pourquoi ce modèle ne peut pas être jugé comme les deux autres
--------------------------------------------------------------
`epi_casque` et `epi_gants_lunettes` se mesurent sur `ppe_dataset/test`, qui
annote leurs concepts. Pour la chaussure, il n'existe **aucun jeu de test
indépendant** : `ppe_dataset` n'annote aucune chaussure, et les seules sources
disponibles ont dû être mises en commun puis redécoupées pour l'entraînement
lui-même (voir `p13_jeux_roboflow.py`).

Sa mAP de validation (0.9343) est donc invérifiable, et probablement flattée :
les sources Roboflow sont augmentées, et un redécoupage sur 938 images laisse
passer des variantes d'une même photo des deux côtés.

Le seul risque qui compte ici
-----------------------------
Ce n'est pas de rater une chaussure, c'est d'en **halluciner**. Un modèle
entraîné sur 750 images qui signale des chaussures de sécurité partout est pire
qu'une classe absente : il pollue la cascade et fabrique de fausses conformités.

Ce script mesure donc le taux de déclenchement sur des images de `ppe_dataset`,
à plusieurs seuils de confiance, et sauvegarde un échantillon annoté.

    ⚠️ Le taux seul ne conclut pas. `ppe_dataset` n'annote pas les chaussures
    mais en CONTIENT : une détection y est peut-être parfaitement correcte.
    C'est pourquoi l'échantillon d'images est produit -- le verdict demande de
    les regarder, et le script le dit plutôt que de trancher à la place.

    python improvements/p15_contre_epreuve_chaussures.py --poids <best.pt>
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import date
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
IMAGES = RACINE / "ppe_detection/data/extracted/ppe_dataset/test/images"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--poids", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seuils", type=float, nargs="+",
                    default=[0.25, 0.40, 0.50, 0.70])
    ap.add_argument("--echantillon", type=int, default=12)
    ap.add_argument("--sortie", default="reports/v3_results/chaussures_candidat.json")
    ap.add_argument("--dossier-images",
                    default="reports/v3_results/chaussures_echantillon")
    args = ap.parse_args()

    poids = Path(args.poids)
    if not poids.is_file():
        print(f"introuvable : {poids}", file=sys.stderr)
        return 1

    from ultralytics import YOLO
    modele = YOLO(str(poids))

    fichiers = sorted(IMAGES.glob("*"))
    if not fichiers:
        print(f"aucune image dans {IMAGES}", file=sys.stderr)
        return 1
    lot = random.Random(0).sample(fichiers, min(args.n, len(fichiers)))
    print(f"contre-epreuve sur {len(lot)} images de ppe_dataset/test\n")

    # Une seule passe au seuil le plus bas : les seuils supérieurs se déduisent
    # des confiances obtenues, inutile de repasser le modele quatre fois.
    seuil_bas = min(args.seuils)
    declenchements = {s: 0 for s in args.seuils}
    confiances: list[tuple[float, Path, int]] = []

    for i, img in enumerate(lot):
        if i % 25 == 0:
            print(f"  {i}/{len(lot)}", end="\r", flush=True)
        res = modele.predict(str(img), conf=seuil_bas, verbose=False)[0]
        if not len(res.boxes):
            continue
        confs = res.boxes.conf.tolist()
        classes = [int(c) for c in res.boxes.cls.tolist()]
        cmax = max(confs)
        confiances.append((cmax, img, classes[confs.index(cmax)]))
        for s in args.seuils:
            if cmax >= s:
                declenchements[s] += 1
    print(" " * 30, end="\r")

    noms = modele.names
    print(f"{'seuil':>7} {'images qui declenchent':>24} {'taux':>8}")
    taux = {}
    for s in sorted(args.seuils):
        t = declenchements[s] / len(lot)
        taux[str(s)] = round(t, 4)
        print(f"{s:>7.2f} {declenchements[s]:>24} {100*t:>7.1f} %")

    # Echantillon annote : les detections les plus CONFIANTES, car ce sont
    # elles qui passeront tous les seuils en production -- et donc celles dont
    # une hallucination coute le plus cher.
    dossier = RACINE / args.dossier_images
    dossier.mkdir(parents=True, exist_ok=True)
    for f in dossier.glob("*.jpg"):
        f.unlink()
    confiances.sort(reverse=True, key=lambda x: x[0])
    echantillon = []
    for rang, (conf, img, cls) in enumerate(confiances[:args.echantillon], 1):
        res = modele.predict(str(img), conf=seuil_bas, verbose=False)[0]
        cible = dossier / f"{rang:02d}_{noms[cls]}_{conf:.2f}_{img.stem}.jpg"
        res.save(filename=str(cible))
        echantillon.append({"rang": rang, "confiance": round(conf, 3),
                            "classe": noms[cls], "image": cible.name})

    resultat = {
        "date": str(date.today()),
        "candidat": "chaussures",
        "poids": str(poids),
        "mesure": f"taux de declenchement sur {len(lot)} images aleatoires de "
                  f"ppe_dataset/test (graine 0)",
        "avertissement": "ppe_dataset n'annote AUCUNE chaussure mais en contient : "
                         "un declenchement n'est pas forcement une erreur. Le taux "
                         "ci-dessous ne conclut pas seul -- regarder l'echantillon.",
        "taux_declenchement": taux,
        "echantillon_a_inspecter": echantillon,
        "dossier_echantillon": str(dossier.relative_to(RACINE)),
    }
    sortie = RACINE / args.sortie
    sortie.parent.mkdir(parents=True, exist_ok=True)
    sortie.write_text(json.dumps(resultat, ensure_ascii=False, indent=2),
                      encoding="utf-8")

    print(f"\n{len(echantillon)} images annotees -> {dossier}")
    print(f"-> {sortie}")
    print("\nA REGARDER : ces detections sont-elles de vraies chaussures de "
          "securite ?\nLe chiffre ne repond pas a cette question, l'oeil si.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
