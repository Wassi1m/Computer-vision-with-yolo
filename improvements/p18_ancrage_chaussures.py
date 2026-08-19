#!/usr/bin/env python3
"""P18 — Ce que la CASCADE produirait vraiment, ancrage compris.

Pourquoi cette mesure manquait
------------------------------
`p15_contre_epreuve_chaussures.py` fait tourner le modèle **nu** : un
`YOLO.predict` direct, sans détection de personne et sans règle de
qualification. C'est la bonne mesure pour juger un modèle en lui-même, et c'est
elle qui a montré, le 2026-08-19, un visage masqué détecté comme
`safety_shoe` à 0.93.

Mais ce n'est pas ce que le moteur produit. Dans `unified_surveillance.py`,
toute détection d'EPI passe par `qualification.associer_a_personne`, qui la
rejette si elle n'est pas confinée dans une personne **et** si sa hauteur est
anatomiquement absurde : `PLAGES_ANATOMIQUES["chaussures"] = (0.55, 1.10)`,
soit les 45 % inférieurs de la personne. Un visage se situe vers 0.10 : il est
donc rejeté par construction, sans qu'aucun ré-entraînement soit nécessaire.

Ce script mesure l'écart entre les deux, sur les mêmes images. Il répond à une
question précise, et à elle seule : **parmi les fausses détections du modèle nu,
combien la cascade laisse-t-elle passer ?**

Ce qu'il ne peut pas corriger
-----------------------------
L'ancrage ne juge que la GÉOMÉTRIE. Une basket de ville détectée au niveau des
pieds d'une personne est parfaitement plausible anatomiquement : elle passe.
Seules disparaissent les confusions grossières -- visage, torse, cône,
carrosserie. La distinction chaussure de sécurité / chaussure ordinaire, elle,
reste entièrement à la charge du modèle.

    python improvements/p18_ancrage_chaussures.py --poids <best.pt>
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qualification as qual  # noqa: E402

RACINE = Path(__file__).resolve().parents[1]
IMAGES = RACINE / "ppe_detection/data/extracted/ppe_dataset/test/images"
DETECTEUR = RACINE / "ppe_detection/models/ppe_detector.pt"
SORTIE_IMG = RACINE / "reports/v3_results/chaussures_ancrees"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--poids", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--echantillon", type=int, default=10)
    ap.add_argument("--sortie", default="reports/v3_results/chaussures_ancrage.json")
    args = ap.parse_args()

    from ultralytics import YOLO
    chaussures = YOLO(args.poids)
    personnes_m = YOLO(str(DETECTEUR))
    # `Person` est la seule classe utile ici ; son indice est lu dans le modele
    # plutot qu'ecrit en dur, la taxonomie ayant deja change une fois.
    idx_personne = next(i for i, n in personnes_m.names.items() if n == "Person")

    fichiers = sorted(IMAGES.glob("*"))
    lot = random.Random(0).sample(fichiers, min(args.n, len(fichiers)))
    print(f"ancrage mesure sur {len(lot)} images de ppe_dataset/test\n")

    brut = Counter()          # images qui declenchent, modele nu
    ancre = Counter()         # images qui declenchent, apres ancrage
    motifs = Counter()        # pourquoi une detection est rejetee
    survivantes: list[tuple[float, Path, str]] = []

    for i, img in enumerate(lot):
        if i % 25 == 0:
            print(f"  {i}/{len(lot)}", end="\r", flush=True)
        res = chaussures.predict(str(img), conf=args.conf, verbose=False)[0]
        if not len(res.boxes):
            continue
        brut["images"] += 1

        pres = personnes_m.predict(str(img), conf=0.25, verbose=False)[0]
        boites_p = [tuple(map(int, b.xyxy[0])) for b in pres.boxes
                    if int(b.cls) == idx_personne]

        gardees = []
        for b in res.boxes:
            boite = tuple(map(int, b.xyxy[0]))
            conf = float(b.conf)
            if not boites_p:
                # Aucune personne : une chaussure de securite PORTEE ne peut pas
                # exister. La cascade rejette.
                motifs["aucune personne detectee"] += 1
                continue
            j = qual.associer_a_personne(boite, boites_p, epi="chaussures")
            if j is None:
                score = max(qual.confinement(boite, p) for p in boites_p)
                motifs["hors personne" if score < 0.50
                       else "hauteur anatomique invraisemblable"] += 1
                continue
            gardees.append((conf, chaussures.names[int(b.cls)]))
        if gardees:
            ancre["images"] += 1
            c, nom = max(gardees)
            survivantes.append((c, img, nom))
    print(" " * 30, end="\r")

    n = len(lot)
    print(f"{'':<34}{'images':>9}{'taux':>9}")
    print(f"{'modele nu (p15)':<34}{brut['images']:>9}{100*brut['images']/n:>8.1f} %")
    print(f"{'apres ancrage a la personne':<34}{ancre['images']:>9}{100*ancre['images']/n:>8.1f} %")
    retire = brut["images"] - ancre["images"]
    if brut["images"]:
        print(f"\n  {retire} images cessent de declencher, soit "
              f"{100*retire/brut['images']:.0f} % des declenchements bruts")
    print("\n  motifs de rejet des detections :")
    for motif, cnt in motifs.most_common():
        print(f"    {motif:<40} {cnt}")

    # Ce qui SURVIT est ce qu'il faut regarder : l'ancrage ne dit rien de la
    # nature de l'objet, seulement de sa position.
    dossier = SORTIE_IMG
    dossier.mkdir(parents=True, exist_ok=True)
    for f in dossier.glob("*.jpg"):
        f.unlink()
    survivantes.sort(reverse=True, key=lambda x: x[0])
    echantillon = []
    for rang, (conf, img, nom) in enumerate(survivantes[:args.echantillon], 1):
        r = chaussures.predict(str(img), conf=args.conf, verbose=False)[0]
        cible = dossier / f"{rang:02d}_{nom}_{conf:.2f}_{img.stem[:34]}.jpg"
        r.save(filename=str(cible))
        echantillon.append({"rang": rang, "confiance": round(conf, 3),
                            "classe": nom, "image": cible.name})

    resultat = {
        "date": str(date.today()),
        "poids": args.poids,
        "mesure": f"{n} images aleatoires de ppe_dataset/test (graine 0), conf={args.conf}",
        "taux_modele_nu": round(brut["images"] / n, 4),
        "taux_apres_ancrage": round(ancre["images"] / n, 4),
        "reduction": round(retire / brut["images"], 4) if brut["images"] else None,
        "motifs_de_rejet": dict(motifs),
        "limite": ("L'ancrage ne juge que la geometrie. Une chaussure de ville detectee au "
                   "niveau des pieds est anatomiquement plausible et passe : la distinction "
                   "chaussure de securite / chaussure ordinaire reste entierement a la charge "
                   "du modele."),
        "echantillon_survivant": echantillon,
        "dossier_echantillon": str(dossier.relative_to(RACINE)),
    }
    sortie = RACINE / args.sortie
    sortie.write_text(json.dumps(resultat, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print(f"\n  {len(echantillon)} images survivantes annotees -> "
          f"{dossier.relative_to(RACINE)}")
    print(f"-> {sortie.relative_to(RACINE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
