#!/usr/bin/env python3
"""P14 — Juge les candidats de la campagne v8 sur un split JAMAIS vu.

Pourquoi ce script existe, et pourquoi il ne lit pas `resume_entrainement.json`
-----------------------------------------------------------------------------
Chaque entraînement publie une mAP mesurée sur SON PROPRE split de validation.
Ce chiffre ne peut pas servir de verdict, pour deux raisons distinctes :

1. Le split `val` du jeu fusionné contient des images de `ppe_dataset`, sur
   lesquelles les poids de départ (`ppe_detector.pt`) ont déjà été entraînés.
2. Plusieurs sources Roboflow sont AUGMENTEES (Safety Gloves v5 annonce 10 459
   images pour ~3 373 originales, soit x3). Si Roboflow a découpé train/val
   APRES augmentation, des variantes d'une même photo se retrouvent des deux
   côtés : la mAP de validation mesure alors de la mémorisation, pas de la
   généralisation. Le candidat `gants` a trouvé son meilleur score dès
   l'époque 1 et ne l'a jamais battu en 22 époques -- exactement ce à quoi
   ressemble une validation qui fuit.

Le verdict porte donc sur `ppe_dataset/test` (4 423 images), filtré et remappé
vers la taxonomie du candidat. Ce split n'a JAMAIS été téléversé sur Kaggle ni
inclus dans aucun jeu d'entraînement -- même règle que pour `masque_gilet.pt`.

Deux indicateurs, jamais confondus (cf. l'en-tête de `models_calsse.txt`)
------------------------------------------------------------------------
- « détecté » : proportion des scènes annotées où le modèle voit l'objet.
  C'est l'indicateur d'EXPLOITATION, et c'est lui qui décide ici.
- « AP@50 »   : précision du rectangle. Utile pour comparer, trompeur pour juger.

Critères de rejet, écrits le 2026-08-17 AVANT le premier résultat
------------------------------------------------------------------
`casque` : validé si Hardhat >= 75 % ET NO-Hardhat >= 80 % de détection de
    scène ; rejeté si l'un des deux passe sous son niveau actuel (65 % / 72 %).
`gants`  : validé seulement s'il bat `ppe_detector.pt`, qui tient déjà
    AP 0.932 / 0.909 / 0.960 / 0.961 sur ces quatre classes.

    python improvements/p14_juger_candidats.py --candidat casque --poids <best.pt>
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import date
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
PPE_DATASET = RACINE / "ppe_detection/data/extracted/ppe_dataset"
TRAVAIL = RACINE / "ppe_detection/data/extracted/_jugement"

# classe ppe_dataset -> classe du candidat, et niveau actuel de detection de
# scene mesure le 2026-08-16 sur ppe_detector.pt (reports/v3_results/).
CANDIDATS = {
    "casque": {
        "noms": ["Hardhat", "NO-Hardhat"],
        "map": {3: 0, 8: 1},
        "actuel": {"Hardhat": 0.65, "NO-Hardhat": 0.72},
        "seuil": {"Hardhat": 0.75, "NO-Hardhat": 0.80},
    },
    "gants": {
        "noms": ["Gloves", "NO-Gloves", "Goggles", "NO-Goggles"],
        "map": {1: 0, 6: 1, 2: 2, 7: 3},
        # Pas de detection de scene publiee pour ces classes : le point de
        # comparaison est l'AP de ppe_detector.pt, deja excellent.
        "ap_actuel": {"Gloves": 0.932, "NO-Gloves": 0.909,
                      "Goggles": 0.960, "NO-Goggles": 0.961},
    },
}


def batir_test(cle: str) -> Path:
    """Construit le split test filtré + remappé. Ne copie que des liens."""
    spec = CANDIDATS[cle]
    dest = TRAVAIL / cle
    if dest.exists():
        shutil.rmtree(dest)
    (dest / "test" / "images").mkdir(parents=True)
    (dest / "test" / "labels").mkdir(parents=True)

    n = 0
    for lbl in sorted((PPE_DATASET / "test" / "labels").glob("*.txt")):
        gardees = []
        for ligne in lbl.read_text(encoding="utf-8").splitlines():
            champs = ligne.split()
            if len(champs) >= 5 and int(champs[0]) in spec["map"]:
                gardees.append((spec["map"][int(champs[0])], champs[1:5]))
        if not gardees:
            continue
        img = next((PPE_DATASET / "test" / "images").glob(lbl.stem + ".*"), None)
        if img is None:
            continue
        n += 1
        # Lien symbolique : le split test pese plusieurs centaines de Mo et
        # n'a aucune raison d'etre duplique a chaque jugement.
        (dest / "test" / "images" / img.name).symlink_to(img.resolve())
        (dest / "test" / "labels" / f"{lbl.stem}.txt").write_text(
            "\n".join(f"{c} {' '.join(co)}" for c, co in gardees) + "\n",
            encoding="utf-8")

    (dest / "data.yaml").write_text(
        f"path: {dest}\ntrain: test/images\nval: test/images\n\n"
        f"nc: {len(spec['noms'])}\nnames: {spec['noms']}\n", encoding="utf-8")
    print(f"split de jugement : {n} images ({dest})")
    return dest


def detection_scene(poids: Path, jeu: Path, spec: dict, conf: float) -> dict:
    """Proportion des scènes annotées où le modèle voit effectivement la classe.

    C'est l'indicateur d'exploitation : il répond à « sur les images qui
    contiennent un casque, combien de fois le modèle en voit-il un ? », sans
    rien dire de la position du rectangle.
    """
    from ultralytics import YOLO
    modele = YOLO(str(poids))
    noms = spec["noms"]
    attendu = {i: 0 for i in range(len(noms))}
    trouve = {i: 0 for i in range(len(noms))}

    labels = sorted((jeu / "test" / "labels").glob("*.txt"))
    for i, lbl in enumerate(labels):
        if i % 200 == 0:
            print(f"  {i}/{len(labels)}", end="\r", flush=True)
        verite = {int(l.split()[0]) for l in lbl.read_text().splitlines() if l.strip()}
        if not verite:
            continue
        img = next((jeu / "test" / "images").glob(lbl.stem + ".*"), None)
        if img is None:
            continue
        res = modele.predict(str(img), conf=conf, verbose=False)[0]
        vues = {int(c) for c in res.boxes.cls.tolist()}
        for c in verite:
            attendu[c] += 1
            if c in vues:
                trouve[c] += 1
    print(" " * 30, end="\r")
    return {noms[c]: (trouve[c] / attendu[c] if attendu[c] else None)
            for c in attendu}, {noms[c]: attendu[c] for c in attendu}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidat", choices=list(CANDIDATS), required=True)
    ap.add_argument("--poids", required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--sortie", default=None)
    args = ap.parse_args()

    poids = Path(args.poids)
    if not poids.is_file():
        print(f"introuvable : {poids}", file=sys.stderr)
        return 1

    spec = CANDIDATS[args.candidat]
    jeu = batir_test(args.candidat)

    from ultralytics import YOLO
    print("\n--- AP@50 (YOLO.val) ---")
    metriques = YOLO(str(poids)).val(data=str(jeu / "data.yaml"), split="val",
                                     imgsz=640, device="cpu", verbose=False)
    ap50 = {spec["noms"][i]: round(float(v), 4)
            for i, v in enumerate(metriques.box.ap50)}

    print("\n--- detection de scene ---")
    detecte, effectifs = detection_scene(poids, jeu, spec, args.conf)

    print(f"\n{'classe':<14} {'scenes':>7} {'detecte':>9} {'AP@50':>8}")
    for nom in spec["noms"]:
        d = detecte[nom]
        print(f"{nom:<14} {effectifs[nom]:>7} "
              f"{(f'{100*d:.1f} %' if d is not None else '--'):>9} "
              f"{ap50.get(nom, float('nan')):>8.3f}")

    resultat = {
        "date": str(date.today()),
        "candidat": args.candidat,
        "poids": str(poids),
        "mesure": f"ppe_dataset/test filtre+remappe ({sum(effectifs.values())} scenes annotees), "
                  f"jamais inclus dans un jeu d'entrainement, conf={args.conf}",
        "detection_scene": {k: (round(v, 4) if v is not None else None)
                            for k, v in detecte.items()},
        "ap50": ap50,
        "mAP50_global": round(float(metriques.box.map50), 4),
        "effectifs": effectifs,
    }

    if "seuil" in spec:
        verdicts = []
        for nom, seuil in spec["seuil"].items():
            d, actuel = detecte[nom], spec["actuel"][nom]
            if d is None:
                verdicts.append(f"{nom} : non mesurable")
            elif d < actuel:
                verdicts.append(f"{nom} : REJET ({100*d:.1f} % < {100*actuel:.0f} % actuel)")
            elif d >= seuil:
                verdicts.append(f"{nom} : VALIDE ({100*d:.1f} % >= {100*seuil:.0f} %)")
            else:
                verdicts.append(f"{nom} : ZONE GRISE ({100*d:.1f} %, entre "
                                f"{100*actuel:.0f} % et {100*seuil:.0f} %)")
        resultat["verdict"] = verdicts
        print("\n--- verdict (criteres ecrits avant le premier resultat) ---")
        for v in verdicts:
            print(f"  {v}")

    if "ap_actuel" in spec:
        comp = {nom: {"candidat": ap50.get(nom), "ppe_detector": ref,
                      "gagne": (ap50.get(nom, 0) > ref)}
                for nom, ref in spec["ap_actuel"].items()}
        resultat["comparaison_ppe_detector"] = comp
        print("\n--- face a ppe_detector.pt ---")
        for nom, c in comp.items():
            print(f"  {nom:<12} candidat {c['candidat']:.3f}  vs  "
                  f"{c['ppe_detector']:.3f}   {'GAGNE' if c['gagne'] else 'PERD'}")

    sortie = Path(args.sortie or
                  RACINE / f"reports/v3_results/{args.candidat}_candidat.json")
    sortie.parent.mkdir(parents=True, exist_ok=True)
    sortie.write_text(json.dumps(resultat, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print(f"\n-> {sortie}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
