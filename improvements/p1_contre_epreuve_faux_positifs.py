#!/usr/bin/env python3
"""P1 — Contre-épreuve : les « faux positifs » gilet sont-ils de vraies erreurs ?

Hypothèse à tester
------------------
Sur le jeu de validation complet, `Safety Vest` plafonne à AP@50 = 49.4 % et
`NO-Safety Vest` à 4.8 %. Sur le sous-ensemble où les gilets sont réellement
annotés, le **même modèle** obtient 89.1 % et 62.8 %. Si l'écart vient bien du
jeu de données et non du modèle, alors sur les images du « lot casques » — celles
qui n'annotent que `Hardhat`/`NO-Hardhat` — `ppe_detector.pt` doit produire un grand
nombre de détections gilet, comptées comme faux positifs par la métrique, alors
qu'elles sont visuellement correctes.

Ce script mesure directement ce taux : combien d'images sans aucune annotation
gilet reçoivent malgré tout une détection gilet confiante, et sauvegarde un
échantillon annoté pour vérification à l'œil.
"""

from pathlib import Path
import argparse
import json
import random

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "ppe_detection/data/extracted/ppe_dataset/val"
VEST, NOVEST = 13, 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--echantillon", type=int, default=6)
    args = ap.parse_args()

    import cv2
    import numpy as np
    from ultralytics import YOLO

    # Images du lot « casques » : elles annotent le casque et rien d'autre, donc
    # tout gilet visible y est structurellement non annoté.
    candidats = []
    for lbl in sorted((DATA / "labels").glob("*.txt")):
        cs = {int(l.split()[0]) for l in lbl.read_text().splitlines() if l.strip()}
        if cs and cs <= {3, 8}:
            candidats.append(lbl)
    random.seed(0)
    echant = random.sample(candidats, min(args.n, len(candidats)))
    print(f"{len(candidats)} images 'casque seul' dans le val ; {len(echant)} tirées")

    model = YOLO(str(ROOT / "ppe_detection/models/ppe_detector.pt"))
    n_img_avec_gilet = 0
    n_det = 0
    confs = []
    apercus = []

    for lbl in echant:
        img_p = next((DATA / "images").glob(lbl.stem + ".*"))
        r = model.predict(str(img_p), conf=args.conf, imgsz=640, verbose=False)[0]
        dets = [(model.names[int(b.cls)], float(b.conf), tuple(map(int, b.xyxy[0])))
                for b in r.boxes if int(b.cls) in (VEST, NOVEST)]
        if dets:
            n_img_avec_gilet += 1
            n_det += len(dets)
            confs += [c for _, c, _ in dets]
            if len(apercus) < args.echantillon:
                im = cv2.imread(str(img_p))
                for nom, c, (x1, y1, x2, y2) in dets:
                    col = (0, 200, 255) if nom == "Safety Vest" else (0, 0, 255)
                    cv2.rectangle(im, (x1, y1), (x2, y2), col, 2)
                    cv2.putText(im, f"{nom} {c:.2f}", (x1, max(14, y1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
                apercus.append(cv2.resize(im, (384, 384)))

    res = {
        "images_testees": len(echant),
        "images_avec_detection_gilet_non_annotee": n_img_avec_gilet,
        "taux": round(n_img_avec_gilet / max(len(echant), 1), 4),
        "detections_gilet_total": n_det,
        "conf_moyenne": round(sum(confs) / len(confs), 4) if confs else None,
        "conf_max": round(max(confs), 4) if confs else None,
        "seuil": args.conf,
    }
    print(json.dumps(res, indent=2, ensure_ascii=False))

    out = ROOT / "reports/v2_results"
    (out / "p1_contre_epreuve.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    if apercus:
        n = len(apercus)
        lignes = [np.hstack(apercus[i:i + 3]) for i in range(0, n - n % 3, 3)]
        if lignes:
            cv2.imwrite(str(out / "p1_contre_epreuve_exemples.png"), np.vstack(lignes))
            print(f"-> {out / 'p1_contre_epreuve_exemples.png'}")


if __name__ == "__main__":
    main()
