#!/usr/bin/env python3
"""P1 étape 1 — Audit de la qualité des annotations `Safety Vest` / `NO-Safety Vest`.

Contexte : ces deux classes sont les plus faibles de `best.pt`
(AP@50 = 4.8 % pour `NO-Safety Vest`, 49.4 % pour `Safety Vest`) alors que le
diagnostic de la v1 a écarté le seuil de confiance comme cause. Avant d'investir
dans un ré-entraînement, il faut écarter un défaut d'annotation.

L'audit est purement statistique (lecture des fichiers de labels, aucune
inférence) et cherche les signatures classiques d'un jeu d'annotations
incohérent :

  1. Volume et répartition train/val/test des deux classes.
  2. Cohérence avec `Person` : un gilet (ou une absence de gilet) doit se
     superposer à une personne. Une boîte gilet sans personne recouvrante est
     suspecte.
  3. Contradictions internes : une même personne annotée à la fois `Safety Vest`
     et `NO-Safety Vest` (IoU élevé entre les deux) est une annotation
     contradictoire, poison direct pour l'apprentissage.
  4. Sous-annotation : images contenant des personnes mais aucune étiquette
     gilet — le modèle y apprend que « personne sans étiquette gilet » est un
     fond valide, ce qui écrase le rappel de `NO-Safety Vest`.
  5. Distribution géométrique des boîtes (aire, ratio, position dans la boîte
     personne) comparée aux classes qui fonctionnent (`Hardhat`).
"""

from pathlib import Path
from collections import Counter, defaultdict
import json
import statistics

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "ppe_detection/data/extracted/ppe_dataset"
NAMES = ['Fall-Detected', 'Gloves', 'Goggles', 'Hardhat', 'Ladder', 'Mask', 'NO-Gloves',
         'NO-Goggles', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone', 'Safety Vest']
IDX = {n: i for i, n in enumerate(NAMES)}
VEST, NOVEST, PERSON, HARDHAT, NOHARDHAT = IDX['Safety Vest'], IDX['NO-Safety Vest'], IDX['Person'], IDX['Hardhat'], IDX['NO-Hardhat']


def to_xyxy(b):
    cx, cy, w, h = b
    return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


def iou(a, b):
    ax1, ay1, ax2, ay2 = to_xyxy(a)
    bx1, by1, bx2, by2 = to_xyxy(b)
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def contained(inner, outer):
    """Fraction de `inner` couverte par `outer`."""
    ax1, ay1, ax2, ay2 = to_xyxy(inner)
    bx1, by1, bx2, by2 = to_xyxy(outer)
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    area = inner[2] * inner[3]
    return (iw * ih) / area if area > 0 else 0.0


def read_split(split):
    for lbl in sorted((DATA / split / "labels").glob("*.txt")):
        rows = []
        for line in lbl.read_text().splitlines():
            p = line.split()
            if len(p) >= 5:
                rows.append((int(p[0]), [float(x) for x in p[1:5]]))
        yield lbl.stem, rows


def audit(split):
    counts = Counter()
    images_with = Counter()
    orphan = Counter()          # boîte gilet non couverte par une personne
    contradictions = 0
    contradiction_files = []
    persons_unlabelled = 0      # personnes sans aucune étiquette gilet
    persons_total = 0
    imgs_person_no_vest = 0
    imgs_with_person = 0
    geom = defaultdict(list)    # aire relative des boîtes par classe
    rel_pos = defaultdict(list)  # position verticale du gilet dans la personne

    for _stem, rows in read_split(split):
        cls_here = {c for c, _ in rows}
        for c in cls_here:
            images_with[c] += 1
        for c, _ in rows:
            counts[c] += 1

        persons = [b for c, b in rows if c == PERSON]
        vests = [(c, b) for c, b in rows if c in (VEST, NOVEST)]

        for c, b in rows:
            if c in (VEST, NOVEST, HARDHAT, NOHARDHAT):
                geom[c].append(b[2] * b[3])

        # 2/5 — rattachement à une personne
        for c, b in vests:
            best_p, best_cov = None, 0.0
            for pb in persons:
                cov = contained(b, pb)
                if cov > best_cov:
                    best_cov, best_p = cov, pb
            if best_cov < 0.5:
                orphan[c] += 1
            elif best_p is not None and best_p[3] > 0:
                rel_pos[c].append((b[1] - (best_p[1] - best_p[3] / 2)) / best_p[3])

        # 3 — contradictions Safety Vest / NO-Safety Vest sur la même zone
        for cb in [b for c, b in vests if c == VEST]:
            for nb in [b for c, b in vests if c == NOVEST]:
                if iou(cb, nb) > 0.5:
                    contradictions += 1
                    contradiction_files.append(_stem)
                    break

        # 4 — personnes non couvertes par une étiquette gilet
        if persons:
            imgs_with_person += 1
            persons_total += len(persons)
            if not vests:
                imgs_person_no_vest += 1
                persons_unlabelled += len(persons)
            else:
                for pb in persons:
                    if not any(contained(b, pb) > 0.5 for _c, b in vests):
                        persons_unlabelled += 1

    def summ(vals):
        if not vals:
            return None
        vals = sorted(vals)
        return {"n": len(vals), "median": round(statistics.median(vals), 5),
                "p10": round(vals[len(vals) // 10], 5), "p90": round(vals[-max(1, len(vals) // 10)], 5)}

    return {
        "split": split,
        "instances": {NAMES[c]: counts[c] for c in (VEST, NOVEST, PERSON, HARDHAT, NOHARDHAT)},
        "images_contenant": {NAMES[c]: images_with[c] for c in (VEST, NOVEST, PERSON, HARDHAT, NOHARDHAT)},
        "boites_gilet_sans_personne": {NAMES[c]: orphan[c] for c in (VEST, NOVEST)},
        "contradictions_vest_vs_novest": contradictions,
        "exemples_contradictions": contradiction_files[:10],
        "images_avec_personne": imgs_with_person,
        "images_personne_sans_aucune_etiquette_gilet": imgs_person_no_vest,
        "personnes_total": persons_total,
        "personnes_sans_etiquette_gilet": persons_unlabelled,
        "aire_relative_boites": {NAMES[c]: summ(geom[c]) for c in (VEST, NOVEST, HARDHAT, NOHARDHAT)},
        "position_verticale_dans_personne": {NAMES[c]: summ(rel_pos[c]) for c in (VEST, NOVEST)},
    }


if __name__ == "__main__":
    out = [audit(s) for s in ("train", "val", "test")]
    dest = ROOT / "reports/v2_results/p1_audit_annotations_gilet.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    for r in out:
        print(json.dumps(r, indent=2, ensure_ascii=False))
    print(f"\n-> {dest}")
