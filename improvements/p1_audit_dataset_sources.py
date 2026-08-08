#!/usr/bin/env python3
"""P1 étape 1bis — Identification des sous-jeux de données hétérogènes.

L'audit d'annotations (p1_audit_vest_annotations.py) a remonté un signal
inattendu : *aucune* boîte gilet n'est couverte par une boîte `Person`, et
*aucune* image contenant `Person` ne porte d'étiquette gilet. Les deux groupes
de classes sont donc disjoints au niveau image, ce qui trahit une fusion de
plusieurs jeux annotés selon des conventions différentes.

Ce script quantifie le phénomène :
  - matrice de co-occurrence des classes au niveau image ;
  - regroupement des images par préfixe de nom de fichier (les exports Roboflow
    conservent le nom source avant le suffixe `.rf.<hash>`), pour rattacher
    chaque convention d'annotation à son lot d'origine.
"""

from pathlib import Path
from collections import Counter, defaultdict
import json
import re

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "ppe_detection/data/extracted/ppe_dataset"
NAMES = ['Fall-Detected', 'Gloves', 'Goggles', 'Hardhat', 'Ladder', 'Mask', 'NO-Gloves',
         'NO-Goggles', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone', 'Safety Vest']

PREFIX_RE = re.compile(r"^([A-Za-z_\-]*?)[-_]?\d*[-_]?(jpg|jpeg|png|JPG|PNG)?$")


def prefix_of(stem: str) -> str:
    base = stem.split(".rf.")[0]
    base = re.sub(r"[-_]?(jpg|jpeg|png|webp)$", "", base, flags=re.I)
    base = re.sub(r"\d+", "#", base)
    return base[:40]


def main(split="train"):
    cooc = defaultdict(Counter)
    per_prefix = defaultdict(Counter)
    prefix_imgs = Counter()

    for lbl in sorted((DATA / split / "labels").glob("*.txt")):
        classes = set()
        for line in lbl.read_text().splitlines():
            p = line.split()
            if len(p) >= 5:
                classes.add(int(p[0]))
        pref = prefix_of(lbl.stem)
        prefix_imgs[pref] += 1
        for c in classes:
            per_prefix[pref][c] += 1
            for d in classes:
                cooc[c][d] += 1

    print(f"=== Co-occurrence au niveau image ({split}) — ligne = classe A, valeur = nb d'images contenant A et B")
    header = "".join(f"{n[:9]:>10}" for n in NAMES)
    print(f"{'':22}{header}")
    for i, n in enumerate(NAMES):
        row = "".join(f"{cooc[i][j]:>10}" for j in range(len(NAMES)))
        print(f"{n:22}{row}")

    print(f"\n=== Lots sources (préfixe de nom de fichier), top 25 par volume")
    print(f"{'prefixe':42}{'images':>8}  classes présentes (nb images)")
    for pref, n in prefix_imgs.most_common(25):
        cls = ", ".join(f"{NAMES[c]}:{v}" for c, v in per_prefix[pref].most_common())
        print(f"{pref:42}{n:>8}  {cls[:150]}")

    out = ROOT / "reports/v2_results/p1_sources_dataset.json"
    out.write_text(json.dumps({
        "split": split,
        "cooccurrence": {NAMES[i]: {NAMES[j]: cooc[i][j] for j in range(len(NAMES)) if cooc[i][j]} for i in range(len(NAMES))},
        "lots": {p: {"images": prefix_imgs[p], "classes": {NAMES[c]: v for c, v in per_prefix[p].items()}}
                 for p, _ in prefix_imgs.most_common(40)},
    }, indent=2, ensure_ascii=False))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
