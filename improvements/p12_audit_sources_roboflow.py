#!/usr/bin/env python3
"""P12 — Audit du contenu réel des jeux Roboflow, avant tout remappage.

Pourquoi cet audit passe AVANT la fusion
----------------------------------------
Le plan v7 annonçait des volumes et des classes tirés de la recherche Roboflow,
pas d'une lecture des pages (bloquées en HTTP 403). L'inspection des `data.yaml`
du 2026-08-17 a déjà montré que **trois jeux sur six** ne correspondaient pas à
leur description. Ce script pousse la vérification d'un cran : les noms de
classes déclarés ne disent rien de ce qu'elles contiennent réellement.

Trois questions bloquantes, chacune capable d'invalider une table de
correspondance (voir `reports/plan_amelioration/v8_campagne_epi.md` §5) :

A. `head` dans Hard Hat Universe désigne-t-il une tête NUE, ou toute tête ?
   Si le jeu annote aussi les têtes casquées comme `head`, alors le mappage
   `head -> NO-Hardhat` apprendrait au modèle que des ouvriers casqués sont des
   infractions -- sur la classe la plus critique du parc. L'indice mesurable :
   la proportion d'images contenant `head` ET `helmet`. Un jeu où `head` = tête
   nue en contient beaucoup (scènes mixtes), mais `head` et `helmet` n'y sont
   jamais sur la MÊME boîte. On mesure donc aussi le recouvrement des boîtes.

B. `Safety-shoes` et `safety_shoe` sont-elles la même chose ? Deux classes pour
   un concept identique dans un jeu de 415 images sent l'artefact d'annotation.

C. Combien d'images de PPEs annotent réellement `shoes` ? Le jeu compte 19 420
   images d'entraînement, mais rien ne dit combien portent une chaussure --
   et c'est ce chiffre seul qui décide si un modèle chaussures est viable.

    python improvements/p12_audit_sources_roboflow.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
SOURCES = RACINE / "ppe_detection/data/extracted/sources_roboflow"

# Index de classes lus dans les data.yaml le 2026-08-17.
CLASSES = {
    "hard_hat_universe": ["head", "helmet", "hi-viz helmet", "hi-viz vest", "person"],
    "ppes": ["glove", "goggles", "helmet", "mask", "no-suit", "no_glove",
             "no_goggles", "no_helmet", "no_mask", "no_shoes", "shoes", "suit"],
    "safety_gloves": ["Gloves", "NO-Gloves"],
    "construction_ppe": ["hat", "no hat", "no vest", "vest"],
    "safety_shoes_detection": ["Safety-shoes", "not_safety_shoe", "safety_shoe"],
    "construction_helmet_detection": ["head"],
}


def lire_labels(jeu: str, split: str = "train"):
    """Rend, pour chaque fichier label, la liste (classe, x, y, w, h)."""
    dossier = SOURCES / jeu / split / "labels"
    if not dossier.is_dir():
        return
    for lbl in sorted(dossier.glob("*.txt")):
        boites = []
        for ligne in lbl.read_text(encoding="utf-8").splitlines():
            champs = ligne.split()
            if len(champs) >= 5:
                boites.append((int(champs[0]), *(float(v) for v in champs[1:5])))
        yield lbl.stem, boites


def iou_xywh(a, b) -> float:
    """IoU entre deux boîtes YOLO normalisées (cx, cy, w, h)."""
    def coins(c):
        cx, cy, w, h = c
        return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    ax1, ay1, ax2, ay2 = coins(a)
    bx1, by1, bx2, by2 = coins(b)
    ix, iy = max(0.0, min(ax2, bx2) - max(ax1, bx1)), max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def compter(jeu: str) -> tuple[Counter, Counter, int]:
    """Instances par classe, images par classe, et nombre total d'images annotées."""
    noms = CLASSES[jeu]
    instances, images = Counter(), Counter()
    total = 0
    for _, boites in lire_labels(jeu):
        if not boites:
            continue
        total += 1
        presentes = set()
        for c, *_ in boites:
            if c < len(noms):
                instances[noms[c]] += 1
                presentes.add(noms[c])
        for n in presentes:
            images[n] += 1
    return instances, images, total


def entete(titre: str) -> None:
    print(f"\n{'=' * 74}\n{titre}\n{'=' * 74}")


def question_a() -> None:
    """`head` = tête nue, ou toute tête ?"""
    entete("A. Hard Hat Universe -- que designe `head` ?")
    noms = CLASSES["hard_hat_universe"]
    idx_head = noms.index("head")
    casques = {noms.index("helmet"), noms.index("hi-viz helmet")}

    mixtes = chevauchements = total = 0
    pire = 0.0
    for _, boites in lire_labels("hard_hat_universe"):
        tetes = [b[1:] for b in boites if b[0] == idx_head]
        casqs = [b[1:] for b in boites if b[0] in casques]
        if not boites:
            continue
        total += 1
        if not (tetes and casqs):
            continue
        mixtes += 1
        for t in tetes:
            for k in casqs:
                r = iou_xywh(t, k)
                pire = max(pire, r)
                if r >= 0.5:
                    chevauchements += 1
                    break

    print(f"images annotees            : {total}")
    print(f"images avec `head` ET casque : {mixtes}  ({100 * mixtes / max(total, 1):.1f} %)")
    print(f"boites `head` recouvrant un casque (IoU >= 0.5) : {chevauchements}")
    print(f"IoU maximal observe head/casque : {pire:.3f}")
    print()
    if chevauchements == 0:
        print("VERDICT : `head` et `helmet` ne designent jamais la meme boite.")
        print("          `head` = tete NUE -> le mappage head -> NO-Hardhat est SUR.")
    else:
        print("VERDICT : ATTENTION -- des boites `head` recouvrent des casques.")
        print("          Le mappage head -> NO-Hardhat empoisonnerait NO-Hardhat.")
        print("          Ecarter ce jeu, ou ne garder que les images sans casque.")


def question_b() -> None:
    """`Safety-shoes` et `safety_shoe` : doublon ?"""
    entete("B. Safety Shoes Detection -- `Safety-shoes` vs `safety_shoe`")
    instances, images, total = compter("safety_shoes_detection")
    print(f"images annotees : {total}\n")
    for nom in CLASSES["safety_shoes_detection"]:
        print(f"  {nom:<18} {instances[nom]:>5} instances  sur {images[nom]:>4} images")
    print()
    vides = [n for n in CLASSES["safety_shoes_detection"] if instances[n] == 0]
    if vides:
        print(f"VERDICT : classe(s) vide(s) -> a ignorer : {', '.join(vides)}")
    else:
        print("VERDICT : les trois classes sont peuplees -- decision manuelle requise")
        print("          (fusionner `Safety-shoes` dans `safety_shoe`, ou l'ignorer).")


def question_c() -> None:
    """Combien d'images de PPEs annotent une chaussure ?"""
    entete("C. PPEs -- volume reel par concept (split train)")
    instances, images, total = compter("ppes")
    print(f"images annotees : {total}\n")
    print(f"  {'classe':<12} {'instances':>10} {'images':>8}")
    for nom in CLASSES["ppes"]:
        print(f"  {nom:<12} {instances[nom]:>10} {images[nom]:>8}")

    chaussures = images["shoes"] + images["no_shoes"]
    print(f"\nimages annotant une chaussure (shoes ou no_shoes) : {chaussures}")
    if chaussures >= 1500:
        print("VERDICT : volume suffisant -> un modele chaussures dedie est viable.")
    else:
        print("VERDICT : volume INSUFFISANT (< 1500) -> le modele chaussures ne vaut")
        print("          pas la session GPU ; reduire la campagne au casque.")


def inventaire() -> None:
    entete("Inventaire complet (toutes sources, split train)")
    for jeu in CLASSES:
        instances, images, total = compter(jeu)
        print(f"\n{jeu}  --  {total} images annotees")
        for nom in CLASSES[jeu]:
            print(f"    {nom:<18} {instances[nom]:>8} instances  sur {images[nom]:>6} images")


def main() -> int:
    if not SOURCES.is_dir():
        print(f"introuvable : {SOURCES}", file=sys.stderr)
        return 1
    question_a()
    question_b()
    question_c()
    inventaire()
    print("\nAudit termine. Les trois verdicts ci-dessus decident des tables de")
    print("correspondance (v8_campagne_epi.md §4).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
