#!/usr/bin/env python3
"""P17 — Contrôle avant vol d'un jeu fusionné, sans consommer une seconde de GPU.

Pourquoi ce script existe
-------------------------
Deux campagnes `safety_shoe` ont été perdues (17 et 19 août 2026), chacune après
plusieurs heures de GPU, et **aucune des deux n'a échoué pour une raison
imprévisible**. Les trois causes étaient présentes dans les données avant même
le lancement, et se voyaient sans entraîner quoi que ce soit :

1. **Aucune opposition annotée.** Les sources étiquetaient « chaussure », jamais
   « chaussure ordinaire » contre « chaussure de sécurité ». Le modèle a donc
   appris à localiser des pieds chaussés, et donnait des baskets de ville pour
   des chaussures de sécurité à 0.86 de confiance.
2. **Une validation qui fuit.** Les exports Roboflow sont AUGMENTÉS : une même
   photo d'origine y apparaît en plusieurs variantes. Découpées au hasard, ces
   variantes se retrouvent des deux côtés, et la mAP mesure de la mémorisation.
   Le candidat du 19 affichait 0.912 de mAP tout en dessinant des boîtes sur des
   cônes de signalisation.
3. **Aucun jeu de test indépendant.** Le verdict reposait sur un proxy (taux de
   déclenchement sur un jeu qui n'annote pas le concept), faute d'avoir réservé
   des images avant d'entraîner.

Ce script mesure ces trois points. Il ne prédit pas la réussite -- rien ne le
peut -- mais il refuse de laisser partir un entraînement dont on sait déjà qu'il
ne peut pas apprendre ce qu'on lui demande.

    python improvements/p17_controle_avant_entrainement.py --jeu chaussures
    python improvements/p17_controle_avant_entrainement.py --jeu chaussures --images 8
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p13_jeux_roboflow import JEUX, collecter  # noqa: E402

RACINE = Path(__file__).resolve().parents[1]
SORTIE_IMG = RACINE / "reports/v3_results/controle_avant_vol"

# En dessous de ce nombre d'images, une classe ne peut pas être apprise de façon
# fiable -- seuil indicatif, tiré de ce qui a échoué : 380 images négatives
# n'avaient pas suffi à faire la différence avec une chaussure ordinaire.
PLANCHER_CLASSE = 800

# Part maximale d'une seule source dans une classe. Au-delà, la classe est
# apprise sur un domaine unique et le modèle ne généralise pas -- c'est ce qui
# avait fait rejeter le candidat gants du 17 août, entraîné sans images locales.
PART_MAX_SOURCE = 0.85


def famille(nom: str) -> str:
    """Photo d'origine dont ce fichier est une variante augmentée.

    Roboflow nomme ses exports `<origine>_jpg.rf.<empreinte>.jpg` : tout ce qui
    précède `.rf.` identifie la photo de départ. Deux variantes d'une même photo
    partagent donc cette racine, et ne doivent JAMAIS être séparées entre
    entraînement et validation.
    """
    return nom.split(".rf.")[0] if ".rf." in nom else nom


def analyser(cle: str) -> dict:
    spec = JEUX[cle]
    noms = spec["noms"]

    par_source: dict[str, Counter] = defaultdict(Counter)
    images_par_source: dict[str, Counter] = defaultdict(Counter)
    familles: dict[str, set] = defaultdict(set)
    echantillons: dict[tuple, list] = defaultdict(list)
    total_images = 0

    for src in spec["sources"]:
        jeu = src["jeu"]
        for split_src in src["splits"]:
            for img, boites in collecter(src, split_src):
                total_images += 1
                familles[jeu].add(famille(img.name))
                vues = set()
                for c, coords in boites:
                    par_source[jeu][c] += 1
                    vues.add(c)
                for c in vues:
                    images_par_source[jeu][c] += 1
                    echantillons[(jeu, c)].append((img, boites))

    return {"noms": noms, "par_source": par_source,
            "images_par_source": images_par_source, "familles": familles,
            "echantillons": echantillons, "total_images": total_images}


def dessiner(a: dict, n_par_case: int, rng: random.Random) -> list[dict]:
    """Écrit des images annotées, pour que la vérification passe par l'œil.

    Les compteurs disent qu'une classe existe ; ils ne disent pas ce qu'elle
    CONTIENT. C'est l'inspection visuelle qui a révélé les deux échecs
    précédents, jamais les chiffres.
    """
    try:
        import cv2
    except ImportError:
        print("  (opencv absent : pas d'echantillon visuel)")
        return []

    if SORTIE_IMG.exists():
        for f in SORTIE_IMG.glob("*.jpg"):
            f.unlink()
    SORTIE_IMG.mkdir(parents=True, exist_ok=True)

    ecrits = []
    for (jeu, c), lot in sorted(a["echantillons"].items()):
        nom_classe = a["noms"][c]
        for img, boites in rng.sample(lot, min(n_par_case, len(lot))):
            im = cv2.imread(str(img))
            if im is None:
                continue
            H, W = im.shape[:2]
            for cl, (x, y, w, h) in ((cl, tuple(float(v) for v in co))
                                     for cl, co in boites):
                x1, y1 = int((x - w / 2) * W), int((y - h / 2) * H)
                x2, y2 = int((x + w / 2) * W), int((y + h / 2) * H)
                # Vert = classe positive, rouge = classe negative : la
                # distinction qu'on vient precisement verifier.
                couleur = (0, 200, 0) if cl == 0 else (0, 0, 230)
                cv2.rectangle(im, (x1, y1), (x2, y2), couleur, 2)
                cv2.putText(im, a["noms"][cl], (x1, max(14, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, couleur, 1)
            cible = SORTIE_IMG / f"{jeu}__{nom_classe}__{img.stem[:28]}.jpg"
            cv2.imwrite(str(cible), im)
            ecrits.append({"jeu": jeu, "classe": nom_classe, "fichier": cible.name})
    return ecrits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jeu", choices=list(JEUX), default="chaussures")
    ap.add_argument("--images", type=int, default=6,
                    help="images annotees par (source, classe)")
    args = ap.parse_args()

    rng = random.Random(0)
    a = analyser(args.jeu)
    noms = a["noms"]
    alertes: list[str] = []

    print(f"\n=== 1. Ce que chaque source apporte, par classe ===")
    entete = f"{'source':<32}" + "".join(f"{n:>22}" for n in noms)
    print(entete)
    totaux_img = Counter()
    for jeu in sorted(a["images_par_source"]):
        ligne = f"{jeu:<32}"
        for c in range(len(noms)):
            ni, nb = a["images_par_source"][jeu][c], a["par_source"][jeu][c]
            ligne += f"{f'{ni} img / {nb} inst':>22}"
            totaux_img[c] += ni
        print(ligne)
    print(f"{'TOTAL':<32}" + "".join(f"{totaux_img[c]:>22}" for c in range(len(noms))))

    print(f"\n=== 2. Volume et diversite par classe ===")
    for c, nom in enumerate(noms):
        n = totaux_img[c]
        if n == 0:
            alertes.append(f"classe '{nom}' VIDE : le modele ne peut rien en apprendre")
            print(f"  {nom:<20} 0 image   <- BLOQUANT")
            continue
        parts = {j: a["images_par_source"][j][c] / n for j in a["images_par_source"]
                 if a["images_par_source"][j][c]}
        dominante, part = max(parts.items(), key=lambda kv: kv[1])
        etat = "ok"
        if n < PLANCHER_CLASSE:
            etat = f"FAIBLE (< {PLANCHER_CLASSE})"
            alertes.append(f"classe '{nom}' : {n} images seulement, sous le plancher "
                           f"de {PLANCHER_CLASSE}")
        if part > PART_MAX_SOURCE and len(parts) > 1:
            etat += f" | MONO-SOURCE ({dominante} = {100*part:.0f} %)"
            alertes.append(f"classe '{nom}' vient a {100*part:.0f} % de {dominante} : "
                           "un seul domaine, generalisation incertaine")
        elif len(parts) == 1:
            etat += f" | SOURCE UNIQUE ({dominante})"
            alertes.append(f"classe '{nom}' n'a qu'une seule source ({dominante})")
        print(f"  {nom:<20} {n:>6} images   sur {len(parts)} source(s)   {etat}")

    print(f"\n=== 3. Variantes augmentees (risque de fuite train/val) ===")
    total_fuite = 0
    for jeu, fam in sorted(a["familles"].items()):
        n_img = sum(a["images_par_source"][jeu].values())
        ratio = n_img / len(fam) if fam else 0
        marque = ""
        if ratio > 1.3:
            marque = f"  <- {ratio:.1f} variantes par photo"
            total_fuite += 1
        print(f"  {jeu:<32} {len(fam):>5} photos d'origine{marque}")
    if total_fuite:
        alertes.append(f"{total_fuite} source(s) augmentee(s) : le decoupage train/val "
                       "DOIT se faire par photo d'origine, pas par fichier, sinon la "
                       "mAP de validation mesurera de la memorisation")

    print(f"\n=== 4. Echantillon a regarder ===")
    ecrits = dessiner(a, args.images, rng)
    print(f"  {len(ecrits)} images annotees -> {SORTIE_IMG.relative_to(RACINE)}")
    print("  Vert = classe positive, rouge = classe negative.")
    print("  A VERIFIER A L'OEIL : les boites rouges sont-elles bien des chaussures")
    print("  ORDINAIRES, et les vertes de vraies chaussures de securite ?")
    print("  Aucun compteur ne repond a cette question.")

    print(f"\n=== VERDICT ===")
    if alertes:
        for m in alertes:
            print(f"  /!\\ {m}")
    else:
        print("  Aucune alerte structurelle.")
    print("\n  Ces controles eliminent les causes des echecs precedents ; ils ne")
    print("  garantissent pas la reussite, que seule la mesure sur un jeu de test")
    print("  independant peut etablir.")

    rapport = RACINE / f"reports/v3_results/controle_avant_vol_{args.jeu}.json"
    rapport.write_text(json.dumps({
        "jeu": args.jeu,
        "images_par_classe": {noms[c]: totaux_img[c] for c in range(len(noms))},
        "par_source": {j: {noms[c]: a["images_par_source"][j][c]
                           for c in range(len(noms))}
                       for j in a["images_par_source"]},
        "photos_origine_par_source": {j: len(f) for j, f in a["familles"].items()},
        "alertes": alertes,
        "echantillon": ecrits,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n-> {rapport.relative_to(RACINE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
