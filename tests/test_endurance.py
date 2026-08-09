#!/usr/bin/env python3
"""Test d'endurance : le moteur tient-il des heures sans se dégrader ?

Un moteur livré à un client tourne des semaines sans supervision. Le nôtre
n'avait jamais tourné plus de quelques secondes : rien ne disait s'il fuit en
mémoire, si son débit s'effondre, ou si ses structures internes grossissent
indéfiniment. Ces défauts-là ne se voient qu'en durée — jamais sur un test
court, jamais en lecture de code.

Le script fait tourner le pipeline complet sur une vidéo rejouée en boucle,
échantillonne périodiquement mémoire et débit, puis conclut sur deux critères :

- **Fuite mémoire** : croissance de la mémoire résidente entre le début (une
  fois les modèles chargés et le régime établi) et la fin.
- **Dérive du débit** : baisse du FPS entre le premier et le dernier quart.

    python tests/test_endurance.py --minutes 30
    python tests/test_endurance.py --minutes 240 --source ma_video.mp4
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SORTIE = ROOT / "reports/v3_results"

# Tolerances : au-dela, on considere qu'il y a un probleme a corriger avant
# livraison. Volontairement larges — l'objectif est de detecter une derive
# franche, pas de traquer quelques Mo de variation d'allocateur.
CROISSANCE_MEMOIRE_MAX_MO = 300
BAISSE_FPS_MAX_PCT = 25
# Le regime n'est etabli qu'apres chargement des modeles et remplissage des
# historiques de lissage : les premieres mesures ne sont pas representatives.
ECHAUFFEMENT_S = 90


def memoire_mo(pid: int) -> float | None:
    """Mémoire résidente du processus, en Mo (lue dans /proc, sans dépendance)."""
    try:
        for ligne in Path(f"/proc/{pid}/status").read_text().splitlines():
            if ligne.startswith("VmRSS:"):
                return int(re.search(r"(\d+)", ligne).group(1)) / 1024
    except (FileNotFoundError, ProcessLookupError, AttributeError):
        return None
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=30)
    ap.add_argument("--source", default=None, help="video a rejouer en boucle")
    ap.add_argument("--intervalle", type=float, default=20, help="secondes entre deux mesures")
    args = ap.parse_args()

    source = args.source
    if source is None:
        # A defaut de video fournie, on en fabrique une a partir d'images du jeu
        # de validation : le test doit pouvoir tourner sans materiel ni donnee
        # supplementaire.
        import cv2
        import glob
        imgs = sorted(glob.glob(str(ROOT / "ppe_detection/data/extracted/ppe_vest_clean_14c/val/images/*.jpg")))[:40]
        if not imgs:
            print("Aucune source video disponible : fournir --source")
            return 2
        source = "/tmp/endurance_source.avi"
        premier = cv2.imread(imgs[0])
        h, w = premier.shape[:2]
        out = cv2.VideoWriter(source, cv2.VideoWriter_fourcc(*"MJPG"), 10, (w, h))
        for p in imgs:
            out.write(cv2.resize(cv2.imread(p), (w, h)))
        out.release()
        print(f"Source de test generee : {source} ({len(imgs)} images)")

    SORTIE.mkdir(parents=True, exist_ok=True)
    journal = SORTIE / "endurance.log"

    cmd = [sys.executable, "unified_surveillance.py", "--source", source, "--boucler",
           "--health-port", "8891", "--log-level", "WARNING"]
    print(f"Lancement : {' '.join(cmd)}")
    with open(journal, "w") as f:
        proc = subprocess.Popen(cmd, cwd=ROOT / "improvements", stdout=f, stderr=subprocess.STDOUT)

    import urllib.request

    mesures = []
    t0 = time.time()
    duree = args.minutes * 60
    try:
        while time.time() - t0 < duree:
            time.sleep(args.intervalle)
            if proc.poll() is not None:
                print(f"\nECHEC : le moteur s'est arrete seul (code {proc.returncode}) "
                      f"apres {time.time() - t0:.0f}s. Voir {journal}")
                return 1
            mem = memoire_mo(proc.pid)
            try:
                with urllib.request.urlopen("http://127.0.0.1:8891/health", timeout=5) as r:
                    sante = json.loads(r.read())
            except Exception as e:
                print(f"  /health injoignable : {e}")
                continue
            m = {"t": round(time.time() - t0, 1), "memoire_mo": round(mem or 0, 1),
                 "fps": sante["fps"], "frames": sante["frames"],
                 "evenements": sante["evenements"], "sain": sante["sain"]}
            mesures.append(m)
            print(f"  {m['t']:7.0f}s  memoire {m['memoire_mo']:7.1f} Mo  "
                  f"fps {m['fps']:5.2f}  frames {m['frames']:6d}  ev {m['evenements']}")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            print("AVERTISSEMENT : arret propre impossible, processus tue de force")

    utiles = [m for m in mesures if m["t"] >= ECHAUFFEMENT_S]
    if len(utiles) < 4:
        print("\nPas assez de mesures apres echauffement : allonger --minutes.")
        return 2

    croissance = utiles[-1]["memoire_mo"] - utiles[0]["memoire_mo"]
    quart = max(len(utiles) // 4, 1)
    fps_debut = sum(m["fps"] for m in utiles[:quart]) / quart
    fps_fin = sum(m["fps"] for m in utiles[-quart:]) / quart
    baisse_pct = 100 * (fps_debut - fps_fin) / fps_debut if fps_debut else 0

    resume = {
        "duree_s": round(time.time() - t0),
        "mesures": mesures,
        "memoire_debut_mo": utiles[0]["memoire_mo"],
        "memoire_fin_mo": utiles[-1]["memoire_mo"],
        "croissance_memoire_mo": round(croissance, 1),
        "fps_premier_quart": round(fps_debut, 2),
        "fps_dernier_quart": round(fps_fin, 2),
        "baisse_fps_pct": round(baisse_pct, 1),
        "frames_totales": utiles[-1]["frames"],
    }
    (SORTIE / "endurance.json").write_text(json.dumps(resume, indent=2, ensure_ascii=False) + "\n")

    print("\n" + "=" * 60)
    print(f"Duree            : {resume['duree_s']}s, {resume['frames_totales']} images analysees")
    print(f"Memoire          : {resume['memoire_debut_mo']:.1f} -> {resume['memoire_fin_mo']:.1f} Mo "
          f"({croissance:+.1f} Mo)")
    print(f"Debit            : {fps_debut:.2f} -> {fps_fin:.2f} FPS ({-baisse_pct:+.1f} %)")

    echecs = []
    if croissance > CROISSANCE_MEMOIRE_MAX_MO:
        echecs.append(f"fuite memoire probable : +{croissance:.0f} Mo "
                      f"(tolerance {CROISSANCE_MEMOIRE_MAX_MO} Mo)")
    if baisse_pct > BAISSE_FPS_MAX_PCT:
        echecs.append(f"derive du debit : -{baisse_pct:.1f} % "
                      f"(tolerance {BAISSE_FPS_MAX_PCT} %)")

    if echecs:
        print("\nPROBLEMES DETECTES :")
        for e in echecs:
            print(f"  - {e}")
        return 1
    print("\nAucune derive detectee sur la duree du test.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
