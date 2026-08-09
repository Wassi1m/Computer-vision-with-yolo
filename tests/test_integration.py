#!/usr/bin/env python3
"""Test d'intégration : le moteur complet, de la vidéo aux évènements émis.

Les tests unitaires (`test_logique_metier.py`) valident chaque brique isolément,
et le garde-fou (`test_non_regression.py`) valide les modèles. Reste ce qui
n'est vérifié par ni l'un ni l'autre : que l'assemblage fonctionne. C'est
précisément là que se logeaient les deux défauts trouvés en construisant ce
pipeline — des coordonnées numpy non sérialisables en JSON, et un chemin de ROI
relatif qui désactivait silencieusement le module porte selon le répertoire de
lancement. Aucun test unitaire ne les aurait vus ; une exécution réelle, si.

Le moteur est lancé comme en production — en sous-processus, headless, avec ses
vraies sorties — puis on vérifie ce qui en sort.

    python -m pytest tests/test_integration.py -q
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VIDEO = ROOT / "tests/donnees/sequence_test.avi"


def _construire_video_si_absente() -> bool:
    """Fabrique la séquence de test à partir d'images du jeu de validation.

    Elle est reconstruite plutôt que versionnée : le dépôt n'embarque pas les
    jeux de données (trop volumineux), et une vidéo binaire de plusieurs Mo dans
    git vieillirait mal. Le test s'ignore proprement si les images sources ne
    sont pas disponibles.
    """
    if VIDEO.exists():
        return True
    src = ROOT / "ppe_detection/data/extracted/ppe_vest_clean_14c/val/images"
    if not src.exists():
        return False
    import cv2
    images = sorted(src.glob("*.jpg"))[:12]
    if not images:
        return False
    VIDEO.parent.mkdir(parents=True, exist_ok=True)
    premier = cv2.imread(str(images[0]))
    h, w = premier.shape[:2]
    sortie = cv2.VideoWriter(str(VIDEO), cv2.VideoWriter_fourcc(*"MJPG"), 5, (w, h))
    for chemin in images:
        sortie.write(cv2.resize(cv2.imread(str(chemin)), (w, h)))
    sortie.release()
    return True


besoin_video = pytest.mark.skipif(
    not _construire_video_si_absente(),
    reason="jeu de donnees absent : impossible de construire la sequence de test")


def _lancer(args: list[str], attendre: bool = True, timeout: int = 300):
    cmd = [sys.executable, "unified_surveillance.py", *args]
    return subprocess.run(cmd, cwd=ROOT / "improvements", capture_output=True,
                          text=True, timeout=timeout) if attendre else None


@besoin_video
def test_le_moteur_traite_une_video_et_produit_des_evenements(tmp_path):
    """Scénario nominal : la chaîne complète tourne et émet des évènements valides."""
    journal = tmp_path / "evenements.jsonl"
    r = _lancer(["--source", str(VIDEO), "--max-frames", "6",
                 "--events", str(journal), "--log-level", "WARNING"])
    assert r.returncode == 0, f"le moteur a echoue :\n{r.stdout}\n{r.stderr}"
    assert journal.exists(), "aucun fichier d'evenements produit"

    evenements = [json.loads(l) for l in journal.read_text().splitlines() if l.strip()]
    assert evenements, "aucun evenement produit sur une video contenant des personnes"

    # Le contrat d'interface (docs/contrat_api.md) impose ces champs : un
    # consommateur qui les attend ne doit jamais recevoir un objet incomplet.
    for ev in evenements:
        for champ in ("t", "frame", "source", "type", "libelle", "conf", "box", "extra"):
            assert champ in ev, f"champ '{champ}' absent de l'evenement {ev}"
        assert isinstance(ev["t"], float)
        assert isinstance(ev["source"], str) and ev["source"]
        assert ev["box"] is None or (isinstance(ev["box"], list) and len(ev["box"]) == 4)


@besoin_video
def test_les_evenements_sont_serialisables_en_json(tmp_path):
    """Régression : les coordonnées numpy (int64) faisaient échouer json.dumps.

    Le défaut ne se voyait pas dans la sortie console — seule l'écriture JSON
    échouait, donc seul un consommateur réel l'aurait constaté.
    """
    journal = tmp_path / "ev.jsonl"
    r = _lancer(["--source", str(VIDEO), "--max-frames", "6",
                 "--events", str(journal), "--log-level", "WARNING"])
    assert "not JSON serializable" not in (r.stdout + r.stderr)
    for ligne in journal.read_text().splitlines():
        if ligne.strip():
            json.loads(ligne)  # leve si la ligne est invalide


@besoin_video
def test_tous_les_modules_disponibles_demarrent(tmp_path):
    """Régression : le module porte échouait silencieusement sur un chemin relatif.

    Un module manquant ne provoque aucune erreur — c'est voulu — donc seule une
    vérification explicite du démarrage peut détecter une régression de ce type.
    """
    r = _lancer(["--source", str(VIDEO), "--max-frames", "3", "--log-level", "INFO"])
    sortie = r.stdout + r.stderr
    for module in ("general", "epi", "chute", "porte"):
        assert f"module {module}" in sortie, f"module {module} absent des logs de demarrage"
        assert f"module {module:<10} indisponible" not in sortie, \
            f"module {module} n'a pas demarre :\n{sortie}"


@besoin_video
def test_arret_propre_sur_signal():
    """Le moteur doit s'arrêter sur SIGTERM : sinon `docker stop` le tue de force."""
    import signal
    proc = subprocess.Popen(
        [sys.executable, "unified_surveillance.py", "--source", str(VIDEO), "--boucler",
         "--log-level", "WARNING"],
        cwd=ROOT / "improvements", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    time.sleep(25)  # laisser les modeles se charger et la boucle demarrer
    assert proc.poll() is None, "le moteur s'est arrete seul avant le signal"
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=45)
    except subprocess.TimeoutExpired:
        proc.kill()
        pytest.fail("le moteur n'a pas repondu a SIGTERM : arret force necessaire")
    assert proc.returncode == 0, f"code de retour inattendu : {proc.returncode}"


@besoin_video
def test_point_de_sante_repond():
    """`/health` est le seul moyen pour un exploitant de savoir si le moteur va bien."""
    proc = subprocess.Popen(
        [sys.executable, "unified_surveillance.py", "--source", str(VIDEO), "--boucler",
         "--health-port", "8893", "--log-level", "WARNING"],
        cwd=ROOT / "improvements", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        etat = None
        for _ in range(40):  # le chargement des modeles prend une dizaine de secondes
            time.sleep(2)
            try:
                with urllib.request.urlopen("http://127.0.0.1:8893/health", timeout=3) as r:
                    etat = json.loads(r.read())
                if etat.get("frames", 0) > 0:
                    break
            except Exception:
                continue
        assert etat is not None, "/health n'a jamais repondu"
        assert etat["flux_connecte"] is True
        assert etat["frames"] > 0, "le moteur ne traite aucune image"
        assert etat["sain"] is True
        assert "modeles" in etat and etat["modeles"]
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()


@besoin_video
def test_mode_retention_presence_ne_publie_aucune_plaque(tmp_path):
    """En mode `presence`, aucun numéro ne doit quitter le moteur."""
    journal = tmp_path / "ev.jsonl"
    _lancer(["--source", str(VIDEO), "--max-frames", "6", "--events", str(journal),
             "--lpr-retention", "presence", "--log-level", "WARNING"])
    if not journal.exists():
        pytest.skip("aucun evenement produit")
    for ligne in journal.read_text().splitlines():
        if not ligne.strip():
            continue
        ev = json.loads(ligne)
        if ev["source"] == "lpr":
            assert "plaque" not in ev["extra"], "un numero de plaque a fuite en mode presence"
            assert ev["extra"].get("retention") == "presence"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
