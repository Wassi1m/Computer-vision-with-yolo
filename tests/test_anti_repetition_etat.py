"""Anti-répétition et changements d'état de qualification (correctif du 2026-09-27).

Défaut relevé par le banc de scénarios (Q9) : la clé d'anti-répétition de
`BusEvenements` (`source|type|libellé`, 3 s) ne contenait pas l'état de
qualification. Un feu ou une chute vus en continu passent de `suspicion` à
`probable` puis `confirme` en moins de 3 s : ces deux changements étaient avalés
et seule `suspicion` était publiée, alors que l'ingestion exige `probable`
(chute) ou `confirme` (feu). Ces tests verrouillent le correctif, avec les
réglages du moteur (anti-répétition par défaut, `MachineEtats()` par défaut).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "improvements"))
sys.path.insert(0, str(ROOT / "surveillance_suite"))

from qualification import MachineEtats  # noqa: E402
from unified_surveillance import BusEvenements, Evenement  # noqa: E402


class _Memoire:
    def __init__(self):
        self.recus = []

    def emettre(self, ev):
        self.recus.append(ev)

    def fermer(self):
        pass


def _etats(source, type_, libelle, secondes=10.0, pas=0.08):
    sortie = _Memoire()
    bus = BusEvenements([sortie], qualification=MachineEtats())
    for k in range(int(secondes / pas)):
        bus.publier(Evenement(pas * k, 2 * k, source, type_, libelle, 0.9, (100, 300, 300, 380)))
    return [e.extra.get("etat") for e in sortie.recus]


def test_une_chute_continue_atteint_probable():
    assert "probable" in _etats("chute", "chute", "PERSONNE AU SOL")


def test_un_feu_continu_atteint_confirme():
    assert "confirme" in _etats("feu", "feu", "FIRE détecté")


def test_l_anti_repetition_filtre_toujours_les_repetitions_d_un_meme_etat():
    """Le correctif ne doit pas inonder la plateforme : un même état n'est
    republié qu'après l'anti-répétition, pas à chaque image."""
    etats = _etats("chute", "chute", "PERSONNE AU SOL", secondes=2.0)
    assert len(etats) <= 4, etats
