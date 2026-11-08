#!/usr/bin/env python3
"""P6 — Point d'entrée vidéo unique pour les deux chaînes de traitement.

Situation corrigée
------------------
Les deux applications d'origine — `ppe_dual_model_backup2.py`, retiré du dépôt
le 2026-08-20 mais présent dans l'historique git, et `surveillance_suite/main.py`
— ouvraient chacune leur propre `cv2.VideoCapture`. Sur une caméra de surveillance
réelle c'est impossible : un flux V4L2 n'accepte généralement qu'un seul
lecteur, et un flux RTSP ouvert deux fois double la bande passante, désynchronise
les deux analyses (elles ne voient pas la même image au même instant) et rend
tout horodatage d'évènement incohérent entre les deux systèmes.

Ce module capture la frame **une seule fois** et la distribue aux deux chaînes.

Ordonnancement
--------------
Tous les analyseurs n'ont pas besoin de tourner à chaque frame : une porte ou un
départ de feu évoluent en secondes, un franchissement de ligne en dizaines de
millisecondes. Chaque analyseur déclare donc son intervalle (`every`), et seuls
ceux dus sur la frame courante sont soumis à un pool de threads commun. Les
autres réaffichent leur dernier résultat. C'est ce qui permet de faire tenir six
modèles sur un CPU : le coût moyen par frame est la somme des coûts divisés par
les intervalles, pas la somme des coûts.

Le pool est partagé (et non un pool par chaîne) pour que le nombre de threads
d'inférence reste borné : au-delà du nombre de cœurs, PyTorch en CPU se met à
se battre contre lui-même et le débit chute.

Usage
-----
    python unified_surveillance.py --source 0
    python unified_surveillance.py --source video.mp4 --no-display --events events.jsonl
    python unified_surveillance.py --source rtsp://... --disable ppe_gloves,lpr
"""

from __future__ import annotations

import argparse
import base64
import hmac
import json
import logging
import os
import signal
import sys
import threading
import uuid
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict, field
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "surveillance_suite"
# La racine elle-meme, et pas seulement `surveillance_suite` : `AnalyseurVisage`
# importe `surveillance_suite.modules.module_visage` par son nom complet. Sans
# cette entree, l'import echouait des que le moteur etait lance depuis
# `improvements/` -- c'est-a-dire toujours, superviseur et conteneur compris --
# et la reconnaissance faciale etait declaree « indisponible » au demarrage
# (constate le 2026-09-23 : `No module named 'surveillance_suite'`).
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SUITE))
sys.path.insert(0, str(SUITE / "modules"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ppe_taxonomy as tax  # noqa: E402
import qualification as qual  # noqa: E402


# ── Journalisation ───────────────────────────────────────────────────────────
# Des `print` ne portent ni niveau, ni horodatage, ni origine : impossible de
# diagnostiquer un incident client à distance, ni de filtrer le bruit. Le format
# reste lisible en console ; `--log-json` produit une ligne JSON par message pour
# les collecteurs de logs.

log = logging.getLogger("moteur")


class FormatJSON(logging.Formatter):
    """Une ligne JSON par message, portant l'identite de la camera.

    Sans `camera_id`, les journaux de plusieurs instances agreges dans un meme
    collecteur sont indistinguables : « flux perdu » ne dit pas QUELLE camera
    est tombee. Le bus le fait deja pour les evenements ; les logs, non.
    """

    def __init__(self, camera_id: str = "", site_id: str = ""):
        super().__init__()
        self.contexte = {}
        if camera_id:
            self.contexte["camera_id"] = camera_id
        if site_id:
            self.contexte["site_id"] = site_id

    def format(self, record):
        return json.dumps({
            "t": record.created,
            "niveau": record.levelname,
            "module": record.name,
            "message": record.getMessage(),
            **self.contexte,
        }, ensure_ascii=False)


def configurer_logs(niveau: str, json_mode: bool, camera_id: str = "", site_id: str = ""):
    h = logging.StreamHandler()
    if json_mode:
        h.setFormatter(FormatJSON(camera_id, site_id))
    else:
        # Le `%` double est indispensable : le format de `logging` est en
        # style %, et un identifiant de camera contenant un `%` ferait echouer
        # le formatage de CHAQUE message -- le moteur deviendrait muet.
        prefixe = f"[{camera_id.replace('%', '%%')}] " if camera_id else ""
        h.setFormatter(logging.Formatter(
            f"%(asctime)s %(levelname)-7s {prefixe}%(name)s: %(message)s",
            datefmt="%H:%M:%S"))
    racine = logging.getLogger()
    racine.handlers[:] = [h]
    racine.setLevel(getattr(logging, niveau.upper(), logging.INFO))


# ── Évènements ───────────────────────────────────────────────────────────────

@dataclass
class Evenement:
    """Un fait daté produit par un analyseur, indépendant de l'affichage."""
    t: float
    frame: int
    source: str          # nom de l'analyseur
    type: str            # 'violation_epi', 'chute', 'franchissement', 'feu', ...
    libelle: str
    conf: float = 0.0
    box: tuple[int, int, int, int] | None = None
    extra: dict = field(default_factory=dict)

    # ── Identité ─────────────────────────────────────────────────────────────
    #
    # `camera_id` est rempli par le bus au moment de la publication, et non par
    # les analyseurs : ceux-ci ignorent tout de la source, et le leur faire
    # porter obligerait a le passer a chacun.
    #
    # Il vient toujours de la CONFIGURATION, jamais d'une deduction depuis
    # l'URL du flux : deux sites peuvent exposer la meme adresse locale, et un
    # identifiant qui change parce qu'on a modifie un port serait pire que pas
    # d'identifiant du tout. Un site reel compte dix a cent cameras ; sans ce
    # champ la plateforme recoit des evenements sans savoir d'ou ils viennent.
    camera_id: str = ""
    site_id: str = ""

    # Identite propre de l'evenement, pour que la plateforme puisse dedupliquer
    # apres un rejeu reseau et relier les mises a jour d'un meme fait. Sans lui,
    # un webhook reemis produit un doublon indiscernable d'un vrai second
    # evenement.
    event_id: str = ""


def _json_defaut(obj):
    """Convertit les types numpy (int64 des coordonnées de boîte, notamment)
    en types Python natifs -- `json.dumps` ne sait pas les sérialiser tels quels."""
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"Non serialisable: {type(obj)}")


# ── Sorties (transport des évènements) ───────────────────────────────────────
#
# Le protocole que la plateforme consommatrice utilisera n'est pas encore arrêté
# (REST + webhook, websocket, ou file de messages). Plutôt que d'attendre cette
# décision, le transport est isolé derrière une interface minimale : le moteur
# publie des `Evenement`, chaque `Sortie` décide quoi en faire. Changer de
# protocole reviendra à ajouter une classe ici, sans toucher au moteur.

class Sortie:
    """Destination d'un flux d'évènements. Ne doit jamais lever : une panne de
    transport ne doit pas arrêter la détection."""

    def emettre(self, ev: "Evenement"):
        raise NotImplementedError

    def fermer(self):
        pass


class SortieConsole(Sortie):
    def emettre(self, ev):
        print(f"[{ev.type.upper()}] {ev.libelle}", flush=True)


class SortieJSONL(Sortie):
    """Fichier JSONL — utile en test, et comme trace locale de secours."""

    def __init__(self, chemin: Path):
        self.chemin = chemin
        self._f = open(chemin, "a", encoding="utf-8")

    def emettre(self, ev):
        self._f.write(json.dumps(asdict(ev), ensure_ascii=False, default=_json_defaut) + "\n")
        self._f.flush()

    def fermer(self):
        self._f.close()


class SortieWebhook(Sortie):
    """POST HTTP par évènement, avec **livraison garantie**.

    L'implémentation précédente envoyait « au mieux » : sur exception, elle
    incrémentait un compteur et l'évènement était perdu. Le raisonnement était
    juste — ne jamais ralentir la détection pour un consommateur indisponible —
    mais la conséquence ne l'était pas : **un départ de feu détecté pendant un
    redémarrage de la plateforme disparaissait sans trace**, et personne, ni
    côté moteur ni côté plateforme, ne pouvait savoir qu'il avait existé.

    Ici, l'évènement est d'abord **écrit sur disque**, puis envoyé par un fil
    dédié qui n'avance sa position de lecture qu'après un accusé du serveur.
    Trois propriétés en découlent :

    - une coupure réseau ne perd rien : les évènements s'accumulent et partent
      au rétablissement ;
    - un arrêt brutal du moteur ne perd rien non plus, la position étant elle
      aussi sur disque : au redémarrage, l'envoi reprend où il s'était arrêté ;
    - l'abandon reste possible après `tentatives_max`, mais il est **journalisé
      et compté**, jamais silencieux.

    Le prix est une livraison « au moins une fois » : un accusé perdu en chemin
    fait renvoyer l'évènement. C'est le bon compromis pour de la sécurité, et
    c'est précisément à cela que sert `event_id` — la plateforme déduplique.
    """

    def __init__(self, url: str, timeout_s: float = 3.0, workers: int = 2,
                 journal: Path | None = None, tentatives_max: int = 8,
                 attente_max_s: float = 60.0, jeton: str = "",
                 compacter_au_dela: int = 64 * 1024 * 1024):
        self.url = url
        self.timeout_s = timeout_s
        self.tentatives_max = tentatives_max
        self.attente_max_s = attente_max_s
        self.jeton = jeton
        # Seuil de compaction. 64 Mo represente de l'ordre de 150 000 evenements
        # deja livres : assez rare pour ne pas compacter sans arret, assez bas
        # pour qu'un disque ne se remplisse jamais.
        self.compacter_au_dela = compacter_au_dela
        self.journal = Path(journal) if journal else Path("evenements_a_livrer.jsonl")
        self.position = self.journal.with_suffix(".pos")
        self.journal.parent.mkdir(parents=True, exist_ok=True)

        self.echecs = 0          # tentatives ratees (cumul)
        self.abandons = 0        # evenements definitivement abandonnes
        self.livres = 0
        self._verrou = threading.Lock()
        self._reveil = threading.Event()
        self._stop = threading.Event()
        self._f = open(self.journal, "a", encoding="utf-8")
        self._fil = threading.Thread(target=self._boucle, name="webhook", daemon=True)
        self._fil.start()

    # ── Ecriture (appelee par le bus, doit rester instantanee) ───────────────

    def emettre(self, ev):
        ligne = json.dumps(asdict(ev), ensure_ascii=False, default=_json_defaut)
        with self._verrou:
            self._f.write(ligne + "\n")
            self._f.flush()
            os.fsync(self._f.fileno())   # sans cela, un arret brutal perd le tampon
        self._reveil.set()

    # ── Livraison (fil dedie) ────────────────────────────────────────────────

    def _lire_position(self) -> int:
        try:
            pos = int(self.position.read_text().strip())
        except Exception:
            return 0
        try:
            taille = self.journal.stat().st_size
        except OSError:
            return pos
        if pos > taille:
            # Position au-dela de la fin du journal : signature d'une compaction
            # interrompue -- le fichier a ete tronque, la position pas encore
            # remise a zero. Reprendre au debut est sur : la compaction ne
            # s'execute qu'a journal integralement livre, donc rien de ce qui
            # precedait n'attend d'etre envoye. Sans ce rattrapage, la lecture
            # reprend a un decalage qui ne correspond plus a rien : les
            # evenements ecrits depuis sont sautes en silence, et le premier lu
            # l'est au milieu d'une ligne (verifie : 10 evenements perdus sur 40
            # et une charge illisible livree a la plateforme).
            log.warning("position de livraison (%d) au-dela de la fin du journal "
                        "(%d octets) : compaction interrompue, reprise au debut",
                        pos, taille)
            self._ecrire_position(0)
            return 0
        return pos

    def _ecrire_position(self, octets: int):
        # Ecriture atomique : un arret entre l'ouverture et l'ecriture laisserait
        # sinon un fichier de position vide, donc un rejeu de tout le journal.
        tmp = self.position.with_suffix(".pos.tmp")
        tmp.write_text(str(octets))
        tmp.replace(self.position)

    def _poster(self, charge: bytes) -> bool:
        import urllib.request
        entetes = {"Content-Type": "application/json"}
        if self.jeton:
            # Sans authentification, quiconque connait l'URL peut injecter de
            # faux evenements dans la plateforme (plan v6 §3.2).
            entetes["Authorization"] = f"Bearer {self.jeton}"
        try:
            req = urllib.request.Request(self.url, data=charge, headers=entetes)
            urllib.request.urlopen(req, timeout=self.timeout_s).close()
            return True
        except Exception as e:
            self.echecs += 1
            log.debug("webhook en echec (%s) : %s", type(e).__name__, e)
            return False

    def _compacter(self, pos: int) -> int:
        """Réécrit le journal sans la partie déjà livrée. Rend la nouvelle position.

        Sans cela le journal croît aussi longtemps que le moteur tourne : sur une
        installation 24/7, la saturation du disque est certaine, seule sa date
        est inconnue. C'était le point bloquant B5 de l'audit du 2026-08-20.

        La compaction ne s'exécute que si TOUT est livré (`pos` en fin de
        fichier). Réécrire un reliquat non livré demanderait de suspendre les
        écritures entrantes ; attendre le prochain moment de calme est plus
        simple et arrive de toute façon sur n'importe quel site réel.

        L'ordre des opérations est ce qui rend l'opération sûre en cas d'arrêt
        brutal : on tronque le journal AVANT de remettre la position à zéro.
        L'ordre inverse rejouerait tout le journal. Un arrêt entre les deux
        laisse une position au-delà de la fin du fichier, que `_lire_position`
        rattrape en reprenant à zéro.
        """
        with self._verrou:
            if self.journal.stat().st_size != pos:
                return pos                    # écrit entre-temps : on repassera
            self._f.truncate(0)
            self._f.seek(0)
            self._f.flush()
            os.fsync(self._f.fileno())
        self._ecrire_position(0)
        log.info("journal de livraison compacte (%d octets liberes, %d evenements livres)",
                 pos, self.livres)
        return 0

    def _boucle(self):
        while not self._stop.is_set():
            pos = self._lire_position()
            envoye = False
            try:
                with open(self.journal, "rb") as f:
                    f.seek(pos)
                    ligne = f.readline()
                    if ligne and ligne.endswith(b"\n"):
                        if self._livrer(ligne.strip()):
                            pos += len(ligne)
                            self._ecrire_position(pos)
                            self.livres += 1
                        envoye = True
            except FileNotFoundError:
                pass
            if not envoye:
                # Rien a livrer : c'est le moment de compacter si le journal a
                # depasse le seuil, puis de dormir jusqu'au prochain evenement.
                if pos >= self.compacter_au_dela:
                    try:
                        self._compacter(pos)
                    except Exception as e:
                        log.error("compaction du journal impossible : %s: %s",
                                  type(e).__name__, e)
                self._reveil.wait(timeout=1.0)
                self._reveil.clear()

    def _livrer(self, charge: bytes) -> bool:
        """Tente la livraison avec temporisation croissante. Renvoie True quand
        la ligne peut etre consideree comme traitee -- livree, ou abandonnee."""
        attente = 1.0
        for tentative in range(1, self.tentatives_max + 1):
            if self._stop.is_set():
                return False
            if self._poster(charge):
                return True
            if tentative < self.tentatives_max:
                # Temporisation croissante : marteler un serveur qui redemarre
                # ne fait que retarder son retablissement.
                self._stop.wait(timeout=attente)
                attente = min(attente * 2, self.attente_max_s)
        self.abandons += 1
        log.error("evenement abandonne apres %d tentatives -- conserve dans %s",
                  self.tentatives_max, self.journal)
        return True

    @property
    def en_attente(self) -> int:
        """Octets de journal pas encore livrés — indicateur à exposer sur /health."""
        try:
            return max(0, self.journal.stat().st_size - self._lire_position())
        except Exception:
            return 0

    def fermer(self):
        """Laisse une chance de vider la file, sans bloquer l'arrêt indéfiniment.

        Ce qui reste dans le journal n'est pas perdu : la position sur disque
        fera reprendre la livraison au prochain démarrage.
        """
        self._reveil.set()
        self._fil.join(timeout=5.0)
        self._stop.set()
        self._reveil.set()
        with self._verrou:
            self._f.close()
        if self.en_attente:
            log.warning("%d octets d'evenements restent a livrer dans %s "
                        "-- ils partiront au prochain demarrage",
                        self.en_attente, self.journal)


class BusEvenements:
    """Collecte les évènements et les distribue aux sorties configurées.

    Séparer la production d'évènements de leur rendu est ce qui rend le pipeline
    utilisable sans écran (serveur, conteneur) : le mode d'affichage ne change
    rien à ce qui est détecté, seulement à ce qui est dessiné.
    """

    # Evenements techniques : ils decrivent l'etat du moteur, pas un fait
    # observe dans la scene. Les faire passer par la machine a etats les
    # retarderait, alors qu'une camera tombee doit se signaler immediatement.
    TYPES_TECHNIQUES = ("flux_perdu", "flux_repris")

    def __init__(self, sorties: list[Sortie] | None = None, anti_repetition_s: float = 3.0,
                 camera_id: str = "", site_id: str = "", qualification=None):
        self.sorties = sorties or [SortieConsole()]
        self._dernier: dict[str, float] = {}
        self._anti_rep = anti_repetition_s
        self.camera_id, self.site_id = camera_id, site_id
        # Machine a etats optionnelle : branchee ici plutot que dans chaque
        # analyseur, pour que tous en beneficient sans etre modifies, et pour
        # qu'elle reste debrayable d'un seul drapeau (--sans-qualification).
        self.qualification = qualification
        self.total = 0
        self.dernier_t = 0.0

    def publier(self, ev: Evenement) -> bool:
        if self.qualification is not None and ev.type not in self.TYPES_TECHNIQUES:
            return self._publier_qualifie(ev)
        return self._publier_brut(ev)

    def _publier_qualifie(self, ev: Evenement) -> bool:
        """Un evenement par changement d'etat, au lieu d'un par image."""
        q = self.qualification
        cle = q.cle_de(ev.source, ev.type, ev.box, self.camera_id)
        piste, nouvel_etat = q.observer(cle, ev.t, ev.conf)
        publie = False
        if nouvel_etat is not None:
            ev.extra = {**ev.extra, **q.preuves(piste, ev.t)}
            publie = self._publier_brut(ev)
        # Les pistes que plus rien n'alimente se terminent : sans ce signal, la
        # plateforme garderait une alerte ouverte indefiniment.
        for finie in q.expirer(ev.t):
            fin = Evenement(ev.t, ev.frame, finie.cle.split("|")[1], f"{ev.type}_termine",
                            f"Fin : {finie.etat} pendant "
                            f"{finie.derniere_vue - finie.debut:.0f} s",
                            extra=q.preuves(finie, finie.derniere_vue))
            self._publier_brut(fin)
        return publie

    def _publier_brut(self, ev: Evenement) -> bool:
        # L'etat de qualification fait partie de la cle : sans lui, le passage
        # suspicion -> probable -> confirme, qui survient en moins de 3 s sur un
        # evenement continu, etait avale par l'anti-repetition et l'alerte
        # n'atteignait jamais l'etat exige par la plateforme (banc, Q9 ; applique
        # le 2026-09-27 a la demande de l'utilisateur). Les evenements bruts
        # n'ont pas d'etat : leur cle ne change pas.
        cle = f"{ev.source}|{ev.type}|{ev.libelle}|{ev.extra.get('etat', '')}"
        if ev.t - self._dernier.get(cle, -1e9) < self._anti_rep:
            return False
        self._dernier[cle] = ev.t
        # L'identite est apposee ici, une seule fois, plutot que dans chaque
        # analyseur : elle ne depend pas de ce qui a ete detecte.
        if not ev.camera_id:
            ev.camera_id = self.camera_id
        if not ev.site_id:
            ev.site_id = self.site_id
        if not ev.event_id:
            ev.event_id = uuid.uuid4().hex
        self.total += 1
        self.dernier_t = ev.t
        for s in self.sorties:
            try:
                s.emettre(ev)
            except Exception as e:
                # Une sortie défaillante ne doit jamais interrompre la détection.
                        log.error("sortie %s en echec: %s: %s", type(s).__name__, type(e).__name__, e)
        return True

    def fermer(self):
        for s in self.sorties:
            s.fermer()


# ── Santé du moteur ──────────────────────────────────────────────────────────

class EtatSante:
    """État courant du moteur, exposé en HTTP sur `/health`.

    Sans cela, un exploitant n'a aucun moyen de savoir à distance si le moteur
    tourne encore, s'il voit toujours la caméra, et à quelle cadence. Le simple
    fait que le processus existe ne prouve rien : il peut tourner en boucle de
    reconnexion sans analyser une seule image.

    Un serveur HTTP minimal de la bibliothèque standard suffit ici — pas de
    dépendance supplémentaire, et cela reste vrai quel que soit le protocole
    retenu plus tard pour le transport des évènements.
    """

    def __init__(self, analyseurs_noms: list[str]):
        self._verrou = threading.Lock()
        self._etat = {
            "demarre_le": time.time(),
            "modeles": analyseurs_noms,
            "flux_connecte": True,
            "reconnexions": 0,
            "retard_flux_s": 0.0,
            "images_ignorees": 0,
            "frames": 0,
            "fps": 0.0,
            "evenements": 0,
            "derniere_detection": None,
        }
        self._serveur = None

    def update(self, frames, fps, bus, cap):
        with self._verrou:
            self._etat.update(
                frames=frames, fps=round(fps, 2), evenements=bus.total,
                derniere_detection=bus.dernier_t or None,
                flux_connecte=cap.connecte, reconnexions=cap.reconnexions,
                # Ecart entre la capture d'une image et le moment ou l'analyse la
                # prend. Les images ignorees disent a quel point l'analyse est
                # distancee par la camera : les deux ensemble repondent a « ce
                # que je vois est-il recent ? », question qu'aucun champ ne
                # permettait de trancher jusqu'ici.
                retard_flux_s=round(cap.retard_s, 3),
                images_ignorees=cap.images_ignorees,
                camera_id=bus.camera_id or None, site_id=bus.site_id or None,
            )
            # Etat de la livraison : ces compteurs existaient sans jamais etre
            # publies, alors qu'ils sont le seul moyen pour un exploitant de
            # voir que la plateforme aval ne recoit plus rien.
            for s in bus.sorties:
                if isinstance(s, SortieWebhook):
                    self._etat["livraison"] = {
                        "livres": s.livres, "echecs": s.echecs,
                        "abandons": s.abandons, "octets_en_attente": s.en_attente,
                    }
                    break

    def instantane(self) -> dict:
        with self._verrou:
            e = dict(self._etat)
        e["uptime_s"] = round(time.time() - e.pop("demarre_le"), 1)
        # Le moteur est « sain » s'il voit le flux ET progresse : un flux
        # connecté mais figé (0 image analysée) n'est pas un état sain.
        e["sain"] = bool(e["flux_connecte"] and e["frames"] > 0)
        # Un abandon de livraison signifie qu'un evenement n'atteindra jamais la
        # plateforme : le moteur detecte correctement mais ne sert plus a rien.
        # Cela doit se voir dans l'etat de sante, pas seulement dans les logs.
        if e.get("livraison", {}).get("abandons"):
            e["sain"] = False
        return e

    # ── Exposition Prometheus ────────────────────────────────────────────────

    @staticmethod
    def _etiquette(valeur: str) -> str:
        """Échappe une valeur d'étiquette selon le format d'exposition."""
        return valeur.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")

    def texte_metrics(self) -> str:
        """État courant au format d'exposition Prometheus.

        `/health` ne donne qu'un instantané : il dit si le moteur va bien
        maintenant, jamais s'il allait mieux il y a une heure. Une supervision a
        besoin de séries — une cadence qui s'effondre lentement, un retard qui
        dérive, un taux d'abandon qui monte ne se voient que dans la durée. Tous
        ces compteurs existaient déjà en mémoire ; seule la mise en forme
        manquait, et elle ne coûte aucune dépendance.
        """
        e = self.instantane()
        etiq = ""
        paires = [(c, e.get(c)) for c in ("camera_id", "site_id")]
        retenues = [f'{c}="{self._etiquette(str(v))}"' for c, v in paires if v]
        if retenues:
            etiq = "{" + ",".join(retenues) + "}"

        livraison = e.get("livraison", {})
        mesures = [
            ("moteur_images_analysees_total", "counter",
             "Images analysees depuis le demarrage", e["frames"]),
            ("moteur_images_ignorees_total", "counter",
             "Images sautees pour ne pas prendre de retard sur le flux",
             e.get("images_ignorees", 0)),
            ("moteur_cadence_images_par_seconde", "gauge",
             "Cadence d'analyse lissee", e["fps"]),
            ("moteur_retard_flux_secondes", "gauge",
             "Age de la derniere image au moment de son analyse",
             e.get("retard_flux_s", 0.0)),
            ("moteur_evenements_total", "counter",
             "Evenements publies sur le bus", e["evenements"]),
            ("moteur_flux_connecte", "gauge",
             "1 si le flux video est lisible, 0 sinon", int(bool(e["flux_connecte"]))),
            ("moteur_reconnexions_total", "counter",
             "Reconnexions au flux video", e["reconnexions"]),
            ("moteur_sain", "gauge",
             "1 si le moteur est sain au sens de /health, 0 sinon", int(bool(e["sain"]))),
            ("moteur_uptime_secondes", "gauge",
             "Duree depuis le demarrage", e["uptime_s"]),
            ("moteur_derniere_detection_timestamp", "gauge",
             "Horodatage Unix du dernier evenement publie, 0 si aucun",
             e.get("derniere_detection") or 0),
        ]
        if livraison:
            mesures += [
                ("moteur_livraison_livres_total", "counter",
                 "Evenements remis au webhook", livraison["livres"]),
                ("moteur_livraison_echecs_total", "counter",
                 "Tentatives de livraison en echec", livraison["echecs"]),
                ("moteur_livraison_abandons_total", "counter",
                 "Evenements definitivement abandonnes", livraison["abandons"]),
                ("moteur_livraison_octets_en_attente", "gauge",
                 "Octets d'evenements pas encore livres", livraison["octets_en_attente"]),
            ]

        lignes = []
        for nom, genre, aide, valeur in mesures:
            lignes += [f"# HELP {nom} {aide}", f"# TYPE {nom} {genre}",
                       f"{nom}{etiq} {valeur}"]
        return "\n".join(lignes) + "\n"

    def demarrer_serveur(self, port: int, adresse: str = "127.0.0.1", jeton: str = ""):
        from http.server import BaseHTTPRequestHandler, HTTPServer

        etat = self

        class Handler(BaseHTTPRequestHandler):
            def _autorise(self) -> bool:
                if not jeton:
                    return True
                attendu = f"Bearer {jeton}"
                fourni = self.headers.get("Authorization", "")
                # Comparaison a temps constant : une comparaison naive laisse
                # deviner le jeton caractere par caractere en mesurant le temps
                # de reponse.
                return hmac.compare_digest(fourni, attendu)

            def do_GET(self):
                chemin = self.path.split("?", 1)[0].rstrip("/")
                if chemin not in ("/health", "/metrics", ""):
                    self.send_error(404)
                    return
                if not self._autorise():
                    self.send_response(401)
                    self.send_header("WWW-Authenticate", 'Bearer realm="moteur"')
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                if chemin == "/metrics":
                    corps = etat.texte_metrics().encode()
                    code, mime = 200, "text/plain; version=0.0.4; charset=utf-8"
                else:
                    instantane = etat.instantane()
                    corps = json.dumps(instantane, ensure_ascii=False).encode()
                    code, mime = (200 if instantane["sain"] else 503), "application/json"
                self.send_response(code)
                self.send_header("Content-Type", mime)
                self.send_header("Content-Length", str(len(corps)))
                self.end_headers()
                self.wfile.write(corps)

            def log_message(self, *a):
                pass  # pas de log par requête : le sondage est périodique

        self._serveur = HTTPServer((adresse, port), Handler)
        threading.Thread(target=self._serveur.serve_forever, daemon=True).start()
        log.info("Sante exposee sur http://%s:%d/health (et /metrics)", adresse, port)
        if adresse not in ("127.0.0.1", "localhost", "::1") and not jeton:
            # Ces points ne sont pas anodins : ils publient la source du flux,
            # l'identite du site et l'etat du moteur. Ouverts au reseau sans
            # jeton, ils renseignent gratuitement quiconque sonde le port.
            log.warning("/health et /metrics sont exposes sur %s SANS authentification : "
                        "utiliser --health-jeton, ou restreindre --health-adresse", adresse)

    def arreter_serveur(self):
        if self._serveur is not None:
            self._serveur.shutdown()


# ── Capture vidéo résiliente ─────────────────────────────────────────────────

class CaptureRobuste:
    """`cv2.VideoCapture` qui survit à une coupure du flux.

    Sans cela, un `cap.read()` en échec sort de la boucle et le processus
    s'arrête : sur un flux RTSP réel (coupure réseau, redémarrage de caméra),
    l'arrêt est définitif et *silencieux* — l'exploitant croit son site
    surveillé alors que plus rien ne tourne. C'est le défaut le plus dangereux
    d'un moteur livré à un client.

    La reconnexion utilise une temporisation progressive (1s, 2s, 4s... plafonnée)
    pour ne pas marteler une caméra déjà en difficulté, et publie un évènement à
    chaque perte et chaque reprise afin que la plateforme consommatrice sache
    distinguer « aucune détection parce que rien ne se passe » de « aucune
    détection parce que le flux est tombé ».

    Un fichier vidéo qui se termine normalement n'est pas une panne : `boucler`
    permet de le relire en boucle (tests d'endurance), sinon la fin du fichier
    termine proprement la lecture.
    """

    DELAI_MIN_S = 1.0
    DELAI_MAX_S = 30.0

    def __init__(self, source, bus=None, boucler=False, echecs_avant_perte=5,
                 vidanger=None):
        self.source = source
        self.bus = bus
        self.boucler = boucler
        self.echecs_avant_perte = echecs_avant_perte
        self.est_fichier = not isinstance(source, int) and not str(source).startswith(
            ("rtsp://", "http://", "https://"))
        self.cap = None
        self.connecte = False
        self._echecs = 0
        self._delai = self.DELAI_MIN_S
        self.reconnexions = 0

        # Retard capture -> analyse. `t_capture` date l'image REELLEMENT lue,
        # et non le moment ou l'analyse la traite : c'est cet horodatage qui doit
        # partir dans les evenements, sinon un retard de flux est indetectable en
        # aval (la plateforme recoit des scenes anciennes datees de maintenant).
        self.t_capture = None
        self.retard_s = 0.0
        self.images_ignorees = 0

        self._vidangeur = None
        self._vidange_morte = False
        self._arret_vidange = threading.Event()
        self._slot = None
        self._slot_verrou = threading.Lock()
        self._image_prete = threading.Event()

        self._ouvrir()

        # Un fichier se lit a la vitesse de l'analyse : aucune image ne s'accumule
        # et les sauter fausserait une mesure. Un flux direct, lui, continue
        # d'avancer que l'analyse suive ou non.
        if vidanger is None:
            vidanger = not self.est_fichier
        if vidanger:
            self._vidangeur = threading.Thread(target=self._vidanger_en_continu,
                                               name="vidange-flux", daemon=True)
            self._vidangeur.start()

    def _ouvrir(self) -> bool:
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.source)
        if isinstance(self.source, int):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not self.est_fichier:
            # Le tampon ne concerne pas que les webcams : sur un flux reseau,
            # c'est precisement lui qui accumule le retard. Ce reglage etait
            # reserve a `isinstance(source, int)`, donc jamais applique au RTSP,
            # la source pour laquelle il compte le plus. Le pilote reste libre
            # de l'ignorer -- c'est courant en RTSP/FFMPEG -- et c'est pourquoi
            # il ne suffit pas : le fil de vidange, lui, ne depend d'aucun
            # reglage de pilote.
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ouvert = self.cap.isOpened()
        if ouvert:
            self._echecs = 0
            self._delai = self.DELAI_MIN_S
            if not self.connecte:
                self.connecte = True
        return ouvert

    def _signaler(self, type_ev: str, libelle: str):
        if self.bus is not None:
            self.bus.publier(Evenement(time.time(), -1, "capture", type_ev, libelle))

    def lire(self, arret=None):
        """Retourne l'image la plus récente, ou None (source épuisée / arrêt).

        Sur un flux direct, l'image vient du fil de vidange : c'est toujours la
        DERNIÈRE captée, jamais la plus ancienne d'une file d'attente. Sur un
        fichier, la lecture reste séquentielle — sauter des images y fausserait
        toute mesure.
        """
        if self._vidangeur is not None and not self._vidange_morte:
            return self._lire_du_tampon(arret)
        frame = self._lire_source(arret)
        if frame is not None:
            self.t_capture = time.time()
            self.retard_s = 0.0
        return frame

    def _lire_du_tampon(self, arret=None):
        """Consomme l'image déposée par le fil de vidange, ou attend la suivante."""
        while True:
            with self._slot_verrou:
                if self._slot is not None:
                    frame, t = self._slot
                    self._slot = None
                    self._image_prete.clear()
                    self.t_capture = t
                    self.retard_s = time.time() - t
                    return frame
            if arret is not None and arret.is_set():
                return None
            if self._vidange_morte:
                # Le fil a succombé : plutôt que de laisser l'appelant attendre
                # une image qui ne viendra jamais, on retombe sur la lecture
                # directe. Le retard n'est plus borné, mais le moteur tourne.
                return self._lire_source(arret)
            self._image_prete.wait(timeout=0.2)

    def _vidanger_en_continu(self):
        """Lit le flux sans discontinuer et ne garde que la dernière image.

        C'est le cœur du correctif : sans ce fil, la lecture avance au rythme de
        l'analyse. Une analyse plus lente que la caméra laisse les images
        s'entasser dans le tampon du décodeur, et le moteur travaille sur des
        scènes de plus en plus vieilles — dérive qui n'apparaît nulle part, car
        rien ne compare la date de capture à la date de traitement.

        Ici les images en trop sont jetées à la source. Le moteur analyse moins
        d'images, mais toujours des images d'il y a un instant.
        """
        try:
            while not self._arret_vidange.is_set():
                frame = self._lire_source(self._arret_vidange)
                if frame is None:
                    break                      # arrêt demandé, ou source épuisée
                with self._slot_verrou:
                    if self._slot is not None:
                        self.images_ignorees += 1
                    self._slot = (frame, time.time())
                    self._image_prete.set()
        except Exception as e:
            log.error("fil de vidange interrompu (%s: %s) : lecture directe, "
                      "le retard du flux n'est plus borne", type(e).__name__, e)
        finally:
            self._vidange_morte = True
            self._image_prete.set()

    def _lire_source(self, arret=None):
        """Lecture séquentielle brute, avec reconnexion et temporisation."""
        while True:
            if arret is not None and arret.is_set():
                return None

            ok, frame = (False, None) if self.cap is None else self.cap.read()
            if ok:
                if not self.connecte:
                    self.connecte = True
                    self.reconnexions += 1
                    self._signaler("flux_repris", f"Flux vidéo rétabli ({self.source})")
                return frame

            # Fin normale d'un fichier : ce n'est pas une panne.
            if self.est_fichier:
                if self.boucler and self.cap is not None:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = self.cap.read()
                    if ok:
                        return frame
                return None

            self._echecs += 1
            if self.connecte and self._echecs >= self.echecs_avant_perte:
                self.connecte = False
                self._signaler("flux_perdu", f"Flux vidéo perdu ({self.source})")

            if self._echecs >= self.echecs_avant_perte:
                if arret is not None:
                    if arret.wait(self._delai):
                        return None
                else:
                    time.sleep(self._delai)
                self._delai = min(self._delai * 2, self.DELAI_MAX_S)
                self._ouvrir()

    def liberer(self):
        if self._vidangeur is not None:
            self._arret_vidange.set()
            self._vidangeur.join(timeout=2.0)
            if self._vidangeur.is_alive():
                # Bloque dans un `read()` que le pilote ne rend pas : relacher la
                # capture sous ses pieds ferait tomber le processus a l'arret.
                log.warning("fil de vidange encore actif a l'arret : capture non relachee")
                return
        if self.cap is not None:
            self.cap.release()


# ── Analyseurs ───────────────────────────────────────────────────────────────

class Analyseur:
    """Contrat commun. `every` = période d'exécution en frames.

    `inference` dit si l'analyseur exécute un modèle. Ceux qui n'en exécutent
    pas (ligne, foule, objet abandonné) ne consomment que le suivi de
    `general` : ils tournent dans la boucle principale, à chaque image, et ne
    passent jamais par le pool où ils risqueraient d'attendre derrière la
    cascade EPI.
    """
    nom = "base"
    every = 1
    inference = True

    def process(self, frame, ctx: dict) -> list[Evenement]:
        raise NotImplementedError

    def draw(self, frame):
        return frame


class ConfirmationTemporelle:
    """Exige qu'une condition soit vue sur plusieurs analyses avant de la déclarer.

    Une détection ponctuelle ne caractérise pas un évènement réel : une chute et
    un départ de feu durent plusieurs images, alors qu'un faux positif est le
    plus souvent isolé. Sans cette confirmation, une seule image bruitée produit
    une alerte — ce que le client perçoit comme un système peu fiable,
    indépendamment des métriques du modèle.

    Le lissage existait déjà pour les EPI et la porte ; cette classe le rend
    disponible aux autres modules sans dupliquer la logique. Coût de calcul nul :
    on ne fait que compter des booléens déjà obtenus.
    """

    def __init__(self, fenetre: int = 4, minimum: int = 2):
        self.fenetre, self.minimum = fenetre, minimum
        self._hist: dict[str, list[bool]] = {}

    def confirmer(self, cle: str, actif: bool) -> bool:
        h = self._hist.setdefault(cle, [])
        h.append(actif)
        if len(h) > self.fenetre:
            del h[0]
        return sum(h) >= self.minimum

    def vieillir(self, cles_vues):
        """Compte une analyse négative pour chaque clé connue absente de celle-ci.

        Sans cet appel, une clé n'est alimentée que les fois où quelque chose
        est détecté à cet endroit : la fenêtre ne contient alors que des `True`
        et « 2 sur les 3 dernières analyses » devient « 2 fois depuis le
        démarrage ». Deux reflets au même endroit à une heure d'écart
        confirmaient ainsi un feu. L'absence de détection est une observation,
        elle doit entrer dans la fenêtre comme les autres.
        """
        vues = set(cles_vues)
        for cle, h in self._hist.items():
            if cle not in vues:
                h.append(False)
                if len(h) > self.fenetre:
                    del h[0]

    def purger(self):
        """Retire les clés dont l'historique est entièrement négatif : évite
        l'accumulation sur un flux de longue durée."""
        for c in [c for c, h in self._hist.items() if not any(h)]:
            del self._hist[c]


class AnalyseurGeneral(Analyseur):
    """Détection + suivi COCO. Produit le contexte partagé par les autres.

    Tourne à chaque frame parce que le suivi (`persist=True`) a besoin de
    continuité : sauter des frames casse l'association d'identifiants et donc le
    comptage de franchissements.
    """
    nom = "general"
    every = 1

    def __init__(self, poids, conf, imgsz):
        from ultralytics import YOLO
        self.model = YOLO(poids)
        self.conf, self.imgsz = conf, imgsz
        self.objets = []

    def process(self, frame, ctx):
        r = self.model.track(frame, persist=True, conf=self.conf, imgsz=self.imgsz, verbose=False)
        objets = []
        if r and r[0].boxes is not None:
            b = r[0].boxes
            for i in range(len(b)):
                objets.append({
                    "box": tuple(b.xyxy[i].cpu().numpy().astype(int)),
                    "label": self.model.names[int(b.cls[i])],
                    "track_id": int(b.id[i]) if b.id is not None else -1,
                })
        self.objets = objets
        ctx["objets"] = objets
        # Les personnes sont transmises avec leur identifiant de suivi, pas
        # seulement leur boîte : c'est cet identifiant qui permet à l'analyseur
        # EPI de rattacher l'historique de lissage à la bonne personne d'une
        # image à l'autre (cf. AnalyseurEPI._histo).
        ctx["personnes"] = [(o["track_id"], o["box"]) for o in objets if o["label"] == "person"]
        return []

    def draw(self, frame):
        for o in self.objets:
            x1, y1, x2, y2 = o["box"]
            c = (255, 150, 0) if o["label"] == "person" else (0, 200, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
            cv2.putText(frame, f"{o['label']} #{o['track_id']}", (x1, max(18, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)
        return frame


class AnalyseurEPI(Analyseur):
    """Chaîne conformité EPI, branchée sur la table de correspondance (P2).

    Contrairement au script d'origine, la traduction classe -> concept passe par
    `ppe_taxonomy`, les seuils sont propres à chaque classe de chaque modèle, et
    les doublons inter-modèles sont fusionnés avant décision.
    """
    nom = "epi"

    def __init__(self, poids_m1, poids_m2=None, poids_m3=None, poids_m4=None,
                 poids_m5=None, poids_m6=None, imgsz=480, every=3,
                 hist_n=12, hist_k=4, ancrage=True):
        from ultralytics import YOLO
        # Derriere un drapeau, comme toute regle de qualification : elle rejette
        # des detections, donc elle doit pouvoir etre comparee et retiree sans
        # redeploiement (garde-fou 3 du plan v5).
        self.ancrage = ancrage
        self.rejetes_ancrage = 0
        self.every = every
        self.imgsz = imgsz
        self.m1 = YOLO(poids_m1)
        self.m2 = YOLO(poids_m2) if poids_m2 else None
        # M3 : modele dedie masque/gilet (2026-08-17), prioritaire sur M1 pour
        # ces deux concepts uniquement -- voir le commentaire de M3 dans
        # ppe_taxonomy.py pour la mesure qui justifie ce choix. M1 reste
        # charge et sert de filet de secours si M3 est absent.
        self.m3 = YOLO(poids_m3) if poids_m3 else None
        # M4 et M5 : modeles dedies casque, puis gants/lunettes (2026-08-18,
        # campagne v8). Meme principe que M3 -- prioritaires sur M1 pour leurs
        # seuls concepts, chacun mesure meilleur que lui sur un split que
        # l'entrainement n'a jamais vu (reports/v3_results/*_candidat.json).
        # Tous restent optionnels : M1 est le filet de secours de la cascade.
        self.m4 = YOLO(poids_m4) if poids_m4 else None
        self.m5 = YOLO(poids_m5) if poids_m5 else None
        # M6 : chaussures de securite, seul modele du parc a couvrir ce concept
        # avec une classe negative. Il DEPEND de l'ancrage pour etre exploitable
        # (voir son commentaire dans ppe_taxonomy.py) : charge sans ancrage, il
        # declenche sur 59 % des images au lieu de 1 %.
        self.m6 = YOLO(poids_m6) if poids_m6 else None
        if self.m6 is not None and not ancrage:
            print("  /!\\ epi_chaussures.pt est charge alors que l'ancrage est "
                  "desactive : ce modele produit alors des detections aberrantes "
                  "(visages, cones). Voir p18_ancrage_chaussures.py.")
        modeles_charges = {tax.M1_NOM: self.m1.names}
        for nom, modele in ((tax.M2_NOM, self.m2), (tax.M3_NOM, self.m3),
                            (tax.M4_NOM, self.m4), (tax.M5_NOM, self.m5),
                            (tax.M6_NOM, self.m6)):
            if modele is not None:
                modeles_charges[nom] = modele.names
        erreurs = tax.verifier_coherence(modeles_charges)
        if erreurs:
            raise RuntimeError("Taxonomie incohérente avec les modèles chargés :\n  " +
                               "\n  ".join(erreurs))

        # Ce que la cascade sait encore faire, concept par concept. Retirer un
        # modèle par drapeau ne provoque aucune erreur : le moteur démarre et
        # se tait, simplement plus aveugle qu'avant. `--sans-chaussures` fait
        # ainsi disparaître un concept ENTIER, aucun autre modèle chargé par
        # défaut ne le couvrant. Un exploitant doit l'apprendre au démarrage,
        # pas en constatant l'absence d'alertes des semaines plus tard.
        self.couverture = {}
        for concept in tax.EPI_CANONIQUES:
            servants = [m for m in tax.PRIORITE_MODELE.get(concept, ())
                        if m in modeles_charges]
            self.couverture[concept] = servants[0] if servants else None
            if not servants:
                print(f"  /!\\ concept '{concept}' NON COUVERT : aucun modèle chargé "
                      f"ne le détecte, il ne produira jamais d'évènement.")
            elif servants[0] != tax.PRIORITE_MODELE[concept][0]:
                print(f"  /!\\ concept '{concept}' servi par {servants[0]} au lieu de "
                      f"{tax.PRIORITE_MODELE[concept][0]} : modèle de repli, moins fiable.")
        # Les modèles de la cascade sont interrogés EN PARALLÈLE, un fil chacun.
        # Ils ne dépendent pas les uns des autres (même image, résultats
        # fusionnés ensuite), et sur CPU une convolution ne profite pas
        # linéairement de ses threads : cinq modèles à trois threads vont plus
        # vite que cinq modèles à la suite sur douze. Mesure du 2026-09-23 sur
        # la séquence de test, en alternance image par image : 708 ms contre
        # 1 859 ms, sorties identiques à 1e-4 près (bruit flottant).
        # Chaque modèle reste servi par un seul fil à la fois : une instance
        # YOLO n'est pas prévue pour l'accès concurrent.
        self._cascade = [(nom, m) for nom, m in (
            (tax.M1_NOM, self.m1), (tax.M2_NOM, self.m2), (tax.M3_NOM, self.m3),
            (tax.M4_NOM, self.m4), (tax.M5_NOM, self.m5), (tax.M6_NOM, self.m6))
            if m is not None]
        self._pool_cascade = (ThreadPoolExecutor(max_workers=len(self._cascade),
                                                 thread_name_prefix="epi")
                              if len(self._cascade) > 1 else None)
        self._seuil_bas = min(c.conf_min for t in tax.TABLES.values() for c in t.values())

        self.hist_n, self.hist_k = hist_n, hist_k
        # Historique indexé par identifiant de suivi (track_id), jamais par
        # position dans la liste des détections : cette liste est reconstruite à
        # chaque image et son ordre n'est pas stable, si bien qu'indexer par
        # position revient à attribuer l'historique d'une personne à une autre —
        # donc à imputer une violation au mauvais individu.
        self._hist: dict[str, dict[str, list[bool]]] = {}
        self._vu_le: dict[str, float] = {}
        self.dets: list[tax.DetectionEPI] = []
        self.violations: dict[str, list[str]] = {}

    def _histo(self, cle: str, epi: str):
        return self._hist.setdefault(cle, {}).setdefault(epi, [False] * self.hist_n)

    def _oublier_absents(self, presents: set[str], maintenant: float, ttl_s: float = 30.0):
        """Purge l'historique des personnes disparues depuis un moment.

        Sans cela, un flux de longue durée accumule un historique par track_id
        jamais réutilisé — fuite mémoire lente sur une caméra 24/7.
        """
        for cle in presents:
            self._vu_le[cle] = maintenant
        for cle in [c for c, t in self._vu_le.items() if maintenant - t > ttl_s]:
            self._hist.pop(cle, None)
            self._vu_le.pop(cle, None)

    def _predire(self, model, frame):
        return model.predict(frame, conf=self._seuil_bas, imgsz=self.imgsz, verbose=False)[0].boxes

    def process(self, frame, ctx):
        if self._pool_cascade is not None:
            sorties = list(self._pool_cascade.map(
                lambda nm: self._predire(nm[1], frame), self._cascade))
        else:
            sorties = [self._predire(m, frame) for _, m in self._cascade]
        # Traduction dans l'ordre fixe de la cascade, quel que soit l'ordre
        # d'achèvement des fils : la fusion doit rester déterministe.
        brutes = []
        for (nom_modele, model), boites in zip(self._cascade, sorties):
            for b in boites:
                d = tax.traduire(nom_modele, model.names[int(b.cls)], float(b.conf),
                                 tuple(map(int, b.xyxy[0])))
                if d:
                    brutes.append(d)
        self.dets = tax.fusionner(brutes)

        # `personnes` : liste de (track_id, box). L'identifiant vient du suivi
        # calculé par l'analyseur général et reste stable d'une image à l'autre.
        personnes = list(ctx.get("personnes") or [])
        if not personnes and self.dets:
            # Aucune personne détectée mais des EPI visibles : on synthétise une
            # zone englobante, sans identité suivie.
            xs = [c for d in self.dets for c in (d.box[0], d.box[2])]
            ys = [c for d in self.dets for c in (d.box[1], d.box[3])]
            personnes = [(-1, (min(xs), min(ys), max(xs), max(ys)))]

        # Le suivi ne fournit pas toujours d'identifiant : au démarrage, et à
        # chaque fois qu'il perd ses cibles (mouvement brusque, occlusion), il
        # renvoie -1 pour tout le monde. Indexer directement par track_id ferait
        # alors s'effondrer plusieurs personnes distinctes sur une même clé — le
        # défaut même que ce correctif vise à supprimer. On retombe donc sur une
        # clé spatiale grossière, distincte par personne et raisonnablement
        # stable d'une image à l'autre pour quelqu'un qui ne court pas.
        cles = []
        for pid, box in personnes:
            if pid >= 0:
                cles.append((f"t{pid}", pid, box))
            else:
                cx, cy = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
                cles.append((f"z{cx // 80}:{cy // 80}", pid, box))

        boites = {cle: box for cle, _, box in cles}
        porte = {cle: set() for cle in boites}
        absent = {cle: set() for cle in boites}
        noms = list(boites)
        for d in self.dets:
            if not d.epi or not boites:
                continue
            if self.ancrage:
                # Association par CONFINEMENT, avec rejet possible.
                #
                # L'ancienne regle -- `max(boites, key=IoU)` -- avait deux
                # defauts qui se cumulaient. D'abord l'IoU est la mauvaise
                # mesure pour un rapport « partie de » : un casque occupe ~3 %
                # de la boite d'une personne, son IoU vaut donc ~0,03 qu'il soit
                # porte ou non, valeur indiscernable du bruit. Ensuite `max`
                # attribue TOUJOURS l'EPI a quelqu'un : un faux positif detecte
                # a l'autre bout de l'image comptait comme equipement porte et
                # masquait une vraie infraction -- le sens d'erreur le plus
                # grave sur un systeme de securite.
                i = qual.associer_a_personne(d.box, [boites[c] for c in noms], epi=d.epi)
                if i is None:
                    self.rejetes_ancrage += 1
                    continue
                cle = noms[i]
            else:
                cle = max(boites, key=lambda c: tax._iou(d.box, boites[c]))
            (porte if d.porte else absent)[cle].add(d.epi)

        self._oublier_absents(set(boites), ctx["t"])

        # Lissage temporel : un EPI n'est réputé porté que s'il a été vu dans au
        # moins hist_k des hist_n dernières frames analysées. Indispensable avec
        # des seuils de confiance bas, sinon chaque frame bruitée bascule l'état.
        evs = []
        self.violations = {}
        for rang, (cle, pid, box) in enumerate(cles, start=1):
            # (concept, motif) et non plus le seul libellé français : celui-ci
            # est destiné à l'affichage, et le contrat d'API interdit de
            # l'analyser par programme. Sans le concept en clair, la plateforme
            # consommatrice n'avait pourtant aucun autre moyen de savoir QUEL
            # équipement manquait -- corrigé le 2026-08-19.
            manquants: list[tuple[str, str]] = []
            # Une absence POSITIVEMENT constatée (classe négative détectée : tête
            # nue, torse sans gilet) prime sur « jamais confirmé ». Auparavant,
            # pour un EPI obligatoire non vu, `jamais_confirme` était posé en
            # premier et masquait `absence_detectee` : un ouvrier sans casque
            # filmé en continu ne produisait que `jamais_confirme`, que la
            # plateforme écarte (`motif_minimum = absence_detectee`,
            # ai_models/ingestion.py) — aucune alerte casque ni gilet n'arrivait.
            # Constaté et corrigé le 2026-09-27 (banc S1/S2, tests/test_epi_motif.py).
            constates = set(absent[cle])
            for epi in tax.EPI_CANONIQUES:
                h = self._histo(cle, epi)
                h.append(epi in porte[cle])
                del h[0]
                stable = sum(h) >= self.hist_k
                if tax.EPI_OBLIGATOIRES[epi] and not stable and epi not in constates:
                    # L'équipement n'a pas été VU assez souvent. Ce n'est pas la
                    # même chose que d'avoir constaté son absence : il peut être
                    # hors champ ou masqué.
                    manquants.append((epi, "jamais_confirme"))
            deja = {epi for epi, _ in manquants}
            for epi in tax.EPI_CANONIQUES:
                if epi in constates and epi not in deja:
                    # Une classe négative a été détectée : l'absence est
                    # positivement constatée, pas seulement supposée.
                    manquants.append((epi, "absence_detectee"))
            if manquants:
                self.violations[cle] = [tax.LIBELLES_FR[e][1] for e, _ in manquants]
                # Le libellé porte l'identifiant de suivi quand il existe : c'est
                # lui qui permet à la plateforme de recoller les évènements
                # successifs à une même personne. Sans suivi, on retombe sur un
                # numéro d'ordre, valable pour la seule image courante.
                qui = f"Personne #{pid}" if pid >= 0 else f"Personne {rang}"
                for epi, motif in manquants:
                    evs.append(Evenement(
                        ctx["t"], ctx["frame"], self.nom, "violation_epi",
                        f"{qui} — {tax.LIBELLES_FR[epi][1]}", box=box,
                        extra={"track_id": pid, "suivi": pid >= 0,
                               "epi": epi, "motif": motif,
                               "obligatoire": tax.EPI_OBLIGATOIRES[epi]}))
        return evs

    def draw(self, frame):
        for d in self.dets:
            if d.epi is None:
                continue
            x1, y1, x2, y2 = d.box
            c = (0, 0, 220) if d.porte is False else (0, 200, 120)
            cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
            cv2.putText(frame, f"{d.libelle} {d.conf:.0%}", (x1, max(14, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, c, 1)
        return frame


class AnalyseurChute(Analyseur):
    """Modèle dédié (`fall_detector.pt`, P5) si fourni, sinon heuristique pose.

    Même schéma de repli que le module porte et que `surveillance_suite/main.py` :
    le modèle dédié (falling 97.6% / stand 93.2% AP50, cf. `p5_final_val.log`)
    est préféré, l'heuristique angle+ratio reste le secours si `poids_dedie`
    est absent.
    """
    nom = "chute"

    def __init__(self, poids_pose=None, poids_dedie=None, imgsz=480, every=2, ratio=1.4, angle=45,
                 fenetre=4, minimum=2):
        from ultralytics import YOLO
        self.imgsz, self.every, self.ratio, self.angle = imgsz, every, ratio, angle
        self.falls = []
        # Une chute dure plusieurs images : on exige `minimum` confirmations sur
        # les `fenetre` dernières analyses avant de publier l'évènement.
        self.confirmation = ConfirmationTemporelle(fenetre, minimum)
        if poids_dedie:
            self.model = YOLO(poids_dedie)
            self._mode = "dedie"
        else:
            from module_fall import detect_falls
            self._detect = detect_falls
            self.model = YOLO(poids_pose)
            self._mode = "pose"

    def process(self, frame, ctx):
        if self._mode == "dedie":
            r = self.model.predict(frame, imgsz=self.imgsz, conf=0.4, verbose=False)
            self.falls = [
                {"box": tuple(b.xyxy[0].cpu().numpy().astype(int)),
                 "fallen": self.model.names[int(b.cls)] == "falling"}
                for b in r[0].boxes
            ]
        else:
            r = self.model(frame, imgsz=self.imgsz, verbose=False)
            self.falls = self._detect(r[0], self.ratio, self.angle)

        # Confirmation par zone : une chute au même endroit sur plusieurs
        # analyses est un évènement, une détection isolée est du bruit. La zone
        # sert de clé faute d'identifiant de suivi sur ce modèle — grille
        # grossière (80 px) pour tolérer le léger déplacement des boîtes.
        # Une observation par zone et par analyse. Deux boîtes « au sol » dans
        # la même zone d'une même image (doublon du détecteur) comptaient pour
        # deux observations, et confirmaient donc une chute sur une image
        # isolée. Une zone est « au sol » si l'une de ses boîtes l'est.
        par_zone: dict[str, dict] = {}
        for f in self.falls:
            x1, y1, x2, y2 = f["box"]
            cle = f"{int((x1 + x2) / 160)}:{int((y1 + y2) / 160)}"
            if cle not in par_zone or (f["fallen"] and not par_zone[cle]["fallen"]):
                par_zone[cle] = f
        evs = []
        for cle, f in par_zone.items():
            if self.confirmation.confirmer(cle, bool(f["fallen"])) and f["fallen"]:
                evs.append(Evenement(ctx["t"], ctx["frame"], self.nom, "chute",
                                     "PERSONNE AU SOL", box=f["box"]))
        vues = set(par_zone)
        # Une zone où plus rien n'est vu compte une analyse négative : sans
        # cela, deux détections isolées au même endroit, à n'importe quel
        # intervalle, suffisaient à confirmer une chute.
        self.confirmation.vieillir(vues)
        self.confirmation.purger()
        return evs

    def draw(self, frame):
        for f in self.falls:
            if f["fallen"]:
                x1, y1, x2, y2 = f["box"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(frame, "PERSONNE AU SOL !", (x1, max(18, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        return frame


class AnalyseurLigne(Analyseur):
    """Ne fait aucune inférence : consomme le suivi déjà calculé par `general`.

    Il tourne donc à chaque frame pour un coût négligeable — c'est justement
    pourquoi le suivi doit, lui, rester à chaque frame.
    """
    nom = "ligne"
    every = 1
    inference = False

    def __init__(self, debut, fin):
        from module_line_crossing import LineCrossingCounter
        self.counter = LineCrossingCounter(debut, fin)
        self.debut, self.fin = debut, fin

    def process(self, frame, ctx):
        evs = []
        for c in self.counter.update(ctx.get("objets", [])):
            evs.append(Evenement(ctx["t"], ctx["frame"], self.nom, "franchissement",
                                 f"{c['label']} #{c['track_id']} a franchi la ligne",
                                 extra={"sens": c.get("direction", "")}))
        return evs

    def draw(self, frame):
        cv2.line(frame, self.debut, self.fin, (255, 255, 0), 2)
        cv2.putText(frame, f"IN:{self.counter.count_in} OUT:{self.counter.count_out}",
                    (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        return frame


class AnalyseurFoule(Analyseur):
    """Effectif et densité de personnes dans une zone au sol.

    Ne fait **aucune inférence** : les personnes et leur suivi viennent du
    détecteur général. Rien à entraîner ici non plus — ce qui manquait n'était
    pas un modèle mais la **géométrie du sol**, sans laquelle « personnes par
    mètre carré » n'a pas de sens (voir `improvements/calibration_sol.py`).

    Deux modes, selon ce qui est disponible :

    - **avec calibration** : chaque personne est projetée au sol par son point
      de contact, l'aire réelle de la zone est connue, la densité est exacte ;
    - **sans calibration** : seul l'effectif est disponible. Le champ `densite`
      vaut alors `null` — il vaut mieux annoncer qu'on ne sait pas que publier
      une valeur inventée.

    Tous les seuils sont des paramètres : ils dépendent du site, du type de
    lieu et de la réglementation applicable, et n'ont aucune valeur universelle.
    """
    nom = "foule"
    every = 1
    inference = False

    def __init__(self, seuil_densite: float = 2.0, seuil_effectif: int = 0,
                 zone=None, calibration=None, fenetre: int = 5, minimum: int = 3):
        self.seuil_densite = seuil_densite      # personnes/m2 (ex. 10 pour 5 m2 = 2.0)
        self.seuil_effectif = seuil_effectif    # 0 = desactive
        self.zone = list(zone) if zone else None
        self.calibration = calibration
        # Une foule ne se forme pas en une image : la confirmation evite qu'un
        # groupe qui passe devant l'objectif, ou une detection instable, ne
        # declenche une alerte.
        self.confirmation = ConfirmationTemporelle(fenetre, minimum)
        self.en_alerte = False
        self.dernier = {"effectif": 0, "densite": None}

        self.aire_m2 = None
        if self.calibration is not None and self.zone:
            self.aire_m2 = self.calibration.aire_m2(self.zone)

    def _dans_zone(self, pied) -> bool:
        if not self.zone:
            return True
        from calibration_sol import dans_polygone
        return dans_polygone(pied, self.zone)

    def process(self, frame, ctx):
        from calibration_sol import point_au_sol

        pieds = [point_au_sol(b) for _, b in ctx.get("personnes", [])]
        dedans = [p for p in pieds if self._dans_zone(p)]
        effectif = len(dedans)

        densite = None
        if self.aire_m2:
            densite = effectif / self.aire_m2
        elif self.calibration is not None and not self.zone:
            # Zone non definie : la densite n'a pas de surface de reference.
            densite = None
        self.dernier = {"effectif": effectif,
                        "densite": round(densite, 3) if densite is not None else None}

        depasse = False
        if densite is not None and self.seuil_densite > 0:
            # Tolerance relative indispensable, et pas une coquetterie : l'aire
            # sort d'une homographie, donc d'un calcul flottant. Une zone reglee
            # a 5 m2 se mesure 5.0000000000000036, et 10 personnes y donnent une
            # densite de 1.9999999999999987 -- juste sous un seuil de 2.0. Sans
            # cette tolerance, « 10 personnes pour 5 m2 » ne declencherait qu'a
            # 11 personnes, defaut invisible en test et introuvable sur le terrain.
            depasse = densite >= self.seuil_densite * (1 - 1e-9)
        if self.seuil_effectif > 0 and effectif >= self.seuil_effectif:
            depasse = True

        confirme = self.confirmation.confirmer("foule", depasse)
        evs = []
        # Front montant / descendant uniquement : sans cela une foule stable
        # produirait un evenement par image (cf. plan v6, deluge d'evenements).
        if confirme and not self.en_alerte:
            self.en_alerte = True
            libelle = (f"Densite {densite:.2f} pers/m2 sur {self.aire_m2:.1f} m2 "
                       f"({effectif} personnes)" if densite is not None
                       else f"{effectif} personnes dans la zone")
            evs.append(Evenement(ctx["t"], ctx["frame"], self.nom, "foule", libelle,
                                 extra={"effectif": effectif,
                                        "densite_pers_m2": self.dernier["densite"],
                                        "aire_m2": round(self.aire_m2, 2) if self.aire_m2 else None,
                                        "seuil_densite": self.seuil_densite,
                                        "seuil_effectif": self.seuil_effectif,
                                        "calibre": self.calibration is not None}))
        elif self.en_alerte and not confirme:
            self.en_alerte = False
            evs.append(Evenement(ctx["t"], ctx["frame"], self.nom, "foule_terminee",
                                 f"Retour sous le seuil ({effectif} personnes)",
                                 extra={"effectif": effectif,
                                        "densite_pers_m2": self.dernier["densite"]}))
        return evs

    def draw(self, frame):
        if self.zone and len(self.zone) >= 3:
            import numpy as np
            pts = np.asarray(self.zone, dtype="int32").reshape(-1, 1, 2)
            cv2.polylines(frame, [pts], True,
                          (0, 0, 255) if self.en_alerte else (0, 200, 200), 2)
        d = self.dernier
        txt = (f"Foule: {d['effectif']} pers"
               + (f" / {d['densite']:.2f} pers/m2" if d["densite"] is not None else ""))
        cv2.putText(frame, txt, (10, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 255) if self.en_alerte else (0, 200, 200), 2)
        return frame


class AnalyseurObjetAbandonne(Analyseur):
    """Objet posé, immobile, et laissé sans surveillance pendant un délai.

    Comme `AnalyseurLigne`, ne fait **aucune inférence** : les classes visées
    (sac à dos, valise, sac à main…) appartiennent déjà aux 80 classes COCO du
    détecteur général, et leur suivi est déjà calculé. Il n'y a donc rien à
    entraîner et le coût est celui de quelques comparaisons de boîtes.

    C'est aussi pourquoi cet analyseur remplace
    `surveillance_suite/detectors/abandoned_object_detector.py`, qui chargeait
    et exécutait *son propre* modèle YOLO — doublant l'inférence la plus coûteuse
    du pipeline pour un résultat déjà disponible.

    Les seuils sont exprimés **relativement à la taille de l'objet** et non en
    pixels absolus comme dans le détecteur d'origine. Un sac à 5 m et le même
    sac à 50 m n'occupent pas le même nombre de pixels : un rayon de proximité
    fixe de 150 px vaudrait quelques centimètres au premier plan et plusieurs
    mètres au fond de la scène. Rapporté à la diagonale de l'objet, le critère
    garde le même sens partout dans l'image.
    """
    nom = "objet_abandonne"
    every = 1
    inference = False

    # Bagagerie et objets encombrants susceptibles d'etre deposes puis oublies.
    # Volontairement restreint : elargir a « bottle » ou « cell phone » noierait
    # l'aval sous des objets sans enjeu de securite.
    CLASSES = ("backpack", "handbag", "suitcase", "bicycle", "skateboard")

    def __init__(self, delai_s: float = 30.0, classes=None,
                 immobilite: float = 0.30, proximite: float = 2.0,
                 oubli_s: float = 10.0):
        self.delai_s = delai_s
        self.classes = set(classes or self.CLASSES)
        self.immobilite = immobilite      # fraction de la diagonale de l'objet
        self.proximite = proximite        # multiple de la diagonale de l'objet
        self.oubli_s = oubli_s
        self.etats: dict[int, dict] = {}  # track_id -> etat

    @staticmethod
    def _centre(b):
        return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2

    @staticmethod
    def _diagonale(b):
        return max(1.0, ((b[2] - b[0]) ** 2 + (b[3] - b[1]) ** 2) ** 0.5)

    @staticmethod
    def _distance_a_boite(point, b) -> float:
        """Distance du point au bord de la boîte, nulle s'il est dedans.

        On mesure jusqu'au *bord* et non jusqu'au centre : une personne est une
        boîte haute, et la distance à son centre placerait ses pieds à plusieurs
        dizaines de pixels d'elle-même.
        """
        x, y = point
        dx = max(b[0] - x, 0, x - b[2])
        dy = max(b[1] - y, 0, y - b[3])
        return (dx * dx + dy * dy) ** 0.5

    def process(self, frame, ctx):
        t = ctx["t"]
        personnes = [b for _, b in ctx.get("personnes", [])]
        evs = []
        vus = set()

        for o in ctx.get("objets", []):
            tid = o.get("track_id", -1)
            if tid < 0 or o.get("label") not in self.classes:
                continue
            vus.add(tid)
            boite = o["box"]
            centre, diag = self._centre(boite), self._diagonale(boite)

            e = self.etats.get(tid)
            if e is None:
                # Un objet deja pose a l'ouverture du flux doit pouvoir alerter :
                # on part du principe qu'il vient d'etre laisse, faute de savoir
                # ce qui s'est passe avant. L'alerte ne partira qu'apres `delai_s`.
                self.etats[tid] = {"ancre": centre, "derniere_personne": t,
                                   "vu": t, "signale": False}
                continue

            e["vu"] = t
            # L'objet s'est-il deplace depuis son point d'ancrage ? Si oui, il est
            # manipule ou transporte : on re-ancre et on repart de zero.
            if (abs(centre[0] - e["ancre"][0]) ** 2
                    + abs(centre[1] - e["ancre"][1]) ** 2) ** 0.5 > self.immobilite * diag:
                e["ancre"] = centre
                e["derniere_personne"] = t
                e["signale"] = False
                continue

            if any(self._distance_a_boite(centre, p) <= self.proximite * diag
                   for p in personnes):
                e["derniere_personne"] = t
                e["signale"] = False
                continue

            seul_depuis = t - e["derniere_personne"]
            if seul_depuis >= self.delai_s and not e["signale"]:
                e["signale"] = True
                evs.append(Evenement(
                    t, ctx["frame"], self.nom, "objet_abandonne",
                    f"{o['label']} #{tid} laissé sans surveillance "
                    f"depuis {int(seul_depuis)} s",
                    conf=float(o.get("conf", 0.0)), box=tuple(boite),
                    extra={"track_id": tid, "classe": o["label"],
                           "secondes_sans_surveillance": round(seul_depuis, 1)}))

        # Un flux de longue duree voit passer des milliers d'identifiants : sans
        # cet oubli, l'etat croitrait indefiniment (cf. plan v6 §2.2, derive
        # memoire).
        for tid in [k for k, e in self.etats.items()
                    if t - e["vu"] > self.oubli_s and k not in vus]:
            del self.etats[tid]

        return evs

    def draw(self, frame):
        return frame


class AnalyseurFeu(Analyseur):
    nom = "feu"

    def __init__(self, poids, conf=0.4, every=5, fenetre=3, minimum=2):
        from module_fire_smoke import FireSmokeDetector
        self.det = FireSmokeDetector(poids, conf=conf)
        self.every = every
        self.res = []
        # Un départ de feu ou de fumée persiste : exiger la confirmation évite
        # qu'un reflet ou une couleur trompeuse sur une seule image ne déclenche
        # une alerte incendie. Fenêtre plus courte que la chute : ce module ne
        # tourne déjà qu'une image sur cinq.
        self.confirmation = ConfirmationTemporelle(fenetre, minimum)

    def process(self, frame, ctx):
        self.res = self.det.detect(frame)
        # Meilleure détection par zone : deux boîtes de fumée qui se
        # chevauchent sur une même image ne sont qu'une observation, pas deux
        # -- sans quoi une image isolée suffisait à « confirmer ».
        par_zone: dict[str, dict] = {}
        for d in self.res:
            x1, y1, x2, y2 = d["box"]
            cle = f"{d['label']}:{int((x1 + x2) / 160)}:{int((y1 + y2) / 160)}"
            if cle not in par_zone or d["conf"] > par_zone[cle]["conf"]:
                par_zone[cle] = d
        evs = []
        for cle, d in par_zone.items():
            if self.confirmation.confirmer(cle, True):
                evs.append(Evenement(ctx["t"], ctx["frame"], self.nom, d["label"],
                                     f"{d['label'].upper()} détecté",
                                     conf=d["conf"], box=tuple(d["box"])))
        # Les zones sans détection sur cette analyse comptent une observation
        # négative : c'est ce qui donne son sens à « 2 sur les 3 dernières ».
        self.confirmation.vieillir(par_zone)
        self.confirmation.purger()
        return evs

    def draw(self, frame):
        for d in self.res:
            x1, y1, x2, y2 = d["box"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 100, 255), 2)
            cv2.putText(frame, f"{d['label'].upper()} {d['conf']:.2f}", (x1, max(18, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 100, 255), 2)
        return frame


class AnalyseurPlaque(Analyseur):
    """Lecture de plaques d'immatriculation.

    Seul analyseur produisant une **donnée à caractère personnel** : un numéro
    d'immatriculation identifie indirectement une personne. Ce que le moteur
    transmet ne peut donc pas être un choix par défaut, il doit être décidé.
    Trois modes, du plus au moins protecteur :

      `presence`     — signale qu'un véhicule est présent, sans lire la plaque.
                       Aucune donnée personnelle ne quitte le moteur.
      `pseudonymise` — transmet une empreinte stable (hachage tronqué) : permet
                       de reconnaître un même véhicule d'un passage à l'autre
                       sans jamais exposer le numéro. Défaut, car il couvre la
                       majorité des besoins de supervision sans le risque.
      `complet`      — transmet le numéro en clair. À n'activer que si le besoin
                       est établi et la base légale documentée.

    Le hachage est salé : sans le sel, une empreinte ne peut pas être rapprochée
    d'une liste de plaques connues par force brute (l'espace des plaques est
    trop petit pour qu'un hachage non salé protège quoi que ce soit).
    """

    nom = "lpr"
    MODES = ("presence", "pseudonymise", "complet")

    def __init__(self, poids, conf=0.4, every=10, mode="pseudonymise", sel=""):
        from module_lpr import LicensePlateReader
        if mode not in self.MODES:
            raise ValueError(f"mode de retention inconnu : {mode} (attendus : {self.MODES})")
        self.reader = LicensePlateReader(poids, conf=conf)
        self.every = every
        self.mode = mode
        self._sel = sel or os.environ.get("MOTEUR_LPR_SEL", "")
        if mode == "pseudonymise" and not self._sel:
            log.warning("LPR pseudonymise sans sel (MOTEUR_LPR_SEL) : les empreintes "
                        "restent vulnerables a une attaque par dictionnaire de plaques")
        self.res = []

    def _libelle(self, texte: str) -> tuple[str, dict]:
        if self.mode == "presence" or not texte:
            return "Vehicule detecte", {}
        if self.mode == "complet":
            return texte, {"plaque": texte}
        import hashlib
        empreinte = hashlib.sha256((self._sel + texte).encode()).hexdigest()[:12]
        return f"Vehicule {empreinte}", {"empreinte": empreinte}

    def process(self, frame, ctx):
        self.res = self.reader.detect_and_read(frame)
        evs = []
        for p in self.res:
            libelle, extra = self._libelle(p.get("text") or "")
            evs.append(Evenement(ctx["t"], ctx["frame"], self.nom, "plaque", libelle,
                                 box=tuple(p["box"]),
                                 extra={"retention": self.mode, **extra}))
        return evs

    def draw(self, frame):
        for p in self.res:
            x1, y1, x2, y2 = p["box"]
            libelle, _ = self._libelle(p.get("text") or "")
            cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 0, 200), 2)
            cv2.putText(frame, libelle, (x1, max(18, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 0, 200), 2)
        return frame


class AnalyseurPorte(Analyseur):
    nom = "porte"

    def __init__(self, classifier, every=3):
        self.clf = classifier
        self.every = every
        self.etat = None

    def process(self, frame, ctx):
        ouvert, conf = self.clf.update(frame)
        evs = []
        if ouvert is not None and ouvert != self.etat:
            self.etat = ouvert
            evs.append(Evenement(ctx["t"], ctx["frame"], self.nom, "porte",
                                 "Porte OUVERTE" if ouvert else "Porte FERMEE", conf=conf or 0.0))
        return evs

    def draw(self, frame):
        return self.clf.draw(frame)


# ── Boucle unifiée ───────────────────────────────────────────────────────────

class AnalyseurVisage(Analyseur):
    """Signature faciale de chaque personne suivie. N'identifie personne.

    Ce module produit un évènement `visage` par piste, portant la **signature**
    du visage (512 nombres) et le `track_id` de la personne. La plateforme fait
    la recherche dans son trombinoscope et attribue un nom ; le moteur ne détient
    aucun gabarit et ne sait donc rien dire de l'identité de qui que ce soit.

    ## Pourquoi le `track_id` est le cœur de la fonction

    C'est lui, et non le nom, qui relie « personne 7 sans casque » à « personne 7
    a le visage X ». `AnalyseurEPI` place déjà `track_id` dans l'`extra` de ses
    violations (§4 bis du contrat) ; en publiant le même identifiant ici, la
    plateforme peut rattacher les deux sans jamais comparer de boîtes ni deviner.

    La conséquence, à connaître : **le rattachement n'est valable que dans une
    même piste**. Si le suivi perd la personne et lui donne un nouveau
    `track_id`, il faudra une nouvelle signature pour la renommer. C'est le
    comportement voulu — perpétuer une identité par-delà une rupture de suivi
    reviendrait à attribuer une infraction à quelqu'un sur la foi d'une
    supposition.

    ## Cadence

    `every=4` : quatre fois moins souvent que le suivi. Un visage exploitable
    reste exploitable pendant plusieurs images, et l'émission n'a lieu qu'au gain
    de qualité (`SuiviSignatures`). Analyser à chaque image aurait multiplié le
    coût sans améliorer une seule signature.
    """
    nom = "visage"
    every = 4

    def __init__(self, paquet=None, seuil=0.5, gpu=True, qualite_minimale=0.35):
        from surveillance_suite.modules.module_visage import (
            DetecteurVisage, PAQUET_DEFAUT, SuiviSignatures)

        self.detecteur = DetecteurVisage(paquet or PAQUET_DEFAUT, seuil=seuil, gpu=gpu)
        if not self.detecteur.disponible:
            # Remonter en `FileNotFoundError` fait traiter le cas par la même
            # voie que les poids manquants dans `construire_analyseurs` : le
            # module est déclaré indisponible, le moteur démarre quand même.
            raise FileNotFoundError(
                "insightface indisponible (paquet %s)" % (paquet or PAQUET_DEFAUT))
        self.suivi = SuiviSignatures()
        self.qualite_minimale = qualite_minimale
        self._derniers = []   # pour `draw`

    def process(self, frame, ctx):
        from surveillance_suite.modules.module_visage import (
            DIMENSION, PAQUET_DEFAUT, encoder, qualite, tete_de)

        personnes = ctx.get("personnes") or []
        self.suivi.oublier([tid for tid, _ in personnes])
        self._derniers = []
        evenements = []

        for track_id, boite in personnes:
            if track_id < 0:
                # Sans identifiant de suivi, la signature ne pourrait être
                # rattachée à aucune infraction : elle ne servirait à rien et
                # coûterait une inférence. Le contrat publie déjà `suivi: false`
                # dans le même cas pour les EPI.
                continue

            fenetre = tete_de(boite, frame.shape)
            if fenetre is None:
                continue
            hx1, hy1, hx2, hy2 = fenetre
            vignette = frame[hy1:hy2, hx1:hx2]
            if vignette.size == 0:
                continue

            for visage in self._visage_de_la_personne(
                    self.detecteur.detecter(vignette), boite, hx1):
                vx1, vy1, vx2, vy2 = visage["box"]
                decoupe = vignette[max(0, vy1):max(0, vy2), max(0, vx1):max(0, vx2)]
                if decoupe.size == 0:
                    continue

                note = qualite(decoupe, visage["score"], visage.get("pose"))
                boite_absolue = (hx1 + vx1, hy1 + vy1, hx1 + vx2, hy1 + vy2)
                self._derniers.append((boite_absolue, track_id, note))

                if note < self.qualite_minimale:
                    continue
                if not self.suivi.merite_emission(track_id, note):
                    continue

                evenements.append(Evenement(
                    t=ctx["t"], frame=ctx["frame"], source="visage", type="visage",
                    libelle=f"Visage capté — personne #{track_id}",
                    conf=round(note, 3),
                    box=boite_absolue,
                    extra={
                        "track_id": track_id,
                        "suivi": True,
                        # La signature, et de quoi refuser de la comparer à un
                        # gabarit produit par un autre modèle. Sans ces deux
                        # champs, un changement de paquet côté moteur ferait
                        # silencieusement chuter tous les scores de la
                        # plateforme, sans qu'aucune erreur ne soit levée.
                        "signature": encoder(visage["signature"]),
                        "modele": self.detecteur.paquet,
                        "dim": DIMENSION,
                        "qualite": round(note, 3),
                        "score_detection": round(visage["score"], 3),
                        "boite_personne": [int(v) for v in boite],
                        # Vignette du visage, pour que la plateforme puisse
                        # montrer QUI elle a vu — indispensable pour trancher
                        # une identité proposée, et pour qu'un « Inconnu #12 »
                        # soit autre chose qu'un numéro. ~3 Ko, émis deux ou
                        # trois fois par piste seulement (voir SuiviSignatures),
                        # pas à chaque image.
                        "vignette": self._vignette(decoupe),
                    },
                ))
        return evenements

    @staticmethod
    def _visage_de_la_personne(visages, boite_personne, decalage_x):
        """Au plus UN visage par personne suivie : celui qui lui appartient.

        La vignette de tête déborde de la boîte de la personne (marge de
        `tete_de`), et deux personnes proches se chevauchent : la vignette de
        l'une contient souvent le visage de l'autre. Tous les visages trouvés
        étaient jusqu'ici émis sous le MÊME `track_id` -- la signature du
        voisin pouvait donc nommer la mauvaise personne, et lui imputer ses
        infractions EPI. On ne garde que le visage dont le centre tombe dans
        la boîte de la personne, le plus proche de son axe vertical ; aucun
        s'il n'y en a pas, plutôt qu'un visage douteux.
        """
        x1, _, x2, _ = (int(v) for v in boite_personne)
        axe = (x1 + x2) / 2
        candidats = []
        for v in visages:
            cx = decalage_x + (v["box"][0] + v["box"][2]) / 2
            if x1 <= cx <= x2:
                candidats.append((abs(cx - axe), v))
        if not candidats:
            return []
        return [min(candidats, key=lambda c: c[0])[1]]

    @staticmethod
    def _vignette(decoupe, cote=128, qualite_jpeg=80):
        """Vignette JPEG du visage, encodée en base64. Chaîne vide si échec.

        128 px : assez pour qu'un opérateur reconnaisse quelqu'un à l'écran, pas
        assez pour ré-identifier hors du système. Un échec d'encodage ne doit
        jamais faire perdre la signature, qui est la donnée utile — d'où le
        repli sur une chaîne vide plutôt qu'une exception.
        """
        try:
            h, l = decoupe.shape[:2]
            if h == 0 or l == 0:
                return ""
            facteur = cote / max(h, l)
            if facteur < 1.0:
                decoupe = cv2.resize(decoupe, (max(1, int(l * facteur)),
                                               max(1, int(h * facteur))),
                                     interpolation=cv2.INTER_AREA)
            ok, tampon = cv2.imencode(".jpg", decoupe,
                                      [int(cv2.IMWRITE_JPEG_QUALITY), qualite_jpeg])
            return base64.b64encode(tampon).decode("ascii") if ok else ""
        except Exception:      # noqa: BLE001
            return ""

    def draw(self, frame):
        for (x1, y1, x2, y2), track_id, note in self._derniers:
            couleur = (0, 220, 220) if note >= self.qualite_minimale else (120, 120, 120)
            cv2.rectangle(frame, (x1, y1), (x2, y2), couleur, 2)
            cv2.putText(frame, f"#{track_id} q={note:.2f}", (x1, max(14, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, couleur, 1)
        return frame


def construire_analyseurs(args, config) -> list[Analyseur]:
    """Instancie les analyseurs disponibles, en ignorant proprement les modèles absents."""
    desactives = {s.strip() for s in args.disable.split(",") if s.strip()}
    analyseurs: list[Analyseur] = []

    def ajouter(nom, fabrique):
        if nom in desactives:
            log.info("module %-10s desactive (--disable)", nom)
            return
        try:
            a = fabrique()
        except FileNotFoundError as e:
            log.warning("module %-10s indisponible : %s", nom, e)
            return
        except Exception as e:  # modèle corrompu, dépendance OCR manquante...
            log.warning("module %-10s indisponible : %s: %s", nom, type(e).__name__, e)
            return
        analyseurs.append(a)
        log.info("module %-10s actif (1 frame sur %d)", nom, a.every)

    def existant(p: Path) -> Path:
        if not p.exists():
            raise FileNotFoundError(str(p))
        return p

    ajouter("general", lambda: AnalyseurGeneral(
        str(existant(SUITE / "models" / config.MODEL_GENERAL)), config.CONF_THRESHOLD, args.imgsz))
    ajouter("ligne", lambda: AnalyseurLigne(config.LINE_START, config.LINE_END))
    def _chute():
        poids_dedie = getattr(config, "MODEL_FALL", None)
        chemin_dedie = SUITE / poids_dedie if poids_dedie else None
        if chemin_dedie and chemin_dedie.exists():
            return AnalyseurChute(poids_dedie=str(chemin_dedie), imgsz=args.imgsz, every=args.every_pose)
        return AnalyseurChute(poids_pose=str(existant(SUITE / "models" / config.MODEL_POSE)),
                              imgsz=args.imgsz, every=args.every_pose,
                              ratio=config.FALL_ASPECT_RATIO_THRESHOLD, angle=config.FALL_ANGLE_THRESHOLD_DEG)
    ajouter("chute", _chute)
    ajouter("objet_abandonne", lambda: AnalyseurObjetAbandonne(delai_s=args.delai_abandon))

    def _foule():
        calib = None
        if args.calibration_sol:
            from calibration_sol import CalibrationSol
            calib = CalibrationSol.depuis_fichier(existant(Path(args.calibration_sol)))
        zone = _points(args.zone_foule)
        if calib is None:
            log.warning("foule : sans --calibration-sol, la densite en pers/m2 "
                        "n'est pas calculable ; seul l'effectif est publie")
        return AnalyseurFoule(seuil_densite=args.seuil_densite,
                              seuil_effectif=args.seuil_effectif,
                              zone=zone, calibration=calib)
    ajouter("foule", _foule)
    ajouter("feu", lambda: AnalyseurFeu(
        str(existant(SUITE / config.MODEL_FIRE_SMOKE)), config.CONF_THRESHOLD, args.every_fire))
    ajouter("lpr", lambda: AnalyseurPlaque(
        str(existant(SUITE / config.MODEL_PLATE)), config.CONF_THRESHOLD, args.every_lpr,
        mode=args.lpr_retention))

    def _porte():
        from module_door_classifier import load_trained_door_classifier
        # chemins absolus (via SUITE) des deux cotes : le pipeline unifie doit
        # pouvoir etre lance depuis n'importe quel repertoire, contrairement a
        # surveillance_suite/main.py qui suppose cwd == surveillance_suite/.
        clf = load_trained_door_classifier(
            model_path=str(existant(SUITE / "models/door_classifier.pt")),
            roi_path=str(existant(SUITE / "data/dataset/roi.json")),
            smoothing_window=config.DOOR_CLASSIFIER_SMOOTHING,
            min_confidence=config.DOOR_CLASSIFIER_MIN_CONF)
        if clf is None:
            raise FileNotFoundError("classifieur porte non chargeable")
        return AnalyseurPorte(clf, args.every_door)
    ajouter("porte", _porte)

    def _epi():
        # Les modeles dedies sont charges s'ils sont presents, sauf refus
        # explicite. Absents, la cascade retombe sur ppe_detector.pt : le moteur
        # demarre toujours, avec les scores publies pour ce modele seul.
        def dedie(nom: str, refuse: bool) -> str | None:
            chemin = ROOT / "ppe_detection/models" / nom
            return None if refuse or not chemin.exists() else str(chemin)

        # `ppe_complement.pt` (M2) n'est PLUS charge par defaut depuis le
        # 2026-08-19. Ses six classes sont desormais toutes devancees dans la
        # cascade -- casque par M4, gants et lunettes par M5, masque et gilet
        # par M3, chaussures par M6 -- et il n'a aucune classe negative, donc il
        # ne peut jamais signaler une infraction. Son seul apport unique etait
        # `safety_shoe`, que M6 couvre en sachant en plus dire l'absence.
        # Le garder coutait une passe d'inference par image pour rien. Le plan
        # v8 §7 prevoyait ce retrait des l'acceptation de M6.
        return AnalyseurEPI(
            str(existant(ROOT / "ppe_detection/models/ppe_detector.pt")),
            dedie("ppe_complement.pt", not args.avec_complement),
            dedie("masque_gilet.pt", args.sans_masque_gilet),
            dedie("epi_casque.pt", args.sans_casque),
            dedie("epi_gants_lunettes.pt", args.sans_gants_lunettes),
            dedie("epi_chaussures.pt", args.sans_chaussures),
            args.imgsz, args.every_epi, ancrage=not args.sans_ancrage_epi)
    ajouter("epi", _epi)

    # La reconnaissance faciale vient EN DERNIER, et ce n'est pas indifférent :
    # `construire_analyseurs` conserve l'ordre, et la boucle principale exécute
    # les analyseurs dans cet ordre. `AnalyseurVisage` consomme
    # `ctx["personnes"]`, que `AnalyseurGeneral` vient de remplir ; le placer
    # avant lui l'aurait fait travailler sur le contexte de l'image précédente.
    if args.visage:
        ajouter("visage", lambda: AnalyseurVisage(
            paquet=args.visage_paquet, seuil=args.visage_seuil,
            gpu=not args.visage_cpu, qualite_minimale=args.visage_qualite))

    return analyseurs


def _points(texte: str | None):
    """Lit une liste de points « x1,y1;x2,y2;… » en pixels.

    Format textuel plutôt qu'un fichier : une zone se règle au déploiement, et
    doit pouvoir passer par une variable d'environnement dans un conteneur.
    """
    if not texte:
        return None
    pts = []
    for couple in texte.split(";"):
        couple = couple.strip()
        if not couple:
            continue
        try:
            x, y = (float(v) for v in couple.split(","))
        except ValueError:
            raise SystemExit(f"zone invalide : « {couple} » n'est pas « x,y »")
        pts.append((x, y))
    if len(pts) < 3:
        raise SystemExit("une zone demande au moins 3 points")
    return pts


def configurer_threads(budget: int) -> int:
    """Répartit le budget de cœurs du processus entre les modèles. Rend les
    threads accordés à chaque appel de modèle.

    Par défaut PyTorch donne TOUS les cœurs de la machine à chaque appel de
    modèle, et chaque fil qui appelle un modèle ouvre sa propre équipe de
    threads. Avec la cascade EPI (cinq modèles simultanés), les analyseurs du
    pool et le suivi, cela faisait plusieurs dizaines de threads de calcul pour
    douze cœurs : ils passaient leur temps à se céder la place (1,27 million de
    changements de contexte pour 60 images, mesure du 2026-09-23). Sur un
    serveur à plusieurs caméras, chaque processus moteur en faisait autant.

    Un quart du budget par appel : c'est le réglage retenu par la mesure
    (cinq modèles à 3 threads sur 12 cœurs, 2,6 fois plus rapide que douze
    threads à la suite), et il garde environ quatre appels simultanés dans le
    budget. Le calcul numérique ne change pas : seul l'ordonnancement change.
    """
    import torch
    par_modele = max(1, budget // 4)
    torch.set_num_threads(par_modele)
    cv2.setNumThreads(par_modele)
    return par_modele


def _coeurs_disponibles() -> int:
    """Cœurs réellement accordés au processus (affinité, `cpuset` du
    conteneur), et non ceux de la machine."""
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def _env(nom: str, defaut, conv=str):
    """Valeur de configuration issue de l'environnement, sinon le défaut.

    Un client doit pouvoir ajuster un seuil ou un chemin sans éditer de code
    Python. Toutes les options ci-dessous acceptent donc une variable
    d'environnement `MOTEUR_*`, que l'argument en ligne de commande surcharge.
    """
    v = os.environ.get(f"MOTEUR_{nom.upper()}")
    if v is None:
        return defaut
    try:
        return conv(v)
    except (TypeError, ValueError):
        log.warning("MOTEUR_%s='%s' invalide, valeur par defaut %r conservee", nom.upper(), v, defaut)
        return defaut


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=_env("source", "0"),
                    help="0 = webcam, chemin video, ou URL RTSP")
    ap.add_argument("--imgsz", type=int, default=_env("imgsz", 480, int))
    ap.add_argument("--every-epi", type=int, default=_env("every_epi", 3, int))
    ap.add_argument("--every-pose", type=int, default=_env("every_pose", 2, int))
    ap.add_argument("--every-fire", type=int, default=_env("every_fire", 5, int))
    ap.add_argument("--delai-abandon", type=float, default=_env("delai_abandon", 30.0, float),
                    help="secondes sans personne a proximite avant de signaler "
                         "un objet abandonne (defaut 30)")
    # Foule -- tous ces seuils dependent du site et de la reglementation
    # applicable ; aucune valeur n'est universelle, d'ou le reglage externe.
    ap.add_argument("--calibration-sol", default=_env("calibration_sol", "", str),
                    help="JSON de calibration (voir outils/calibrer_sol.py). "
                         "Sans lui, la densite en pers/m2 n'est pas calculable")
    ap.add_argument("--zone-foule", default=_env("zone_foule", "", str),
                    help="polygone de la zone surveillee, « x1,y1;x2,y2;... » "
                         "en pixels. Defaut : toute l'image")
    ap.add_argument("--seuil-densite", type=float, default=_env("seuil_densite", 2.0, float),
                    help="personnes/m2 declenchant l'alerte (defaut 2.0, "
                         "soit 10 personnes pour 5 m2). 0 pour desactiver")
    ap.add_argument("--seuil-effectif", type=int, default=_env("seuil_effectif", 0, int),
                    help="nombre de personnes declenchant l'alerte, "
                         "independamment de la surface. 0 pour desactiver")
    ap.add_argument("--every-lpr", type=int, default=_env("every_lpr", 10, int))
    ap.add_argument("--every-door", type=int, default=_env("every_door", 3, int))
    ap.add_argument("--conf", type=float, default=_env("conf", None, float),
                    help="seuil de confiance global (surcharge config.CONF_THRESHOLD)")
    ap.add_argument("--log-level", default=_env("log_level", "INFO"))
    ap.add_argument("--log-json", action="store_true", default=_env("log_json", False, lambda v: v == "1"))
    ap.add_argument("--max-echecs", type=int, default=_env("max_echecs", 5, int),
                    help="echecs consecutifs avant desactivation d'un module")
    ap.add_argument("--lpr-retention", choices=AnalyseurPlaque.MODES,
                    default=_env("lpr_retention", "pseudonymise"),
                    help="ce que le moteur transmet des plaques lues (defaut : pseudonymise)")
    # `--sans-gants` etait un nom herite et trompeur : il ne retirait pas les
    # gants mais `ppe_complement.pt` en entier. Conserve en alias sans effet
    # pour ne pas casser les scripts existants, puisque ce modele n'est de
    # toute facon plus charge par defaut.
    ap.add_argument("--sans-gants", action="store_true",
                    help="(obsolete, sans effet) ppe_complement.pt n'est plus charge par "
                         "defaut depuis le 2026-08-19 ; utiliser --sans-gants-lunettes pour "
                         "retirer le modele qui sert reellement les gants")
    ap.add_argument("--avec-complement", action="store_true",
                    help="recharge ppe_complement.pt (M2) dans la cascade. Il est redondant "
                         "sur ses six classes et n'a aucune classe negative : a n'utiliser "
                         "que pour comparer avec l'ancien comportement")
    ap.add_argument("--sans-masque-gilet", action="store_true",
                    help="retire masque_gilet.pt de la cascade EPI (revient a ppe_detector.pt "
                         "seul pour ces deux concepts, moins fiable -- voir ppe_taxonomy.py)")
    ap.add_argument("--sans-casque", action="store_true",
                    help="retire epi_casque.pt de la cascade EPI (ppe_detector.pt seul retombe "
                         "a 65 %% / 72 %% de detection de scene -- voir ppe_taxonomy.py)")
    ap.add_argument("--sans-gants-lunettes", action="store_true",
                    help="retire epi_gants_lunettes.pt de la cascade EPI (revient a "
                         "ppe_detector.pt seul pour gants et lunettes)")
    ap.add_argument("--verifier-gpu", action="store_true",
                    default=_env("verifier_gpu", False, lambda v: str(v).lower()
                                 in ("1", "true", "oui", "yes")),
                    help="refuse de demarrer si CUDA est indisponible. A activer en "
                         "production GPU : sans cela un repli silencieux sur CPU divise "
                         "la cadence par dix sans qu'aucun signal ne l'indique")
    ap.add_argument("--sans-chaussures", action="store_true",
                    help="retire epi_chaussures.pt de la cascade EPI ; le concept "
                         "'chaussures' n'est alors PLUS COUVERT DU TOUT, aucun autre modele "
                         "charge par defaut ne le detecte")
    ap.add_argument("--disable", default="", help="analyseurs a desactiver, separes par des virgules")
    # Headless par defaut : c'est le mode de production. L'affichage reste
    # disponible a la demande pour les tests, les demonstrations client et le
    # diagnostic visuel d'une scene.
    ap.add_argument("--display", action="store_true",
                    help="affiche la video annotee (test/demo ; production = headless)")
    ap.add_argument("--events", type=Path, default=_env("events", None, Path),
                    help="fichier JSONL des evenements")
    ap.add_argument("--webhook", default=_env("webhook", None),
                    help="URL POST recevant chaque evenement en JSON")
    # Identite -- toujours issue de la configuration, jamais deduite du flux.
    ap.add_argument("--camera-id", default=_env("camera_id", "", str),
                    help="identifiant de la camera, appose sur chaque evenement. "
                         "INDISPENSABLE des qu'un site compte plusieurs cameras")
    ap.add_argument("--site-id", default=_env("site_id", "", str),
                    help="identifiant du site, appose sur chaque evenement")
    # Livraison garantie
    ap.add_argument("--journal-livraison", type=Path,
                    default=_env("journal_livraison", None, Path),
                    help="journal des evenements a livrer (defaut : a cote du "
                         "processus). Permet la reprise apres coupure ou redemarrage")
    ap.add_argument("--webhook-jeton", default=_env("webhook_jeton", "", str),
                    help="jeton porteur envoye en en-tete Authorization")
    ap.add_argument("--webhook-tentatives", type=int,
                    default=_env("webhook_tentatives", 8, int),
                    help="tentatives avant abandon journalise (defaut 8)")
    # Qualification -- derriere un drapeau, pour comparer en parallele et
    # revenir en arriere sans redeploiement (garde-fou 3 du plan v5).
    ap.add_argument("--sans-ancrage-epi", action="store_true",
                    help="revient a l'association EPI par IoU, sans rejet "
                         "(comportement d'avant le 2026-08-13)")
    # ── Reconnaissance faciale ───────────────────────────────────────────────
    # Désactivée par défaut, délibérément. C'est le seul module qui produise une
    # donnée biométrique : il ne doit jamais s'activer parce qu'on a oublié de
    # le désactiver. Les autres modules suivent la règle inverse (actifs si
    # leurs poids sont là) parce qu'un casque détecté n'engage personne.
    ap.add_argument("--visage", action="store_true",
                    default=_env("visage", False, lambda v: v == "1"),
                    help="active la signature faciale (donnee biometrique : "
                         "verifier la base legale avant d'activer)")
    ap.add_argument("--visage-paquet", default=_env("visage_paquet", "buffalo_l", str),
                    help="paquet InsightFace (buffalo_l par defaut : SCRFD + ArcFace R50)")
    ap.add_argument("--visage-seuil", type=float, default=_env("visage_seuil", 0.5, float),
                    help="seuil de detection d'un visage")
    ap.add_argument("--visage-qualite", type=float,
                    default=_env("visage_qualite", 0.35, float),
                    help="qualite minimale d'un visage pour que sa signature soit emise. "
                         "Sous ce niveau la signature est instable et polluerait la recherche")
    ap.add_argument("--visage-cpu", action="store_true",
                    help="force la reconnaissance faciale sur CPU (essais uniquement : "
                         "ArcFace sur CPU ne tient pas le temps reel)")
    ap.add_argument("--sans-qualification", action="store_true",
                    help="desactive la machine a etats : un evenement par image, "
                         "comportement d'avant le 2026-08-13")
    ap.add_argument("--seuil-probable", type=int, default=_env("seuil_probable", 3, int),
                    help="observations avant de passer de suspicion a probable")
    ap.add_argument("--seuil-confirme", type=int, default=_env("seuil_confirme", 8, int),
                    help="observations avant de passer de probable a confirme")
    ap.add_argument("--fin-evenement-s", type=float, default=_env("fin_evenement_s", 5.0, float),
                    help="secondes sans observation avant de clore un evenement")
    ap.add_argument("--boucler", action="store_true",
                    help="relit un fichier video en boucle (tests d'endurance)")
    ap.add_argument("--max-frames", type=int, default=_env("max_frames", 0, int), help="0 = illimite")
    ap.add_argument("--workers", type=int, default=_env("workers", 4, int),
                    help="fils d'analyse minimum ; le pool en ouvre au moins un par "
                         "analyseur a modele, pour qu'aucun n'attende derriere un autre")
    ap.add_argument("--threads", type=int, default=_env("threads", 0, int),
                    help="coeurs CPU accordes a ce processus (0 = tous ceux qu'il voit). "
                         "Le superviseur le fixe a sa part quand plusieurs cameras "
                         "partagent la machine")
    ap.add_argument("--synchrone", action="store_true",
                    default=_env("synchrone", False, lambda v: v == "1"),
                    help="sur un flux direct, attend la fin de tous les analyseurs avant "
                         "l'image suivante (comportement d'avant le 2026-09-23). Par "
                         "defaut, les analyseurs lourds tournent en arriere-plan et le "
                         "suivi continue a chaque image")
    ap.add_argument("--health-port", type=int, default=_env("health_port", 0, int),
                    help="expose GET /health et GET /metrics sur ce port (0 = desactive)")
    ap.add_argument("--health-adresse", default=_env("health_adresse", "127.0.0.1", str),
                    help="interface d'ecoute de /health et /metrics. Boucle locale par "
                         "defaut : ces points publient la source du flux, l'identite du "
                         "site et l'etat du moteur. Mettre 0.0.0.0 pour un sondage "
                         "externe (conteneur, Kubernetes), avec --health-jeton")
    ap.add_argument("--health-jeton", default=_env("health_jeton", "", str),
                    help="exige un en-tete Authorization: Bearer <jeton> sur /health "
                         "et /metrics")
    args = ap.parse_args()

    configurer_logs(args.log_level, args.log_json, args.camera_id, args.site_id)

    budget = args.threads if args.threads > 0 else _coeurs_disponibles()
    par_modele = configurer_threads(budget)
    log.info("budget CPU : %d coeur(s), %d thread(s) par appel de modele",
             budget, par_modele)

    # Un moteur qui retombe sur le CPU ne produit aucune erreur : il tourne, il
    # detecte, il est simplement dix fois trop lent. Le retard se voit maintenant
    # (`retard_flux_s` sur /health, horodatage pris a la capture), mais encore
    # faut-il regarder : refuser de demarrer reste la seule facon de ne pas
    # decouvrir la meprise apres coup.
    # Mesure du 2026-08-19 : la cascade complete tient 0,91 image/s sur CPU.
    if args.verifier_gpu:
        try:
            import torch
            dispo = torch.cuda.is_available()
            nom = torch.cuda.get_device_name(0) if dispo else "aucun"
        except Exception as e:                       # torch absent ou casse
            log.error("verification GPU impossible : %s: %s", type(e).__name__, e)
            return 2
        if not dispo:
            log.error("GPU demande (--verifier-gpu) mais CUDA indisponible. "
                      "Causes usuelles : conteneur lance sans `--gpus all`, ou "
                      "image construite avec les roues torch '+cpu' "
                      "(voir requirements-service-gpu.txt). Arret.")
            return 2
        log.info("GPU actif : %s", nom)

    import config
    # Le seuil global reste dans config.py (valeur de reference documentee),
    # mais devient surchargeable sans editer le fichier.
    if args.conf is not None:
        config.CONF_THRESHOLD = args.conf

    # Arret propre : en conteneur ou sous systemd, l'arret arrive par signal, pas
    # par une touche clavier. On sort de la boucle a la fin de l'image courante
    # pour liberer camera et threads proprement, plutot que d'etre tue net.
    arret = threading.Event()

    def _demander_arret(signum, _frame):
        log.info("Signal %s recu, arret en cours...", signal.Signals(signum).name)
        arret.set()

    signal.signal(signal.SIGINT, _demander_arret)
    signal.signal(signal.SIGTERM, _demander_arret)

    sorties: list[Sortie] = []
    if args.events:
        sorties.append(SortieJSONL(args.events))
    webhook = None
    if args.webhook:
        webhook = SortieWebhook(args.webhook, journal=args.journal_livraison,
                                jeton=args.webhook_jeton,
                                tentatives_max=args.webhook_tentatives)
        sorties.append(webhook)
    if not sorties:
        sorties.append(SortieConsole())
    if not args.camera_id:
        # Un identifiant vide passe silencieusement en mono-camera et devient un
        # defaut invisible des la deuxieme : mieux vaut le dire au demarrage.
        log.warning("aucun --camera-id : les evenements ne diront pas de quelle "
                    "camera ils viennent (indispensable en multi-camera)")
    qualif = None
    if not args.sans_qualification:
        from qualification import MachineEtats
        qualif = MachineEtats(seuil_probable=args.seuil_probable,
                              seuil_confirme=args.seuil_confirme,
                              fin_apres_s=args.fin_evenement_s)
        log.info("qualification active : suspicion -> probable (%d obs) -> confirme (%d obs)",
                 args.seuil_probable, args.seuil_confirme)
    else:
        log.warning("qualification desactivee : un evenement par image, "
                    "~137 600 par jour et par camera (mesure)")
    bus = BusEvenements(sorties, camera_id=args.camera_id, site_id=args.site_id,
                        qualification=qualif)

    source = int(args.source) if args.source.isdigit() else args.source
    cap = CaptureRobuste(source, bus=bus, boucler=args.boucler)
    if not cap.connecte:
        # Un fichier absent est une erreur de configuration : echouer tout de
        # suite est plus utile que boucler indefiniment. Une camera ou un flux
        # reseau, en revanche, peut simplement ne pas etre encore pret au
        # demarrage du service (ordre de demarrage, caméra qui redemarre) : on
        # laisse la boucle de reconnexion faire son travail.
        if cap.est_fichier:
            raise SystemExit(f"Fichier video introuvable ou illisible : {source}")
        log.warning("Source video pas encore disponible (%s) : demarrage quand meme, "
                    "reconnexion en cours...", source)

    log.info("Initialisation des analyseurs...")
    analyseurs = construire_analyseurs(args, config)
    if not analyseurs:
        raise SystemExit("Aucun analyseur disponible.")

    # `general` alimente le contexte des autres : il est exécuté avant eux, pas
    # en parallèle, sinon `ligne` et `epi` travailleraient sur le suivi de la
    # frame précédente.
    general = next((a for a in analyseurs if a.nom == "general"), None)
    autres = [a for a in analyseurs if a is not general]
    # Sans modèle, un analyseur ne coûte que quelques comparaisons de boîtes :
    # il tourne dans la boucle, à chaque image, sur le suivi qui vient d'être
    # calculé. Les autres passent par le pool.
    legers = [a for a in autres if not a.inference]
    lourds = [a for a in autres if a.inference]

    # Ordonnancement asynchrone sur flux direct.
    #
    # Jusqu'ici chaque image attendait la fin de TOUS les analyseurs dus avant
    # de passer à la suivante : une image sur trois, la boucle restait bloquée
    # près d'une seconde sur la cascade EPI, et le suivi des personnes avec
    # elle. Or le suivi est ce qui porte l'identité (`track_id`) dont
    # dépendent l'historique EPI par personne, le franchissement de ligne et le
    # rattachement d'un visage : une piste qui saute des images perd plus
    # souvent sa cible et change d'identifiant.
    #
    # Désormais, sur un flux direct, un analyseur lourd reçoit l'image ET le
    # contexte de suivi de la même image, tourne en arrière-plan, et ses
    # évènements sont publiés dès qu'il a fini, datés de SON image. Pendant ce
    # temps, le suivi continue. S'il est encore occupé quand son tour revient,
    # il saute ce tour au lieu de bloquer tout le monde.
    #
    # Un fichier reste traité de façon synchrone et reproductible : c'est ce
    # que les mesures et les tests attendent, et rien n'y presse.
    asynchrone = not cap.est_fichier and not args.synchrone
    pool = ThreadPoolExecutor(max_workers=max(args.workers, len(lourds), 1),
                              thread_name_prefix="analyse")
    log.info("ordonnancement %s : %d analyseur(s) a modele, %d sans modele",
             "asynchrone (flux direct)" if asynchrone else "synchrone",
             len(lourds), len(legers))

    etat_sante = EtatSante([a.nom for a in analyseurs])
    if args.health_port:
        etat_sante.demarrer_serveur(args.health_port, args.health_adresse,
                                    args.health_jeton)

    frame_idx = 0
    t_debut = time.time()
    couts: dict[str, float] = {a.nom: 0.0 for a in analyseurs}
    executions: dict[str, int] = {a.nom: 0 for a in analyseurs}
    echecs: dict[str, int] = {a.nom: 0 for a in analyseurs}
    desactives_a_chaud: set[str] = set()
    dernier_lancement: dict[str, int] = {a.nom: 0 for a in autres}
    en_cours: dict[str, tuple] = {}      # nom -> (future, analyseur)
    MAX_ECHECS = args.max_echecs
    fps_liss = 0.0
    prev = time.time()

    def chronometre(a, frame, ctx):
        t0 = time.time()
        evs = a.process(frame, ctx)
        return evs, time.time() - t0

    def traiter(a, obtenir):
        """Publie les évènements d'un analyseur, ou comptabilise son échec.

        Toujours appelé depuis la boucle principale : le bus, sa machine à
        états et ses sorties ne voient jamais deux fils à la fois.
        """
        try:
            evs, duree = obtenir()
            couts[a.nom] += duree
            executions[a.nom] += 1
            for ev in evs:
                bus.publier(ev)
            echecs[a.nom] = 0
        except Exception as e:
            # Un module qui échoue ne doit pas emporter les autres : on
            # journalise, on continue, et s'il échoue de façon répétée on
            # le désactive en le signalant. Mieux vaut cinq modules qui
            # fonctionnent qu'un pipeline entier arrêté par le sixième.
            echecs[a.nom] += 1
            log.error("module %s en echec (%d/%d): %s: %s",
                      a.nom, echecs[a.nom], MAX_ECHECS, type(e).__name__, e)
            if echecs[a.nom] >= MAX_ECHECS:
                desactives_a_chaud.add(a.nom)
                log.error("module %s desactive apres %d echecs consecutifs",
                          a.nom, MAX_ECHECS)
                bus.publier(Evenement(
                    time.time(), frame_idx, "moteur", "module_desactive",
                    f"Module {a.nom} désactivé après {MAX_ECHECS} échecs consécutifs"))

    def recolter(attendre: bool):
        # Dans l'ordre de lancement : à attente égale, la publication reste
        # celle du comportement synchrone.
        for nom, (fut, a) in list(en_cours.items()):
            if attendre or fut.done():
                del en_cours[nom]
                traiter(a, fut.result)

    try:
        while not arret.is_set():
            frame = cap.lire(arret)
            if frame is None:
                break  # source epuisee (fichier termine) ou arret demande
            frame_idx += 1
            # Date de capture, pas date de traitement : c'est la seule qui
            # situe la scene dans le temps pour la plateforme aval.
            ctx = {"frame": frame_idx, "t": cap.t_capture or time.time()}

            recolter(attendre=False)

            if general is not None:
                t0 = time.time()
                general.process(frame, ctx)
                couts["general"] += time.time() - t0
                executions["general"] += 1

            for a in legers:
                if a.nom in desactives_a_chaud or frame_idx - dernier_lancement[a.nom] < a.every:
                    continue
                dernier_lancement[a.nom] = frame_idx
                traiter(a, lambda a=a: chronometre(a, frame, ctx))

            for a in lourds:
                if (a.nom in desactives_a_chaud or a.nom in en_cours
                        or frame_idx - dernier_lancement[a.nom] < a.every):
                    continue
                dernier_lancement[a.nom] = frame_idx
                en_cours[a.nom] = (pool.submit(chronometre, a, frame, ctx), a)

            if not asynchrone:
                recolter(attendre=True)

            now = time.time()
            fps = 1.0 / max(now - prev, 1e-6)
            prev = now
            fps_liss = fps if frame_idx == 1 else 0.9 * fps_liss + 0.1 * fps

            etat_sante.update(frame_idx, fps_liss, bus, cap)

            if args.display:
                # Dessiner sur une copie : en asynchrone, un analyseur lit
                # peut-être encore cette image, et y tracer des rectangles
                # fausserait sa détection.
                if asynchrone:
                    frame = frame.copy()
                for a in analyseurs:
                    frame = a.draw(frame)
                cv2.putText(frame, f"FPS:{fps_liss:.1f}  frame {frame_idx}", (10, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.imshow("Surveillance unifiee (EPI + securite)", frame)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
            elif frame_idx % 20 == 0:
                log.info("frame %d  fps=%.2f  evenements=%d", frame_idx, fps_liss, bus.total)

            if args.max_frames and frame_idx >= args.max_frames:
                break
    finally:
        # Les analyses encore en vol portent peut-être un évènement : on les
        # laisse finir et on le publie avant de fermer le bus.
        try:
            recolter(attendre=True)
        except Exception as e:  # noqa: BLE001 -- l'arret doit aller au bout
            log.error("recolte finale impossible : %s: %s", type(e).__name__, e)
        cap.liberer()
        if args.display:
            cv2.destroyAllWindows()
        pool.shutdown(wait=True)
        etat_sante.arreter_serveur()
        bus.fermer()

    duree = time.time() - t_debut
    log.info("%d frames en %.1fs -> %.2f FPS moyen", frame_idx, duree, frame_idx / max(duree, 1e-6))
    log.info("%d evenements%s", bus.total, f" -> {args.events}" if args.events else "")
    for nom, c in sorted(couts.items(), key=lambda kv: -kv[1]):
        # Le nombre d'executions dit ce que le cout seul cache : en asynchrone,
        # un analyseur trop lent pour sa cadence saute des tours.
        log.info("cout %-10s %6.1fs  %5.1f%%  %d execution(s)", nom, c,
                 100 * c / max(duree, 1e-6), executions[nom])


if __name__ == "__main__":
    # `sys.exit(main())` et non `main()` : la valeur de retour est le code de
    # sortie du processus, et c'est le seul signal qu'un orchestrateur sait
    # lire. Appeler `main()` sans le propager faisait sortir en 0 meme sur
    # refus de demarrage -- un conteneur en echec de configuration etait alors
    # rapporte comme s'etant termine normalement, et jamais relance.
    sys.exit(main())
