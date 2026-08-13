#!/usr/bin/env python3
"""P6 — Point d'entrée vidéo unique pour les deux chaînes de traitement.

Situation corrigée
------------------
`ppe_detection/scripts/ppe_dual_model_backup2.py` et `surveillance_suite/main.py`
ouvrent chacun leur propre `cv2.VideoCapture`. Sur une caméra de surveillance
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
import json
import logging
import os
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict, field
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "surveillance_suite"
sys.path.insert(0, str(SUITE))
sys.path.insert(0, str(SUITE / "modules"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ppe_taxonomy as tax  # noqa: E402


# ── Journalisation ───────────────────────────────────────────────────────────
# Des `print` ne portent ni niveau, ni horodatage, ni origine : impossible de
# diagnostiquer un incident client à distance, ni de filtrer le bruit. Le format
# reste lisible en console ; `--log-json` produit une ligne JSON par message pour
# les collecteurs de logs.

log = logging.getLogger("moteur")


class FormatJSON(logging.Formatter):
    def format(self, record):
        return json.dumps({
            "t": record.created,
            "niveau": record.levelname,
            "module": record.name,
            "message": record.getMessage(),
        }, ensure_ascii=False)


def configurer_logs(niveau: str, json_mode: bool):
    h = logging.StreamHandler()
    h.setFormatter(FormatJSON() if json_mode
                   else logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s",
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
    """POST HTTP par évènement, en tâche de fond.

    Implémentation par défaut en attendant la décision de protocole : c'est le
    mode d'intégration le plus universel. L'envoi est asynchrone et les échecs
    sont comptés, jamais propagés — le moteur ne doit pas ralentir ni s'arrêter
    parce que le consommateur est momentanément indisponible.
    """

    def __init__(self, url: str, timeout_s: float = 3.0, workers: int = 2):
        self.url = url
        self.timeout_s = timeout_s
        self._pool = ThreadPoolExecutor(max_workers=workers)
        self.echecs = 0

    def _envoyer(self, charge: bytes):
        import urllib.request
        try:
            req = urllib.request.Request(
                self.url, data=charge, headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=self.timeout_s).close()
        except Exception:
            self.echecs += 1

    def emettre(self, ev):
        charge = json.dumps(asdict(ev), ensure_ascii=False, default=_json_defaut).encode()
        self._pool.submit(self._envoyer, charge)

    def fermer(self):
        self._pool.shutdown(wait=True)


class BusEvenements:
    """Collecte les évènements et les distribue aux sorties configurées.

    Séparer la production d'évènements de leur rendu est ce qui rend le pipeline
    utilisable sans écran (serveur, conteneur) : le mode d'affichage ne change
    rien à ce qui est détecté, seulement à ce qui est dessiné.
    """

    def __init__(self, sorties: list[Sortie] | None = None, anti_repetition_s: float = 3.0):
        self.sorties = sorties or [SortieConsole()]
        self._dernier: dict[str, float] = {}
        self._anti_rep = anti_repetition_s
        self.total = 0
        self.dernier_t = 0.0

    def publier(self, ev: Evenement) -> bool:
        cle = f"{ev.source}|{ev.type}|{ev.libelle}"
        if ev.t - self._dernier.get(cle, -1e9) < self._anti_rep:
            return False
        self._dernier[cle] = ev.t
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
            )

    def instantane(self) -> dict:
        with self._verrou:
            e = dict(self._etat)
        e["uptime_s"] = round(time.time() - e.pop("demarre_le"), 1)
        # Le moteur est « sain » s'il voit le flux ET progresse : un flux
        # connecté mais figé (0 image analysée) n'est pas un état sain.
        e["sain"] = bool(e["flux_connecte"] and e["frames"] > 0)
        return e

    def demarrer_serveur(self, port: int):
        from http.server import BaseHTTPRequestHandler, HTTPServer

        etat = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path.rstrip("/") not in ("/health", ""):
                    self.send_error(404)
                    return
                corps = json.dumps(etat.instantane(), ensure_ascii=False).encode()
                self.send_response(200 if etat.instantane()["sain"] else 503)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(corps)))
                self.end_headers()
                self.wfile.write(corps)

            def log_message(self, *a):
                pass  # pas de log par requête : le sondage est périodique

        self._serveur = HTTPServer(("0.0.0.0", port), Handler)
        threading.Thread(target=self._serveur.serve_forever, daemon=True).start()
        log.info("Sante exposee sur http://0.0.0.0:%d/health", port)

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

    def __init__(self, source, bus=None, boucler=False, echecs_avant_perte=5):
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
        self._ouvrir()

    def _ouvrir(self) -> bool:
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.source)
        if isinstance(self.source, int):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
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
        """Retourne une frame, ou None si la source est épuisée / l'arrêt demandé.

        Bloque pendant les tentatives de reconnexion, en vérifiant `arret` pour
        rester interruptible par un signal.
        """
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
        if self.cap is not None:
            self.cap.release()


# ── Analyseurs ───────────────────────────────────────────────────────────────

class Analyseur:
    """Contrat commun. `every` = période d'exécution en frames."""
    nom = "base"
    every = 1

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

    def __init__(self, poids_m1, poids_m2=None, imgsz=480, every=3,
                 hist_n=12, hist_k=4):
        from ultralytics import YOLO
        self.every = every
        self.imgsz = imgsz
        self.m1 = YOLO(poids_m1)
        self.m2 = YOLO(poids_m2) if poids_m2 else None
        erreurs = tax.verifier_coherence(
            self.m1.names, self.m2.names if self.m2 else dict(enumerate(tax.M2)))
        if self.m2 and erreurs:
            raise RuntimeError("Taxonomie incohérente avec les modèles chargés :\n  " +
                               "\n  ".join(erreurs))
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

    def process(self, frame, ctx):
        seuil_bas = min(c.conf_min for t in tax.TABLES.values() for c in t.values())
        brutes = []
        for nom_modele, model in (("best.pt", self.m1), ("best_gloves.pt", self.m2)):
            if model is None:
                continue
            for b in model.predict(frame, conf=seuil_bas, imgsz=self.imgsz, verbose=False)[0].boxes:
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
        for d in self.dets:
            if not d.epi or not boites:
                continue
            cle = max(boites, key=lambda c: tax._iou(d.box, boites[c]))
            (porte if d.porte else absent)[cle].add(d.epi)

        self._oublier_absents(set(boites), ctx["t"])

        # Lissage temporel : un EPI n'est réputé porté que s'il a été vu dans au
        # moins hist_k des hist_n dernières frames analysées. Indispensable avec
        # des seuils de confiance bas, sinon chaque frame bruitée bascule l'état.
        evs = []
        self.violations = {}
        for rang, (cle, pid, box) in enumerate(cles, start=1):
            v = []
            for epi in tax.EPI_CANONIQUES:
                h = self._histo(cle, epi)
                h.append(epi in porte[cle])
                del h[0]
                stable = sum(h) >= self.hist_k
                if tax.EPI_OBLIGATOIRES[epi] and not stable:
                    v.append(tax.LIBELLES_FR[epi][1])
            for epi in absent[cle]:
                lib = tax.LIBELLES_FR[epi][1]
                if lib not in v:
                    v.append(lib)
            if v:
                self.violations[cle] = v
                # Le libellé porte l'identifiant de suivi quand il existe : c'est
                # lui qui permet à la plateforme de recoller les évènements
                # successifs à une même personne. Sans suivi, on retombe sur un
                # numéro d'ordre, valable pour la seule image courante.
                qui = f"Personne #{pid}" if pid >= 0 else f"Personne {rang}"
                for lib in v:
                    evs.append(Evenement(ctx["t"], ctx["frame"], self.nom, "violation_epi",
                                         f"{qui} — {lib}", box=box,
                                         extra={"track_id": pid, "suivi": pid >= 0}))
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
        evs = []
        for f in self.falls:
            x1, y1, x2, y2 = f["box"]
            cle = f"{int((x1 + x2) / 160)}:{int((y1 + y2) / 160)}"
            if self.confirmation.confirmer(cle, bool(f["fallen"])) and f["fallen"]:
                evs.append(Evenement(ctx["t"], ctx["frame"], self.nom, "chute",
                                     "PERSONNE AU SOL", box=f["box"]))
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
        evs = []
        for d in self.res:
            x1, y1, x2, y2 = d["box"]
            cle = f"{d['label']}:{int((x1 + x2) / 160)}:{int((y1 + y2) / 160)}"
            if self.confirmation.confirmer(cle, True):
                evs.append(Evenement(ctx["t"], ctx["frame"], self.nom, d["label"],
                                     f"{d['label'].upper()} détecté",
                                     conf=d["conf"], box=tuple(d["box"])))
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

    ajouter("epi", lambda: AnalyseurEPI(
        str(existant(ROOT / "ppe_detection/models/best.pt")),
        None if args.sans_gants else str(existant(ROOT / "ppe_detection/models/best_gloves.pt")),
        args.imgsz, args.every_epi))

    return analyseurs


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
    ap.add_argument("--sans-gants", action="store_true",
                    help="retire best_gloves.pt de la cascade EPI (seul apport : chaussures)")
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
    ap.add_argument("--boucler", action="store_true",
                    help="relit un fichier video en boucle (tests d'endurance)")
    ap.add_argument("--max-frames", type=int, default=_env("max_frames", 0, int), help="0 = illimite")
    ap.add_argument("--workers", type=int, default=_env("workers", 4, int))
    ap.add_argument("--health-port", type=int, default=_env("health_port", 0, int),
                    help="expose GET /health sur ce port (0 = desactive)")
    args = ap.parse_args()

    configurer_logs(args.log_level, args.log_json)

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
    if args.webhook:
        sorties.append(SortieWebhook(args.webhook))
    if not sorties:
        sorties.append(SortieConsole())
    bus = BusEvenements(sorties)

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

    pool = ThreadPoolExecutor(max_workers=args.workers)

    etat_sante = EtatSante([a.nom for a in analyseurs])
    if args.health_port:
        etat_sante.demarrer_serveur(args.health_port)

    # `general` alimente le contexte des autres : il est exécuté avant eux, pas
    # en parallèle, sinon `ligne` et `epi` travailleraient sur le suivi de la
    # frame précédente.
    general = next((a for a in analyseurs if a.nom == "general"), None)
    autres = [a for a in analyseurs if a is not general]

    frame_idx = 0
    t_debut = time.time()
    couts: dict[str, float] = {a.nom: 0.0 for a in analyseurs}
    echecs: dict[str, int] = {a.nom: 0 for a in analyseurs}
    desactives_a_chaud: set[str] = set()
    MAX_ECHECS = args.max_echecs
    fps_liss = 0.0
    prev = time.time()

    try:
        while not arret.is_set():
            frame = cap.lire(arret)
            if frame is None:
                break  # source epuisee (fichier termine) ou arret demande
            frame_idx += 1
            ctx = {"frame": frame_idx, "t": time.time()}

            if general is not None:
                t0 = time.time()
                general.process(frame, ctx)
                couts["general"] += time.time() - t0

            dus = [a for a in autres if frame_idx % a.every == 0 and a.nom not in desactives_a_chaud]
            futures = {pool.submit(a.process, frame, ctx): (a, time.time()) for a in dus}
            for fut, (a, t0) in futures.items():
                try:
                    for ev in fut.result():
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
                couts[a.nom] += time.time() - t0

            now = time.time()
            fps = 1.0 / max(now - prev, 1e-6)
            prev = now
            fps_liss = fps if frame_idx == 1 else 0.9 * fps_liss + 0.1 * fps

            etat_sante.update(frame_idx, fps_liss, bus, cap)

            if args.display:
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
        log.info("cout %-10s %6.1fs  %5.1f%%", nom, c, 100 * c / max(duree, 1e-6))


if __name__ == "__main__":
    main()
