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
import sys
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


class BusEvenements:
    """Collecte les évènements et les écrit en JSONL.

    Séparer la production d'évènements de leur rendu est ce qui rend le pipeline
    utilisable sans écran (serveur, conteneur) : `--no-display` ne change rien à
    ce qui est détecté, seulement à ce qui est dessiné.
    """

    def __init__(self, chemin: Path | None = None, anti_repetition_s: float = 3.0):
        self.chemin = chemin
        self._f = open(chemin, "a", encoding="utf-8") if chemin else None
        self._dernier: dict[str, float] = {}
        self._anti_rep = anti_repetition_s
        self.total = 0

    def publier(self, ev: Evenement) -> bool:
        cle = f"{ev.source}|{ev.type}|{ev.libelle}"
        if ev.t - self._dernier.get(cle, -1e9) < self._anti_rep:
            return False
        self._dernier[cle] = ev.t
        self.total += 1
        if self._f:
            self._f.write(json.dumps(asdict(ev), ensure_ascii=False, default=_json_defaut) + "\n")
            self._f.flush()
        else:
            print(f"[{ev.type.upper()}] {ev.libelle}", flush=True)
        return True

    def fermer(self):
        if self._f:
            self._f.close()


# ── Analyseurs ───────────────────────────────────────────────────────────────

class Analyseur:
    """Contrat commun. `every` = période d'exécution en frames."""
    nom = "base"
    every = 1

    def process(self, frame, ctx: dict) -> list[Evenement]:
        raise NotImplementedError

    def draw(self, frame):
        return frame


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
        ctx["personnes"] = [o["box"] for o in objets if o["label"] == "person"]
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
        self._hist: dict[int, dict[str, list[bool]]] = {}
        self.dets: list[tax.DetectionEPI] = []
        self.violations: dict[int, list[str]] = {}

    def _histo(self, pid, epi):
        h = self._hist.setdefault(pid, {}).setdefault(epi, [False] * self.hist_n)
        return h

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

        personnes = ctx.get("personnes") or []
        if not personnes and self.dets:
            xs = [c for d in self.dets for c in (d.box[0], d.box[2])]
            ys = [c for d in self.dets for c in (d.box[1], d.box[3])]
            personnes = [(min(xs), min(ys), max(xs), max(ys))]

        porte = {i: set() for i in range(len(personnes))}
        absent = {i: set() for i in range(len(personnes))}
        for d in self.dets:
            if not d.epi or not personnes:
                continue
            pid = max(range(len(personnes)), key=lambda i: tax._iou(d.box, personnes[i]))
            (porte if d.porte else absent)[pid].add(d.epi)

        # Lissage temporel : un EPI n'est réputé porté que s'il a été vu dans au
        # moins hist_k des hist_n dernières frames analysées. Indispensable avec
        # des seuils de confiance bas, sinon chaque frame bruitée bascule l'état.
        evs = []
        self.violations = {}
        for pid in range(len(personnes)):
            v = []
            for epi in tax.EPI_CANONIQUES:
                h = self._histo(pid, epi)
                h.append(epi in porte[pid])
                del h[0]
                stable = sum(h) >= self.hist_k
                if tax.EPI_OBLIGATOIRES[epi] and not stable:
                    v.append(tax.LIBELLES_FR[epi][1])
            for epi in absent[pid]:
                lib = tax.LIBELLES_FR[epi][1]
                if lib not in v:
                    v.append(lib)
            if v:
                self.violations[pid] = v
                for lib in v:
                    evs.append(Evenement(ctx["t"], ctx["frame"], self.nom, "violation_epi",
                                         f"Personne {pid + 1} — {lib}",
                                         box=personnes[pid]))
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

    def __init__(self, poids_pose=None, poids_dedie=None, imgsz=480, every=2, ratio=1.4, angle=45):
        from ultralytics import YOLO
        self.imgsz, self.every, self.ratio, self.angle = imgsz, every, ratio, angle
        self.falls = []
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
        return [Evenement(ctx["t"], ctx["frame"], self.nom, "chute", "PERSONNE AU SOL", box=f["box"])
                for f in self.falls if f["fallen"]]

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


class AnalyseurFeu(Analyseur):
    nom = "feu"

    def __init__(self, poids, conf=0.4, every=5):
        from module_fire_smoke import FireSmokeDetector
        self.det = FireSmokeDetector(poids, conf=conf)
        self.every = every
        self.res = []

    def process(self, frame, ctx):
        self.res = self.det.detect(frame)
        return [Evenement(ctx["t"], ctx["frame"], self.nom, d["label"],
                          f"{d['label'].upper()} détecté", conf=d["conf"], box=tuple(d["box"]))
                for d in self.res]

    def draw(self, frame):
        for d in self.res:
            x1, y1, x2, y2 = d["box"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 100, 255), 2)
            cv2.putText(frame, f"{d['label'].upper()} {d['conf']:.2f}", (x1, max(18, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 100, 255), 2)
        return frame


class AnalyseurPlaque(Analyseur):
    nom = "lpr"

    def __init__(self, poids, conf=0.4, every=10):
        from module_lpr import LicensePlateReader
        self.reader = LicensePlateReader(poids, conf=conf)
        self.every = every
        self.res = []

    def process(self, frame, ctx):
        self.res = self.reader.detect_and_read(frame)
        return [Evenement(ctx["t"], ctx["frame"], self.nom, "plaque",
                          p.get("text") or "plaque", box=tuple(p["box"]))
                for p in self.res]

    def draw(self, frame):
        for p in self.res:
            x1, y1, x2, y2 = p["box"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 0, 200), 2)
            cv2.putText(frame, p.get("text", ""), (x1, max(18, y1 - 8)),
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
            print(f"  {nom:10} desactive (--disable)")
            return
        try:
            a = fabrique()
        except FileNotFoundError as e:
            print(f"  {nom:10} indisponible : {e}")
            return
        except Exception as e:  # modèle corrompu, dépendance OCR manquante...
            print(f"  {nom:10} indisponible : {type(e).__name__}: {e}")
            return
        analyseurs.append(a)
        print(f"  {nom:10} actif (1 frame sur {a.every})")

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
    ajouter("feu", lambda: AnalyseurFeu(
        str(existant(SUITE / config.MODEL_FIRE_SMOKE)), config.CONF_THRESHOLD, args.every_fire))
    ajouter("lpr", lambda: AnalyseurPlaque(
        str(existant(SUITE / config.MODEL_PLATE)), config.CONF_THRESHOLD, args.every_lpr))

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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="0", help="0 = webcam, chemin video, ou URL RTSP")
    ap.add_argument("--imgsz", type=int, default=480)
    ap.add_argument("--every-epi", type=int, default=3)
    ap.add_argument("--every-pose", type=int, default=2)
    ap.add_argument("--every-fire", type=int, default=5)
    ap.add_argument("--every-lpr", type=int, default=10)
    ap.add_argument("--every-door", type=int, default=3)
    ap.add_argument("--sans-gants", action="store_true",
                    help="retire best_gloves.pt de la cascade EPI (seul apport : chaussures)")
    ap.add_argument("--disable", default="", help="analyseurs a desactiver, separes par des virgules")
    ap.add_argument("--no-display", action="store_true", help="mode sans affichage (serveur)")
    ap.add_argument("--events", type=Path, default=None, help="fichier JSONL des evenements")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = illimite")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    import config

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Source video inaccessible : {source}")
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print("Analyseurs :")
    analyseurs = construire_analyseurs(args, config)
    if not analyseurs:
        raise SystemExit("Aucun analyseur disponible.")

    bus = BusEvenements(args.events)
    pool = ThreadPoolExecutor(max_workers=args.workers)

    # `general` alimente le contexte des autres : il est exécuté avant eux, pas
    # en parallèle, sinon `ligne` et `epi` travailleraient sur le suivi de la
    # frame précédente.
    general = next((a for a in analyseurs if a.nom == "general"), None)
    autres = [a for a in analyseurs if a is not general]

    frame_idx = 0
    t_debut = time.time()
    couts: dict[str, float] = {a.nom: 0.0 for a in analyseurs}
    fps_liss = 0.0
    prev = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            ctx = {"frame": frame_idx, "t": time.time()}

            if general is not None:
                t0 = time.time()
                general.process(frame, ctx)
                couts["general"] += time.time() - t0

            dus = [a for a in autres if frame_idx % a.every == 0]
            futures = {pool.submit(a.process, frame, ctx): (a, time.time()) for a in dus}
            for fut, (a, t0) in futures.items():
                try:
                    for ev in fut.result():
                        bus.publier(ev)
                except Exception as e:
                    print(f"[{a.nom}] erreur: {type(e).__name__}: {e}", flush=True)
                couts[a.nom] += time.time() - t0

            now = time.time()
            fps = 1.0 / max(now - prev, 1e-6)
            prev = now
            fps_liss = fps if frame_idx == 1 else 0.9 * fps_liss + 0.1 * fps

            if not args.no_display:
                for a in analyseurs:
                    frame = a.draw(frame)
                cv2.putText(frame, f"FPS:{fps_liss:.1f}  frame {frame_idx}", (10, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.imshow("Surveillance unifiee (EPI + securite)", frame)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
            elif frame_idx % 20 == 0:
                print(f"frame {frame_idx}  fps={fps_liss:.2f}  evenements={bus.total}", flush=True)

            if args.max_frames and frame_idx >= args.max_frames:
                break
    finally:
        cap.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        pool.shutdown(wait=True)
        bus.fermer()

    duree = time.time() - t_debut
    print(f"\n{frame_idx} frames en {duree:.1f}s -> {frame_idx / max(duree, 1e-6):.2f} FPS moyen")
    print(f"{bus.total} evenements" + (f" -> {args.events}" if args.events else ""))
    print("\nCout cumule par analyseur (part du temps total) :")
    for nom, c in sorted(couts.items(), key=lambda kv: -kv[1]):
        print(f"  {nom:10} {c:7.1f}s  {100 * c / max(duree, 1e-6):5.1f}%")


if __name__ == "__main__":
    main()
