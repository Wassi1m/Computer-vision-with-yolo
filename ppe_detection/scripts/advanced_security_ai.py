"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          SYSTÈME IA SÉCURITÉ INDUSTRIELLE — VERSION AVANCÉE                 ║
║  Modules : PPE | Fumée/Feu | Chute | Ligne Virtuelle | Présence |           ║
║            Objet Abandonné | LPR | Porte Ouverte/Fermée                     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from ultralytics import YOLO
from collections import deque, defaultdict
import cv2, time, json, os, re
from pathlib import Path
from datetime import datetime
import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
#  CHARGEMENT DES MODÈLES
# ══════════════════════════════════════════════════════════════════════════════

print("═" * 60)
print("  Chargement des modèles IA...")
print("═" * 60)

model_person = YOLO("yolov8n.pt")   # Détection personnes (COCO)
model1       = YOLO("best.pt")       # PPE modèle 1
model2       = YOLO("best_gloves.pt")# PPE modèle 2

# Modèle dédié feu/fumée — téléchargeable ou substitué par best.pt
# Si vous avez un modèle fire_smoke.pt spécialisé, remplacez ici
try:
    model_fire = YOLO("fire_smoke.pt")
    print("  [OK] Modèle feu/fumée chargé : fire_smoke.pt")
except Exception:
    model_fire = None
    print("  [--] fire_smoke.pt absent → détection feu via classes COCO uniquement")

# Modèle LPR (lecture de plaques)
try:
    model_lpr = YOLO("yolov8n.pt")   # Remplacer par un modèle LPR dédié si disponible
    print("  [OK] Modèle LPR chargé (yolov8n — remplacer par lpr.pt pour meilleur résultat)")
except Exception:
    model_lpr = None
    print("  [--] Modèle LPR absent")

print("  [OK] Modèles PPE chargés\n")


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION GÉNÉRALE
# ══════════════════════════════════════════════════════════════════════════════

DEBUG = False

SCREENSHOTS = Path("security_screenshots")
SCREENSHOTS.mkdir(exist_ok=True)

LOGS_DIR = Path("security_logs")
LOGS_DIR.mkdir(exist_ok=True)

# Couleurs BGR pour les personnes
COULEURS_PERSONNES = [
    (0, 255, 255), (0, 100, 255), (255, 50, 50), (50, 255, 50),
    (255, 0, 200), (0, 200, 0),  (200, 0, 255), (255, 200, 0),
    (0, 0, 255),   (220, 150, 0),(0, 255, 150), (255, 0, 80),
]

# ══════════════════════════════════════════════════════════════════════════════
#  MODULE 1 — PPE (EPI) — INCHANGÉ + AMÉLIORÉ
# ══════════════════════════════════════════════════════════════════════════════

TRADUCTION_PPE = {
    "Hardhat": "Casque porté", "NO-Hardhat": "SANS CASQUE !",
    "Mask": "Masque porté",   "NO-Mask": "SANS MASQUE !",
    "Goggles": "Lunettes",    "NO-Goggles": "SANS LUNETTES !",
    "Gloves": "Gants portés", "NO-Gloves": "SANS GANTS !",
    "Safety Vest": "Gilet porté", "NO-Safety Vest": "SANS GILET !",
    "Fall-Detected": "CHUTE DÉTECTÉE !", "Person": "Personne",
    "Ladder": "Échelle", "Safety Cone": "Cône sécurité",
    "helmet": "Casque porté", "mask": "Masque porté",
    "goggles": "Lunettes", "Vest": "Gilet porté",
    "safety_shoe": "Chaussures sécurité",
}

MOTS_CLES_EPI = {
    "casque":     ["hardhat", "helmet"],
    "masque":     ["mask"],
    "lunettes":   ["goggle"],
    "gilet":      ["safety vest", "vest"],
    "gants":      ["glove"],
    "chaussures": ["safety_shoe", "shoe"],
}

EPI_OBLIGATOIRES = {
    "casque": True, "masque": False, "lunettes": False,
    "gilet": True,  "gants": False,  "chaussures": False,
}

CONF_EPI = {
    "casque": 0.15, "masque": 0.08, "lunettes": 0.08,
    "gilet": 0.12,  "gants": 0.08,  "chaussures": 0.20,
}

LABELS_ABSENCE = {
    "casque": "SANS CASQUE !", "masque": "SANS MASQUE !",
    "lunettes": "SANS LUNETTES !", "gilet": "SANS GILET !",
    "gants": "SANS GANTS !", "chaussures": "SANS CHAUSSURES !",
}

HIST_N, HIST_K = 12, 4
_historique_ppe: dict = {}


def get_hist_ppe(pidx: int, epi: str) -> deque:
    if pidx not in _historique_ppe:
        _historique_ppe[pidx] = {}
    if epi not in _historique_ppe[pidx]:
        _historique_ppe[pidx][epi] = deque([False] * HIST_N, maxlen=HIST_N)
    return _historique_ppe[pidx][epi]


def reset_hist_ppe(n_personnes: int):
    for pidx in list(_historique_ppe.keys()):
        if pidx >= n_personnes:
            del _historique_ppe[pidx]


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE 2 — DÉTECTION FUMÉE / INCENDIE 🔥
# ══════════════════════════════════════════════════════════════════════════════

class SmokeFire_Detector:
    """
    Détection feu/fumée par deux approches complémentaires :
    1. Analyse de couleur HSV (rouge/orange intense + zones grises)
    2. Modèle YOLO si disponible (fire_smoke.pt)
    """
    # Seuils couleur HSV pour flammes
    FLAME_HSV_LOWER1 = np.array([0,   120, 120])
    FLAME_HSV_UPPER1 = np.array([15,  255, 255])
    FLAME_HSV_LOWER2 = np.array([160, 120, 120])
    FLAME_HSV_UPPER2 = np.array([180, 255, 255])
    FLAME_ORANGE_L   = np.array([10,  120, 120])
    FLAME_ORANGE_U   = np.array([35,  255, 255])

    # Seuils pour fumée (zones grises saturées > seuil luminosité)
    SMOKE_S_MAX      = 40
    SMOKE_V_MIN      = 80
    SMOKE_V_MAX      = 210

    AREA_MIN_FLAME   = 800    # px² minimum pour valider une flamme
    AREA_MIN_SMOKE   = 2000   # px² minimum pour valider la fumée

    HIST             = deque([False] * 8, maxlen=8)
    HIST_K           = 3      # 3/8 frames pour confirmer

    def __init__(self):
        self.detections: list[dict] = []

    def detect(self, frame: np.ndarray, model=None) -> list[dict]:
        self.detections = []
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # — Flammes (rouge/orange) —
        m1 = cv2.inRange(hsv, self.FLAME_HSV_LOWER1, self.FLAME_HSV_UPPER1)
        m2 = cv2.inRange(hsv, self.FLAME_HSV_LOWER2, self.FLAME_HSV_UPPER2)
        m3 = cv2.inRange(hsv, self.FLAME_ORANGE_L,   self.FLAME_ORANGE_U)
        fire_mask = cv2.bitwise_or(cv2.bitwise_or(m1, m2), m3)
        fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_OPEN,
                                     np.ones((5, 5), np.uint8))
        fire_mask = cv2.dilate(fire_mask, np.ones((7, 7), np.uint8), iterations=2)
        cnts_fire, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts_fire:
            if cv2.contourArea(c) >= self.AREA_MIN_FLAME:
                x, y, fw, fh = cv2.boundingRect(c)
                self.detections.append({"type": "INCENDIE 🔥", "box": (x, y, x+fw, y+fh),
                                        "conf": 0.90, "color": (0, 40, 220)})

        # — Fumée (gris dense) —
        smoke_cond = ((hsv[:, :, 1] < self.SMOKE_S_MAX) &
                      (hsv[:, :, 2] > self.SMOKE_V_MIN) &
                      (hsv[:, :, 2] < self.SMOKE_V_MAX))
        smoke_mask = (smoke_cond.astype(np.uint8) * 255)
        smoke_mask = cv2.morphologyEx(smoke_mask, cv2.MORPH_OPEN,
                                      np.ones((9, 9), np.uint8))
        smoke_mask = cv2.dilate(smoke_mask, np.ones((15, 15), np.uint8), iterations=3)
        cnts_smoke, _ = cv2.findContours(smoke_mask, cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts_smoke:
            if cv2.contourArea(c) >= self.AREA_MIN_SMOKE:
                x, y, sw, sh = cv2.boundingRect(c)
                self.detections.append({"type": "FUMÉE 💨", "box": (x, y, x+sw, y+sh),
                                        "conf": 0.75, "color": (100, 100, 100)})

        # — Modèle YOLO dédié si disponible —
        if model is not None:
            classes_fire = {"fire", "smoke", "feu", "fumee", "flame"}
            for box in model.predict(frame, conf=0.40, verbose=False)[0].boxes:
                cls = model.names[int(box.cls)].lower()
                if any(k in cls for k in classes_fire):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    self.detections.append({
                        "type": "FEU YOLO 🔥", "box": (x1, y1, x2, y2),
                        "conf": float(box.conf), "color": (0, 0, 200)
                    })

        detected_now = len(self.detections) > 0
        self.HIST.append(detected_now)
        confirmed = sum(self.HIST) >= self.HIST_K
        return self.detections if confirmed else []


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE 3 — PERSONNE ÉVANOUIE / TOMBÉE 🚑
# ══════════════════════════════════════════════════════════════════════════════

class FallDetector:
    """
    Détection de chute par ratio largeur/hauteur de la bounding box.
    Une personne debout : H >> W  (ratio ~0.3)
    Une personne allongée : W >> H (ratio >1.2)
    """
    RATIO_THRESH    = 1.2    # W/H au-delà duquel = allongé
    CONF_MIN        = 0.40   # Confiance min modèle personnes
    HIST_N          = 10
    HIST_K          = 5      # 5/10 frames pour confirmer une chute

    # Délai anti-spam alerte (secondes)
    ALERT_COOLDOWN  = 5.0

    def __init__(self):
        self._hists: dict[int, deque] = {}
        self._last_alert: dict[int, float] = {}

    def _get_hist(self, pidx: int) -> deque:
        if pidx not in self._hists:
            self._hists[pidx] = deque([False] * self.HIST_N, maxlen=self.HIST_N)
        return self._hists[pidx]

    def process(self, personnes: list[tuple]) -> list[tuple[int, bool]]:
        """Retourne [(pidx, is_fallen)] pour chaque personne."""
        results = []
        now = time.time()
        for pidx, (x1, y1, x2, y2) in enumerate(personnes):
            w, h = x2 - x1, y2 - y1
            if h == 0:
                continue
            ratio = w / h
            fallen_now = ratio >= self.RATIO_THRESH
            hist = self._get_hist(pidx)
            hist.append(fallen_now)
            confirmed = sum(hist) >= self.HIST_K

            # Anti-spam
            cooldown_ok = (now - self._last_alert.get(pidx, 0)) >= self.ALERT_COOLDOWN
            if confirmed and cooldown_ok:
                self._last_alert[pidx] = now
            results.append((pidx, confirmed))
        return results


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE 4 — FRANCHISSEMENT LIGNE VIRTUELLE 🚧
# ══════════════════════════════════════════════════════════════════════════════

class VirtualLineDetector:
    """
    Ligne virtuelle configurable. Détecte le franchissement par le centre bas
    (pied) des bounding boxes de personnes.
    Mode édition : cliquer 2 points pour tracer la ligne.
    """

    def __init__(self):
        self.lines: list[dict] = []          # [{p1, p2, name, cross_count, hist}]
        self._editing       = False
        self._pending_p1    = None
        self._line_name_ctr = 1
        self._crossed_ids: set = set()

    def start_edit(self):
        self._editing    = True
        self._pending_p1 = None
        print("  [LIGNE] Cliquez P1 puis P2 pour tracer la ligne.")

    def on_mouse(self, event, x, y, flags, param):
        if not self._editing:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            if self._pending_p1 is None:
                self._pending_p1 = (x, y)
                print(f"  [LIGNE] P1 = {self._pending_p1} — cliquez P2")
            else:
                p2 = (x, y)
                name = f"Ligne {self._line_name_ctr}"
                self.lines.append({
                    "p1": self._pending_p1, "p2": p2,
                    "name": name, "cross_count": 0,
                    "hist": deque([False] * 6, maxlen=6),
                    "active": False,
                })
                print(f"  [LIGNE] '{name}' créée : {self._pending_p1} → {p2}")
                self._pending_p1 = None
                self._line_name_ctr += 1
                self._editing = False

    @staticmethod
    def _sign(p, a, b) -> int:
        """Signe du produit vectoriel (b-a) × (p-a)."""
        return int(np.sign((b[0] - a[0]) * (p[1] - a[1]) -
                           (b[1] - a[1]) * (p[0] - a[0])))

    @staticmethod
    def _segments_intersect(p1, p2, p3, p4) -> bool:
        d1 = VirtualLineDetector._sign(p3, p4, p1)
        d2 = VirtualLineDetector._sign(p3, p4, p2)
        d3 = VirtualLineDetector._sign(p1, p2, p3)
        d4 = VirtualLineDetector._sign(p1, p2, p4)
        return (d1 != d2) and (d3 != d4)

    def process(self, personnes: list[tuple], prev_personnes: list[tuple]) -> list[tuple]:
        """Retourne [(pidx, line_name)] pour chaque franchissement détecté."""
        events = []
        for line in self.lines:
            p1, p2 = line["p1"], line["p2"]
            crossed_this_frame = False
            for pidx, (x1, y1, x2, y2) in enumerate(personnes):
                # Centre bas (pieds) de la personne
                foot_curr = ((x1 + x2) // 2, y2)
                if pidx < len(prev_personnes):
                    px1, py1, px2, py2 = prev_personnes[pidx]
                    foot_prev = ((px1 + px2) // 2, py2)
                    # Test de croisement entre le déplacement du pied et la ligne
                    if self._segments_intersect(foot_prev, foot_curr, p1, p2):
                        crossed_this_frame = True
                        events.append((pidx, line["name"]))
                        line["cross_count"] += 1
            line["hist"].append(crossed_this_frame)
            line["active"] = any(line["hist"])
        return events

    def draw(self, frame: np.ndarray):
        for line in self.lines:
            color = (0, 30, 200) if line["active"] else (0, 200, 200)
            thick = 3 if line["active"] else 2
            cv2.line(frame, line["p1"], line["p2"], color, thick)
            mid = ((line["p1"][0] + line["p2"][0]) // 2,
                   (line["p1"][1] + line["p2"][1]) // 2)
            cv2.putText(frame, f"{line['name']} ({line['cross_count']})",
                        (mid[0] - 40, mid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        if self._editing and self._pending_p1:
            cv2.circle(frame, self._pending_p1, 6, (0, 255, 255), -1)
            cv2.putText(frame, "Cliquez P2...", (self._pending_p1[0] + 10,
                        self._pending_p1[1]), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 255, 255), 2)


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE 5 — AGENT À SON POSTE / ABSENT 👮
# ══════════════════════════════════════════════════════════════════════════════

class PostDetector:
    """
    Zones de surveillance rectangulaires (postes).
    Vérifie la présence d'au moins une personne dans chaque zone.
    Alerte si absent > ABSENCE_TIMEOUT secondes.
    """
    ABSENCE_TIMEOUT = 10.0  # secondes avant alerte

    def __init__(self):
        self.zones: list[dict] = []
        self._editing     = False
        self._pending_p1  = None
        self._zone_ctr    = 1

    def start_edit(self):
        self._editing    = True
        self._pending_p1 = None
        print("  [POSTE] Cliquez P1 puis P2 pour définir la zone de poste.")

    def on_mouse(self, event, x, y, flags, param):
        if not self._editing:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            if self._pending_p1 is None:
                self._pending_p1 = (x, y)
            else:
                p2 = (x, y)
                x1 = min(self._pending_p1[0], p2[0])
                y1 = min(self._pending_p1[1], p2[1])
                x2 = max(self._pending_p1[0], p2[0])
                y2 = max(self._pending_p1[1], p2[1])
                name = f"Poste {self._zone_ctr}"
                self.zones.append({
                    "box": (x1, y1, x2, y2), "name": name,
                    "present": False, "last_seen": time.time(),
                    "absent_since": None,
                })
                print(f"  [POSTE] '{name}' défini : ({x1},{y1}) → ({x2},{y2})")
                self._pending_p1 = None
                self._zone_ctr  += 1
                self._editing    = False

    def process(self, personnes: list[tuple]) -> list[dict]:
        """Met à jour l'état présence/absence et retourne les zones en alerte."""
        now = time.time()
        alerts = []
        for zone in self.zones:
            zx1, zy1, zx2, zy2 = zone["box"]
            present = False
            for (px1, py1, px2, py2) in personnes:
                # Centre personne dans la zone ?
                cx, cy = (px1 + px2) // 2, (py1 + py2) // 2
                if zx1 <= cx <= zx2 and zy1 <= cy <= zy2:
                    present = True
                    break
            zone["present"] = present
            if present:
                zone["last_seen"]    = now
                zone["absent_since"] = None
            else:
                if zone["absent_since"] is None:
                    zone["absent_since"] = now
                absent_dur = now - zone["absent_since"]
                if absent_dur >= self.ABSENCE_TIMEOUT:
                    alerts.append({**zone, "absent_duration": absent_dur})
        return alerts

    def draw(self, frame: np.ndarray):
        for zone in self.zones:
            x1, y1, x2, y2 = zone["box"]
            if zone["present"]:
                color, label = (0, 200, 0), f"✓ {zone['name']}"
            else:
                absent_s = int(time.time() - zone["absent_since"]) if zone["absent_since"] else 0
                color = (0, 30, 200) if absent_s >= self.ABSENCE_TIMEOUT else (0, 160, 255)
                label = f"ABSENT {absent_s}s — {zone['name']}" if absent_s > 0 else zone["name"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1 + 4, y1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        if self._editing and self._pending_p1:
            cv2.circle(frame, self._pending_p1, 6, (0, 255, 0), -1)


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE 6 — OBJET ABANDONNÉ 💼
# ══════════════════════════════════════════════════════════════════════════════

class AbandonedObjectDetector:
    """
    Détecte les objets statiques dans la scène qui persistent sans
    propriétaire nearby. Utilise la soustraction de fond (MOG2) +
    association personne / objet par distance.
    """
    STATIC_THRESHOLD    = 15       # Pixels de mouvement minimum pour "mobile"
    ABANDONED_TIMEOUT   = 20.0     # secondes immobile = abandonné
    OWNER_DIST_MAX      = 200      # Pixels : distance max personne ↔ objet
    OBJECT_AREA_MIN     = 1500     # px² minimum (filtre bruit)
    MERGE_DIST          = 60       # Fusionne objets proches

    def __init__(self):
        self._bg        = cv2.createBackgroundSubtractorMOG2(
                            history=500, varThreshold=25, detectShadows=False)
        self._objects: dict[int, dict] = {}   # id → {box, first_seen, last_moved}
        self._next_id   = 0

    @staticmethod
    def _box_center(box):
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @staticmethod
    def _box_dist(b1, b2):
        c1, c2 = AbandonedObjectDetector._box_center(b1), \
                 AbandonedObjectDetector._box_center(b2)
        return ((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2) ** 0.5

    def process(self, frame: np.ndarray, personnes: list[tuple]) -> list[dict]:
        now  = time.time()
        fg   = self._bg.apply(frame)
        # Enlever bruit morphologique
        kernel = np.ones((5, 5), np.uint8)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
        fg = cv2.dilate(fg, np.ones((7, 7), np.uint8), iterations=2)
        cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        current_blobs: list[tuple] = []
        for c in cnts:
            if cv2.contourArea(c) >= self.OBJECT_AREA_MIN:
                x, y, bw, bh = cv2.boundingRect(c)
                current_blobs.append((x, y, x + bw, y + bh))

        # Fusion des blobs proches
        merged: list[tuple] = []
        used = [False] * len(current_blobs)
        for i, b1 in enumerate(current_blobs):
            if used[i]:
                continue
            group = [b1]
            for j, b2 in enumerate(current_blobs):
                if i != j and not used[j] and self._box_dist(b1, b2) < self.MERGE_DIST:
                    group.append(b2)
                    used[j] = True
            used[i] = True
            xs = [g[0] for g in group] + [g[2] for g in group]
            ys = [g[1] for g in group] + [g[3] for g in group]
            merged.append((min(xs), min(ys), max(xs), max(ys)))

        # Match blobs → objets trackés
        matched_ids: set = set()
        for blob in merged:
            best_id, best_d = None, float("inf")
            for oid, obj in self._objects.items():
                d = self._box_dist(obj["box"], blob)
                if d < best_d:
                    best_d = d
                    best_id = oid
            if best_id is not None and best_d < self.MERGE_DIST * 1.5:
                self._objects[best_id]["last_moved"] = now
                self._objects[best_id]["box"]        = blob
                matched_ids.add(best_id)
            else:
                # Nouvel objet
                self._objects[self._next_id] = {
                    "box": blob, "first_seen": now,
                    "last_moved": now, "id": self._next_id
                }
                matched_ids.add(self._next_id)
                self._next_id += 1

        # Supprimer objets non vus depuis longtemps
        to_del = [oid for oid, obj in self._objects.items()
                  if now - obj["last_moved"] > self.ABANDONED_TIMEOUT * 3]
        for oid in to_del:
            del self._objects[oid]

        # Identifier abandonnés (statiques + sans propriétaire proche)
        abandoned = []
        for oid, obj in self._objects.items():
            static_dur = now - obj["last_moved"]
            if static_dur < self.ABANDONED_TIMEOUT:
                continue
            # Vérification : personne proche ?
            box = obj["box"]
            has_owner = any(
                self._box_dist(box, (px1, py1, px2, py2)) < self.OWNER_DIST_MAX
                for (px1, py1, px2, py2) in personnes
            )
            if not has_owner:
                abandoned.append({**obj, "static_seconds": static_dur})

        return abandoned


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE 7 — DÉTECTION AUTOMATIQUE DE MATRICULES (LPR) 🚗
# ══════════════════════════════════════════════════════════════════════════════

class LPRDetector:
    """
    Lecture automatique de plaques d'immatriculation.
    Approche : détection ROI via YOLO "car" (classe COCO) → ROI plaques
    → amélioration contraste → OCR via Tesseract (si disponible).
    Sans Tesseract, affiche la zone de plaque détectée.
    """
    # Classes COCO véhicules
    VEHICLE_CLASSES     = {2, 3, 5, 7}   # car, motorcycle, bus, truck

    # Ratio plaque / véhicule estimé
    PLATE_ROI_Y_START   = 0.70   # bas du véhicule
    PLATE_ROI_Y_END     = 0.95
    PLATE_ROI_X_MARGIN  = 0.20

    HIST_N              = 5

    def __init__(self):
        self._ocr_available = False
        try:
            import pytesseract
            self._pytesseract = pytesseract
            self._ocr_available = True
            print("  [OK] Tesseract OCR disponible → lecture plaques active")
        except ImportError:
            print("  [--] pytesseract absent → détection région plaque uniquement")

    def _preprocess_plate(self, roi: np.ndarray) -> np.ndarray:
        """Prétraitement pour améliorer la lisibilité de la plaque."""
        gray  = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # Redimensionner × 3 pour l'OCR
        h, w  = gray.shape
        gray  = cv2.resize(gray, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)
        # CLAHE pour améliorer le contraste
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray  = clahe.apply(gray)
        # Binarisation adaptative
        gray  = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 11, 2)
        return gray

    def _read_plate(self, roi: np.ndarray) -> str:
        if not self._ocr_available:
            return ""
        proc = self._preprocess_plate(roi)
        config = r"--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
        try:
            text = self._pytesseract.image_to_string(proc, config=config).strip()
            # Filtrer : plaque valide ≥ 4 chars alphanumériques
            clean = re.sub(r"[^A-Z0-9-]", "", text.upper())
            return clean if len(clean) >= 4 else ""
        except Exception:
            return ""

    def detect(self, frame: np.ndarray, model_yolo) -> list[dict]:
        detections = []
        if model_yolo is None:
            return detections

        for box in model_yolo.predict(frame, conf=0.40, verbose=False)[0].boxes:
            cls_id = int(box.cls)
            if cls_id not in self.VEHICLE_CLASSES:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            veh_w, veh_h    = x2 - x1, y2 - y1

            # ROI plaque estimée (bas du véhicule)
            px1 = x1 + int(veh_w * self.PLATE_ROI_X_MARGIN)
            px2 = x2 - int(veh_w * self.PLATE_ROI_X_MARGIN)
            py1 = y1 + int(veh_h * self.PLATE_ROI_Y_START)
            py2 = y1 + int(veh_h * self.PLATE_ROI_Y_END)

            px1 = max(0, px1); py1 = max(0, py1)
            px2 = min(frame.shape[1], px2)
            py2 = min(frame.shape[0], py2)

            plate_roi = frame[py1:py2, px1:px2]
            if plate_roi.size == 0:
                continue

            text = self._read_plate(plate_roi)
            detections.append({
                "vehicle_box": (x1, y1, x2, y2),
                "plate_box":   (px1, py1, px2, py2),
                "plate_text":  text or "—",
                "conf":        float(box.conf),
                "class":       model_yolo.names[cls_id],
            })
        return detections


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE 8 — DÉTECTION PORTE OUVERTE / FERMÉE 🚪
# ══════════════════════════════════════════════════════════════════════════════

class DoorDetector:
    """
    Détection de l'état des portes par zone définie manuellement.
    Chaque zone surveille le changement de luminosité/mouvement
    pour inférer l'état ouvert/fermé.
    """
    OPEN_CHANGE_THRESH  = 20.0   # Écart moyen de luminosité → porte ouverte
    HIST_N              = 8
    HIST_K              = 4

    def __init__(self):
        self.zones: list[dict] = []
        self._editing     = False
        self._pending_p1  = None
        self._zone_ctr    = 1
        self._ref_frames: dict[int, np.ndarray] = {}

    def start_edit(self):
        self._editing    = True
        self._pending_p1 = None
        print("  [PORTE] Cliquez P1 puis P2 pour définir la zone porte.")

    def on_mouse(self, event, x, y, flags, param):
        if not self._editing:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            if self._pending_p1 is None:
                self._pending_p1 = (x, y)
            else:
                p2 = (x, y)
                x1 = min(self._pending_p1[0], p2[0])
                y1 = min(self._pending_p1[1], p2[1])
                x2 = max(self._pending_p1[0], p2[0])
                y2 = max(self._pending_p1[1], p2[1])
                zidx = len(self.zones)
                name = f"Porte {self._zone_ctr}"
                self.zones.append({
                    "box": (x1, y1, x2, y2), "name": name,
                    "open": False, "confidence": 0.0,
                    "hist": deque([False] * self.HIST_N, maxlen=self.HIST_N),
                    "idx": zidx,
                })
                print(f"  [PORTE] '{name}' définie : ({x1},{y1}) → ({x2},{y2})")
                self._pending_p1 = None
                self._zone_ctr  += 1
                self._editing    = False

    def process(self, frame: np.ndarray) -> list[dict]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = []
        for zone in self.zones:
            x1, y1, x2, y2 = zone["box"]
            roi = gray[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            zidx = zone["idx"]
            if zidx not in self._ref_frames:
                self._ref_frames[zidx] = roi.copy().astype(np.float32)
                continue
            ref  = self._ref_frames[zidx]
            diff = np.abs(roi.astype(np.float32) - ref)
            mean_diff = float(np.mean(diff))
            # Mettre à jour la référence lentement (adaptation lente)
            alpha = 0.02
            self._ref_frames[zidx] = (1 - alpha) * ref + alpha * roi.astype(np.float32)

            is_open = mean_diff >= self.OPEN_CHANGE_THRESH
            zone["hist"].append(is_open)
            zone["open"]       = sum(zone["hist"]) >= self.HIST_K
            zone["confidence"] = sum(zone["hist"]) / self.HIST_N
            results.append(zone)
        return results

    def draw(self, frame: np.ndarray):
        for zone in self.zones:
            x1, y1, x2, y2 = zone["box"]
            state  = "OUVERTE 🔓" if zone["open"] else "FERMÉE 🔒"
            color  = (0, 100, 220) if zone["open"] else (0, 200, 0)
            thick  = 3 if zone["open"] else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
            label  = f"{zone['name']} — {state}"
            cv2.putText(frame, label, (x1 + 4, y1 + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2)
        if self._editing and self._pending_p1:
            cv2.circle(frame, self._pending_p1, 6, (255, 0, 255), -1)


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITAIRES COMMUNS
# ══════════════════════════════════════════════════════════════════════════════

def iou(boxA, boxB) -> float:
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    aA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    aB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    union = aA + aB - inter
    return inter / union if union > 0 else 0.0


def distance_centres(boxA, boxB) -> float:
    cxA = (boxA[0]+boxA[2])/2; cyA = (boxA[1]+boxA[3])/2
    cxB = (boxB[0]+boxB[2])/2; cyB = (boxB[1]+boxB[3])/2
    return ((cxA-cxB)**2 + (cyA-cyB)**2) ** 0.5


def assigner_personne(ppe_box, personnes) -> int:
    if not personnes:
        return -1
    best_idx, best_iou = 0, 0.0
    for i, p in enumerate(personnes):
        s = iou(ppe_box, p)
        if s > best_iou:
            best_iou = s; best_idx = i
    if best_iou > 0.05:
        return best_idx
    best_idx, best_dist = 0, float("inf")
    for i, p in enumerate(personnes):
        d = distance_centres(ppe_box, p)
        if d < best_dist:
            best_dist = d; best_idx = i
    return best_idx


def dessiner_label(frame, texte, x1, y1, color, font_scale=0.55, ep=2):
    (tw, th), _ = cv2.getTextSize(texte, cv2.FONT_HERSHEY_SIMPLEX, font_scale, ep)
    cv2.rectangle(frame, (x1, y1-th-10), (x1+tw+8, y1), color, -1)
    cv2.putText(frame, texte, (x1+4, y1-5),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), ep)


def log_event(module: str, message: str, data: dict = None):
    """Enregistre les événements importants dans un fichier JSON."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "module":    module,
        "message":   message,
        "data":      data or {},
    }
    log_file = LOGS_DIR / f"events_{datetime.now().strftime('%Y%m%d')}.jsonl"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def draw_hud_header(frame: np.ndarray, fps: float, n_persons: int,
                    mode_flags: dict, w: int, h: int):
    """Barre de titre + FPS + compteurs."""
    cv2.rectangle(frame, (0, 0), (w, 34), (20, 20, 30), -1)
    modules_actifs = [k for k, v in mode_flags.items() if v]
    txt = f"FPS:{fps:.0f}  |  {n_persons} pers.  |  Modules: {', '.join(modules_actifs) or 'PPE'}"
    cv2.putText(frame, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200,220,255), 1)


def draw_alert_sidebar(frame: np.ndarray, alerts: list[str], w: int):
    """Colonne d'alertes en haut à gauche."""
    y = 50
    for alert in alerts:
        bg_w = len(alert) * 12 + 24
        cv2.rectangle(frame, (0, y-24), (bg_w, y+8), (30, 0, 80), -1)
        cv2.rectangle(frame, (0, y-24), (5,    y+8), (0, 0, 220), -1)
        cv2.putText(frame, alert, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255,255,255), 2)
        y += 36


def draw_ppe_panel(frame: np.ndarray, personnes, ppe_stable, violations_pp,
                   w: int, h: int):
    """Panneau EPI bas-droit."""
    epi_items = [("Casque","casque"),("Masque","masque"),("Lunettes","lunettes"),
                 ("Gilet","gilet"),("Gants","gants"),("Chaussures","chaussures")]
    n = len(personnes)
    lignes   = n * (len(epi_items) + 1) + 3
    panel_h  = lignes * 20 + 24
    panel_x  = w - 270
    panel_y0 = h - panel_h - 44
    cv2.rectangle(frame, (panel_x-8, panel_y0), (w-4, h-42), (18,18,28), -1)
    cv2.putText(frame, "═ EPI Status ═", (panel_x, panel_y0+16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100,200,255), 1)
    for pidx in range(n):
        ppe    = ppe_stable[pidx]
        color  = COULEURS_PERSONNES[pidx % len(COULEURS_PERSONNES)]
        base_y = panel_y0 + 18 + pidx * (len(epi_items)+1) * 20
        cv2.putText(frame, f"  P{pidx+1}", (panel_x, base_y+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 2)
        for j, (nom, key) in enumerate(epi_items):
            ok    = key in ppe
            icone = "[OK]" if ok else "[ X]"
            col   = (0,200,0) if ok else (0,0,200)
            cv2.putText(frame, f"  {icone} {nom}",
                        (panel_x, base_y+20+(j+1)*20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1)


# ══════════════════════════════════════════════════════════════════════════════
#  GESTIONNAIRE DE SOURIS (multiplexage des modules)
# ══════════════════════════════════════════════════════════════════════════════

class MouseRouter:
    def __init__(self):
        self._active_module = None
        self._modules       = {}

    def register(self, name: str, handler):
        self._modules[name] = handler

    def activate(self, name: str | None):
        self._active_module = name
        print(f"  [SOURIS] Module actif : {name or 'aucun'}")

    def callback(self, event, x, y, flags, param):
        if self._active_module and self._active_module in self._modules:
            self._modules[self._active_module](event, x, y, flags, param)


# ══════════════════════════════════════════════════════════════════════════════
#  BOUCLE PRINCIPALE
# ══════════════════════════════════════════════════════════════════════════════

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERREUR : Caméra non disponible.")
        return

    # — Instanciation des modules —
    smoke_fire   = SmokeFire_Detector()
    fall_det     = FallDetector()
    line_det     = VirtualLineDetector()
    post_det     = PostDetector()
    abandon_det  = AbandonedObjectDetector()
    lpr_det      = LPRDetector()
    door_det     = DoorDetector()

    mouse_router = MouseRouter()
    mouse_router.register("ligne",   line_det.on_mouse)
    mouse_router.register("poste",   post_det.on_mouse)
    mouse_router.register("porte",   door_det.on_mouse)

    # — Modes actifs —
    MODES = {
        "PPE":        True,
        "Feu/Fumée":  True,
        "Chute":      True,
        "Ligne":      True,
        "Poste":      True,
        "Abandon":    True,
        "LPR":        False,   # Désactivé par défaut (lourd CPU)
        "Porte":      True,
    }

    # — Fenêtre & callbacks —
    WIN = "Système IA Sécurité — Multi-Module"
    cv2.namedWindow(WIN)
    cv2.setMouseCallback(WIN, mouse_router.callback)

    prev_personnes: list[tuple] = []
    prev_time  = time.time()
    last_log   = 0.0
    lpr_buffer: list[str] = []   # Historique des plaques lues

    print("\n═" * 60)
    print("  RACCOURCIS CLAVIER")
    print("  Q      → Quitter")
    print("  S      → Screenshot")
    print("  D      → Mode Debug")
    print("  F      → Toggle Feu/Fumée")
    print("  C      → Toggle Chute")
    print("  L      → Tracer une Ligne virtuelle")
    print("  P      → Définir un Poste")
    print("  O      → Définir une zone Porte")
    print("  A      → Toggle Objets Abandonnés")
    print("  R      → Toggle LPR (plaques)")
    print("  X      → Effacer toutes les zones")
    print("═" * 60 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        now  = time.time()
        fps  = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        global_alerts: list[str] = []

        # ── 1. Détection personnes ────────────────────────────────────────
        personnes: list[tuple] = []
        for box in model_person.predict(frame, conf=0.35, classes=[0],
                                        verbose=False)[0].boxes:
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            personnes.append((x1,y1,x2,y2))

        # ── 2. PPE ───────────────────────────────────────────────────────
        ppe_detections = []
        ppe_par_personne  = [set() for _ in range(max(len(personnes),1))]
        ppe_stable        = [set() for _ in range(max(len(personnes),1))]
        violations_pp     = [[] for _ in range(max(len(personnes),1))]

        if MODES["PPE"]:
            conf_g = 0.01 if DEBUG else 0.03
            for model, _ in [(model1,"M1"), (model2,"M2")]:
                for box in model.predict(frame, conf=conf_g, verbose=False)[0].boxes:
                    cls = model.names[int(box.cls)]
                    sc  = float(box.conf)
                    x1,y1,x2,y2 = map(int, box.xyxy[0])
                    cls_low = cls.lower()
                    if "person" in cls_low:
                        if not personnes:
                            personnes.append((x1,y1,x2,y2))
                        continue
                    if sc >= 0.03:
                        ppe_detections.append((cls,sc,x1,y1,x2,y2))

            if not personnes and ppe_detections:
                xs = [d[2] for d in ppe_detections]+[d[4] for d in ppe_detections]
                ys = [d[3] for d in ppe_detections]+[d[5] for d in ppe_detections]
                personnes = [(min(xs),min(ys),max(xs),max(ys))]

            n = len(personnes)
            reset_hist_ppe(n)
            ppe_par_personne  = [set() for _ in range(n)]
            ppe_stable        = [set() for _ in range(n)]
            violations_pp     = [[] for _ in range(n)]

            ppe_avec_personne = []
            for (cls,sc,x1,y1,x2,y2) in ppe_detections:
                cls_low = cls.lower()
                pidx    = assigner_personne((x1,y1,x2,y2), personnes)
                if pidx == -1: pidx = 0
                est_neg = cls_low.startswith(("no-","no ","without"))
                for epi, mots in MOTS_CLES_EPI.items():
                    if any(k in cls_low for k in mots) and not est_neg:
                        if sc >= CONF_EPI.get(epi,0.10):
                            ppe_par_personne[pidx].add(epi)
                is_danger = cls in {"NO-Hardhat","NO-Mask","NO-Goggles",
                                    "NO-Gloves","NO-Safety Vest","Fall-Detected"} \
                            or "no-" in cls_low[:4]
                ppe_avec_personne.append((cls,sc,x1,y1,x2,y2,pidx,is_danger))

            for pidx in range(n):
                for epi in MOTS_CLES_EPI:
                    h_epi = get_hist_ppe(pidx, epi)
                    h_epi.append(epi in ppe_par_personne[pidx])
                    if sum(h_epi) >= HIST_K:
                        ppe_stable[pidx].add(epi)

            for pidx in range(n):
                for (cls,sc,x1,y1,x2,y2,p,is_danger) in ppe_avec_personne:
                    if p == pidx and is_danger:
                        lbl = TRADUCTION_PPE.get(cls,cls)
                        if lbl not in violations_pp[pidx]:
                            violations_pp[pidx].append(lbl)
                for epi, oblig in EPI_OBLIGATOIRES.items():
                    if oblig and epi not in ppe_stable[pidx]:
                        v = LABELS_ABSENCE[epi]
                        if v not in violations_pp[pidx]:
                            violations_pp[pidx].append(v)

            # Alertes PPE
            for pidx in range(n):
                for v in violations_pp[pidx]:
                    global_alerts.append(f"P{pidx+1} | {v}")

        # ── 3. Feu / Fumée ───────────────────────────────────────────────
        fire_events = []
        if MODES["Feu/Fumée"]:
            fire_events = smoke_fire.detect(frame, model_fire)
            for ev in fire_events:
                global_alerts.append(f"⚠ {ev['type']}")
                log_event("FEU_FUMEE", ev["type"])

        # ── 4. Chute / Personne évanouie ─────────────────────────────────
        fall_results = []
        if MODES["Chute"]:
            fall_results = fall_det.process(personnes)
            for pidx, fallen in fall_results:
                if fallen:
                    global_alerts.append(f"P{pidx+1} | 🚑 PERSONNE TOMBÉE !")
                    log_event("CHUTE", f"Personne {pidx+1} tombée/allongée")

        # ── 5. Franchissement de ligne ────────────────────────────────────
        line_events = []
        if MODES["Ligne"] and line_det.lines:
            line_events = line_det.process(personnes, prev_personnes)
            for pidx, lname in line_events:
                global_alerts.append(f"P{pidx+1} | 🚧 {lname} franchie !")
                log_event("LIGNE", f"P{pidx+1} a franchi {lname}")

        # ── 6. Poste / Présence ───────────────────────────────────────────
        post_alerts = []
        if MODES["Poste"] and post_det.zones:
            post_alerts = post_det.process(personnes)
            for za in post_alerts:
                msg = f"👮 {za['name']} ABSENT {int(za['absent_duration'])}s"
                global_alerts.append(msg)
                if int(za['absent_duration']) % 30 == 0:
                    log_event("POSTE", msg)

        # ── 7. Objet abandonné ────────────────────────────────────────────
        abandon_events = []
        if MODES["Abandon"]:
            abandon_events = abandon_det.process(frame, personnes)
            for obj in abandon_events:
                global_alerts.append(f"💼 OBJET ABANDONNÉ {int(obj['static_seconds'])}s")
                log_event("ABANDON", f"Objet statique depuis {obj['static_seconds']:.0f}s")

        # ── 8. LPR ───────────────────────────────────────────────────────
        lpr_events = []
        if MODES["LPR"]:
            lpr_events = lpr_det.detect(frame, model_person)
            for lpr in lpr_events:
                if lpr["plate_text"] and lpr["plate_text"] != "—":
                    if lpr["plate_text"] not in lpr_buffer:
                        lpr_buffer.insert(0, lpr["plate_text"])
                        lpr_buffer[:] = lpr_buffer[:10]
                        log_event("LPR", f"Plaque lue: {lpr['plate_text']}")

        # ── 9. Portes ─────────────────────────────────────────────────────
        if MODES["Porte"] and door_det.zones:
            door_det.process(frame)

        # ════════════════════════════════════════════════════════════════
        #  DESSIN
        # ════════════════════════════════════════════════════════════════

        # Feu/Fumée
        for ev in fire_events:
            x1,y1,x2,y2 = ev["box"]
            cv2.rectangle(frame, (x1,y1), (x2,y2), ev["color"], 3)
            dessiner_label(frame, ev["type"], x1, y1, ev["color"], 0.60)

        # Objets abandonnés
        for obj in abandon_events:
            x1,y1,x2,y2 = obj["box"]
            dur = int(obj["static_seconds"])
            pulse = int(now * 3) % 2 == 0
            color = (0,0,200) if pulse else (0,0,130)
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 3)
            dessiner_label(frame, f"ABANDONNÉ {dur}s", x1, y1, color, 0.55)

        # LPR
        for lpr in lpr_events:
            vx1,vy1,vx2,vy2 = lpr["vehicle_box"]
            px1,py1,px2,py2 = lpr["plate_box"]
            cv2.rectangle(frame, (vx1,vy1), (vx2,vy2), (255,200,0), 2)
            cv2.rectangle(frame, (px1,py1), (px2,py2), (0,200,255), 2)
            dessiner_label(frame, f"🚗 {lpr['plate_text']}", px1, py1,
                           (0,160,200), 0.58)

        # LPR buffer (historique)
        if MODES["LPR"] and lpr_buffer:
            bx = w - 240; by = 40
            cv2.rectangle(frame, (bx-5, by-20), (w-5, by+len(lpr_buffer)*22+10),
                          (20,20,40), -1)
            cv2.putText(frame, "Plaques lues :", (bx, by),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,200,255), 1)
            for i, pl in enumerate(lpr_buffer):
                cv2.putText(frame, pl, (bx, by+20+i*22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200,255,200), 1)

        # Personnes (PPE)
        if MODES["PPE"]:
            for pidx, (px1,py1,px2,py2) in enumerate(personnes):
                color = COULEURS_PERSONNES[pidx % len(COULEURS_PERSONNES)]
                bc    = (0,0,200) if violations_pp and violations_pp[pidx] else color
                cv2.rectangle(frame, (px1,py1), (px2,py2), bc, 3)
                dessiner_label(frame, f"P{pidx+1}", px1, py1, bc, 0.60)

            # Chute par-dessus la bounding box
            for pidx, fallen in fall_results:
                if fallen and pidx < len(personnes):
                    px1,py1,px2,py2 = personnes[pidx]
                    cv2.rectangle(frame, (px1,py1), (px2,py2), (0,0,220), 4)
                    dessiner_label(frame, "🚑 TOMBÉ", px1, py2-30,
                                   (0,0,200), 0.65)

            # Boîtes EPI
            for (cls,sc,x1,y1,x2,y2,pidx,is_danger) in \
                    (ppe_avec_personne if 'ppe_avec_personne' in dir() else []):
                if sc < 0.08: continue
                color = COULEURS_PERSONNES[pidx % len(COULEURS_PERSONNES)]
                col   = (0,0,200) if is_danger else color
                lbl   = f"{TRADUCTION_PPE.get(cls,cls)} {sc:.0%}"
                cv2.rectangle(frame, (x1,y1), (x2,y2), col, 2)
                dessiner_label(frame, lbl, x1, y1, col, 0.48)

        # Ligne virtuelle
        if MODES["Ligne"]:
            line_det.draw(frame)

        # Postes
        if MODES["Poste"]:
            post_det.draw(frame)

        # Portes
        if MODES["Porte"]:
            door_det.draw(frame)

        # Panneau EPI
        if MODES["PPE"] and personnes:
            draw_ppe_panel(frame, personnes, ppe_stable, violations_pp, w, h)

        # Alertes globales
        if global_alerts:
            draw_alert_sidebar(frame, global_alerts[:12], w)

        # HUD header
        draw_hud_header(frame, fps, len(personnes), MODES, w, h)

        # Barre de statut bas
        cv2.rectangle(frame, (0,h-38), (w,h), (20,20,30), -1)
        helps = "Q=Quitter  S=Screenshot  D=Debug  L=Ligne  P=Poste  O=Porte  F=Feu  C=Chute  R=LPR  A=Abandon  X=Effacer"
        cv2.putText(frame, helps, (6,h-12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160,180,200), 1)

        cv2.imshow(WIN, frame)
        prev_personnes = list(personnes)

        # ── Gestion clavier ──────────────────────────────────────────────
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        elif k == ord('s'):
            p = SCREENSHOTS / f"sec_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(str(p), frame)
            print(f"  Screenshot → {p}")
        elif k == ord('d'):
            DEBUG = not DEBUG
            print(f"  DEBUG {'ON' if DEBUG else 'OFF'}")
        elif k == ord('f'):
            MODES["Feu/Fumée"] = not MODES["Feu/Fumée"]
            print(f"  Feu/Fumée: {'ON' if MODES['Feu/Fumée'] else 'OFF'}")
        elif k == ord('c'):
            MODES["Chute"] = not MODES["Chute"]
            print(f"  Chute: {'ON' if MODES['Chute'] else 'OFF'}")
        elif k == ord('l'):
            MODES["Ligne"] = True
            line_det.start_edit()
            mouse_router.activate("ligne")
        elif k == ord('p'):
            MODES["Poste"] = True
            post_det.start_edit()
            mouse_router.activate("poste")
        elif k == ord('o'):
            MODES["Porte"] = True
            door_det.start_edit()
            mouse_router.activate("porte")
        elif k == ord('a'):
            MODES["Abandon"] = not MODES["Abandon"]
            print(f"  Abandon: {'ON' if MODES['Abandon'] else 'OFF'}")
        elif k == ord('r'):
            MODES["LPR"] = not MODES["LPR"]
            print(f"  LPR: {'ON' if MODES['LPR'] else 'OFF'}")
        elif k == ord('x'):
            line_det.lines.clear()
            post_det.zones.clear()
            door_det.zones.clear()
            mouse_router.activate(None)
            print("  Toutes les zones effacées.")
        elif k == 27:   # Echap
            mouse_router.activate(None)

    cap.release()
    cv2.destroyAllWindows()
    print("\n  Système arrêté. Logs : security_logs/")


if __name__ == "__main__":
    main()
