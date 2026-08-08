from ultralytics import YOLO
from collections import deque
import cv2, time
from pathlib import Path
from datetime import datetime

print("Chargement modele personnes..."); model_person = YOLO("yolov8n.pt")
print("Chargement modele 1..."); model1 = YOLO("best.pt")
print("Chargement modele 2..."); model2 = YOLO("best_gloves.pt")
print("Modeles prets !\n")

# ── Classes connues ──────────────────────────────────────────────────────────
# M1 (best.pt)        : Fall-Detected, Gloves, Goggles, Hardhat, Ladder, Mask,
#                       NO-Gloves, NO-Goggles, NO-Hardhat, NO-Mask,
#                       NO-Safety Vest, Person, Safety Cone, Safety Vest
# M2 (best_gloves.pt) : Gloves, Vest, goggles, helmet, mask, safety_shoe
# ─────────────────────────────────────────────────────────────────────────────

# ── P2 : table de correspondance M2 -> M1 ─────────────────────────────────────
# Les deux modèles couvrent en grande partie les mêmes EPI sous des noms
# différents (`Hardhat`/`helmet`, `Safety Vest`/`Vest`...). Sans cette table,
# le même gilet physique produit 2 boîtes superposées (une par modèle) à
# l'affichage, et 2x le calcul pour rien. On ne peut PAS ré-entraîner un
# modèle unique (aucun dataset local pour les classes propres à M2, en
# particulier `safety_shoe` -- cf. `improvements/p2_table_correspondance_epi.py`
# pour le diagnostic complet) : cette table est le correctif retenu, sans
# ré-entraînement. `safety_shoe` reste le seul apport net de M2, jamais
# dédoublonnée (aucun équivalent dans M1).
CORRESPONDANCE_M2_VERS_M1 = {
    "gloves": "Gloves", "vest": "Safety Vest", "goggles": "Goggles",
    "helmet": "Hardhat", "mask": "Mask",
}
IOU_DEDUP_SEUIL = 0.4

DEBUG = False   # Touche D en direct pour activer le mode diagnostic

# ── Traductions vers le français (clés = noms de classes exacts des modèles) ──
TRADUCTION = {
    # M1
    "Hardhat":        "Casque porte",
    "NO-Hardhat":     "SANS CASQUE !",
    "Mask":           "Masque porte",
    "NO-Mask":        "SANS MASQUE !",
    "Goggles":        "Lunettes portees",
    "NO-Goggles":     "SANS LUNETTES !",
    "Gloves":         "Gants portes",
    "NO-Gloves":      "SANS GANTS !",
    "Safety Vest":    "Gilet porte",
    "NO-Safety Vest": "SANS GILET !",
    "Fall-Detected":  "CHUTE DETECTEE !",
    "Person":         "Personne",
    "Ladder":         "Echelle",
    "Safety Cone":    "Cone securite",
    # M2
    "helmet":         "Casque porte",
    "mask":           "Masque porte",
    "goggles":        "Lunettes portees",
    "Vest":           "Gilet porte",
    "safety_shoe":    "Chaussures securite",
}

# ── Classes qui déclenchent une alerte directe ──
CLASSES_DANGER = {
    "NO-Hardhat", "NO-Mask", "NO-Goggles",
    "NO-Gloves", "NO-Safety Vest", "Fall-Detected",
}

# ── Association mot-clé → EPI (basée sur les noms de classes réels) ──
MOTS_CLES_EPI = {
    "casque":     ["hardhat", "helmet"],
    "masque":     ["mask"],
    "lunettes":   ["goggle"],
    "gilet":      ["safety vest", "vest"],
    "gants":      ["glove"],
    "chaussures": ["safety_shoe", "shoe"],
}

# ── EPI dont l'ABSENCE génère automatiquement une alerte ──
EPI_OBLIGATOIRES = {
    "casque":     True,
    "masque":     False,   # contexte-dépendant
    "lunettes":   False,   # rare en situation réelle
    "gilet":      True,
    "gants":      False,   # lissage nécessaire, désactivé par défaut
    "chaussures": False,
}

# ── Seuils de confiance calibrés sur logs réels ──────────────────────────────
CONF_EPI = {
    "casque":     0.15,   # Hardhat max ~0.55 ; helmet max ~0.04 → seuil bas
    "masque":     0.08,   # Mask max ~0.21 → seuil bas indispensable
    "lunettes":   0.08,   # Goggles M1 correct ; goggles M2 max ~0.04
    "gilet":      0.12,   # Safety Vest max ~0.32 ; Vest max ~0.13
    "gants":      0.08,   # Gloves max ~0.19
    "chaussures": 0.20,   # safety_shoe — pas encore observé en logs
}

# ── Lissage temporel ──────────────────────────────────────────────────────────
# EPI validé = détecté dans HIST_K des HIST_N dernières frames
# Evite les faux positifs avec des seuils de confiance très bas
HIST_N = 12
HIST_K = 4

_historique: dict = {}

def get_hist(pidx: int, epi: str) -> deque:
    if pidx not in _historique:
        _historique[pidx] = {}
    if epi not in _historique[pidx]:
        _historique[pidx][epi] = deque([False] * HIST_N, maxlen=HIST_N)
    return _historique[pidx][epi]

def reset_hist_si_besoin(n_personnes: int):
    for pidx in list(_historique.keys()):
        if pidx >= n_personnes:
            del _historique[pidx]

# ── Couleurs par personne (format BGR) ───────────────────────────────────────
COULEURS_PERSONNES = [
    (0,   255, 255),  # P1  — Jaune
    (0,   100, 255),  # P2  — Orange
    (255,  50,  50),  # P3  — Bleu royal
    (50,  255,  50),  # P4  — Vert lime
    (255,   0, 200),  # P5  — Magenta
    (0,   200,   0),  # P6  — Vert
    (200,   0, 255),  # P7  — Violet
    (255, 200,   0),  # P8  — Cyan
    (0,     0, 255),  # P9  — Rouge
    (220, 150,   0),  # P10 — Bleu acier
    (0,   255, 150),  # P11 — Vert-jaune
    (255,   0,  80),  # P12 — Indigo
]

LABELS_ABSENCE = {
    "casque":     "SANS CASQUE !",
    "masque":     "SANS MASQUE !",
    "lunettes":   "SANS LUNETTES !",
    "gilet":      "SANS GILET !",
    "gants":      "SANS GANTS !",
    "chaussures": "SANS CHAUSSURES !",
}

SCREENSHOTS = Path("ppe_screenshots")
SCREENSHOTS.mkdir(exist_ok=True)


# ── Fonctions utilitaires ─────────────────────────────────────────────────────

def iou(boxA, boxB):
    """Intersection sur Union de deux boîtes (x1,y1,x2,y2)."""
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0


def distance_centres(boxA, boxB):
    """Distance euclidienne entre les centres de deux boîtes."""
    cxA = (boxA[0] + boxA[2]) / 2; cyA = (boxA[1] + boxA[3]) / 2
    cxB = (boxB[0] + boxB[2]) / 2; cyB = (boxB[1] + boxB[3]) / 2
    return ((cxA - cxB) ** 2 + (cyA - cyB) ** 2) ** 0.5


def assigner_personne(ppe_box, personnes):
    """Associe une boîte EPI à la personne la plus proche (IoU puis distance)."""
    if not personnes:
        return -1
    meilleur_idx, meilleur_iou = 0, 0.0
    for i, p in enumerate(personnes):
        s = iou(ppe_box, p)
        if s > meilleur_iou:
            meilleur_iou = s
            meilleur_idx = i
    if meilleur_iou > 0.05:
        return meilleur_idx
    meilleur_idx, meilleur_dist = 0, float("inf")
    for i, p in enumerate(personnes):
        d = distance_centres(ppe_box, p)
        if d < meilleur_dist:
            meilleur_dist = d
            meilleur_idx = i
    return meilleur_idx


def dessiner_label(frame, texte, x1, y1, color, font_scale=0.55, ep=2):
    """Dessine un fond coloré puis le texte en blanc."""
    (tw, th), _ = cv2.getTextSize(texte, cv2.FONT_HERSHEY_SIMPLEX, font_scale, ep)
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 8, y1), color, -1)
    cv2.putText(frame, texte, (x1 + 4, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), ep)


# ── Boucle principale ─────────────────────────────────────────────────────────

cap = cv2.VideoCapture(0)
prev_time = time.time()
last_log  = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    ppe_detections = []
    personnes = []

    # 1) Détection des personnes avec le modèle COCO général
    for box in model_person.predict(frame, conf=0.35, classes=[0], verbose=False)[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        personnes.append((x1, y1, x2, y2))

    # 2) Collecte des EPI avec les 2 modèles spécialisés
    conf_globale = 0.01 if DEBUG else 0.03
    brutes_debug = []
    detections_m1 = []  # (cls, sc, x1, y1, x2, y2) -- pour dédup IoU de M2 (P2)

    for model, lm in [(model1, "M1"), (model2, "M2")]:
        for box in model.predict(frame, conf=conf_globale, verbose=False)[0].boxes:
            cls = model.names[int(box.cls)]
            sc  = float(box.conf)
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if DEBUG:
                brutes_debug.append((cls, sc, x1, y1, x2, y2, lm))

            cls_low = cls.lower()
            if "person" in cls_low or "personne" in cls_low:
                if not personnes:
                    personnes.append((x1, y1, x2, y2))
                continue

            if sc < 0.03:
                continue

            # P2 : une détection M2 qui correspond à un concept déjà couvert par
            # M1 est ignorée si elle chevauche (IoU) une détection M1 du même
            # concept -- M1 sert de référence (mieux mesuré, cf. P1). Seule
            # `safety_shoe` (sans équivalent dans M1) est toujours conservée.
            if lm == "M2":
                equiv = CORRESPONDANCE_M2_VERS_M1.get(cls_low)
                if equiv is not None:
                    doublon = any(
                        d_cls == equiv and iou((x1, y1, x2, y2), (dx1, dy1, dx2, dy2)) >= IOU_DEDUP_SEUIL
                        for d_cls, _, dx1, dy1, dx2, dy2 in detections_m1
                    )
                    if doublon:
                        continue

            ppe_detections.append((cls, sc, x1, y1, x2, y2))
            if lm == "M1":
                detections_m1.append((cls, sc, x1, y1, x2, y2))

    # Fallback : personne virtuelle si aucune détectée mais EPI visibles
    if not personnes and ppe_detections:
        xs = [d[2] for d in ppe_detections] + [d[4] for d in ppe_detections]
        ys = [d[3] for d in ppe_detections] + [d[5] for d in ppe_detections]
        personnes = [(min(xs), min(ys), max(xs), max(ys))]

    n = len(personnes)
    reset_hist_si_besoin(n)

    ppe_par_personne  = [set() for _ in range(n)]
    ppe_avec_personne = []

    # 3) Association EPI → Personne + enregistrement des EPI portés
    for (cls, sc, x1, y1, x2, y2) in ppe_detections:
        cls_low = cls.lower()
        pidx = assigner_personne((x1, y1, x2, y2), personnes)
        if pidx == -1:
            pidx = 0

        est_negatif = (
            cls_low.startswith("no-") or
            cls_low.startswith("no ") or
            cls_low.startswith("without")
        )
        for epi, mots in MOTS_CLES_EPI.items():
            if any(k in cls_low for k in mots) and not est_negatif:
                if sc >= CONF_EPI.get(epi, 0.10):
                    ppe_par_personne[pidx].add(epi)

        is_danger = cls in CLASSES_DANGER or "no-" in cls_low[:4]
        ppe_avec_personne.append((cls, sc, x1, y1, x2, y2, pidx, is_danger))

    # 4) Lissage temporel → ppe_stable
    ppe_stable = [set() for _ in range(n)]
    for pidx in range(n):
        for epi in MOTS_CLES_EPI:
            h_epi = get_hist(pidx, epi)
            h_epi.append(epi in ppe_par_personne[pidx])
            if sum(h_epi) >= HIST_K:
                ppe_stable[pidx].add(epi)

    # 5) Calcul des violations (basé sur ppe_stable)
    violations_par_personne = [[] for _ in range(n)]
    for pidx in range(n):
        ppe = ppe_stable[pidx]
        for (cls, sc, x1, y1, x2, y2, p, is_danger) in ppe_avec_personne:
            if p == pidx and is_danger:
                lbl = TRADUCTION.get(cls, cls)
                if lbl not in violations_par_personne[pidx]:
                    violations_par_personne[pidx].append(lbl)
        for epi, obligatoire in EPI_OBLIGATOIRES.items():
            if obligatoire and epi not in ppe:
                v = LABELS_ABSENCE[epi]
                if v not in violations_par_personne[pidx]:
                    violations_par_personne[pidx].append(v)

    # ── Log debug console (1 fois/s) ──────────────────────────────────────────
    now = time.time()
    if DEBUG and now - last_log >= 1.0:
        last_log = now
        print(f"\n--- Détections brutes ({len(brutes_debug)}) ---")
        for cls, sc, *_, lm in sorted(brutes_debug, key=lambda x: -x[1]):
            epi_match = next(
                (e for e, mots in MOTS_CLES_EPI.items()
                 if any(k in cls.lower() for k in mots)), "—"
            )
            print(f"  [{lm}] {cls:<20} conf={sc:.2f}  → {epi_match}")

    # ── DESSIN ────────────────────────────────────────────────────────────────

    # Mode DEBUG : boîtes grises pour les détections faibles
    if DEBUG:
        for (cls, sc, x1, y1, x2, y2, lm) in brutes_debug:
            if sc < 0.10:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 100, 100), 1)
                cv2.putText(frame, f"[{lm}]{cls} {sc:.0%}",
                            (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (160, 160, 160), 1)

    # Boîtes des personnes
    for pidx, (px1, py1, px2, py2) in enumerate(personnes):
        couleur = COULEURS_PERSONNES[pidx % len(COULEURS_PERSONNES)]
        bc = (0, 0, 220) if violations_par_personne[pidx] else couleur
        cv2.rectangle(frame, (px1, py1), (px2, py2), bc, 3)
        dessiner_label(frame, f"Personne {pidx + 1}", px1, py1, bc, font_scale=0.62)

    # Boîtes des EPI (seuil d'affichage ≥ 0.08 pour éviter le bruit visuel)
    for (cls, sc, x1, y1, x2, y2, pidx, is_danger) in ppe_avec_personne:
        if sc < 0.08:
            continue
        couleur = COULEURS_PERSONNES[pidx % len(COULEURS_PERSONNES)]
        color   = (0, 0, 220) if is_danger else couleur
        label   = f"{TRADUCTION.get(cls, cls)} {sc:.0%}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        dessiner_label(frame, label, x1, y1, color, font_scale=0.50)

    # Alertes (colonne gauche)
    fps = 1.0 / max(time.time() - prev_time, 1e-6)
    prev_time = time.time()
    y_alert = 35
    for pidx in range(n):
        couleur = COULEURS_PERSONNES[pidx % len(COULEURS_PERSONNES)]
        for v in violations_par_personne[pidx]:
            txt  = f"P{pidx + 1} — {v}"
            bg_w = len(txt) * 13 + 20
            cv2.rectangle(frame, (0, y_alert - 28), (bg_w, y_alert + 8), (30, 0, 100), -1)
            cv2.rectangle(frame, (0, y_alert - 28), (6,    y_alert + 8), couleur,       -1)
            cv2.putText(frame, txt, (10, y_alert),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.66, (255, 255, 255), 2)
            y_alert += 38

    # Panneau EPI (bas droite)
    epi_items = [
        ("Casque",     "casque"),
        ("Masque",     "masque"),
        ("Lunettes",   "lunettes"),
        ("Gilet",      "gilet"),
        ("Gants",      "gants"),
        ("Chaussures", "chaussures"),
    ]
    lignes   = n * (len(epi_items) + 1) + 2
    panel_h  = lignes * 21 + 20
    panel_x  = w - 275
    panel_y0 = h - panel_h - 50
    cv2.rectangle(frame, (panel_x - 8, panel_y0), (w - 5, h - 50), (20, 20, 20), -1)

    mode_txt = "DEBUG | D=off" if DEBUG else "D=debug"
    cv2.putText(frame, f"== EPI Status [{mode_txt}] ==",
                (panel_x, panel_y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                (0, 200, 255) if DEBUG else (200, 200, 200), 1)

    for pidx in range(n):
        ppe     = ppe_stable[pidx]
        couleur = COULEURS_PERSONNES[pidx % len(COULEURS_PERSONNES)]
        base_y  = panel_y0 + 20 + pidx * (len(epi_items) + 1) * 21
        cv2.putText(frame, f"  Personne {pidx + 1}",
                    (panel_x, base_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, couleur, 2)
        for j, (nom, key) in enumerate(epi_items):
            porte = key in ppe
            icone = "[OK]" if porte else "[ X]"
            col   = (0, 200, 0) if porte else (0, 0, 220)
            cv2.putText(frame, f"  {icone} {nom}",
                        (panel_x, base_y + 20 + (j + 1) * 21),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, col, 1)

    # Barre de statut
    cv2.rectangle(frame, (0, h - 40), (w, h), (20, 20, 20), -1)
    cv2.putText(frame,
                f"FPS:{fps:.0f}  |  {n} pers.  |  Q=Quitter  S=Screenshot  D=Debug",
                (8, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (180, 180, 180), 1)

    cv2.imshow("Detection EPI - Multi-Personnes", frame)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break
    elif k == ord('s'):
        p = SCREENSHOTS / f"ppe_{datetime.now().strftime('%H%M%S')}.jpg"
        cv2.imwrite(str(p), frame)
        print(f"Screenshot : {p}")
    elif k == ord('d'):
        DEBUG = not DEBUG
        print(f"DEBUG {'ON' if DEBUG else 'OFF'}")

cap.release()
cv2.destroyAllWindows()
