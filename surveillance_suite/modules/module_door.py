"""
Module 5 (v2 - version robuste) : Détection porte ouverte / fermée.

Améliorations vs v1 :
1. SSIM (Structural Similarity) au lieu du simple diff de pixels
   -> beaucoup plus robuste au bruit caméra et aux variations de luminosité,
      car il compare la STRUCTURE de l'image, pas juste l'intensité brute.
2. CLAHE (égalisation d'histogramme adaptative) avant comparaison
   -> neutralise les changements d'exposition automatique de la caméra.
3. Calibration MULTI-FRAMES moyennée (au lieu d'une seule image)
   -> élimine le bruit ponctuel d'un seul frame de référence.
4. Anti-flicker / stabilité temporelle
   -> l'état ne change que si N frames consécutives confirment le
      changement, ce qui élimine les faux positifs "clignotants".
5. Score affiché en direct pour faciliter le réglage du seuil.

Dépendance : scikit-image (déjà dans requirements.txt du projet)
    pip install scikit-image
Si absent, le module retombe automatiquement sur un diff de pixels classique.
"""

import cv2
import numpy as np

try:
    from skimage.metrics import structural_similarity as ssim
    _SSIM_AVAILABLE = True
except ImportError:
    _SSIM_AVAILABLE = False


class DoorStateDetector:
    def __init__(self, roi, diff_threshold=0.12, stability_frames=5, calibration_samples=15):
        """
        roi                 : (x1, y1, x2, y2) zone de la porte, la plus
                               serrée possible (juste la porte, pas le mur autour)
        diff_threshold      : seuil de dissimilarité (0 = identique, 1 = totalement
                               différent). Avec SSIM, 0.10-0.20 est un bon point
                               de départ. Augmente si trop de faux positifs,
                               diminue si la porte ouverte n'est pas détectée.
        stability_frames    : nombre de frames consécutives nécessaires pour
                               confirmer un changement d'état (anti-flicker)
        calibration_samples : nombre de frames moyennées pendant la calibration
        """
        self.roi = roi
        self.diff_threshold = diff_threshold
        self.stability_frames = stability_frames
        self.calibration_samples = calibration_samples

        self.reference = None
        self.is_open = False
        self.last_score = 0.0

        self._pending_state = False
        self._pending_count = 0

        self._calibrating = False
        self._calib_buffer = []

        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # ------------------------------------------------------------------
    # Prétraitement commun (utilisé pour la calibration ET la comparaison)
    # ------------------------------------------------------------------
    def _preprocess(self, frame):
        x1, y1, x2, y2 = self.roi
        crop = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = self._clahe.apply(gray)              # normalise la luminosité
        gray = cv2.GaussianBlur(gray, (5, 5), 0)     # réduit le bruit fin
        return gray

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------
    def start_calibration(self):
        """Démarre une calibration étalée sur plusieurs frames (recommandé)."""
        self._calibrating = True
        self._calib_buffer = []

    def calibrate(self, frame):
        """
        Appelle cette méthode à chaque frame pendant la calibration.
        Retourne True quand la calibration est TERMINÉE, False si elle
        est encore en cours (continue d'appeler avec les frames suivantes).

        Si start_calibration() n'a pas été appelé avant, calibre en une
        seule frame (comportement simple, moins robuste).
        """
        gray = self._preprocess(frame)

        if not self._calibrating:
            self.reference = gray.astype(np.float32)
            return True

        self._calib_buffer.append(gray.astype(np.float32))
        if len(self._calib_buffer) >= self.calibration_samples:
            self.reference = np.mean(self._calib_buffer, axis=0)
            self._calibrating = False
            self._calib_buffer = []
            return True
        return False

    @property
    def is_calibrating(self):
        return self._calibrating

    # ------------------------------------------------------------------
    # Mise à jour / détection
    # ------------------------------------------------------------------
    def update(self, frame, occluded=False):
        """
        Retourne (is_open: bool, diff_score: float).

        occluded : si True, la frame est ignorée pour la décision (ex: une
                   personne masque le ROI). L'état affiché reste celui
                   précédemment confirmé, et les compteurs de stabilité ne
                   sont ni incrémentés ni réinitialisés -> dès que la
                   personne s'écarte, la détection reprend normalement
                   sans redémarrer à zéro.
        """
        if self.reference is None:
            return False, 0.0

        if occluded:
            return self.is_open, self.last_score

        gray = self._preprocess(frame)
        ref = self.reference.astype(np.uint8)

        if _SSIM_AVAILABLE:
            similarity = ssim(ref, gray)
            diff_score = 1.0 - similarity   # 0 = identique, 1 = très différent
        else:
            diff = cv2.absdiff(ref, gray)
            diff_score = float(np.mean(diff)) / 255.0

        self.last_score = diff_score
        raw_state = diff_score > self.diff_threshold

        # --- Anti-flicker : il faut N frames consécutives d'accord ---
        if raw_state == self._pending_state:
            self._pending_count += 1
        else:
            self._pending_state = raw_state
            self._pending_count = 1

        if self._pending_count >= self.stability_frames:
            self.is_open = raw_state

        return self.is_open, diff_score

    # ------------------------------------------------------------------
    # Affichage
    # ------------------------------------------------------------------
    def draw(self, frame):
        x1, y1, x2, y2 = self.roi
        if self.is_calibrating:
            color, label = (0, 200, 255), f"Calibration... ({len(self._calib_buffer)}/{self.calibration_samples})"
        elif self.is_open:
            color, label = (0, 0, 255), "PORTE OUVERTE"
        else:
            color, label = (0, 200, 0), "Porte fermee"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label}  score={self.last_score:.3f}", (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        return frame


def compute_occlusion_ratio(roi, boxes):
    """
    Calcule quelle fraction du ROI de la porte est recouverte par au moins
    une des boîtes données (ex: boîtes de personnes détectées par YOLO).

    roi   : (x1, y1, x2, y2)
    boxes : liste de (x1, y1, x2, y2)

    Retourne un float entre 0.0 (aucun recouvrement) et 1.0 (ROI entièrement
    recouvert).
    """
    rx1, ry1, rx2, ry2 = roi
    roi_area = max(1, (rx2 - rx1) * (ry2 - ry1))

    # Masque binaire pour gérer correctement le recouvrement de plusieurs boîtes
    mask = np.zeros((ry2 - ry1, rx2 - rx1), dtype=np.uint8)
    for (bx1, by1, bx2, by2) in boxes:
        ix1, iy1 = max(rx1, bx1), max(ry1, by1)
        ix2, iy2 = min(rx2, bx2), min(ry2, by2)
        if ix2 > ix1 and iy2 > iy1:
            mask[iy1 - ry1: iy2 - ry1, ix1 - rx1: ix2 - rx1] = 1

    covered = int(mask.sum())
    return covered / roi_area


def select_roi_interactively(frame, window_name="Selectionne la porte puis ENTREE"):
    """
    Ouvre une fenêtre pour sélectionner le ROI de la porte à la souris.
    Clique-glisse un rectangle autour de la porte, puis appuie sur ENTREE
    (ou ESPACE). Appuie sur 'c' pour annuler.
    Retourne (x1, y1, x2, y2).
    """
    r = cv2.selectROI(window_name, frame, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(window_name)
    x, y, w, h = r
    if w == 0 or h == 0:
        raise ValueError("Aucune zone sélectionnée pour le ROI de la porte.")
    return (int(x), int(y), int(x + w), int(y + h))
