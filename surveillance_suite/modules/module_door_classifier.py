"""
Étape 3/3 - Module d'inférence : porte ouverte/fermée via classifieur entraîné.

Remplace l'ancienne heuristique SSIM par un modèle appris sur les images de
TA propre porte -> beaucoup plus robuste au bruit caméra, à l'éclairage et
aux occlusions partielles.

Utilise en plus un vote majoritaire sur une fenêtre glissante de frames pour
éviter les changements d'état trop rapides (anti-flicker).
"""

import json
import os
from collections import deque

import cv2
from ultralytics import YOLO


class DoorClassifier:
    def __init__(self, model_path, roi, smoothing_window=7, min_confidence=0.6):
        self.model = YOLO(model_path)
        self.roi = roi
        self.smoothing_window = smoothing_window
        self.min_confidence = min_confidence
        self.history = deque(maxlen=smoothing_window)
        self.is_open = False
        self.last_confidence = 0.0

    def update(self, frame):
        """Retourne (is_open: bool, confidence: float)."""
        x1, y1, x2, y2 = self.roi
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return self.is_open, self.last_confidence

        result = self.model.predict(crop, verbose=False)[0]
        probs = result.probs
        top_idx = int(probs.top1)
        top_label = result.names[top_idx]
        top_conf = float(probs.top1conf)

        # On ne prend en compte la prédiction que si elle est assez confiante,
        # sinon on la traite comme "incertain" et on garde l'historique tel quel
        if top_conf >= self.min_confidence:
            self.history.append(top_label == "open")

        self.last_confidence = top_conf

        if self.history:
            # Vote majoritaire sur la fenêtre glissante
            self.is_open = sum(self.history) > (len(self.history) / 2)

        return self.is_open, self.last_confidence

    def draw(self, frame):
        x1, y1, x2, y2 = self.roi
        color = (0, 0, 255) if self.is_open else (0, 200, 0)
        label = "PORTE OUVERTE" if self.is_open else "Porte fermee"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label}  conf={self.last_confidence:.2f}",
                    (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        return frame


def load_trained_door_classifier(model_path="models/door_classifier.pt",
                                  roi_path="dataset/roi.json",
                                  smoothing_window=7, min_confidence=0.6):
    """
    Charge le classifieur entraîné + son ROI associé, si disponibles.
    Retourne None si le modèle n'a pas encore été entraîné (permet un
    fallback propre vers l'ancienne méthode heuristique dans main.py).
    """
    if not os.path.exists(model_path) or not os.path.exists(roi_path):
        return None

    with open(roi_path) as f:
        roi = tuple(json.load(f)["roi"])

    return DoorClassifier(model_path, roi, smoothing_window, min_confidence)
