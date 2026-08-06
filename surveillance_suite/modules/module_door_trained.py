"""
Module porte utilisant le modèle de DÉTECTION entraîné (door_state.pt,
mAP50=0.995). Remplace l'ancienne heuristique SSIM de module_door.py.

Contrairement à l'heuristique, ce modèle a appris à reconnaître une porte
ouverte/fermée sur des images variées -> robuste aux occlusions partielles,
variations de luminosité, angle de caméra, etc.
"""

from collections import deque

import cv2
from ultralytics import YOLO


class DoorStateDetectorML:
    def __init__(self, model_path="models/door_state.pt", conf=0.5, smoothing_window=5):
        self.model = YOLO(model_path)
        self.conf = conf
        self.smoothing_window = smoothing_window
        self.history = deque(maxlen=smoothing_window)
        self.is_open = False
        self.last_confidence = 0.0
        self.last_box = None

    def update(self, frame):
        """Retourne (is_open: bool, confidence: float)."""
        results = self.model.predict(frame, conf=self.conf, verbose=False)
        boxes = results[0].boxes if results else None

        if boxes is not None and len(boxes) > 0:
            # Prend la detection la plus confiante si plusieurs portes/etats detectes
            best_idx = int(boxes.conf.argmax())
            label = self.model.names[int(boxes.cls[best_idx])]
            conf = float(boxes.conf[best_idx])
            self.last_box = tuple(boxes.xyxy[best_idx].cpu().numpy().astype(int))
            self.last_confidence = conf
            self.history.append(label.lower() == "open")
        else:
            self.last_box = None
            # aucune detection cette frame -> ne modifie pas l'historique

        if self.history:
            self.is_open = sum(self.history) > (len(self.history) / 2)

        return self.is_open, self.last_confidence

    def draw(self, frame):
        color = (0, 0, 255) if self.is_open else (0, 200, 0)
        label = "PORTE OUVERTE" if self.is_open else "Porte fermee"
        if self.last_box is not None:
            x1, y1, x2, y2 = self.last_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label}  conf={self.last_confidence:.2f}",
                        (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        else:
            cv2.putText(frame, f"{label} (pas de detection)", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        return frame
