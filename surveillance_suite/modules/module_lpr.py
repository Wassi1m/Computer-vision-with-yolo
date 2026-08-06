"""
Module 4 : Détection automatique de matricules (LPR - License Plate Recognition).

Deux étapes :
1. DÉTECTION de la plaque dans l'image -> nécessite un modèle spécialisé
   (YOLO26 standard ne connaît pas la classe "license_plate").
   Modèles gratuits dispo sur Roboflow Universe, ex: recherche
   "license plate detection yolov8".
2. OCR (lecture du texte) sur la zone détectée -> EasyOCR (gratuit, local).

Installation :
    pip install easyocr --break-system-packages

Place ton modèle de détection de plaques dans models/license_plate.pt
(chemin défini dans config.py).
"""

import cv2
from ultralytics import YOLO

try:
    import easyocr
    _OCR_AVAILABLE = True
except ImportError:
    _OCR_AVAILABLE = False


class LicensePlateReader:
    def __init__(self, model_path, conf=0.4, languages=("en",)):
        self.model = YOLO(model_path)
        self.conf = conf
        self.reader = easyocr.Reader(list(languages), gpu=False) if _OCR_AVAILABLE else None

    def detect_and_read(self, frame):
        """Retourne une liste de dicts {box, text, ocr_conf}."""
        if not _OCR_AVAILABLE:
            raise RuntimeError("easyocr n'est pas installé : pip install easyocr --break-system-packages")

        results = self.model.predict(frame, conf=self.conf, verbose=False)
        plates = []
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                # Pré-traitement simple pour améliorer l'OCR
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

                ocr_result = self.reader.readtext(gray)
                text, ocr_conf = "", 0.0
                if ocr_result:
                    # on prend le résultat avec la meilleure confiance
                    best = max(ocr_result, key=lambda r: r[2])
                    text, ocr_conf = best[1].upper().replace(" ", ""), best[2]

                plates.append(
                    {"box": (x1, y1, x2, y2), "text": text, "ocr_conf": ocr_conf}
                )
        return plates
