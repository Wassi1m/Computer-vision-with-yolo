"""
Module 1 : Détection fumée / incendie.

IMPORTANT : YOLO26 standard (COCO) ne connaît pas les classes "fire"/"smoke".
Il faut un modèle entraîné spécifiquement. Options gratuites :

1. Télécharger un modèle pré-entraîné existant, par ex. sur Roboflow Universe :
   https://universe.roboflow.com/  -> chercher "fire smoke detection yolov8"
   (plusieurs modèles gratuits avec poids .pt téléchargeables)

2. Entraîner le tien avec très peu de code, en partant de yolo26n.pt :

   from ultralytics import YOLO
   model = YOLO("yolo26n.pt")
   model.train(data="fire_smoke_dataset/data.yaml", epochs=100, imgsz=640)

   (dataset annoté "fire"/"smoke" téléchargeable gratuitement sur Roboflow
   Universe au format YOLO, quelques centaines à quelques milliers d'images
   suffisent pour un premier modèle correct)

Une fois le fichier .pt obtenu, place-le dans models/fire_smoke.pt
(chemin défini dans config.py) et ce module fonctionnera directement.
"""

from ultralytics import YOLO


class FireSmokeDetector:
    def __init__(self, model_path, conf=0.4):
        self.model = YOLO(model_path)
        self.conf = conf

    def detect(self, frame):
        """Retourne une liste de dicts {box, label, conf}."""
        results = self.model.predict(frame, conf=self.conf, verbose=False)
        detections = []
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
                cls_id = int(boxes.cls[i])
                label = self.model.names[cls_id]   # ex: "fire" ou "smoke"
                conf = float(boxes.conf[i])
                detections.append({"box": xyxy, "label": label, "conf": conf})
        return detections
