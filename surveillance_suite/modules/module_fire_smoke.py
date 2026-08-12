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

# Certains modèles distinguent la fumée lointaine de la fumée proche, parce que
# la première est un objet visuellement différent (voile diffus, faible
# contraste) qu'un entraînement gagne à traiter à part. En exploitation cette
# distinction n'existe pas : de la fumée est de la fumée, et l'aval attend
# l'évènement `smoke`. Sans ce repli, un modèle à trois classes émettrait un
# label `smoke_distant` que rien ne consomme -- la fumée lointaine, justement
# celle qui permet la détection la plus précoce, ne déclencherait jamais rien.
LABELS_EQUIVALENTS = {"smoke_distant": "smoke"}


class FireSmokeDetector:
    def __init__(self, model_path, conf=0.4):
        self.model = YOLO(model_path)
        self.conf = conf

    def detect(self, frame):
        """Retourne une liste de dicts {box, label, conf, label_modele}."""
        results = self.model.predict(frame, conf=self.conf, verbose=False)
        detections = []
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
                cls_id = int(boxes.cls[i])
                brut = self.model.names[cls_id]   # ex: "fire", "smoke", "smoke_distant"
                conf = float(boxes.conf[i])
                detections.append({
                    "box": xyxy,
                    "label": LABELS_EQUIVALENTS.get(brut, brut),
                    # Le label d'origine reste disponible : la couche de
                    # qualification (plan v5) s'en sert comme indice de distance,
                    # et le perdre ici serait irrécupérable en aval.
                    "label_modele": brut,
                    "conf": conf,
                })
        return detections
