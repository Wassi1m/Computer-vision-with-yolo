"""
Détection d'objet abandonné (sac, valise, bagage sans personne à proximité
pendant une durée prolongée).

Principe (state machine par objet suivi) :
1. YOLO détecte + suit (track_id) les objets cibles (sac, valise, etc.) et
   les personnes.
2. Pour chaque objet suivi, on garde un historique de position -> si le
   déplacement max sur la fenêtre récente reste sous un seuil, l'objet est
   "stationnaire".
3. À chaque frame, si une personne est à proximité de l'objet, on met à
   jour son "dernier instant surveillé".
4. Si un objet est stationnaire ET que le temps écoulé depuis la dernière
   présence humaine à proximité dépasse un seuil -> alerte "abandonné".

Aucun entraînement nécessaire : les classes (sac à dos, valise, sac à main)
existent déjà dans COCO, donc dans le modèle général déjà utilisé.
"""

import time
from collections import deque

import cv2
from ultralytics import YOLO

DEFAULT_OBJECT_CLASSES = {
    "backpack", "handbag", "suitcase", "umbrella",   # bagagerie classique
    "bottle", "laptop", "cell phone", "book",         # objets personnels courants
    "bicycle", "skateboard",                           # objets encombrants
}


class AbandonedObjectDetector:
    def __init__(
        self,
        model_path="yolo26n.pt",
        conf=0.4,
        object_classes=None,
        proximity_radius_px=150,
        stationary_threshold_px=25,
        stationary_window_s=5.0,
        unattended_time_threshold_s=30.0,
        position_history_len=30,
        debug=False,
    ):
        self.model = YOLO(model_path)
        self.conf = conf
        self.object_classes = object_classes or DEFAULT_OBJECT_CLASSES
        self.proximity_radius_px = proximity_radius_px
        self.stationary_threshold_px = stationary_threshold_px
        self.stationary_window_s = stationary_window_s
        self.unattended_time_threshold_s = unattended_time_threshold_s
        self.debug = debug

        # Etat par track_id d'objet surveille
        self.position_history = {}   # track_id -> deque[(t, cx, cy)]
        self.last_person_nearby_time = {}  # track_id -> timestamp
        self.first_seen_time = {}    # track_id -> timestamp (pour eviter alerte immediate)
        self._history_len = position_history_len

    @staticmethod
    def _center(box):
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @staticmethod
    def _distance(p1, p2):
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

    def _is_stationary(self, track_id):
        history = self.position_history.get(track_id)
        if not history or len(history) < 2:
            return False
        now = time.time()
        recent = [(t, x, y) for (t, x, y) in history if now - t <= self.stationary_window_s]
        if len(recent) < 2:
            return False
        xs = [x for _, x, _ in recent]
        ys = [y for _, _, y in recent]
        max_disp = ((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2) ** 0.5
        return max_disp <= self.stationary_threshold_px

    def predict(self, frame):
        """
        Retourne un dict :
            {
              "objects": [{"box","track_id","label","stationary","unattended_s","abandoned"}],
              "alerts": [...],
            }
        """
        results = self.model.track(frame, persist=True, conf=self.conf, verbose=False)
        now = time.time()

        person_centers = []
        candidate_objects = []

        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                box = tuple(boxes.xyxy[i].cpu().numpy().astype(int))
                label = self.model.names[int(boxes.cls[i])]
                track_id = int(boxes.id[i]) if boxes.id is not None else -1

                if label == "person":
                    person_centers.append(self._center(box))
                elif label in self.object_classes and track_id != -1:
                    candidate_objects.append({"box": box, "track_id": track_id, "label": label})

        if self.debug:
            all_detections = []
            if results and results[0].boxes is not None:
                boxes = results[0].boxes
                for i in range(len(boxes)):
                    label = self.model.names[int(boxes.cls[i])]
                    conf_val = float(boxes.conf[i])
                    all_detections.append(f"{label}:{conf_val:.2f}")
            print(f"[DEBUG] Detections brutes (conf>={self.conf}): {all_detections}")
            print(f"[DEBUG] Personnes: {len(person_centers)} | "
                  f"Objets candidats (classes surveillees): {len(candidate_objects)} "
                  f"-> {[o['label'] for o in candidate_objects]}")
            print(f"[DEBUG] Classes surveillees: {sorted(self.object_classes)}")

        objects_result = []
        alerts = []

        for obj in candidate_objects:
            tid = obj["track_id"]
            center = self._center(obj["box"])

            if tid not in self.position_history:
                self.position_history[tid] = deque(maxlen=self._history_len)
                self.first_seen_time[tid] = now
                self.last_person_nearby_time[tid] = now  # au depart, pas d'alerte immediate
            self.position_history[tid].append((now, center[0], center[1]))

            # Une personne est-elle proche de cet objet MAINTENANT ?
            person_nearby = any(self._distance(center, p) <= self.proximity_radius_px for p in person_centers)
            if person_nearby:
                self.last_person_nearby_time[tid] = now

            stationary = self._is_stationary(tid)
            unattended_s = now - self.last_person_nearby_time[tid]
            abandoned = stationary and unattended_s >= self.unattended_time_threshold_s

            objects_result.append({
                "box": obj["box"], "track_id": tid, "label": obj["label"],
                "stationary": stationary, "unattended_s": unattended_s, "abandoned": abandoned,
            })

            if abandoned:
                alerts.append({
                    "level": "critical",
                    "message": f"Objet abandonne detecte: {obj['label']} #{tid} "
                               f"(sans surveillance depuis {unattended_s:.0f}s)",
                })

        # Nettoyage : oublie les objets qui ont disparu (evite fuite memoire long terme)
        active_ids = {obj["track_id"] for obj in candidate_objects}
        for tid in list(self.position_history.keys()):
            if tid not in active_ids:
                self.position_history.pop(tid, None)
                self.last_person_nearby_time.pop(tid, None)
                self.first_seen_time.pop(tid, None)

        return {"objects": objects_result, "alerts": alerts}

    def draw(self, frame, result: dict):
        for obj in result["objects"]:
            x1, y1, x2, y2 = obj["box"]
            if obj["abandoned"]:
                color = (0, 0, 255)
                label = f"ABANDONNE! {obj['label']} #{obj['track_id']} ({obj['unattended_s']:.0f}s)"
            elif obj["stationary"]:
                color = (0, 165, 255)
                label = f"{obj['label']} #{obj['track_id']} immobile ({obj['unattended_s']:.0f}s)"
            else:
                color = (0, 200, 0)
                label = f"{obj['label']} #{obj['track_id']}"

            thickness = 3 if obj["abandoned"] else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(frame, label, (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame
