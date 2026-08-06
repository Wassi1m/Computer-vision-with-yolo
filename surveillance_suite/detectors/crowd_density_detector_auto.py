"""
Détection de foule et de proximité, 100% AUTOMATIQUE (aucune calibration
manuelle, aucun clic, aucun rectangle à définir).

Principe :
- YOLO détecte les personnes -> boîtes englobantes
- On suppose une taille humaine moyenne (1.70m par défaut, réglable)
- Pour chaque personne détectée, sa hauteur en pixels donne l'échelle
  locale : pixels_par_metre = hauteur_boite_px / taille_humaine_m
  (plus la personne est grande à l'écran, plus elle est proche de la
  caméra -> échelle plus grande, ce qui compense la perspective)
- Pour une paire de personnes, on utilise l'échelle moyenne des deux pour
  convertir leur distance en pixels vers une distance estimée en mètres
- Alerte si distance estimée < seuil, alerte "foule" si trop de monde

Limites (approximation, pas une vraie calibration) :
- Suppose que tout le monde a une taille proche de la moyenne
- Suppose une camera pas trop inclinee (fonctionne mieux si la camera
  est a hauteur d'homme, vue quasi-frontale, que en plongee forte)
- Moins precis qu'une calibration par homographie, mais zero setup
  necessaire -> ideal pour un deploiement rapide multi-cameras

Usage :
    python run_crowd_detection_auto.py
"""

from itertools import combinations

import cv2
from ultralytics import YOLO


class AutoCrowdDetector:
    def __init__(
        self,
        model_path="yolo26n.pt",
        conf=0.4,
        imgsz=640,
        average_human_height_m=1.70,
        min_distance_m=1.0,
        crowd_count_threshold=5,
    ):
        self.model = YOLO(model_path)
        self.conf = conf
        self.imgsz = imgsz
        self.average_human_height_m = average_human_height_m
        self.min_distance_m = min_distance_m
        self.crowd_count_threshold = crowd_count_threshold

    @staticmethod
    def _foot_point(box):
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2, y2)

    def _pixels_per_meter(self, box):
        x1, y1, x2, y2 = box
        height_px = max(y2 - y1, 1)
        return height_px / self.average_human_height_m

    def _predict_boxes_simple(self, frame):
        """Inference standard (1 passage sur l'image entiere)."""
        results = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz, verbose=False, classes=[0])
        boxes = []
        if results and results[0].boxes is not None:
            for i in range(len(results[0].boxes)):
                boxes.append(tuple(results[0].boxes.xyxy[i].cpu().numpy().astype(int)))
        return boxes

    def _predict_boxes_tiled(self, frame, grid=(3, 3), overlap=0.2):
        """
        Inference par tuiles : decoupe l'image en grid[0] x grid[1] tuiles
        avec chevauchement, lance la detection sur chaque tuile en pleine
        resolution -> detecte des personnes bien plus petites/lointaines
        qu'une inference sur l'image entiere redimensionnee.
        """
        h, w = frame.shape[:2]
        rows, cols = grid
        tile_h, tile_w = h / rows, w / cols
        overlap_h, overlap_w = int(tile_h * overlap), int(tile_w * overlap)

        all_boxes = []
        for r in range(rows):
            for c in range(cols):
                y1 = max(0, int(r * tile_h - overlap_h))
                y2 = min(h, int((r + 1) * tile_h + overlap_h))
                x1 = max(0, int(c * tile_w - overlap_w))
                x2 = min(w, int((c + 1) * tile_w + overlap_w))

                tile = frame[y1:y2, x1:x2]
                if tile.size == 0:
                    continue

                results = self.model.predict(tile, conf=self.conf, imgsz=self.imgsz, verbose=False, classes=[0])
                if results and results[0].boxes is not None:
                    for i in range(len(results[0].boxes)):
                        bx1, by1, bx2, by2 = results[0].boxes.xyxy[i].cpu().numpy().astype(int)
                        all_boxes.append((bx1 + x1, by1 + y1, bx2 + x1, by2 + y1))

        return self._merge_overlapping_boxes(all_boxes)

    @staticmethod
    def _iou(box_a, box_b):
        xa1, ya1, xa2, ya2 = box_a
        xb1, yb1, xb2, yb2 = box_b
        inter_x1, inter_y1 = max(xa1, xb1), max(ya1, yb1)
        inter_x2, inter_y2 = min(xa2, xb2), min(ya2, yb2)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        area_a = max(0, xa2 - xa1) * max(0, ya2 - ya1)
        area_b = max(0, xb2 - xb1) * max(0, yb2 - yb1)
        union = area_a + area_b - inter_area
        return inter_area / union if union > 0 else 0.0

    def _merge_overlapping_boxes(self, boxes, iou_threshold=0.4):
        """Elimine les doublons de detection dus au chevauchement entre tuiles."""
        boxes = sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
        kept = []
        for box in boxes:
            if all(self._iou(box, k) < iou_threshold for k in kept):
                kept.append(box)
        return kept

    def predict(self, frame, use_tiling=False, tile_grid=(3, 3)):
        if use_tiling:
            raw_boxes = self._predict_boxes_tiled(frame, grid=tile_grid)
        else:
            raw_boxes = self._predict_boxes_simple(frame)

        people = []
        for box in raw_boxes:
            people.append({
                "box": box,
                "foot_px": self._foot_point(box),
                "px_per_m": self._pixels_per_meter(box),
                "too_close": False,
            })

        close_pairs = []
        for (i, p1), (j, p2) in combinations(enumerate(people), 2):
            x1, y1 = p1["foot_px"]
            x2, y2 = p2["foot_px"]
            pixel_dist = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

            avg_scale = (p1["px_per_m"] + p2["px_per_m"]) / 2
            dist_m = pixel_dist / avg_scale if avg_scale > 0 else float("inf")

            if dist_m < self.min_distance_m:
                close_pairs.append((i, j, dist_m))
                people[i]["too_close"] = True
                people[j]["too_close"] = True

        count = len(people)
        is_crowd = count >= self.crowd_count_threshold

        alerts = []
        if is_crowd:
            alerts.append({"level": "warning", "message": f"Foule detectee: {count} personnes"})
        if close_pairs:
            alerts.append({
                "level": "info",
                "message": f"{len(close_pairs)} paire(s) trop proches (<{self.min_distance_m}m estime)",
            })

        return {
            "people": people,
            "count": count,
            "is_crowd": is_crowd,
            "close_pairs": close_pairs,
            "alerts": alerts,
        }

    def draw(self, frame, result: dict):
        for p in result["people"]:
            x1, y1, x2, y2 = p["box"]
            color = (0, 0, 255) if p["too_close"] else (0, 200, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        for i, j, dist_m in result["close_pairs"]:
            box_i, box_j = result["people"][i]["box"], result["people"][j]["box"]
            ci = ((box_i[0] + box_i[2]) // 2, box_i[3])
            cj = ((box_j[0] + box_j[2]) // 2, box_j[3])
            cv2.line(frame, ci, cj, (0, 0, 255), 2)
            mid = ((ci[0] + cj[0]) // 2, (ci[1] + cj[1]) // 2)
            cv2.putText(frame, f"~{dist_m:.1f}m", mid, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        status_color = (0, 0, 255) if result["is_crowd"] else (0, 200, 0)
        status_text = "FOULE DETECTEE" if result["is_crowd"] else "Densite normale"
        cv2.putText(frame, f"{status_text} | {result['count']} personne(s)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)
        return frame
