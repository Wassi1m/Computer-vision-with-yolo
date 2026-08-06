"""
Module 2 : Détection de personne tombée / évanouie.

Principe :
- YOLO26-pose donne 17 points clés (COCO keypoints) par personne.
- On calcule l'angle du "tronc" (ligne épaules -> hanches) par rapport
  à la verticale, + le ratio largeur/hauteur de la boîte englobante.
- Une personne debout : tronc quasi vertical, boîte haute et étroite.
- Une personne tombée : tronc proche de l'horizontale, boîte large et basse.
- On combine les 2 critères pour réduire les faux positifs.
"""

import math
import numpy as np

# Indices COCO keypoints (format Ultralytics)
L_SHOULDER, R_SHOULDER = 5, 6
L_HIP, R_HIP = 11, 12


def compute_trunk_angle(keypoints):
    """Retourne l'angle du tronc (0° = vertical, 90° = horizontal), ou None."""
    try:
        ls, rs = keypoints[L_SHOULDER], keypoints[R_SHOULDER]
        lh, rh = keypoints[L_HIP], keypoints[R_HIP]
    except IndexError:
        return None

    # Ignore si un point clé n'a pas été détecté (confiance nulle -> (0,0))
    pts = [ls, rs, lh, rh]
    if any(p[0] == 0 and p[1] == 0 for p in pts):
        return None

    shoulder_mid = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
    hip_mid = ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2)

    dx = hip_mid[0] - shoulder_mid[0]
    dy = hip_mid[1] - shoulder_mid[1]
    angle_from_vertical = math.degrees(math.atan2(abs(dx), abs(dy) + 1e-6))
    return angle_from_vertical


def detect_falls(pose_result, aspect_ratio_thresh, angle_thresh_deg):
    """
    pose_result : sortie brute d'un appel model(frame) avec un modèle -pose.
    Retourne une liste de dicts : {box, fallen(bool), angle, aspect_ratio}
    """
    detections = []
    if pose_result.boxes is None:
        return detections

    boxes = pose_result.boxes
    keypoints_all = pose_result.keypoints

    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i])
        if pose_result.names[cls_id] != "person":
            continue

        xyxy = boxes.xyxy[i].cpu().numpy()
        x1, y1, x2, y2 = xyxy
        w, h = x2 - x1, y2 - y1
        aspect_ratio = w / max(h, 1e-6)

        angle = None
        if keypoints_all is not None:
            kpts = keypoints_all.xy[i].cpu().numpy()
            angle = compute_trunk_angle(kpts)

        # Les deux critères doivent être confirmés pour réduire les faux
        # positifs (voir docstring) : si l'angle est indisponible (occlusion,
        # personne accroupie/penchée...), on ne peut pas le confirmer, donc on
        # ne déclare PAS de chute plutôt que de se rabattre uniquement sur le
        # ratio largeur/hauteur (qui seul est trop bruité).
        fallen = (
            aspect_ratio > aspect_ratio_thresh
            and angle is not None
            and angle > angle_thresh_deg
        )

        detections.append(
            {
                "box": xyxy.astype(int),
                "fallen": fallen,
                "angle": angle,
                "aspect_ratio": aspect_ratio,
            }
        )
    return detections
