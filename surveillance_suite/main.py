"""
Système de surveillance intelligente - 5 modules combinés :
  1. Détection fumée / incendie          (optionnel, nécessite modèle custom)
  2. Personne évanouie / tombée          (YOLO26-pose)
  3. Franchissement de ligne virtuelle   (YOLO26 + tracking)
  4. Détection automatique de matricules (optionnel, nécessite modèle custom)
  5. Porte ouverte / fermée              (heuristique par zone)

Usage :
    python main.py                     # webcam, tous les modules dispo
    python main.py --calibrate-door    # capture d'abord l'état "porte fermée"
    python main.py --no-fire --no-lpr  # désactive les modules nécessitant
                                        # un modèle custom si tu ne les as pas
"""

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
from ultralytics import YOLO

import config
from module_fall import detect_falls
from module_line_crossing import LineCrossingCounter
from module_door import DoorStateDetector, select_roi_interactively, compute_occlusion_ratio
from module_door_classifier import load_trained_door_classifier
from module_door_trained import DoorStateDetectorML

# Modules optionnels (import protégé, désactivés si modèle absent)
try:
    from module_fire_smoke import FireSmokeDetector
except ImportError:
    FireSmokeDetector = None

try:
    from module_lpr import LicensePlateReader
except ImportError:
    LicensePlateReader = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(config.CAMERA_SOURCE))
    parser.add_argument("--no-fire", action="store_true", help="désactive le module fumée/feu")
    parser.add_argument("--no-lpr", action="store_true", help="désactive le module plaques")
    parser.add_argument("--calibrate-door", action="store_true",
                         help="capture la 1ère frame comme référence 'porte fermée'")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la source vidéo: {source}")

    # Résolution de capture réduite -> accélère TOUT le pipeline en aval
    # (moins de pixels à traiter par chaque modèle). 480x360 reste largement
    # suffisant pour la détection, le tracking et les classifieurs.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # réduit la latence (pas de frames en attente)

    # ---- Chargement des modèles ----
    print("Chargement du modèle général YOLO26...")
    model_general = YOLO(config.MODEL_GENERAL)

    print("Chargement du modèle pose YOLO26 (détection de chute)...")
    model_pose = YOLO(config.MODEL_POSE)

    fire_detector = None
    if not args.no_fire and FireSmokeDetector and os.path.exists(config.MODEL_FIRE_SMOKE or ""):
        print("Chargement du modèle fumée/feu...")
        fire_detector = FireSmokeDetector(config.MODEL_FIRE_SMOKE, conf=config.CONF_THRESHOLD)
    else:
        print("Module fumée/feu désactivé (modèle absent ou --no-fire).")

    lpr_reader = None
    if not args.no_lpr and LicensePlateReader and os.path.exists(config.MODEL_PLATE or ""):
        print("Chargement du modèle de plaques + OCR...")
        lpr_reader = LicensePlateReader(config.MODEL_PLATE, conf=config.CONF_THRESHOLD)
    else:
        print("Module LPR désactivé (modèle absent ou --no-lpr).")

    line_counter = LineCrossingCounter(config.LINE_START, config.LINE_END)

    # ---- Module porte : classifieur entraîné sur TA porte en priorité
    # (door_classifier.pt, 2 classes open/closed, via collect_door_samples.py +
    # train_door_classifier.py). door_state.pt (Roboflow) est ECARTE pour cette
    # decision : verification faite, ce dataset n'a qu'1 classe "door" (presence
    # uniquement), il ne peut donc pas distinguer ouvert/ferme. Heuristique SSIM
    # en tout dernier recours si aucun classifieur entraine n'existe. ----
    door_classifier = load_trained_door_classifier(
        smoothing_window=config.DOOR_CLASSIFIER_SMOOTHING,
        min_confidence=config.DOOR_CLASSIFIER_MIN_CONF,
    )
    door_ml = None
    door_detector = None

    if door_classifier is not None:
        print("Classifieur porte entraîné trouvé (door_classifier.pt) -> utilisation (recommandé).")
    else:
        print("Aucun classifieur entraîné trouvé (models/door_classifier.pt manquant).")
        print("-> Lance: python collect_door_samples.py puis python train_door_classifier.py")
        print("-> En attendant, utilisation de l'heuristique SSIM de secours (moins fiable).")

        print("Aucun classifieur entraîné trouvé (models/door_classifier.pt manquant).")
        print("-> Utilisation de l'heuristique SSIM de secours (moins fiable).")
        print("   Pour de meilleurs résultats : python collect_door_samples.py puis "
              "python train_door_classifier.py")

        door_roi = config.DOOR_ROI
        if door_roi is None:
            ok, first_frame = cap.read()
            if not ok:
                raise RuntimeError("Impossible de lire la caméra pour la sélection du ROI.")
            print("Sélectionne la zone de la porte à la souris (clique-glisse), puis ENTREE.")
            door_roi = select_roi_interactively(first_frame)
            print(f"ROI porte sélectionné : {door_roi}")

        door_detector = DoorStateDetector(
            door_roi,
            diff_threshold=config.DOOR_DIFF_THRESHOLD,
            stability_frames=config.DOOR_STABILITY_FRAMES,
            calibration_samples=config.DOOR_CALIBRATION_SAMPLES,
        )

    frame_idx = 0
    prev_t = time.time()
    executor = ThreadPoolExecutor(max_workers=3)

    # Fréquences d'exécution des modules moins critiques en latence -> réduit
    # la charge totale par frame (le mouvement/tracking reste analysé à
    # chaque frame, le reste n'a pas besoin de cette précision)
    POSE_EVERY_N = 2     # chute : 1 frame sur 2 suffit largement
    FIRE_EVERY_N = 5      # fumée/feu : evenement lent, pas besoin de chaque frame
    DOOR_EVERY_N = 3       # porte : change rarement d'etat frame a frame
    INFER_IMGSZ = 480       # resolution d'inference reduite -> beaucoup plus rapide

    last_falls = []
    last_fire_dets = []
    last_door_result = None

    if door_detector is not None and args.calibrate_door:
        door_detector.start_calibration()
        print(f"Calibration en cours ({config.DOOR_CALIBRATION_SAMPLES} frames) - "
              f"assure-toi que la porte est FERMEE et immobile.")

    print("Appuie sur 'q' pour quitter." + ("" if door_detector is None else
          " 'c' pour (re)calibrer la porte (fermee)."))

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1


        # Calibration multi-frames de la porte, si en cours (heuristique de secours uniquement)
        if door_detector is not None and door_detector.is_calibrating:
            finished = door_detector.calibrate(frame)
            if finished:
                print("Calibration de la porte terminee.")

        # ---- Module 2 + 3 : détection générale (mouvement/tracking) + pose (chute)
        # exécutés EN PARALLÈLE (le CPU/OpenCV/PyTorch libère le GIL pendant les
        # calculs lourds, donc le threading apporte un vrai gain de latence ici) ----
        general_future = executor.submit(
            model_general.track, frame, persist=True, conf=config.CONF_THRESHOLD,
            imgsz=INFER_IMGSZ, verbose=False,
        )

        run_pose_this_frame = frame_idx % POSE_EVERY_N == 0
        if run_pose_this_frame:
            pose_future = executor.submit(model_pose, frame, imgsz=INFER_IMGSZ, verbose=False)

        track_results = general_future.result()
        tracked_objects = []
        if track_results and track_results[0].boxes is not None:
            boxes = track_results[0].boxes
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
                cls_id = int(boxes.cls[i])
                label = model_general.names[cls_id]
                track_id = int(boxes.id[i]) if boxes.id is not None else -1
                tracked_objects.append({"box": tuple(xyxy), "label": label, "track_id": track_id})

                color = (0, 200, 0) if label != "person" else (255, 150, 0)
                x1, y1, x2, y2 = xyxy
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} #{track_id}", (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Module 3 : franchissement de ligne
        cv2.line(frame, config.LINE_START, config.LINE_END, (255, 255, 0), 2)
        crossed = line_counter.update(tracked_objects)
        for c in crossed:
            print(f"[LIGNE] {c['label']} #{c['track_id']} a franchi la ligne")
        cv2.putText(frame, f"IN: {line_counter.count_in}  OUT: {line_counter.count_out}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Module 2 : détection de chute (pose) - 1 frame sur POSE_EVERY_N,
        # on reaffiche le dernier resultat connu les autres frames
        if run_pose_this_frame:
            pose_results = pose_future.result()
            last_falls = detect_falls(pose_results[0], config.FALL_ASPECT_RATIO_THRESHOLD,
                                       config.FALL_ANGLE_THRESHOLD_DEG)
        for f in last_falls:
            x1, y1, x2, y2 = f["box"]
            if f["fallen"]:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(frame, "PERSONNE AU SOL !", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Module 1 : fumée / feu - 1 frame sur FIRE_EVERY_N (evenement lent)
        if fire_detector and frame_idx % FIRE_EVERY_N == 0:
            last_fire_dets = fire_detector.detect(frame)
        for d in last_fire_dets:
            x1, y1, x2, y2 = d["box"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 100, 255), 2)
            cv2.putText(frame, f"{d['label'].upper()} {d['conf']:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)

        # Module 4 : plaques (LPR) - seulement sur les objets "car"/"truck" pour la perf
        if lpr_reader and frame_idx % 5 == 0:  # 1 frame sur 5, l'OCR est lourd
            plates = lpr_reader.detect_and_read(frame)
            for p in plates:
                x1, y1, x2, y2 = p["box"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 0, 200), 2)
                cv2.putText(frame, p["text"], (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 0, 200), 2)

        # Module 5 : porte - 1 frame sur DOOR_EVERY_N (l'etat change lentement)
        run_door_this_frame = frame_idx % DOOR_EVERY_N == 0
        if door_ml is not None:
            if run_door_this_frame:
                is_open, conf = door_ml.update(frame)
            frame = door_ml.draw(frame)
        elif door_classifier is not None:
            if run_door_this_frame:
                is_open, conf = door_classifier.update(frame)
            frame = door_classifier.draw(frame)
        elif door_detector is not None:
            # On ignore la mise à jour si une personne masque une part trop
            # importante du ROI (sinon on confond "personne devant la porte"
            # avec "la porte a changé d'état")
            if run_door_this_frame:
                person_boxes = [obj["box"] for obj in tracked_objects if obj["label"] == "person"]
                occlusion_ratio = compute_occlusion_ratio(door_detector.roi, person_boxes) if person_boxes else 0.0
                is_occluded = occlusion_ratio > config.DOOR_OCCLUSION_THRESHOLD

                if not door_detector.is_calibrating:
                    is_open, diff_score = door_detector.update(frame, occluded=is_occluded)
            frame = door_detector.draw(frame)

        # FPS
        now = time.time()
        fps = 1.0 / max(now - prev_t, 1e-6)
        prev_t = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Surveillance intelligente", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c") and door_detector is not None:
            door_detector.start_calibration()
            print("Recalibration de la porte lancee - assure-toi qu'elle est FERMEE.")

    cap.release()
    cv2.destroyAllWindows()
    executor.shutdown(wait=True)


if __name__ == "__main__":
    main()
