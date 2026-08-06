"""
Lance la détection d'objets abandonnés en direct.

Usage :
    python run_abandoned_object_detection.py
    python run_abandoned_object_detection.py --unattended-time 10 --proximity 200

Pour tester rapidement (au lieu d'attendre 30s réelles) :
    python run_abandoned_object_detection.py --unattended-time 8
"""

import argparse

import cv2

import config
from abandoned_object_detector import AbandonedObjectDetector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(config.CAMERA_SOURCE))
    parser.add_argument("--unattended-time", type=float, default=30.0,
                         help="secondes sans personne proche avant alerte")
    parser.add_argument("--proximity", type=float, default=150.0,
                         help="rayon en pixels pour considerer une personne 'proche'")
    parser.add_argument("--stationary-threshold", type=float, default=25.0,
                         help="tolerance de mouvement en pixels pour etre considere immobile")
    parser.add_argument("--model", default="yolo26s.pt",
                         help="yolo26n.pt = rapide mais rate certains objets ambigus; "
                              "yolo26s.pt = plus lent mais bien plus fiable (recommande pour ce module)")
    parser.add_argument("--conf", type=float, default=0.3, help="seuil de confiance (abaisse a 0.15-0.2 si besoin)")
    parser.add_argument("--imgsz", type=int, default=960, help="resolution d'inference (plus grand = mieux mais plus lent)")
    parser.add_argument("--debug", action="store_true", help="affiche toutes les detections brutes a chaque frame")
    args = parser.parse_args()

    detector = AbandonedObjectDetector(
        model_path=args.model,
        conf=args.conf,
        proximity_radius_px=args.proximity,
        stationary_threshold_px=args.stationary_threshold,
        unattended_time_threshold_s=args.unattended_time,
        debug=args.debug,
    )

    source = int(args.source) if str(args.source).isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError("Impossible d'ouvrir la camera")

    print("Detection d'objets abandonnes en cours. 'q' pour quitter.")
    print(f"Seuil: {args.unattended_time}s sans personne proche (rayon {args.proximity}px)")

    seen_alerts = set()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        result = detector.predict(frame)
        frame = detector.draw(frame, result)

        for alert in result["alerts"]:
            key = alert["message"]
            if key not in seen_alerts:
                print(f"[ALERTE] {alert['message']}")
                seen_alerts.add(key)

        cv2.imshow("Detection objets abandonnes", frame)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
