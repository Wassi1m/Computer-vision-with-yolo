"""
Lance la détection de foule automatique - zéro calibration nécessaire.

Usage :
    python run_crowd_detection_auto.py
    python run_crowd_detection_auto.py --min-distance 1.5 --crowd-threshold 5
"""

import argparse

import cv2

import config
from crowd_density_detector_auto import AutoCrowdDetector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(config.CAMERA_SOURCE))
    parser.add_argument("--min-distance", type=float, default=1.0, help="distance minimale estimee en metres")
    parser.add_argument("--crowd-threshold", type=int, default=5, help="nb de personnes pour declarer une foule")
    parser.add_argument("--avg-height", type=float, default=1.70, help="taille humaine moyenne supposee (m)")
    parser.add_argument("--imgsz", type=int, default=960, help="resolution d'inference (plus grand = detecte mieux les petites personnes, plus lent)")
    parser.add_argument("--tiled", action="store_true", help="active l'inference par tuiles (foules denses, personnes lointaines)")
    parser.add_argument("--tile-grid", type=int, nargs=2, default=[3, 3], metavar=("ROWS", "COLS"))
    args = parser.parse_args()

    detector = AutoCrowdDetector(
        model_path=config.MODEL_GENERAL,
        conf=config.CONF_THRESHOLD,
        imgsz=args.imgsz,
        average_human_height_m=args.avg_height,
        min_distance_m=args.min_distance,
        crowd_count_threshold=args.crowd_threshold,
    )

    source = int(args.source) if str(args.source).isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError("Impossible d'ouvrir la camera")

    print("Detection de foule automatique en cours. 'q' pour quitter.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        result = detector.predict(frame, use_tiling=args.tiled, tile_grid=tuple(args.tile_grid))
        frame = detector.draw(frame, result)

        for alert in result["alerts"]:
            print(f"[ALERTE] {alert['message']}")

        cv2.imshow("Detection de foule (automatique)", frame)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
