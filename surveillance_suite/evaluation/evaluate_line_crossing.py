"""
Évalue la précision du module de franchissement de ligne.

Principe : tu traverses la ligne un nombre de fois connu (toi-même tu
comptes mentalement ou à voix haute), le script compare son comptage au
tien -> donne un taux d'erreur de comptage.

Usage :
    python evaluate_line_crossing.py

Pendant l'exécution :
    Traverse la ligne plusieurs fois dans les deux sens, en comptant à voix
    haute ou sur tes doigts combien de fois TU penses avoir traversé.
    'q' pour arrêter et comparer avec le compteur du script.
"""

import cv2
from ultralytics import YOLO

import config
from module_line_crossing import LineCrossingCounter


def main():
    model = YOLO(config.MODEL_GENERAL)
    source = int(config.CAMERA_SOURCE) if str(config.CAMERA_SOURCE).isdigit() else config.CAMERA_SOURCE
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError("Impossible d'ouvrir la camera")

    counter = LineCrossingCounter(config.LINE_START, config.LINE_END)

    print("Traverse la ligne plusieurs fois (compte toi-meme a voix haute).")
    print("Appuie sur 'q' quand tu as fini, pour voir le resultat du script.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.track(frame, persist=True, conf=config.CONF_THRESHOLD, verbose=False)
        tracked_objects = []
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
                label = model.names[int(boxes.cls[i])]
                track_id = int(boxes.id[i]) if boxes.id is not None else -1
                tracked_objects.append({"box": tuple(xyxy), "label": label, "track_id": track_id})
                x1, y1, x2, y2 = xyxy
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 150, 0), 2)

        counter.update(tracked_objects)

        # DEBUG : affiche la position et le track_id de chaque personne suivie
        for obj in tracked_objects:
            if obj["label"] == "person":
                x1, y1, x2, y2 = obj["box"]
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(frame, f"ID:{obj['track_id']} y={cy}", (cx + 10, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)


        cv2.line(frame, config.LINE_START, config.LINE_END, (255, 255, 0), 2)
        cv2.putText(frame, f"IN: {counter.count_in}  OUT: {counter.count_out}  "
                            f"TOTAL: {counter.count_in + counter.count_out}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow("Evaluation franchissement de ligne", frame)

        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    total_detected = counter.count_in + counter.count_out
    print("\n" + "=" * 50)
    print(f"Le script a compte {total_detected} franchissements "
          f"(IN={counter.count_in}, OUT={counter.count_out})")
    real_count = input("Combien de fois AS-TU REELLEMENT traverse la ligne ? ")
    try:
        real_count = int(real_count)
        error = abs(total_detected - real_count)
        accuracy = 1 - (error / max(real_count, 1))
        print(f"Erreur: {error} franchissement(s) | Precision de comptage: {accuracy * 100:.1f}%")
    except ValueError:
        print("Valeur non reconnue, comparaison manuelle a faire toi-meme.")
    print("=" * 50)


if __name__ == "__main__":
    main()
