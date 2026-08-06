"""
Évalue la précision du module de détection de chute en conditions réelles.

Principe : le script tourne en direct avec ta webcam. Toi, l'humain, tu sais
la VÉRITÉ (tu es debout, tu es tombé, etc.). Tu appuies sur une touche pour
dire "en ce moment je suis au sol" ou "en ce moment je suis debout", et le
script compare ça à ce que le modèle a prédit -> calcule Precision/Recall.

Usage :
    python evaluate_fall_detection.py

Pendant l'exécution :
    'f' = j'affirme que je suis actuellement TOMBE/AU SOL (verite terrain)
    'n' = j'affirme que je suis actuellement DEBOUT/NORMAL (verite terrain)
    'q' = quitter et afficher les resultats

Fais varier les situations pendant le test : reste debout un moment, tombe
volontairement (sur un tapis/matelas !), assieds-toi par terre (cas limite
frequent de faux positif), relève-toi, etc.
"""

import cv2
from ultralytics import YOLO

import config
from module_fall import detect_falls


def main():
    model_pose = YOLO(config.MODEL_POSE)
    source = int(config.CAMERA_SOURCE) if str(config.CAMERA_SOURCE).isdigit() else config.CAMERA_SOURCE
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError("Impossible d'ouvrir la camera")

    ground_truth = None  # None = pas encore renseigne, True = tombe, False = debout
    tp = fp = tn = fn = 0
    total_labeled = 0

    print("Appuie sur 'f' quand tu es reellement AU SOL, 'n' quand tu es DEBOUT, 'q' pour finir.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model_pose(frame, verbose=False)
        falls = detect_falls(results[0], config.FALL_ASPECT_RATIO_THRESHOLD, config.FALL_ANGLE_THRESHOLD_DEG)
        predicted_fallen = any(f["fallen"] for f in falls)

        display = frame.copy()
        for f in falls:
            x1, y1, x2, y2 = f["box"]
            color = (0, 0, 255) if f["fallen"] else (0, 200, 0)
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

        gt_text = "?" if ground_truth is None else ("TOMBE" if ground_truth else "DEBOUT")
        pred_text = "TOMBE" if predicted_fallen else "DEBOUT"
        cv2.putText(display, f"Verite terrain (f/n): {gt_text}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(display, f"Prediction modele: {pred_text}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(display, f"TP:{tp} FP:{fp} TN:{tn} FN:{fn}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        cv2.imshow("Evaluation detection de chute", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("f"):
            ground_truth = True
        elif key == ord("n"):
            ground_truth = False

        if ground_truth is not None:
            total_labeled += 1
            if ground_truth and predicted_fallen:
                tp += 1
            elif ground_truth and not predicted_fallen:
                fn += 1
            elif not ground_truth and predicted_fallen:
                fp += 1
            else:
                tn += 1

    cap.release()
    cv2.destroyAllWindows()

    print("\n" + "=" * 50)
    print("RESULTATS")
    print("=" * 50)
    print(f"Frames evaluees: {total_labeled}")
    print(f"TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    if (tp + fp) > 0:
        precision = tp / (tp + fp)
        print(f"Precision: {precision:.3f}")
    if (tp + fn) > 0:
        recall = tp / (tp + fn)
        print(f"Recall: {recall:.3f}")
    if total_labeled > 0:
        accuracy = (tp + tn) / total_labeled
        print(f"Accuracy: {accuracy:.3f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
