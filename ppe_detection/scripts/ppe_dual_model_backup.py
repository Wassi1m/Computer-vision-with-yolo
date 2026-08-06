from ultralytics import YOLO
import cv2, time
from pathlib import Path
from datetime import datetime

print("Chargement modele 1..."); model1 = YOLO("best.pt")
print("Chargement modele 2..."); model2 = YOLO("best_gloves.pt")
print("Modeles prets !\n")

TRADUCTION = {
    "Hardhat":"Casque porte",        "NO-Hardhat":"SANS CASQUE !",
    "Mask":"Masque porte",           "NO-Mask":"SANS MASQUE !",
    "Safety Vest":"Gilet porte",     "NO-Safety Vest":"SANS GILET !",
    "NO-Hardhat":"SANS CASQUE !",    "NO-Gloves":"SANS GANTS !",
    "NO-Goggles":"SANS LUNETTES !",  "Gloves":"Gants portes",
    "Goggles":"Lunettes portees",    "Person":"Personne",
    "Fall-Detected":"CHUTE !",
    "helmet":"Casque porte",         "no-hardhat":"SANS CASQUE !",
    "vest":"Gilet porte",            "no-vest":"SANS GILET !",
    "gloves":"Gants portes",         "goggles":"Lunettes portees",
    "mask":"Masque porte",
}

CLASSES_DANGER = {
    "NO-Hardhat","no-hardhat","NO-Mask","NO-Safety Vest",
    "no-vest","NO-Gloves","NO-Goggles","Fall-Detected",
    "__SANS_LUNETTES__","__SANS_GILET__","__SANS_GANTS__"
}

SCREENSHOTS = Path("ppe_screenshots"); SCREENSHOTS.mkdir(exist_ok=True)
cap = cv2.VideoCapture(0); prev = time.time()

while True:
    ret, frame = cap.read()
    if not ret: break

    violations   = []
    detections   = []

    # -- Collecte toutes les détections des 2 modèles --
    ppe_detectes = set()

    for model, conf_seuil in [(model1, 0.40), (model2, 0.60)]:
        results = model.predict(frame, conf=conf_seuil, verbose=False)[0]
        for box in results.boxes:
            cls = model.names[int(box.cls)]
            sc  = float(box.conf)
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            detections.append((cls, sc, x1, y1, x2, y2))

            cls_low = cls.lower()
            if any(k in cls_low for k in ["goggle","lunette"]) and not cls_low.startswith("no"):
                ppe_detectes.add("lunettes")
            if any(k in cls_low for k in ["vest","gilet"]) and not cls_low.startswith("no"):
                ppe_detectes.add("gilet")
            if any(k in cls_low for k in ["glove","gant"]) and sc >= 0.60:
                ppe_detectes.add("gants")
            if any(k in cls_low for k in ["hardhat","helmet","casque"]) and not cls_low.startswith("no"):
                ppe_detectes.add("casque")
            if any(k in cls_low for k in ["mask","masque"]) and not cls_low.startswith("no"):
                ppe_detectes.add("masque")

    # -- Vérification : personne détectée sans certain EPI ? --
    personne_visible = any(
        "person" in d[0].lower() or "personne" in d[0].lower()
        for d in detections
    ) or len(detections) > 0

    absences_auto = []
    if personne_visible:
        if "lunettes" not in ppe_detectes:
            absences_auto.append(("SANS LUNETTES !", "__SANS_LUNETTES__"))
        if "gilet" not in ppe_detectes:
            absences_auto.append(("SANS GILET !", "__SANS_GILET__"))
        if "gants" not in ppe_detectes:
            absences_auto.append(("SANS GANTS !", "__SANS_GANTS__"))

    # -- Dessin des boîtes de détection --
    for (cls, sc, x1, y1, x2, y2) in detections:
        is_danger = cls in CLASSES_DANGER or any(
            n in cls for n in ["NO-","no-"]
        )
        color = (0,0,220) if is_danger else ((200,150,0) if "erson" in cls else (0,200,0))
        label = f"{TRADUCTION.get(cls, cls)} {sc:.0%}"
        cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
        (tw,th),_ = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.60,2)
        cv2.rectangle(frame,(x1,y1-th-10),(x1+tw+8,y1),color,-1)
        cv2.putText(frame,label,(x1+4,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.60,(255,255,255),2)
        if is_danger:
            violations.append(TRADUCTION.get(cls,cls))

    # -- Ajouter les absences automatiques aux violations --
    for label_fr, _ in absences_auto:
        if label_fr not in violations:
            violations.append(label_fr)

    # -- Affichage des alertes --
    fps = 1.0/max(time.time()-prev,1e-6); prev=time.time()
    h,w = frame.shape[:2]

    for i, v in enumerate(violations):
        y = 35 + i * 38
        bg_w = len(v)*15 + 20
        cv2.rectangle(frame,(0,y-28),(bg_w,y+8),(0,0,160),-1)
        cv2.putText(frame, f"ALERTE: {v}", (8,y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75,(255,255,255),2)

    # -- Panneau EPI en bas à droite --
    panel_x = w - 260
    panel_items = [
        ("Casque",   "casque"   in ppe_detectes),
        ("Masque",   "masque"   in ppe_detectes),
        ("Lunettes", "lunettes" in ppe_detectes),
        ("Gilet",    "gilet"    in ppe_detectes),
        ("Gants",    "gants"    in ppe_detectes),
    ]
    cv2.rectangle(frame,(panel_x-5, h-175),(w-5, h-45),(20,20,20),-1)
    cv2.putText(frame,"EPI Status:",(panel_x, h-155),
                cv2.FONT_HERSHEY_SIMPLEX,0.55,(200,200,200),1)
    for i,(nom, porte) in enumerate(panel_items):
        y_p = h - 130 + i*23
        icone = "OK" if porte else "X "
        color_p = (0,200,0) if porte else (0,0,220)
        cv2.putText(frame, f"[{icone}] {nom}", (panel_x, y_p),
                    cv2.FONT_HERSHEY_SIMPLEX,0.52, color_p, 2)

    cv2.rectangle(frame,(0,h-40),(panel_x-10,h),(20,20,20),-1)
    cv2.putText(frame,f"FPS:{fps:.0f}  Q=Quitter  S=Screenshot",
                (8,h-12),cv2.FONT_HERSHEY_SIMPLEX,0.50,(180,180,180),1)

    cv2.imshow("Detection EPI - Securite Chantier", frame)
    k = cv2.waitKey(1)&0xFF
    if k==ord('q'): break
    elif k==ord('s'):
        p=SCREENSHOTS/f"ppe_{datetime.now().strftime('%H%M%S')}.jpg"
        cv2.imwrite(str(p),frame); print(f"Screenshot: {p}")

cap.release(); cv2.destroyAllWindows()
