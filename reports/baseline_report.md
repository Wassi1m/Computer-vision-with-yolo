# Rapport de baseline — Étape 2

Date : 2026-08-07
Machine d'évaluation : CPU Intel i7-1355U (13ᵉ gen, 12 cœurs logiques), 15 Go RAM, pas de GPU. `ultralytics==8.4.115`, `torch==2.13.0+cpu`.

Ce rapport couvre l'Étape 2 de la mission (évaluation quantitative de l'état actuel). Il complète `PROJECT_ANALYSIS.md` (Étape 1 — architecture et rôle de chaque modèle).

---

## 1. Description du projet

Le dépôt contient deux systèmes de vision par ordinateur indépendants :

- **`ppe_detection/`** — conformité EPI (casque, gilet, gants, masque, lunettes) sur chantier/site industriel. Modèle principal : `best.pt` (YOLOv8m, 14 classes). Modèle secondaire : `best_gloves.pt` (YOLOv8n, 6 classes).
- **`surveillance_suite/`** — sécurité générale de site (incendie/fumée, plaques d'immatriculation, chute, porte, objets abandonnés). Modèles fine-tunés : `fire_smoke.pt`, `license_plate.pt`. Les autres modules (chute, porte, ligne, objets abandonnés) reposent sur un modèle générique YOLO26 pré-entraîné ou une heuristique CV classique, sans dataset de validation dédié — ils sont donc **hors périmètre de cette évaluation quantitative** (voir §6).

Si la caméra cible doit à la fois vérifier le port des EPI et assurer la sécurité générale, `ppe_detection/scripts/ppe_dual_model_backup2.py` est le pipeline à utiliser pour le premier besoin ; `surveillance_suite/main.py` pour le second.

## 2. Dataset de validation

| Dataset | Origine | Statut | Taille |
|---|---|---|---|
| `ppe_dataset` (14 classes) | Roboflow *PPE Combined Model v4*, fourni dans le dépôt (`ppe_detection/data/ppe_dataset.zip`) | Extrait, `data.yaml` reconstruit manuellement (absent du zip) | 44 002 images (30 765 train / 8 814 val / 4 423 test) |
| `ppe_dataset_subset` | Sous-ensemble du dataset ci-dessus (mêmes images val/test, aucune modification) | Créé pour cette évaluation — la validation complète sur les 44 002 images a été interrompue après plusieurs heures sur CPU (voir §5.4) | 1 469 images val + 1 106 images test |
| `Fire -smoke Detection.v1i.yolo26` | Roboflow, fourni dans le dépôt (`surveillance_suite/data/dataset/`) | Déjà extrait | 10 589 train / **1 017 valid** / 521 test |
| `License Plate Detector.v5i.yolo26` | Roboflow, fourni dans le dépôt | Déjà extrait | 610 train / **10 valid** / 10 test |

**Aucun téléchargement externe n'a été nécessaire** : les datasets de validation étaient déjà présents dans le dépôt sous forme d'archives zip ou de dossiers Roboflow. Pour `ppe_dataset`, la seule étape manuelle a été la reconstruction du `data.yaml` (noms de classes tirés des métadonnées de `best.pt`), absent de l'archive.

**Vérification de la validité des données** : les scans Ultralytics (`val: Scanning ... 0 corrupt`) confirment 0 image corrompue sur les deux datasets. Le split `valid` de `License Plate Detector` ne contient que **10 images** sur 630 au total — un split volontairement minuscule côté Roboflow, insuffisant pour une mesure de mAP statistiquement fiable (voir §4.3).

## 3. Méthodologie

- **`ppe_detection`** : réutilisation du script existant `ppe_detection/scripts/validate_model.py` (déjà bien conçu — mAP, P/R/F1 par classe, matrice de confusion, rapport HTML), exécuté sur `ppe_dataset_subset` avec les réglages par défaut du script (`conf=0.25`, `iou=0.50`, `imgsz=640`).
- **`surveillance_suite`** : aucun script de validation quantitative n'existait pour `fire_smoke.pt` / `license_plate.pt` (seuls des scripts d'évaluation manuelle en temps réel existent pour chute/ligne). Validation effectuée via un script minimal appelant directement `YOLO(...).val()` d'Ultralytics avec les mêmes réglages (`conf=0.25`, `iou=0.50`, `imgsz=640`) sur le split `valid` déjà fourni.
- **Vitesse/FPS/mémoire** : benchmark dédié (script séparé), 3 warmup + 20 itérations par modèle, une image par modèle, `imgsz=480` (résolution réellement utilisée en production par `surveillance_suite/main.py`), CPU non partagé avec un autre job au moment de la mesure.

## 4. Résultats — Précision

### 4.1 `ppe_detection/models/best.pt` (YOLOv8m, 14 classes)

| Split | Images | Instances | mAP@50 | mAP@50:95 | Précision | Rappel | F1 |
|---|---|---|---|---|---|---|---|
| val | 1 469 | 3 699 | **72.8%** | **50.2%** | 72.8% | 82.2% | 76.5% |
| test | 1 106 | 2 808 | **75.3%** | **51.5%** | 75.8% | 85.1% | 79.5% |

Les deux splits sont cohérents (écart < 3 pts), ce qui écarte un biais de sélection du sous-ensemble.

**Détail par classe (split val)** :

| Classe | Précision | Rappel | F1 | AP@50 |
|---|---|---|---|---|
| Ladder | 0.939 | 0.969 | 0.954 | **0.964** |
| Goggles | 0.847 | 0.965 | 0.902 | 0.943 |
| Gloves | 0.838 | 0.940 | 0.886 | 0.915 |
| Person | 0.905 | 0.905 | 0.905 | 0.901 |
| NO-Goggles | 0.831 | 0.937 | 0.881 | 0.899 |
| Hardhat | 0.824 | 0.914 | 0.867 | 0.855 |
| NO-Gloves | 0.843 | 0.853 | 0.848 | 0.810 |
| Fall-Detected | 0.821 | 0.840 | 0.830 | 0.810 |
| NO-Hardhat | 0.617 | 0.910 | 0.736 | 0.718 |
| Safety Cone | 0.801 | 0.667 | 0.728 | 0.636 |
| NO-Mask | 0.625 | 0.879 | 0.731 | 0.664 |
| Mask | 0.510 | 0.899 | 0.651 | 0.529 |
| Safety Vest | 0.537 | 0.659 | 0.592 | 0.494 |
| **NO-Safety Vest** | 0.251 | 0.170 | 0.202 | **0.048** |

**Constat majeur** : la classe **`NO-Safety Vest` est quasiment non fonctionnelle (AP@50 = 4.8%)** — c'est pourtant l'une des deux seules classes qui déclenchent une alerte métier dans `ppe_dual_model_backup2.py` (avec `NO-Hardhat`, lui-même à seulement 71.8% AP@50). `Safety Vest` (le pendant positif) est également faible (49.4%). Ce résultat confirme quantitativement ce que les seuils de confiance ultra-bas codés en dur dans le pipeline (`CONF_EPI` 0.08–0.20, `conf=0.03` à l'inférence) laissaient supposer en Étape 1 : **le modèle ne détecte pas fiablement l'absence de gilet de sécurité**, un point faible critique pour un cas d'usage de conformité EPI.

Graphiques : `reports/assets/ppe_detection/best_confusion_matrix_val.png`, `best_per_class_val.png`, `best_summary_gauges_val.png` (+ équivalents `_test.png`).

### 4.2 `ppe_detection/models/best_gloves.pt` (YOLOv8n, 6 classes)

**Validation impossible telle quelle — erreur bloquante.** `best_gloves.pt` a été entraîné sur une taxonomie de 6 classes (`Gloves, Vest, goggles, helmet, mask, safety_shoe`) différente des 14 classes de `ppe_dataset` (`Hardhat, Safety Vest, NO-Hardhat`...). Lancer `model.val()` sur ce modèle avec le `data.yaml` à 14 classes fait planter Ultralytics :

```
IndexError: index 8 is out of bounds for axis 1 with size 7
  (ultralytics/utils/metrics.py:474, ConfusionMatrix.process_batch)
```

Les indices de classe des labels de vérité-terrain (0–13) dépassent la taille de la matrice de confusion du modèle (dimensionnée pour 6 classes + 1 "background"). Ce n'est pas un bug de configuration mais la confirmation directe du point faible n°2 déjà identifié en Étape 1 : **les deux modèles EPI utilisent des taxonomies de classes incompatibles**, et il n'existe aucun dataset de validation propre à `best_gloves.pt`. Le valider correctement nécessiterait soit un dataset annoté selon sa propre taxonomie, soit une table de correspondance classe-à-classe (non présente dans le code actuel) — c'est un chantier à part entière, hors périmètre d'une simple exécution de validation.

### 4.3 `surveillance_suite/models/fire_smoke.pt`

| Classe | Images | Instances | Précision | Rappel | mAP@50 | mAP@50:95 |
|---|---|---|---|---|---|---|
| **Global** | 1 017 | 1 118 | 57.6% | 54.0% | **46.4%** | **15.6%** |
| fire | 943 | 1 021 | 78.3% | 65.9% | 65.2% | 24.4% |
| smoke | 94 | 97 | 36.8% | 42.1% | 27.6% | 6.8% |

La détection de fumée est nettement moins fiable que celle du feu (mAP@50 27.6% vs 65.2%), et la classe est sous-représentée dans le split de validation (97 instances vs 1 021 pour `fire`), ce qui limite déjà la confiance qu'on peut placer dans le modèle sur ce cas précis. À conf=0.25 (seuil standard), le modèle global reste en dessous des standards habituels de production (mAP@50 < 50%).

Graphiques : `reports/assets/fire_smoke/confusion_matrix.png`, `PR_curve.png`.

### 4.4 `surveillance_suite/models/license_plate.pt`

| Classe | Images | Instances | Précision | Rappel | mAP@50 | mAP@50:95 |
|---|---|---|---|---|---|---|
| Global | 10 | 13 | 87.5% | 53.8% | 53.5% | 29.1% |

**⚠️ Résultat non fiable statistiquement** : le split `valid` fourni par le dataset Roboflow ne contient que **10 images / 13 instances**, toutes de la classe `licence` — les classes `num_plate` et `number_plate` (probablement des doublons de nommage du même concept, déjà relevé en Étape 1) n'apparaissent pas dans ce split et n'ont donc aucune métrique calculable. Une mesure fiable nécessiterait de reconstituer un split de validation plus large, par exemple en repartageant train/val (610 images disponibles en train) ou en fusionnant les 3 classes redondantes avant réentraînement.

Graphiques : `reports/assets/license_plate/confusion_matrix.png`, `PR_curve.png`.

## 5. Résultats — Performance / ressources

### 5.1 Taille des modèles

| Modèle | Taille |
|---|---|
| `ppe_detection/best.pt` | 49.6 MB |
| `ppe_detection/best_gloves.pt` | 5.4 MB |
| `ppe_detection/yolov8n.pt` | 6.2 MB |
| `surveillance_suite/fire_smoke.pt` | 5.1 MB |
| `surveillance_suite/license_plate.pt` | 5.1 MB |
| `surveillance_suite/yolo26n.pt` | 5.3 MB |
| `surveillance_suite/yolo26n-pose.pt` | 7.5 MB |
| `surveillance_suite/yolo26s.pt` | 19.5 MB |

### 5.2 Vitesse d'inférence / FPS (CPU, imgsz=480 — résolution de production)

Mesuré sur image unique, moyenne de 20 itérations après 3 warmups, CPU non partagé.

| Modèle | ms/image | FPS | ΔRSS mémoire process |
|---|---|---|---|
| `ppe_detection/best.pt` (YOLOv8m) | **206.8 ms** | **4.84** | +232.5 MB |
| `ppe_detection/best_gloves.pt` (YOLOv8n) | 36.6 ms | 27.3 | +9.1 MB |
| `ppe_detection/yolov8n.pt` (COCO) | 40.6 ms | 24.7 | +1.2 MB |
| `surveillance_suite/fire_smoke.pt` | 36.7 ms | 27.2 | +13.1 MB |
| `surveillance_suite/license_plate.pt` | 33.1 ms | 30.3 | -15.2 MB (bruit mesure) |
| `surveillance_suite/yolo26n.pt` | 36.0 ms | 27.8 | +7.2 MB |
| `surveillance_suite/yolo26n-pose.pt` | 42.1 ms | 23.8 | -7.2 MB (bruit mesure) |

**Constat critique pour le déploiement caméra** : `ppe_detection/scripts/ppe_dual_model_backup2.py` exécute **3 modèles YOLO en séquence sur chaque frame** (`yolov8n.pt` + `best.pt` + `best_gloves.pt`). En sommant leurs temps mesurés à imgsz=480 : ~207 + ~41 + ~37 ≈ **285 ms/frame ≈ 3.5 FPS** au mieux sur ce CPU, avant même le post-traitement (association IoU, lissage temporel). C'est `best.pt` (YOLOv8m, 25.8M paramètres) qui domine ce coût — à elle seule cette étape consomme plus des deux tiers du budget temps. Pour une caméra de surveillance temps réel (15-30 FPS attendus), ce pipeline est **actuellement inadapté à un déploiement CPU sans optimisation** (quantification, export ONNX/TensorRT, GPU dédié, ou réduction à un seul modèle EPI).

À `imgsz=640` (réglage par défaut de `validate_model.py`, utilisé pour la validation §4.1), `best.pt` mesure 546-606 ms/image, soit ~1.6-1.8 FPS — cohérent avec le ratio (640/480)² ≈ 1.78 par rapport à la mesure à 480px.

`surveillance_suite/main.py` reste dans une fourchette plus raisonnable (~27-28 FPS par modèle prenant chaque frame ou 1 frame sur N selon le module), car il n'utilise que des modèles YOLOv8n/YOLO26n légers.

### 5.3 Mémoire

Le RSS process pour un seul modèle chargé (232 MB pour `best.pt`, quelques MB pour les modèles nano) donne un ordre de grandeur, mais les deltas mesurés sont bruités (garbage collector Python, allocation PyTorch paresseuse) — à considérer comme indicatifs. Le pipeline `ppe_dual_model_backup2.py` charge 3 modèles simultanément ; l'empreinte mémoire combinée est donc supérieure à la somme naïve de ces mesures individuelles (allocations PyTorch partagées), à mesurer en conditions réelles si la RAM disponible sur la caméra cible est contrainte.

### 5.4 Note méthodologique — validation complète interrompue

Une première tentative de validation sur le dataset `ppe_dataset` complet (44 002 images) a été lancée puis interrompue manuellement après plusieurs heures sans avoir atteint la fin du split `val` (8 814 images) : au débit mesuré ici (~0.6-0.9 s/image en validation avec metrics, imgsz=640, CPU), valider les 8 814 images du split val complet aurait pris **~1h30 à 2h rien que pour `best.pt`**, et il aurait fallu répéter l'opération pour `test` et pour chaque modèle. Le sous-ensemble de 1 469 (val) + 1 106 (test) images utilisé pour ce rapport donne des résultats cohérents entre les deux splits (§4.1) et constitue une estimation raisonnable à cette échéance, mais **une validation exhaustive sur GPU serait nécessaire pour confirmer ces chiffres à l'échelle complète du dataset**, en particulier pour les classes rares comme `NO-Safety Vest`.

## 6. Modules non couverts par cette évaluation

Aucun dataset de validation n'existe pour les modules suivants de `surveillance_suite` — leurs performances n'ont donc **pas pu être quantifiées** dans ce rapport (déjà signalé en Étape 1, §6 point 6 de `PROJECT_ANALYSIS.md`) :

- Détection de chute (`module_fall.py`, heuristique pose) — évaluation manuelle uniquement (`evaluation/evaluate_fall_detection.py`)
- Franchissement de ligne (`module_line_crossing.py`) — idem
- État de porte (`module_door.py`) — un dataset Roboflow complet (6 968 images, Open/Closed/Semi) existe pourtant dans le dépôt mais n'a jamais servi à entraîner `door_classifier.pt` (absent), donc rien à valider
- Objet abandonné (`abandoned_object_detector.py`, soustraction de fond MOG2) — pas de dataset dédié, pas de modèle appris

## 7. Synthèse — priorités pour l'Étape 4 (plan d'amélioration)

Classées par impact estimé sur la fiabilité en conditions réelles de caméra de surveillance :

1. **`NO-Safety Vest` (AP@50 4.8%) et `NO-Hardhat` (AP@50 71.8%, mais seule alerte fiable)** : ce sont les deux seules classes qui déclenchent une alerte de non-conformité dans le pipeline actuel — l'une des deux est quasiment inopérante. Priorité n°1.
2. **Budget temps du pipeline EPI (~3.5 FPS sur CPU)** : incompatible avec un flux caméra temps réel sans optimisation (export ONNX/TensorRT, quantification INT8, ou modèle plus léger).
3. **Incohérence de taxonomie entre `best.pt` et `best_gloves.pt`** : empêche toute validation ou fusion propre des deux modèles ; à trancher (un seul modèle réentraîné vs. table de correspondance de classes).
4. **`fire_smoke.pt`, classe `smoke`** (AP@50 27.6%) sous-performante et sous-représentée dans le dataset de validation.
5. **`license_plate.pt`** : split de validation à reconstituer (10 images actuellement) avant de pouvoir juger la qualité réelle du modèle.
6. **Modules chute/porte/ligne/objet abandonné** : aucune métrique automatisée n'existe ; le module porte a un dataset prêt à l'emploi et non exploité (gain rapide possible).

---

*Fichiers sources des métriques : `reports/assets/`. Rapport HTML interactif complet pour `best.pt` : `ppe_detection/scripts/validation_results/rapport_validation_20260807_112801.html`.*
