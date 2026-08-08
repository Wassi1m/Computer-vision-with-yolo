
# Analyse complète du projet — Étape 1

Date : 2026-08-07

## 1. Vue d'ensemble

Le dépôt contient **deux projets Computer Vision indépendants**, non intégrés entre eux, qui partagent la même famille d'outils (Ultralytics YOLO + OpenCV) mais n'ont **aucun code, dataset ni modèle en commun** :

| | `ppe_detection/` | `surveillance_suite/` |
|---|---|---|
| Objectif métier | Conformité EPI (Équipements de Protection Individuelle) sur un site industriel/chantier | Sécurité générale d'un site (incendie, chute, intrusion, objets abandonnés, plaques, porte) |
| Modèle(s) cœur | 2 modèles YOLOv8 EPI dédiés | 6+ modèles YOLO26 (génériques + fine-tunés) + heuristiques CV classiques |
| Statut | Modèles déjà entraînés, prêts à l'emploi | Modèles déjà entraînés pour 2 modules ; 4 modules reposent sur un modèle générique ou une heuristique, pas d'entraînement dédié |
| Dataset de validation dispo | Oui (44 002 images annotées, zippé, pas encore extrait) | Partiel (fire/smoke et LPR oui, chute/porte/ligne non) |

**Aucun module de détection EPI n'existe dans `surveillance_suite`.** Si l'objectif final est une caméra de chantier qui doit à la fois vérifier le port des EPI *et* assurer la sécurité générale (incendie, chute...), les deux projets devraient à terme fusionner leur pipeline d'inférence (un seul flux vidéo, plusieurs têtes de détection) plutôt que de tourner comme deux applications séparées consommant chacune leur propre flux caméra. C'est actuellement le cas : chacun ouvre indépendamment `cv2.VideoCapture(0)`.

## 2. `ppe_detection/` — Détection EPI

### 2.1 Modèles

| Fichier | Architecture | Classes | Taille | Source |
|---|---|---|---|---|
| `models/best.pt` | YOLOv8m (25.9M paramètres, ~79 GFLOPs) | 14 classes (voir §2.2) | 52.1 MB | `Hexmon/vyra-yolo-ppe-detection` (Hugging Face), entraîné sur *PPE Combined Model v4* (Roboflow, ~44k images, 100 epochs) |
| `models/best_gloves.pt` | YOLOv8n | 6 classes : `Gloves, Vest, goggles, helmet, mask, safety_shoe` | 5.4 MB | `Tanishjain9/yolov8n-ppe-detection-6classes` (Hugging Face) |
| `models/yolov8n.pt` | YOLOv8n COCO générique | 80 classes COCO | ~6 MB | Poids Ultralytics standard, utilisé uniquement pour détecter la classe `person` |

Les deux modèles EPI sont **redondants et partiellement contradictoires** : ils couvrent des EPI qui se chevauchent (casque, gants, gilet, masque, lunettes) mais avec des noms de classes différents (`Hardhat` vs `helmet`, `Safety Vest` vs `Vest`...) et des niveaux de confiance très différents en pratique (`CONF_EPI` dans le code va de 0.08 à 0.20, calibré manuellement "sur logs réels" — signe que les seuils par défaut des modèles ne sont pas fiables tels quels).

### 2.2 Classes du modèle principal (`best.pt`)

`Fall-Detected, Gloves, Goggles, Hardhat, Ladder, Mask, NO-Gloves, NO-Goggles, NO-Hardhat, NO-Mask, NO-Safety Vest, Person, Safety Cone, Safety Vest`

Absence notable : pas de classe `NO-Gloves`/`NO-Goggles` fiable en pratique (voir §2.5), pas de classe "chaussures de sécurité", pas de "harnais".

### 2.3 Pipeline d'inférence (`scripts/ppe_dual_model_backup2.py`)

```
Frame caméra
   │
   ├─► YOLOv8n (yolov8n.pt)   → détection des personnes (classe COCO 0)
   ├─► YOLOv8m (best.pt)      → détection EPI + personnes (conf=0.03, très bas)
   └─► YOLOv8n (best_gloves.pt) → détection EPI complémentaires (conf=0.03)
   │
   ▼ Post-traitement
   1. Association EPI → personne par IoU, fallback par distance au centre
   2. Filtrage par seuil de confiance PAR TYPE D'EPI (CONF_EPI, 0.08–0.20)
   3. Lissage temporel : un EPI n'est "validé" que s'il est vu ≥4 fois
      sur les 12 dernières frames (fenêtre glissante, evite le flicker)
   4. Règle métier : un EPI n'est obligatoire que si EPI_OBLIGATOIRES[epi]=True
      (seuls casque et gilet déclenchent une alerte d'absence aujourd'hui —
      gants/lunettes/masque/chaussures sont désactivés par défaut)
   5. Dessin des boîtes, panneau de statut par personne, alertes
```

C'est un pipeline **entièrement heuristique en aval du modèle** : trois passes d'inférence à conf ultra-basse (0.03) suivies d'un système de règles fait main pour compenser le manque de fiabilité brute des modèles. C'est fonctionnel mais fragile — voir faiblesses en §5.

`scripts/advanced_security_ai.py` est une extension du même pipeline PPE avec 5 modules de sécurité supplémentaires **non-appris** (détection feu/fumée par seuillage HSV, chute par ratio largeur/hauteur, ligne virtuelle, poste/présence, objet abandonné par soustraction de fond MOG2) — un prototype qui préfigure ce que `surveillance_suite` fait en mieux (modèles dédiés au lieu d'heuristiques couleur pour le feu, pose réelle au lieu d'un simple ratio pour la chute).

### 2.4 Dataset (`data/`)

- **`ppe_dataset.zip`** (2.53 GB, non extrait) : format YOLO standard, déjà splitté :
  - `train` : 30 765 images | `val` : 8 814 images | `test` : 4 423 images (**44 002 images au total**, cohérent avec la fiche technique)
  - Pas de `data.yaml` inclus dans le zip → à reconstruire manuellement à partir des 14 classes de `best.pt` avant de pouvoir lancer une validation Ultralytics.
- **`data_set.zip`** (6.72 GB, non extrait) : sac de datasets Roboflow bruts, **non utilisés par aucun script actuel** :
  - `Construction Site Safety.v30-raw-images_latestversion.yolov8/` — 717 images, 25 classes (dont EPI + véhicules de chantier) — alternative/complément possible au dataset principal
  - `fall detection 2.v2i.yolov8/` — 511 images, 2 classes (`falling`, `stand`) — dataset dédié chute, plus riche que l'heuristique ratio actuelle
  - `License Plate Recognition.v13i.yolov8 (1).zip` — zip imbriqué de 3.2 GB, non exploré (contenu non inventorié)
  - `archive/fire_dataset/` — 756 images `fire_images` + 245 `non_fire_images`, format **classification** (pas de bounding boxes) donc inutilisable tel quel pour du YOLO detection sans ré-annotation

### 2.5 Scripts

- `scripts/validate_model.py` : script de validation complet déjà présent et bien conçu (calcule mAP50/mAP50-95/precision/recall/F1 par classe, matrice de confusion, génère graphiques + rapport HTML). **C'est l'outil qu'on va réutiliser pour l'Étape 2.** Aucun script d'entraînement dédié n'existe côté `ppe_detection` (les `.pt` ont été téléchargés tout faits depuis Hugging Face, jamais ré-entraînés localement).
- Les commentaires dans `ppe_dual_model_backup2.py` (`CONF_EPI`) révèlent que certaines classes ont des scores de confiance maximaux très faibles en conditions réelles (`Mask max ~0.21`, `goggles M2 max ~0.04`, `Vest max ~0.13`) — signe fort de **sous-performance du modèle sur certaines classes**, à confirmer quantitativement en Étape 2.

## 3. `surveillance_suite/` — Sécurité générale

### 3.1 Modèles

| Fichier | Rôle | Statut |
|---|---|---|
| `models/yolo26n.pt` | Détection générique COCO (personnes, véhicules, objets) + tracking (`model.track`) | Poids pré-entraîné, non spécialisé |
| `models/yolo26n-pose.pt` | Pose (17 keypoints COCO) → module chute | Poids pré-entraîné, non spécialisé |
| `models/yolo26s.pt` | Alternative plus lourde/précise à `yolo26n.pt`, utilisée par le détecteur d'objets abandonnés | Poids pré-entraîné, non spécialisé |
| `models/fire_smoke.pt` | Détection feu/fumée (2 classes) | **Fine-tuné** sur le dataset local (voir §3.3) |
| `models/license_plate.pt` | Détection de plaques (3 classes, avant OCR EasyOCR) | **Fine-tuné** sur le dataset local |
| `models/yolo11n-cls.pt` | Modèle de classification présent dans `models/` | **Orphelin** — aucun script ne le charge, à vérifier/supprimer |
| `models/door_classifier.pt`, `models/door_state.pt` | Référencés dans `module_door_classifier.py` / `module_door_trained.py` | **Absents du dépôt** — le code fait un fallback automatique vers l'heuristique SSIM (`module_door.py`) quand ces fichiers manquent |

### 3.2 Pipeline (`main.py`, orchestrateur des 5 modules actifs)

```
cap.read() → frame 480×360 (résolution volontairement réduite pour la latence)
   │
   ├─ Thread A : model_general.track(frame, imgsz=480)  → tracking personnes/objets (chaque frame)
   ├─ Thread B : model_pose(frame, imgsz=480)             → pose (1 frame / 2)
   ├─ (séquentiel) fire_detector.detect(frame)             → feu/fumée (1 frame / 5), si modèle présent
   ├─ (séquentiel) lpr_reader.detect_and_read(frame)       → plaques + OCR (1 frame / 5), si modèle présent
   └─ (séquentiel) door_*.update(frame)                    → état porte (1 frame / 3)
   │
   ▼ Post-traitement par module
   - Chute : angle du tronc (keypoints épaules/hanches) ET ratio largeur/hauteur, les deux
     doivent concorder (réduit les faux positifs vs l'heuristique ratio seul de ppe_detection)
   - Ligne : intersection de segments + cooldown anti double-comptage (RECROSS_MARGIN)
   - Porte : SSIM + CLAHE + calibration multi-frames + anti-flicker (fallback), ou
     classifieur/détecteur entraîné si présent (non présent actuellement)
   - Objet abandonné : soustraction de fond MOG2 + proximité personne/objet + timeout
```

Architecture nettement plus mature que `ppe_detection` : threading pour paralléliser les modèles lourds, sous-échantillonnage de fréquence par module selon sa criticité temporelle, dégradation propre (modules désactivés proprement si le modèle `.pt` correspondant est absent plutôt que de planter), config centralisée (`config.py`).

### 3.3 Datasets (`surveillance_suite/data/dataset/`, déjà extraits)

| Dataset | Images | Classes | Utilisé pour |
|---|---|---|---|
| `Fire -smoke Detection.v1i.yolo26/` | 12 127 | `fire, smoke` | `models/fire_smoke.pt` |
| `Fire smoke yolo.v4i.yolo26/` | non compté (5.2 MB, petit) | — | dataset alternatif, apparemment inutilisé |
| `License Plate Detector.v5i.yolo26/` | 630 | `licence, num_plate, number_plate` (3 classes qui semblent être des doublons/variantes du même concept — à nettoyer) | `models/license_plate.pt` |
| `Door - Open - Closed -.v1i.folder/` | dataset Roboflow classification, **6 968 images réparties en 3 classes `Open/Closed/Semi`** (train 6108, valid 577, test 283) | Open/Closed/Semi | **Non utilisé** — présent et complet, mais jamais consommé par un script d'entraînement |
| `dataset/train/{open,closed}`, `dataset/val/{open,closed}` (racine) | Dossiers destinés à recevoir les échantillons de **ta propre porte**, collectés via un script `collect_door_samples.py` (mentionné dans `main.py` mais absent du dépôt) | open/closed | Vides (2 images au total) |

Le module porte est donc la partie la plus faible du pipeline malgré une opportunité facile à saisir : un dataset Roboflow complet et prêt à l'emploi (6 968 images, 3 classes) est présent dans le dépôt mais n'a **jamais été utilisé pour entraîner `door_classifier.pt`**, qui est absent de `models/`. En production, ce module tourne donc uniquement sur l'heuristique SSIM, explicitement documentée comme "moins fiable" dans le code lui-même. C'est une amélioration à fort impact et faible coût (voir `improvement_plan.md`).

Il n'existe **aucun dataset de validation** pour : chute (pose), franchissement de ligne, foule/densité, objet abandonné. Ces modules ont uniquement des scripts d'évaluation *manuels* en temps réel (`evaluation/evaluate_fall_detection.py`, `evaluate_line_crossing.py`) où un humain doit se filmer et étiqueter au clavier — pas reproductible, pas automatisable, aucune métrique historique disponible.

### 3.4 Entraînement

`training/train_yolo_detector.py` : script générique réutilisable (`--data data.yaml --output --epochs --imgsz --batch`) qui a servi à produire `fire_smoke.pt` et `license_plate.pt` par transfer learning depuis `yolo26n.pt`. Bien fait : patience=30, augmentations (`degrees`, `mixup`, `copy_paste`) adaptées aux petits datasets.

## 4. Dépendances et configuration

- `ppe_detection/requirements.txt` : `ultralytics, torch, torchvision, tensorflow, keras, opencv-python, numpy, matplotlib` — **`tensorflow`/`keras` sont listés mais non utilisés par aucun script** (tout est fait avec PyTorch/Ultralytics) : dépendance morte à retirer.
- `surveillance_suite/requirements.txt` : `ultralytics, torch, torchvision, opencv-python, numpy, easyocr, scikit-image, lap` — cohérent avec le code, pas de dépendance inutile identifiée.
- Aucun `torch`/`ultralytics` n'était installé sur cette machine avant l'Étape 2 (pas de GPU disponible — CPU 12 cœurs, 15 Go RAM).
- Aucun fichier de config d'entraînement (pas de `pyproject.toml`, pas de CI, pas de tests automatisés) dans les deux projets.

## 5. Quel modèle pour la caméra de surveillance ?

Les deux systèmes sont candidats, pour des usages différents et actuellement non fusionnés :

- **`ppe_detection/scripts/ppe_dual_model_backup2.py` (best.pt + best_gloves.pt)** : à utiliser si l'objectif prioritaire est le contrôle de conformité EPI (chantier, usine).
- **`surveillance_suite/main.py`** : à utiliser si l'objectif prioritaire est la sécurité générale du site (intrusion, chute, feu, objets abandonnés, plaques).

Si la caméra doit couvrir les deux besoins, il faudra à terme un seul point d'entrée vidéo qui appelle les deux pipelines de post-traitement sur les mêmes frames (actuellement chacun ouvre sa propre capture caméra et tourne en boucle indépendante) — c'est un axe d'amélioration pipeline identifié pour l'Étape 4.

## 6. Points faibles actuels du pipeline (synthèse)

1. **Confiances de détection ultra-basses utilisées en production** (`conf=0.03` sur les 2 modèles EPI, `CONF_EPI` 0.08–0.20 par classe) : les modèles bruts ne sont pas assez précis pour être utilisés à un seuil standard (0.25–0.5) et le code compense par du post-traitement — signe que les modèles eux-mêmes ont une précision insuffisante sur certaines classes plutôt qu'un simple problème de réglage.
2. **Deux modèles EPI redondants et incohérents entre eux** (noms de classes différents, confiances différentes) au lieu d'un seul modèle ré-entraîné proprement sur un dataset combiné.
3. **`data.yaml` manquant** pour `ppe_dataset.zip` — aucune validation Ultralytics standard n'est possible sans le reconstruire.
4. **Datasets non exploités** : dataset chute dédié (511 img), dataset EPI alternatif (717 img, 25 classes), disponibles dans `data_set.zip` mais jamais utilisés pour enrichir/valider les modèles.
5. **Module porte non entraîné malgré un dataset prêt à l'emploi** : 6 968 images Roboflow (Open/Closed/Semi) dorment dans `data/dataset/` sans jamais avoir servi à produire `models/door_classifier.pt` (absent) → le module tourne uniquement sur l'heuristique SSIM, la plus fragile aux conditions d'éclairage réelles d'une caméra de surveillance.
6. **Aucune métrique automatisée/reproductible** pour 4 des 8 modules `surveillance_suite` (chute, ligne, foule, objet abandonné) — évaluation uniquement manuelle en direct.
7. **Dépendances mortes** (`tensorflow`, `keras` dans `ppe_detection/requirements.txt`) et **modèle orphelin** (`yolo11n-cls.pt`).
8. **Pas de gestion de la robustesse caméra réelle** : pas de préconisation testée sur faible luminosité, contre-jour, pluie/brouillard, occultation partielle — pertinent vu que le déploiement cible est une caméra de surveillance extérieure/chantier.
9. **Deux pipelines non fusionnés** consommant chacun leur propre flux caméra, aucune stratégie de déploiement unifiée (pas de service, pas de conteneur, pas de config RTSP par défaut au-delà d'un commentaire dans `config.py`).

---

*Ce document couvre l'Étape 1 de la mission. L'Étape 2 (validation quantitative du dataset et calcul des métriques) suit dans `reports/baseline_report.md`.*
