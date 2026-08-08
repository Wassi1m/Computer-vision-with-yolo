# Plan d'amélioration — Étape 4

Date : 2026-08-07
Basé sur les constats quantitatifs de `reports/baseline_report.md` (Étape 2/3).

Ce document distingue les actions **déjà implémentées** dans cette session (réalisables sans GPU, à faible coût) des actions **proposées** qui nécessitent davantage de ressources (GPU, ré-annotation, collecte de données) et sortent du périmètre exécutable ici.

---

## Partie A — Actions implémentées dans cette session

### A.1 Entraînement de `door_classifier.pt` (gain le plus significatif)

**Constat** : le dataset Roboflow `Door - Open - Closed -.v1i.folder/` (6 968 images, 3 classes Open/Closed/Semi, équilibré : ~2000 images/classe) était présent dans le dépôt mais n'avait **jamais servi à produire `door_classifier.pt`**. Le module porte tournait donc uniquement sur l'heuristique SSIM (`module_door.py`), explicitement documentée comme moins fiable dans le code. Le modèle `yolo11n-cls.pt` présent dans `models/` était lui un poids ImageNet-1000-classes générique, jamais chargé par aucun script — un point mort exact pour amorcer ce travail par transfer learning.

**Action réalisée** :
```
yolo classify train data="surveillance_suite/data/dataset/Door - Open - Closed -.v1i.folder" \
  model=models/yolo11n-cls.pt epochs=15 imgsz=224 batch=32 patience=5
```
15 epochs complets, ~48 min sur CPU (12 cœurs).

**Résultat** :

| Split | Images | top1-accuracy |
|---|---|---|
| valid | 577 | **95.1%** |
| test (jamais vu à l'entraînement) | 283 | **97.2%** (top5=100%) |

**Bugs d'intégration découverts et corrigés en cours de route** (le modèle entraîné n'aurait pas fonctionné correctement sans ces deux correctifs) :
1. `surveillance_suite/modules/module_door_classifier.py` comparait `top_label == "open"` en minuscule strict, alors que les classes issues des noms de dossiers Roboflow sont `Closed`, `Open`, `Semi` (capitalisées) — la comparaison ne matchait donc jamais. Corrigé en `top_label.lower() == "open"`.
2. `load_trained_door_classifier()` cherchait le fichier ROI à `dataset/roi.json` (chemin relatif au cwd de `main.py`), alors que le fichier réel se trouve à `data/dataset/roi.json`. Corrigé.

**Déploiement** : poids copiés vers `surveillance_suite/models/door_classifier.pt`. Testé de bout en bout (`load_trained_door_classifier()` + `.update()` sur une frame factice) : le module se charge et s'exécute sans erreur. `main.py` l'utilisera automatiquement au prochain lancement (`door_classifier is not None` devient vrai), sans autre changement de code — c'est la branche prioritaire dans la logique de fallback déjà présente.

**Limite connue conservée telle quelle** : la logique binaire `is_open` de `DoorClassifier.update()` traite `Semi` (porte entrouverte) comme "non ouverte" par simplification (elle n'apparaît dans aucune branche `if`). Ce n'est pas un bug introduit ici — c'était déjà la conception du fichier avant cette session — mais si la distinction "entrouverte" est utile métier, une vraie logique à 3 états serait à ajouter séparément.

Artefacts : `reports/assets/door_classifier/confusion_matrix.png`, `results.png`.

### A.2 Dataset `license_plate` rééquilibré et revalidé

**Constat (baseline_report.md §4.4)** : le split `valid` fourni ne contenait que 10 images sur 630 au total — mesure non fiable.

**Action réalisée** : fusion des 630 images (train+valid+test confondus) puis re-split aléatoire 80/10/10 (seed fixe = 42, reproductible) → `surveillance_suite/data/dataset/License Plate Detector.v5i.yolo26_resplit/` (504/63/63), dataset original conservé intact à côté.

**Résultat de la revalidation sur ce split correct** :

| | Ancien split (n=10) | Nouveau split (n=63) |
|---|---|---|
| mAP@50 | 53.5% | **72.6%** |
| mAP@50:95 | 29.1% | **43.3%** |
| Précision | 87.5% | 75.2% |
| Rappel | 53.8% | 79.1% |

Le modèle est en réalité **meilleur** que ce que le split initial (10 images) laissait supposer, mais avec un rappel plus faible que le laissait croire la petite mesure biaisée — ce nouveau chiffre est la référence à utiliser désormais.

**Constat additionnel découvert pendant cette analyse** : sur les 630 images, la classe `num_plate` (index 1) n'a que **18 instances au total** (16 train / 0 valid / 2 test) contre 515 pour `licence` et 549 pour `number_plate` — confirmation chiffrée que `licence` et `number_plate` sont deux étiquettes redondantes pour le même concept visuel (déjà suspecté en Étape 1), et que `num_plate` est un troisième nom quasi-inutilisé, probablement une erreur d'annotation ou un reliquat d'une itération antérieure du schéma de classes.

**Recommandation pour aller plus loin** (non implémentée ici, nécessite ré-annotation) : fusionner `licence` et `number_plate` en une seule classe et retirer/réattribuer les 18 instances `num_plate`, puis ré-entraîner — un modèle à 2 classes propres (plaque / pas-plaque) sera plus simple à apprendre et plus facile à interpréter en production qu'un modèle à 3 classes dont une n'existe quasiment pas.

### A.3 Nettoyage de dépendances mortes

`ppe_detection/requirements.txt` : suppression de `tensorflow` et `keras` (confirmé par `grep` : aucun `import tensorflow`/`import keras` dans tout `ppe_detection/`, tout le code utilise PyTorch/Ultralytics). Réduit le temps d'installation et la surface d'attaque/maintenance sans aucun risque de régression.

### A.4 Diagnostic quantitatif : seuil de confiance vs. sous-apprentissage réel

**Question posée** : le point faible le plus critique du baseline (`NO-Safety Vest`, AP@50=4.8% à conf=0.25) est-il un problème de seuil mal calibré, ou une vraie faiblesse du modèle entraîné ?

**Méthode** : re-validation de `best.pt` sur le même split val à `conf=0.05` (au lieu de 0.25).

| Classe | AP@50 à conf=0.25 | AP@50 à conf=0.05 | Δ |
|---|---|---|---|
| NO-Safety Vest | 4.8% | 12.9% | +8.1 pts |
| Safety Vest | 49.4% | 60.0% | +10.6 pts |
| mAP global (14 classes) | 72.8% | 76.3% | +3.5 pts |

**Conclusion** : baisser le seuil aide (le pipeline de production le fait déjà, avec `conf=0.03` et des seuils par classe `CONF_EPI` 0.08–0.20 — cette session en confirme empiriquement le bien-fondé), mais **ne comble pas l'écart** : même à conf=0.05, `NO-Safety Vest` reste à 12.9% quand la plupart des autres classes sont entre 70% et 96%. La matrice de confusion (à conf=0.25) montre que l'essentiel des instances manquées partent en "background" (aucune boîte prédite), pas en confusion avec une autre classe — le modèle ne détecte simplement pas la zone, il ne se trompe pas de label. C'est donc une **vraie limite du modèle entraîné** sur cette classe (features insuffisamment discriminantes), pas un problème de réglage. Voir Partie B.1 pour la piste de correction (nécessite un ré-entraînement, hors périmètre CPU de cette session).

---

## Partie B — Actions proposées (nécessitent GPU / plus de données / plus de temps)

Classées par impact estimé pour un déploiement caméra de surveillance EPI, décroissant.

### B.1 Ré-entraîner `best.pt` avec un focus sur `Safety Vest` / `NO-Safety Vest`

Ce sont les deux classes qui portent l'alerte de non-conformité "gilet" — actuellement la moins fiable du modèle. Pistes concrètes, à combiner :
- Vérifier manuellement un échantillon d'annotations `NO-Safety Vest` du dataset (678 images train) : les boîtes "absence de gilet" sont par nature plus ambiguës à annoter cohéremment (quelle zone du corps ? quel niveau de couleur/texture "compte" comme un vêtement normal vs un gilet ?) qu'une classe positive comme casque ou échelle — une passe de relecture qualité sur un échantillon serait la première étape avant tout ré-entraînement, pour écarter un problème d'annotation plutôt que de modèle.
- Fine-tuning ciblé avec pondération de perte plus élevée sur ces deux classes, ou sur-échantillonnage (oversampling) des images contenant `Safety Vest`/`NO-Safety Vest` dans les batches d'entraînement.
- Augmentation de données spécifique (variations de couleur/texture de gilet — c'est un vêtement dont l'apparence varie énormément par rapport à un casque, plus uniforme visuellement).
- **Estimation de coût** : sur cette machine (CPU, pas de GPU), fine-tuner YOLOv8m sur 30 765 images même pour seulement quelques epochs prendrait probablement plusieurs heures à un jour — à faire sur GPU (même un GPU grand public type RTX réduirait ce temps à moins d'une heure).

### B.2 Unifier la taxonomie `best.pt` / `best_gloves.pt`

Cette session a confirmé par le crash `IndexError` (baseline_report §4.2) que les deux modèles sont aujourd'hui **impossibles à valider ou fusionner proprement** l'un avec l'autre : classes différentes, aucune table de correspondance dans le code. Deux options, à trancher avant tout travail :
- **Option 1 (recommandée)** : ré-entraîner un seul modèle sur l'union des deux taxonomies (14 + 6 classes, avec fusion des doublons sémantiques : `Hardhat`≈`helmet`, `Safety Vest`≈`Vest`, `Goggles`≈`goggles`, `Mask`≈`mask`, `Gloves`≈`Gloves`) — supprime la redondance de calcul (un seul forward pass au lieu de deux) et lève l'incohérence de seuils entre les deux modèles actuels.
- **Option 2** : garder les deux modèles séparés mais écrire une table de correspondance de classes explicite pour permettre la validation croisée et la fusion cohérente des détections dans `ppe_dual_model_backup2.py` (actuellement fait à la main via association IoU sans vérifier la cohérence sémantique entre les deux jeux de classes).

### B.3 Vitesse du pipeline EPI pour la caméra cible

Mesuré dans le baseline (§5.2) : ~3.5 FPS pour les 3 modèles en cascade de `ppe_dual_model_backup2.py` sur ce CPU, dominé par `best.pt` (YOLOv8m). Pistes par ordre de coût croissant :
1. **Gratuit / immédiat** : réduire `imgsz` de 640 à 480 (déjà mesuré : ~1.8x plus rapide, 606→207 ms/image) — implique de revalider la précision à cette résolution avant de l'adopter en production (non fait dans cette session, mAP mesuré ici l'était à imgsz=640).
2. **Faible coût** : export ONNX (`model.export(format="onnx")`) + inférence via `onnxruntime` — gain typique de 1.3-2x sur CPU par rapport à PyTorch eager, sans réentraînement.
3. **Coût moyen** : quantification INT8 post-training — gain supplémentaire mais avec une perte de précision à quantifier soigneusement, en particulier sur les classes déjà fragiles (§B.1) qu'il ne faut pas dégrader davantage.
4. **Solution de fond** : si la caméra cible dispose d'un GPU/NPU embarqué (Jetson, Coral, etc.), l'export TensorRT ou l'exécution sur accélérateur dédié est la vraie solution — le CPU seul restera un goulot d'étranglement pour du temps réel à 3 modèles.
5. **Architecturale** : appliquer la Partie B.2 (un seul modèle EPI au lieu de deux) réduit directement d'un tiers le nombre de forward pass par frame.

### B.4 `fire_smoke.pt`, classe `smoke`

AP@50 = 27.6% contre 65.2% pour `fire`, sur seulement 97 instances de validation (contre 1 021 pour `fire`) — sous-représentation du dataset source. Piste : enrichir spécifiquement les images de fumée (recherche de datasets complémentaires ou collecte ciblée), la fumée étant par nature plus variable visuellement (densité, couleur, translucidité) que les flammes.

### B.5 Modules sans dataset de validation (`fall`, `line_crossing`, `abandoned_object`)

Aucune métrique automatisée n'existe pour ces trois modules — seule une évaluation manuelle en temps réel est possible aujourd'hui (`evaluation/evaluate_fall_detection.py`, `evaluate_line_crossing.py`). Recommandation : au minimum, enregistrer un jeu de clips vidéo annotés (même une vingtaine de séquences avec vérité-terrain image par image) pour chaque module, afin de sortir de l'évaluation purement qualitative avant tout travail d'amélioration — sans mesure de référence, impossible de savoir si un futur changement améliore ou dégrade ces modules.

### B.6 Fusion des deux pipelines caméra

`ppe_detection` et `surveillance_suite` ouvrent chacun leur propre `cv2.VideoCapture(0)` indépendamment (déjà noté en Étape 1). Si le déploiement final doit couvrir à la fois conformité EPI et sécurité générale sur la même caméra, il faudra un point d'entrée vidéo unique appelant les deux chaînes de post-traitement sur les mêmes frames — actuellement hors périmètre "amélioration de précision" mais bloquant pour un déploiement caméra unique tel que décrit dans la mission.

---

## Synthèse

| Action | Statut | Effort | Impact |
|---|---|---|---|
| `door_classifier.pt` entraîné + bugs corrigés | ✅ Fait | Faible (CPU, ~48 min) | Élevé — module porte passe d'une heuristique SSIM à un modèle appris à 97.2% test acc. |
| Dataset `license_plate` rééquilibré | ✅ Fait | Faible | Moyen — mesure désormais fiable (n=63 vs n=10) |
| Dépendances mortes retirées | ✅ Fait | Trivial | Faible mais gratuit |
| Diagnostic seuil vs sous-apprentissage | ✅ Fait | Faible | Élevé — oriente correctement B.1 (ne pas se contenter de baisser un seuil) |
| B.1 Ré-entraînement `Safety Vest`/`NO-Safety Vest` | ⏳ Proposé | Élevé (GPU requis) | Le plus élevé — cible directement le point faible n°1 |
| B.2 Unification taxonomie EPI | ⏳ Proposé | Élevé (GPU requis) | Élevé — condition pour toute fusion propre des 2 modèles |
| B.3 Optimisation vitesse (ONNX/imgsz/quantization) | ⏳ Proposé | Faible à moyen | Élevé pour la faisabilité temps réel caméra |
| B.4 Enrichissement dataset `smoke` | ⏳ Proposé | Moyen (collecte données) | Moyen |
| B.5 Datasets de validation fall/line/abandoned | ⏳ Proposé | Moyen (collecte+annotation) | Moyen — prérequis à toute amélioration mesurable de ces modules |
| B.6 Fusion des deux pipelines caméra | ⏳ Proposé | Élevé (architecture) | Bloquant pour déploiement caméra unique |
