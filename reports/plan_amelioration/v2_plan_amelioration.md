# Plan d'amélioration v2 — améliorations restantes à faire

Date : 2026-08-07
Fait suite à `v1_plan_amelioration.md` (même dossier). v1 a été **implémenté** dans la session précédente : `door_classifier.pt` entraîné (97.2% test acc), split `license_plate` rééquilibré et revalidé (mAP@50 72.6%), dépendances mortes retirées, diagnostic seuil de confiance vs sous-apprentissage confirmé sur `NO-Safety Vest`.

Ce document v2 part de l'état **actuel** (post-v1) et liste uniquement ce qu'il reste à faire, priorisé. C'est la version à suivre pour la suite du travail.

---

## État actuel — récapitulatif chiffré (après v1)

| Composant | Métrique clé | État |
|---|---|---|
| `best.pt` (EPI, 14 classes) | mAP@50 = 72.8% (val) / 75.3% (test) | ✅ Mesuré, ⚠️ classes gilet très faibles |
| ↳ `NO-Safety Vest` | AP@50 = 4.8% (conf=0.25) / 12.9% (conf=0.05) | 🔴 Point faible n°1, non résolu |
| ↳ `Safety Vest` | AP@50 = 49.4% / 60.0% | 🟠 Faible, non résolu |
| `best_gloves.pt` (EPI, 6 classes) | — | 🔴 Toujours impossible à valider (taxonomie incompatible), non résolu |
| `fire_smoke.pt` | mAP@50 = 46.4% (fire 65.2%, smoke 27.6%) | 🟠 Classe smoke faible, non résolu |
| `license_plate.pt` | mAP@50 = 72.6% (n=63) | ✅ Mesure fiabilisée en v1, modèle lui-même non amélioré |
| `door_classifier.pt` | top1 = 97.2% (test) | ✅ Résolu en v1 |
| Fall / ligne / objet abandonné / foule | — | 🔴 Toujours aucune métrique, non résolu |
| Pipeline EPI (3 modèles cascadés) | ~3.5 FPS (CPU, imgsz=480) | 🔴 Toujours insuffisant pour temps réel, non résolu |
| Fusion des 2 pipelines caméra | — | 🔴 Toujours 2 flux caméra séparés, non résolu |

**Conclusion : oui, il reste des améliorations nécessaires.** Elles étaient déjà identifiées en v1 (Partie B) mais non exécutées faute de GPU/données sur cette machine. Elles restent valables et sont reprises ci-dessous, priorisées.

---

## Priorité 1 — Fiabiliser la détection gilet de sécurité (`Safety Vest` / `NO-Safety Vest`)

C'est le point le plus critique pour un déploiement conformité EPI : ce sont les deux seules classes avec `NO-Hardhat` qui déclenchent une alerte métier, et `NO-Safety Vest` est quasi inutilisable en l'état (AP@50 4.8-12.9%).

**Diagnostic déjà établi (v1, A.4)** : ce n'est pas un problème de seuil de confiance — abaisser le seuil de 0.25 à 0.05 n'a récupéré que +8 points, très loin des 70-96% des autres classes. Le modèle rate la zone (boîte non prédite), il ne se trompe pas de classe.

**À faire, dans cet ordre** :
1. Relecture qualité manuelle d'un échantillon d'annotations `NO-Safety Vest`/`Safety Vest` (quelques dizaines d'images tirées des 678/2135 images train concernées) pour écarter un problème d'annotation incohérente avant d'investir dans un ré-entraînement.
2. Fine-tuning ciblé de `best.pt` avec sur-échantillonnage de ces deux classes et augmentation de données orientée (variations de couleur/texture de gilet).
3. Nécessite un GPU — sur CPU, même quelques epochs de fine-tuning sur les 30 765 images d'entraînement prendraient plusieurs heures à un jour.

## Priorité 2 — Unifier ou faire cohabiter proprement `best.pt` et `best_gloves.pt`

Le crash `IndexError` obtenu en tentant de valider `best_gloves.pt` sur le dataset 14-classes (v1, confirmé en Étape 2) démontre que les deux modèles EPI ne peuvent aujourd'hui ni être comparés, ni être fusionnés de façon fiable dans `ppe_dual_model_backup2.py` — leur association actuelle se fait par IoU sans vérifier la cohérence sémantique des classes.

**À faire** — trancher entre :
- Ré-entraîner un seul modèle sur l'union des deux taxonomies (recommandé : supprime la redondance de calcul et l'incohérence de seuils) ; ou
- Écrire une table de correspondance de classes explicite entre les deux modèles pour permettre une fusion cohérente sans ré-entraînement.

## Priorité 3 — Rendre le pipeline EPI compatible temps réel caméra

Mesuré : ~3.5 FPS avec les 3 modèles en cascade sur CPU, dominé par `best.pt` (YOLOv8m, 78.7 GFLOPs). Actions par coût croissant :
1. Revalider la précision à `imgsz=480` (actuellement mesurée seulement à 640) avant de l'adopter — gain vitesse déjà mesuré (~1.8x) mais impact précision non quantifié.
2. Export ONNX + `onnxruntime` (gain 1.3-2x sur CPU, pas de ré-entraînement nécessaire) — la prochaine action la moins coûteuse à tenter.
3. Quantification INT8 post-training, à valider soigneusement sur les classes déjà fragiles (Priorité 1) pour ne pas les dégrader davantage.
4. Si la caméra cible dispose d'un GPU/NPU embarqué (Jetson, Coral...), migrer vers TensorRT ou équivalent — solution de fond, le CPU restera un plafond dur sinon.
5. La Priorité 2 (un seul modèle EPI) réduit mécaniquement d'un tiers le nombre de forward pass par frame.

## Priorité 4 — Enrichir le dataset `smoke` de `fire_smoke.pt`

AP@50 = 27.6% contre 65.2% pour `fire`, avec seulement 97 instances de validation contre 1 021 — sous-représentation nette. À faire : collecte ou recherche de données complémentaires de fumée (densité/couleur/translucidité variées), puis ré-entraînement.

## Priorité 5 — Construire des datasets de validation pour fall / ligne / objet abandonné / foule

Ces modules n'ont toujours aucune métrique automatisée reproductible — seule une évaluation manuelle en temps réel existe. Sans mesure de référence, impossible de savoir si un futur changement les améliore ou les dégrade. À faire au minimum : enregistrer et annoter une vingtaine de séquences vidéo par module pour sortir de l'évaluation purement qualitative.

## Priorité 6 — Fusionner les deux pipelines caméra

`ppe_detection` et `surveillance_suite` ouvrent chacun leur propre flux caméra indépendamment. Pour un déploiement sur une seule caméra de surveillance couvrant à la fois conformité EPI et sécurité générale, il faudra un point d'entrée vidéo unique appelant les deux chaînes de traitement sur les mêmes frames. Non prioritaire tant que Priorité 1-3 ne sont pas traitées, mais bloquant pour le déploiement final tel que décrit dans la mission.

## Priorité 7 — Nettoyer la taxonomie `license_plate.pt`

Confirmé en v1 : `licence` et `number_plate` sont deux étiquettes redondantes du même concept (515 et 549 instances), `num_plate` est quasi inutilisé (18 instances/630 images). À faire : fusionner `licence`/`number_plate` en une seule classe, retirer ou ré-attribuer les 18 instances `num_plate`, puis ré-entraîner un modèle à 2 classes (plaque / pas-plaque), plus simple et plus interprétable.

---

## Ce qui bloque une exécution immédiate de la majorité de ces points

Priorités 1, 2 et 4 nécessitent un **ré-entraînement** de modèles YOLO de taille moyenne (YOLOv8m pour `best.pt`) sur des dizaines de milliers d'images — infaisable en temps raisonnable sur cette machine CPU-only (12 cœurs, pas de GPU). Elles nécessitent soit un accès GPU (local ou cloud), soit d'accepter des temps d'entraînement de plusieurs heures à un jour par expérience. Les priorités 3 (partiellement), 5, 6 et 7 sont réalisables sans GPU et peuvent être engagées dès maintenant si souhaité.
