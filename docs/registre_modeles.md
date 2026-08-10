# Registre des modèles

Ce document répond à une seule question, celle qu'on se pose toujours trop tard :
**quel modèle tourne exactement, d'où vient-il, et que vaut-il ?**

L'information existait, mais dispersée entre les rapports, l'historique git et
les notebooks Colab. Elle est ici rassemblée et fait référence.

Les métriques sont celles figées dans `tests/reference_modeles.json`, qui sert
de garde-fou automatique (`python tests/test_non_regression.py`).

---

## Modèles en service

### `ppe_detection/models/best.pt` — conformité EPI

| | |
|---|---|
| **Rôle** | Détection des EPI et de leur absence (14 classes) |
| **Architecture** | YOLOv8m (25,9 M paramètres, ~79 GFLOPs) |
| **Version** | P1 — fine-tuning du 2026-08-08 |
| **Origine** | `Hexmon/vyra-yolo-ppe-detection` (Hugging Face), puis fine-tuné |
| **Données** | `ppe_vest_clean_14c` — sous-ensemble des images où le gilet est réellement annoté (voir ci-dessous) |
| **Entraînement** | Colab T4, 40 époques, lr0=0.001, batch 32 — `reports/colab_package/p1_vest/p1_finetune_vest_colab.ipynb` |
| **Métriques** | `Safety Vest` 92.5% · `NO-Safety Vest` 84.2% · mAP@50 88.3% |
| **Version précédente** | Voir l'historique git (commit `023f92f` et antérieurs) |

**À savoir** : le jeu d'origine `ppe_dataset` est un patchwork de lots annotés
chacun sur un seul concept. Environ 14 300 images sur 30 765 enseignaient au
modèle que la zone du gilet est du fond. Le fine-tuning a donc porté sur le
seul sous-ensemble où le gilet est effectivement annoté — c'est ce qui a fait
passer `NO-Safety Vest` de 4.8 % à 84.2 %. Ré-entraîner sur le jeu complet
reproduirait le défaut.

### `ppe_detection/models/best_gloves.pt` — EPI complémentaires

| | |
|---|---|
| **Rôle** | 6 classes EPI, dont `safety_shoe` |
| **Architecture** | YOLOv8n |
| **Origine** | `Tanishjain9/yolov8n-ppe-detection-6classes` (Hugging Face), jamais ré-entraîné |
| **Métriques** | **Non mesurées** — aucun jeu de validation local ne couvre sa taxonomie |

**À savoir** : ce modèle ne couvre aucun concept absent de `best.pt` **sauf**
`safety_shoe`, et ne possède aucune classe négative (il ne peut donc jamais
signaler une non-conformité). Le retirer de la cascade fait gagner 15 % de
FPS. Si les chaussures de sécurité ne font pas partie du référentiel du client,
il est purement redondant. La correspondance de classes entre les deux modèles
est explicitée dans `improvements/ppe_taxonomy.py`.

### `surveillance_suite/models/fall_detector.pt` — détection de chute

| | |
|---|---|
| **Rôle** | 2 classes : `falling`, `stand` |
| **Architecture** | YOLO26n |
| **Version** | P8 — fine-tuning du 2026-08-10 (reprend le poids P5) |
| **Origine** | Fine-tuning P5 sur `fall_detection_enriched_robuste` |
| **Données** | Jeu P5 (945 images) + 60 % de copies dégradées (nuit/contre-jour/flou/intempérie) — `improvements/p8_dataset_nuit.py` |
| **Entraînement** | Colab T4, 60 époques — `reports/colab_package/p8_chute/p8_train_chute_colab.ipynb` |
| **Métriques** | `falling` 99.2% · `stand` 98.8% · mAP@50 99.0% (split `test`) — contre 95.4 % avant P8 |

**À savoir** : la fusion des deux jeux a demandé un remappage de classes. Les
labels DeZan utilisent trois identifiants (0/1/2) sans documentation ; l'analyse
des noms de fichiers a établi que 0 = chute et {1,2} = deux activités sans
chute, remappées en `stand` (voir `improvements/p5_merge_fall_dataset.py`).

**P8** : fine-tuning de robustesse aux conditions dégradées (motivé par
`reports/v3_results/robustesse_conditions_reelles.json`, qui montrait un
effondrement à 49.2 % de mAP@50 en faible luminosité). Le gain observé ici sur
`test` (conditions normales) est probablement en partie dû au fait que ce split
est plus petit (51 images) — à confirmer par une nouvelle mesure de robustesse
en conditions dégradées avant de considérer le point clos. Ancien poids
conservé en `fall_detector_pre_p8.pt` (hors dépôt, voir `.gitignore`).

### `surveillance_suite/models/fire_smoke.pt` — feu et fumée

| | |
|---|---|
| **Rôle** | 2 classes : `fire`, `smoke` |
| **Architecture** | YOLO26n |
| **Version** | P4 — fine-tuning du 2026-08-08 |
| **Données** | `fire_smoke_enriched` : jeu Roboflow d'origine (12 127 images) + 4 800 images de fumée `pyronear/pyro-sdis` (Hugging Face, Apache-2.0) |
| **Entraînement** | Colab T4, 41 époques (patience atteinte) |
| **Métriques** | `fire` 91.6% · `smoke` 39.1% · mAP@50 65.3% (split `test`, non enrichi) |

**À savoir** : `smoke` reste le point faible du parc. Le complément pyro-sdis a
apporté +11.5 points, mais ses images sont des panaches de feu de forêt vus de
loin depuis une tour de guet — un domaine éloigné d'une caméra de site. Le
split `test` a été délibérément laissé intact pour que la comparaison avec la
mesure d'origine (27.6 %) reste valable.

### `surveillance_suite/models/license_plate.pt` — plaques

| | |
|---|---|
| **Rôle** | 1 classe : `plate` |
| **Architecture** | YOLO26n |
| **Version** | P8 — fine-tuning du 2026-08-10 (reprend le poids P7) |
| **Origine** | Fine-tuning P7 sur `license_plate_unified_robuste` |
| **Données** | Jeu P7 (504 images train) + 60 % de copies dégradées (nuit/contre-jour/flou/intempérie) — `improvements/p8_dataset_nuit.py` |
| **Entraînement** | Colab T4, 60 époques — `reports/colab_package/p8_plaque/p8_train_plaque_colab.ipynb` |
| **Métriques** | mAP@50 86.2% (`test`) — contre 85.3 % avant P8 |

**À savoir** : le modèle d'origine avait trois classes redondantes du même
concept (`licence`, `num_plate`, `number_plate`), qui se disputaient les mêmes
boîtes et diluaient la mAP. Leur fusion en une seule classe explique l'essentiel
du gain P7. Le module consommateur (`module_lpr.py`) ne lit pas les identifiants
de classe : le remplacement lui a été transparent.

**P8** : fine-tuning de robustesse aux conditions dégradées (motivé par
`reports/v3_results/robustesse_conditions_reelles.json`, qui montrait un
effondrement à 41.8 % de mAP@50 en faible luminosité — le pire cas mesuré,
aggravé par le flou de mouvement, −75 %). Le gain en conditions normales est
faible mais positif ; l'apport réel se mesure en conditions dégradées, à
vérifier par une nouvelle passe de `tests/mesure_robustesse.py`. Ancien poids
conservé en `license_plate_pre_p8.pt` (hors dépôt, voir `.gitignore`).

### `surveillance_suite/models/door_classifier.pt` — état de porte

| | |
|---|---|
| **Rôle** | Classification `open` / `closed` / `semi` |
| **Version** | v1 — 2026-08-07 |
| **Données** | `Door - Open - Closed -.v1i.folder` (6 968 images Roboflow, présentes dans le dépôt mais jamais utilisées auparavant) |
| **Métriques** | top-1 97.2 % (`test`) |

**À savoir** : ce module dispose aussi d'une heuristique SSIM de secours,
utilisée automatiquement si le modèle est absent. Elle est explicitement moins
fiable — sensible aux conditions d'éclairage. Le classifieur a besoin du fichier
`surveillance_suite/data/dataset/roi.json` en plus des poids.

### Modèles génériques (non entraînés ici)

`yolo26n.pt` (détection et suivi COCO), `yolo26n-pose.pt` (secours du module
chute), `yolo26s.pt`, `yolov8n.pt`. Poids Ultralytics standard, utilisés tels
quels.

`yolo11n-cls.pt` est **orphelin** : aucun script ne le charge. Candidat à la
suppression.

---

## Traçabilité d'une version

Les fichiers `.pt` ne sont **pas** versionnés dans git (142 Mo à eux seuls,
répartis sur une dizaine de fichiers — trop lourd pour un dépôt de code, et sans
valeur ajoutée face à un stockage dédié). Ils vivent uniquement sur les machines
qui en ont besoin. Ce que git conserve, et qui fait référence, c'est
`tests/reference_modeles.json` : les métriques et la provenance de chaque
version, committées et donc traçables même quand le binaire ne l'est pas.

| Question | Où trouver la réponse |
|---|---|
| Que vaut-il ? | `tests/reference_modeles.json` |
| Comment a-t-il été construit ? | Notebook Colab et script de préparation des données cités ci-dessus |
| Quel fichier exact est en service ? | À déterminer par une somme de contrôle (`sha256sum <chemin>`) consignée à chaque déploiement — voir §« Avant de remplacer un modèle » |
| Où trouver une version antérieure ? | Sur le poste ou le stockage où elle a été conservée avant remplacement (voir `docs/exploitation.md`, §retour arrière) : ce n'est **pas** git |

## Avant de remplacer un modèle

1. **Conserver l'ancien fichier** hors du dépôt (copie horodatée sur le poste ou
   le stockage de modèles) avant de l'écraser — c'est la seule sauvegarde
   possible puisque git ne le suit pas.
2. Lancer `python tests/test_non_regression.py --modele <nom>` avec le **nouveau**
   modèle en place.
3. Un échec signifie une perte de plus de 2 points d'AP@50 sur une classe suivie :
   ne pas déployer, revenir à la copie conservée à l'étape 1.
4. En cas de succès, mettre à jour la référence avec `--maj`, puis commiter
   `tests/reference_modeles.json` (le fichier `.pt` lui-même reste hors dépôt).

**Ne jamais abaisser une valeur de référence pour faire passer le test** : c'est
exactement la régression que ce dispositif existe pour détecter.
