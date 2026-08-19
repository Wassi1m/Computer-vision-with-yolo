# Registre des modèles

Ce document répond à une seule question, celle qu'on se pose toujours trop tard :
**quel modèle tourne exactement, d'où vient-il, et que vaut-il ?**

L'information existait, mais dispersée entre les rapports, l'historique git et
les notebooks Colab. Elle est ici rassemblée et fait référence.

Les métriques sont celles figées dans `tests/reference_modeles.json`, qui sert
de garde-fou automatique (`python tests/test_non_regression.py`).

---

## Modèles en service

### `ppe_detection/models/ppe_detector.pt` — conformité EPI

| | |
|---|---|
| **Rôle** | Détection des EPI et de leur absence (14 classes) |
| **Architecture** | YOLOv8m (25,9 M paramètres, ~79 GFLOPs) |
| **Version** | P11 — ré-entraînement 14 classes du 2026-08-15 |
| **Origine** | Fine-tuning du modèle P1, lui-même issu de `Hexmon/vyra-yolo-ppe-detection` |
| **Données** | `ppe_14c_equilibre` — 13 174 images, ~2 500 instances par classe et l'intégralité du gilet (`improvements/p10_sous_ensemble_epi.py`) |
| **Entraînement** | Kaggle T4, 78 époques sur 80 (session coupée à 12 h), `optimizer=SGD`, `lr0=0.001`, batch 16 |
| **Métriques** | `Safety Vest` 91.7% · `NO-Safety Vest` 77.9% (jeu gilet) · mAP@50 **69.8 % sur les 14 classes** |
| **Version précédente** | `ppe_detector_pre_14c.pt` (hors dépôt, voir `.gitignore`) |

**Pourquoi ce ré-entraînement.** Le modèle P1 avait été fine-tuné sur
`ppe_vest_clean_14c`, le sous-ensemble des images où le gilet est annoté. Ce
choix était défendable — il avait fait passer `NO-Safety Vest` de 4.8 % à
84.2 % — mais il a eu une conséquence non mesurée à l'époque : en s'entraînant
longtemps sur 2 728 images ne contenant que du gilet, le réseau a réaffecté sa
capacité et **effacé les douze autres classes**. Constaté le 2026-08-13 : AP@50
à exactement 0.0000 pour casque, gants, masque, lunettes, cône et personne, et
rien ne sortait même au seuil de confiance 0.01.

**Ce que le ré-entraînement a changé**, sur les 4 423 images du split `test` :
mAP@50 **0.0427 → 0.6976**, les douze classes remontant de zéro à 0.42–0.96.

**Le prix payé, et il est réel** : `NO-Safety Vest` recule de 84.2 % à 77.9 %
sur le jeu gilet. Cette classe ne compte que 1 435 instances contre 4 499 pour
`Safety Vest` ; elle souffre le plus du partage de capacité avec douze classes
de plus. C'est la classe qui signale l'infraction, donc celle où l'erreur coûte
le plus cher — la correction par sur-échantillonnage est à faire au prochain
entraînement.

**Leçon de méthode** : un jeu d'évaluation ne juge que ce qu'il annote. Mesuré
sur le seul `ppe_vest_clean_14c`, ce ré-entraînement paraissait être une
régression ; mesuré sur le seul `ppe_dataset`, un triomphe. Les deux mesures
étaient nécessaires pour voir l'arbitrage réel.

### `ppe_detection/models/ppe_complement.pt` — EPI complémentaires

| | |
|---|---|
| **Rôle** | 6 classes EPI, dont `safety_shoe` |
| **Architecture** | YOLOv8n |
| **Origine** | `Tanishjain9/yolov8n-ppe-detection-6classes` (Hugging Face), jamais ré-entraîné |
| **Métriques** | **Non mesurées** — aucun jeu de validation local ne couvre sa taxonomie |

**À savoir** : ce modèle ne couvre aucun concept absent de `ppe_detector.pt` **sauf**
`safety_shoe`, et ne possède aucune classe négative (il ne peut donc jamais
signaler une non-conformité). Le retirer de la cascade fait gagner 15 % de
FPS. Si les chaussures de sécurité ne font pas partie du référentiel du client,
il est purement redondant. La correspondance de classes entre les deux modèles
est explicitée dans `improvements/ppe_taxonomy.py`.

### `ppe_detection/models/masque_gilet.pt` — masque et gilet dédiés

| | |
|---|---|
| **Rôle** | 4 classes : `Mask`, `NO-Mask`, `Safety Vest`, `NO-Safety Vest` — prioritaire sur `ppe_detector.pt` pour ces deux concepts uniquement |
| **Architecture** | YOLOv8m, transfert depuis `ppe_detector.pt` (tête réinitialisée à 4 classes) |
| **Version** | Entraînement du 2026-08-17 |
| **Origine** | `improvements/p11_jeu_masque_gilet.py` — uniquement les images de `ppe_dataset` qui annotent réellement masque ou gilet |
| **Données** | 4 698 images train + 1 361 val (0 image contradictoire) ; split test (655 images) resté local, jamais entraîné dessus |
| **Entraînement** | Kaggle T4, 48 époques (arrêt anticipé, meilleure à l'époque 20), `optimizer=SGD`, `lr0=0.001`, 1h17 |
| **Métriques** (split test local, jamais vu) | `Mask` 97.2% · `NO-Mask` 96.4% · `Safety Vest` 93.2% · `NO-Safety Vest` 77.1% · mAP@50 **91.0 %** |

**Pourquoi ce modèle existe.** Le 2026-08-16, un ré-entraînement de
`ppe_detector.pt` ciblant `NO-Mask`/`NO-Safety Vest` par duplication d'images a
été **rejeté** : `NO-Mask` a reculé, `NO-Safety Vest` a stagné (voir
`reports/v3_results/epi_14c_candidat_20260816.json`). Diagnostic établi par
`improvements/p1_eval_par_concept.py` : sur un sous-ensemble qui annote
réellement ces deux concepts, `ppe_detector.pt` atteignait déjà 0.96/0.92/0.89/0.67
d'AP50 — contre 0.55/0.60/0.58/0.17 publiés sur `ppe_dataset` complet. Ce
n'était pas un problème de capacité mais de bruit d'annotation : la plupart des
~29 000 autres images de `ppe_dataset` n'annotent ni le masque ni le gilet, et
chaque détection correcte y était comptée comme un faux positif pendant
l'entraînement. Un modèle dédié, entraîné uniquement sur les images cohérentes,
n'a aucune image contradictoire à apprendre — et rien d'autre à oublier,
contrairement à un réglage de `ppe_detector.pt`.

**Intégration.** Branché en cascade dans `improvements/unified_surveillance.py`
(`AnalyseurEPI`, poids M3) avec priorité par concept dans
`improvements/ppe_taxonomy.py` (`PRIORITE_MODELE`). `ppe_detector.pt` reste
chargé et inchangé : si `masque_gilet.pt` est absent ou ne détecte rien sur une
image, `ppe_detector.pt` prend le relais automatiquement. Désactivable via
`--sans-masque-gilet`.

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
| **Version** | P4 — fine-tuning du 2026-08-08, mené à son terme le 2026-08-09 |
| **Données** | `fire_smoke_enriched` : jeu Roboflow d'origine (12 127 images) + 4 800 images de fumée `pyronear/pyro-sdis` (Hugging Face, Apache-2.0) |
| **Entraînement** | Colab T4, 60 époques (41 au premier jet, poursuivi jusqu'à 60) |
| **Métriques** | `fire` 90.4% · `smoke` 40.7% · mAP@50 65.5% (split `test`, non enrichi) |

**À savoir** : `smoke` reste le point faible du parc. Le complément pyro-sdis a
apporté +11.5 points, mais ses images sont des panaches de feu de forêt vus de
loin depuis une tour de guet — un domaine éloigné d'une caméra de site. Le
split `test` a été délibérément laissé intact pour que la comparaison avec la
mesure d'origine (27.6 %) reste valable.

**Ne pas juger ce modèle sur sa mAP@50.** Mesuré au niveau de la scène
(`tests/mesure_operationnelle.py`, jeu `fire_smoke_v9`, seuil 0.10) il repère
**96.7 %** des scènes contenant de la fumée pour 2.0 % de fausse alarme, très
loin de ce que ses 40.7 % d'AP laissent croire : la mAP sanctionne la position
du rectangle autour d'un panache, qui n'a pas de contour net, et non le service
rendu. Référence figée dans
`reports/v3_results/operationnel_fire_smoke_avant.json` — c'est **ce** chiffre
qu'un remplaçant doit battre.

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
