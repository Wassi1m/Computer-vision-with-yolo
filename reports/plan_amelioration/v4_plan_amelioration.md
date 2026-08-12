# Plan d'amélioration v4 — temps réel, multi-scénarios, mise en production

Date : 2026-08-10
Fait suite à `v3_plan_amelioration.md` (même dossier), **entièrement implémenté** :
service headless, transport webhook/JSONL, contrat d'interface documenté,
`/health`, reconnexion automatique du flux, isolation des modules en erreur,
tests de non-régression et de logique métier, endurance 30 min sans fuite,
registre des modèles, robustesse mesurée puis corrigée par fine-tuning P8 sur
3 modèles/4 (EPI, chute, plaque).

## Périmètre de ce projet

Inchangé depuis v3 : ce dépôt est le **moteur de détection IA**, et uniquement
lui. Alerting, notifications, tableaux de bord et interface utilisateur sont
hors périmètre — assurés par la plateforme consommatrice via l'API décrite
dans `docs/contrat_api.md`. Toute action de ce plan est jugée à l'aune d'une
question : *est-ce que ça rapproche d'un moteur temps réel, multi-scénarios,
livrable sans régression ?*

**Objectif explicite de ce plan** : le client final veut détecter, **en temps
réel**, **tous les scénarios en même temps**, sur des **caméras de
surveillance**. C'est le fil conducteur de toutes les priorités ci-dessous.

## Garde-fou : ne pas casser un moteur qui fonctionne

Le moteur passe aujourd'hui l'endurance (30 min, mémoire stable) et la
non-régression (17/17 tests logique métier). C'est un acquis fragile à
préserver. **Aucune action de ce plan ne s'applique sans repasser par le
filet de sécurité déjà en place** :

1. Avant de remplacer un `.pt` : sauvegarde horodatée hors dépôt (`cp ... _pre_v4.pt`),
   comme documenté dans `docs/exploitation.md` §6.
2. Avant tout déploiement : `python tests/test_non_regression.py` doit passer
   sans régression sur une classe suivie (tolérance 2 points d'AP@50). Un
   échec = ne pas déployer, restaurer la sauvegarde.
3. Avant toute modification touchant la boucle principale
   (`unified_surveillance.py`) : relancer `tests/test_integration.py` et
   `tests/test_logique_metier.py` (rapides, sans GPU).
4. Pour tout changement d'architecture ou de cascade (ex : retrait d'un
   modèle, remplacement de `best.pt`) : le déployer d'abord **en parallèle**
   de l'existant (flag optionnel, comme `--sans-gants` déjà présent) plutôt
   que de remplacer directement — bascule en défaut seulement après
   validation mesurée.
5. Après tout changement touchant la boucle vidéo ou la gestion mémoire : un
   nouveau test d'endurance (au moins 30 min, idéalement plus long que le
   précédent) avant de considérer le point clos.
6. Sur la VM GCP : toujours passer par `deploy/00_tout_entrainer.sh` ou les
   scripts unitaires (`arreter_en_sortant` intégré) — jamais de manipulation
   manuelle qui laisserait l'instance tourner facturée pour rien.

**Aucune action ci-dessous ne justifie de sauter une de ces six étapes, même
sous pression de délai.**

---

## État actuel — ce qui est déjà solide

| Composant | Métrique | État |
|---|---|---|
| Pipeline unifié (`unified_surveillance.py`) | 6 modules, 1 flux caméra, headless par défaut | ✅ |
| API / contrat d'interface | webhook + JSONL, schéma documenté | ✅ |
| `/health`, reconnexion, isolation par module | vérifiés dans le code | ✅ |
| Non-régression + logique métier | 17/17 tests passent | ✅ |
| Endurance | 30 min, mémoire stable (+30 Mo), FPS stable | ✅ |
| `best.pt` (EPI) | mAP50 88.5% jour / 85.3% nuit (P8) | ✅ |
| `fall_detector.pt` (chute) | mAP50 99.1% jour / 97.6% nuit (P8) | ✅ |
| `license_plate.pt` (plaque) | mAP50 85.4% jour / 76.9% nuit (P8) | ✅ |
| `fire_smoke.pt` (feu/fumée) | mAP50 73.9% jour / **32.0% nuit** | 🟠 seul modèle sans P8 |
| Débit CPU, tous modules actifs | **~5-7 FPS** (endurance réelle) | 🔴 pas temps réel |

**Les 3/4 modèles retravaillés tiennent bien la nuit. Ce qui manque
maintenant n'est plus la justesse des modèles pris un par un, c'est la
vitesse du pipeline complet et sa capacité à couvrir plusieurs caméras et
plusieurs scénarios simultanément, en production réelle.**

---

## Classification des problèmes par niveau d'importance

| Niveau | Signification |
|---|---|
| 🔴 **Bloquant** | Incompatible avec l'objectif "temps réel, tous scénarios, production" tel quel |
| 🟠 **Critique** | Livrable, mais risque réel (scénario non fiable, ou point non tranché avec l'aval) |
| 🟡 **Important** | Pas de risque immédiat, dette qui coûtera cher plus tard |
| 🔵 **Souhaitable** | Gain de valeur, sans urgence |

### Vue d'ensemble

| # | Problème | Catégorie | Niveau |
|---|---|---|---|
| 1.1 | Cible de déploiement (CPU vs GPU) non tranchée — conditionne tout le reste | Décision | 🔴 Bloquant |
| 1.2 | Débit ~5-7 FPS en continu, loin du temps réel vidéo | Performance | 🔴 Bloquant |
| 1.3 | `best.pt` (YOLOv8m) consomme à lui seul ~78% du temps de la cascade EPI | Performance | 🔴 Bloquant |
| 2.1 | `smoke` toujours faible la nuit (32%), seul modèle sans traitement P8 | Précision | 🟠 Critique |
| 2.2 | Robustesse validée uniquement sur dégradation synthétique, jamais sur vraies images de nuit | Précision | 🟠 Critique |
| 2.3 | Une seule caméra par processus, pas de `camera_id` dans le format d'évènement | Intégration | 🟠 Critique |
| 2.4 | 4 questions ouvertes du contrat API non tranchées avec la plateforme aval | Intégration | 🟠 Critique |
| 2.5 | `best_gloves.pt` dans la cascade par défaut (+15% FPS possible, gain quasi gratuit) | Performance | 🟠 Critique |
| 3.1 | Scénarios ligne/objet abandonné/foule sans dataset ni métrique automatisée | Précision | 🟡 Important |
| 3.2 | `best.pt` : seules les classes gilet ont été retravaillées en profondeur (héritage v1/v2) | Précision | 🟡 Important |
| 4.1 | `yolo11n-cls.pt` orphelin, `Fire smoke yolo.v4i` dataset inutilisé | Nettoyage | 🔵 Souhaitable |

---

## Priorité 1 — Rendre le pipeline réellement temps réel

C'est le seul axe qui touche directement la demande "temps réel + tous les
scénarios en même temps". Aucune quantité d'entraînement supplémentaire ne
le résout : le goulot est architectural (un modèle lourd dans une boucle
CPU), pas un manque de données.

**1.1 — 🔴 Trancher la cible de déploiement.**
Deux chemins mutuellement exclusifs, à décider avec le client/la plateforme
avant d'investir dans l'un ou l'autre :

- **GPU disponible chez le client ou en serveur dédié** → priorité 1.2 devient
  quasi gratuite (le L4 utilisé aujourd'hui pour l'entraînement ferait tomber
  les 207ms de `best.pt` à ~10-20ms). Il s'agit alors de déployer le
  *moteur*, pas seulement d'entraîner dessus — adapter le `Dockerfile` pour
  une image CUDA, valider le pipeline complet sur GPU avec le test
  d'endurance avant bascule.
- **CPU uniquement (edge, caméra embarquée, contrainte client)** → 1.3 est la
  seule voie.

Sans cette décision, tout travail de performance ci-dessous risque d'être
refait en double.

**1.2 — 🔴 Mesurer le temps réel visé.**
"Temps réel" pour de la sécurité n'impose pas forcément 25-30 FPS : une chute
ou un départ de feu ne durent pas une fraction de seconde. Avant d'investir
dans la vitesse, faire chiffrer par le client/la plateforme la cadence
minimale réellement nécessaire par scénario (l'EPI peut tolérer plus de
latence qu'une détection de chute, par exemple). Cela évite de sur-optimiser
un objectif jamais formulé explicitement.

**1.3 — 🔴 Alléger `best.pt` si la cible reste CPU.**
`best.pt` (YOLOv8m, 25,9M paramètres) coûte 207ms/image contre 35-40ms pour
tous les autres modèles de la cascade (5-8M paramètres). Deux options, à
tester **en parallèle de l'existant** (garde-fou n°4) avant toute bascule :
- Réentraîner sur une architecture plus légère (YOLOv8n/s ou YOLO26n comme
  les autres modules) avec les mêmes classes et le même dataset — cohérent
  avec le reste du parc, gain de FPS attendu important.
- Si la précision se dégrade trop, explorer la quantification (déjà noté en
  v3 : ONNX n'avait rien apporté sur cette machine — à re-mesurer seulement
  si l'architecture change, pas avant).

**2.5 — 🟠 Retirer `best_gloves.pt` de la cascade par défaut.**
Gain mesuré +15% FPS, apport propre limité à `safety_shoe` (aucun dataset
local pour la valider). Le flag `--sans-gants` existe déjà : inverser le
défaut est un changement à faible risque, à valider par le test
d'intégration avant de le figer.

## Priorité 2 — Fiabiliser ce qui reste faible ou non tranché

**2.1 — 🟠 Traiter `smoke` comme les 3 autres modèles P8.**
Recette déjà rodée aujourd'hui (dataset dégradé + fine-tuning sur la VM GPU +
validation non-régression + mesure de robustesse). C'est le seul des 4
modèles principaux qui reste en dessous de 40% la nuit. Aucune dépendance
externe, réalisable immédiatement.

**2.2 — 🟠 Valider la robustesse sur de vraies images de nuit.**
Le gain P8 mesuré aujourd'hui est réel mais mesuré sur des dégradations
synthétiques (assombrissement d'images de jour) — le code le documente
lui-même (`p8_dataset_nuit.py`). Dès qu'un extrait de vraies séquences
nocturnes de caméra client est disponible, le faire passer dans
`tests/mesure_robustesse.py` pour confirmer que le gain généralise. Ne pas
promettre le chiffre de 85%/98%/77% nuit à un client avant cette
confirmation.

**2.3 — 🟠 Ajouter un identifiant caméra au format d'évènement.**
Champ `camera_id` (ou équivalent) dans le schéma JSON de `contrat_api.md`,
alimenté par un paramètre `--camera-id`/`MOTEUR_CAMERA_ID`. Nécessaire dès
qu'il y a plus d'une caméra — documenter aussi comment orchestrer plusieurs
instances (une par flux) en attendant un vrai support multi-flux dans un même
processus, qui serait un chantier bien plus lourd et n'est pas justifié tant
que le nombre de caméras cibles n'est pas connu.

**2.4 — 🟠 Trancher les 4 questions ouvertes avec la plateforme aval.**
Listées dans `docs/contrat_api.md` §9 : protocole garanti (webhook suffit-il
ou faut-il une file de messages), format d'identifiant caméra/site,
authentification, politique de rétention des plaques. Ce sont des décisions,
pas du code — mais sans elles, l'intégration réelle ne peut pas démarrer même
si le moteur est prêt.

## Priorité 3 — Couvrir les scénarios non encore mesurés

**3.1 — 🟡 Ligne, objet abandonné, foule : dataset et métrique.**
Ces modules tournent sur heuristique (OpenCV classique) sans jeu de
validation ni test automatisé — évaluation manuelle uniquement aujourd'hui.
Si ces scénarios font partie du périmètre client "tous les scénarios", ils
ont besoin du même traitement que EPI/chute/feu/plaque : un jeu de test
annoté, une mesure chiffrée, une référence figée. À chiffrer en effort une
fois le périmètre client confirmé — ne pas lancer sans savoir si ces
scénarios sont réellement demandés.

**3.2 — 🟡 `best.pt` : classes hors gilet jamais retravaillées en profondeur.**
Le fine-tuning historique (v1) a ciblé le gilet ; les autres classes
(casque, gants, lunettes, masque) restent sur l'entraînement d'origine. Bon
niveau actuel (92-98% selon la classe, voir `registre_modeles.md`) mais
jamais remis en question depuis. Faible priorité tant qu'aucune régression
n'est observée en usage réel.

## Priorité 4 — Nettoyage (sans urgence)

**4.1 — 🔵** `yolo11n-cls.pt` orphelin (aucun script ne le charge) et
`surveillance_suite/data/dataset/Fire smoke yolo.v4i` (dataset alternatif
inutilisé) : candidats à la suppression, à confirmer avant de supprimer quoi
que ce soit qui pourrait être une référence future.

---

## Ordre d'exécution recommandé

**Étape 1 — la décision qui conditionne tout (1.1, 1.2).** Sans cible de
déploiement et sans cadence cible par scénario, tout travail de performance
risque d'optimiser la mauvaise chose. C'est une décision externe (client +
plateforme), à obtenir avant d'investir du temps d'ingénierie.

**Étape 2 — les gains rapides et sans risque, en parallèle de l'étape 1**
(2.1 `smoke` sur la VM GPU, 2.5 retrait des gants par défaut). Aucun des deux
ne dépend de la décision de déploiement, aucun ne casse l'existant si le
garde-fou (tests + endurance) est respecté à chaque étape.

**Étape 3 — une fois la cible connue** : 1.3 (alléger `best.pt` si CPU
confirmé, ou porter le déploiement sur GPU si GPU confirmé), 2.3
(identifiant caméra), 2.4 (trancher avec la plateforme aval).

**Étape 4 — dès que des données réelles sont disponibles** : 2.2 (validation
nuit réelle), 3.1 (scénarios ligne/objet/foule, si dans le périmètre client).

**Étape 5 — en continu, sans bloquer le reste** : 3.2, 4.1.

## Sur la proposition d'entraîner davantage de modèles sur la VM GPU

Utile et à poursuivre — mais seulement pour ce qu'elle résout réellement :
la **justesse** (comme `smoke`, priorité 2.1), pas la **vitesse**. La VM GPU
a un second usage, plus impactant pour "temps réel", qui n'a pas encore été
exploité : servir de **cible d'inférence en production**, pas seulement
d'entraînement. C'est exactement l'objet de la décision 1.1.
