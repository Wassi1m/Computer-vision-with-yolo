# Plan d'amélioration v3 — fiabiliser le moteur de détection pour livraison client

Date : 2026-08-08
Fait suite à `v2_plan_amelioration.md` (même dossier), **entièrement implémenté** :
gilet `NO-Safety Vest` 4.8% → 84.2% AP@50, plaques 72.6% → 90.9%, fumée 27.6% →
39.1%, nouveau détecteur de chute dédié (97.6%), table de correspondance EPI
branchée, pipeline caméra unifié testé de bout en bout.

## Périmètre de ce projet

Ce dépôt est le **moteur de détection IA**, et uniquement lui. L'alerting, les
notifications, les tableaux de bord et l'interface utilisateur sont hors
périmètre : ils seront assurés par une autre plateforme qui consommera ce
moteur **via API**. Toute priorité ci-dessous est évaluée à l'aune d'une seule
question : *est-ce que ça rend les détections plus justes, plus stables, ou
plus intégrables par un tiers ?*

Deux décisions de cadrage :

- **Le mode caméra avec affichage est conservé** comme mode de test et de
  démonstration. Le service headless est le mode de production ; l'affichage
  reste disponible en option (`--display`) pour valider visuellement une scène,
  faire une démonstration client ou diagnostiquer un comportement.
- **Le déploiement serveur/GPU est remis à la fin.** D'ici là, tout entraînement
  se fait sur Colab (gratuit, T4) selon la procédure rodée en v2, et le moteur
  doit rester exploitable sur CPU.

## État actuel — ce qui est déjà solide

| Composant | Métrique | État |
|---|---|---|
| `best.pt` — gilet | `Safety Vest` 92.5% / `NO-Safety Vest` 84.2% AP@50 | ✅ Fiabilisé en v2 |
| `best.pt` — autres EPI | casque 95.0/93.6%, gants 95.1/91.5%, lunettes 98.1/95.8%, masque 96.2/92.4% | ✅ Bon niveau |
| `license_plate.pt` | mAP@50 90.9% (val) / 85.3% (test) | ✅ Fiabilisé en v2 |
| `fall_detector.pt` | `falling` 97.6% / `stand` 93.2% AP@50 | ✅ Créé en v2 |
| `door_classifier.pt` | top1 97.2% | ✅ Fiabilisé en v1 |
| `fire_smoke.pt` | `fire` 91.6% / `smoke` 39.1% AP@50 | 🟠 `smoke` reste faible |
| Pipeline unifié | 6 modules, flux caméra unique, ~3.8 FPS CPU | ✅ Fonctionne, ⚠️ jamais testé en continu |

**Les modèles sont bons. Ce qui manque, c'est tout ce qui garantit qu'ils le
restent** — et un moyen pour un tiers de les consommer.

---

## Classification des problèmes par niveau d'importance

Quatre niveaux, définis par la conséquence concrète si le point n'est pas traité :

| Niveau | Signification |
|---|---|
| 🔴 **Bloquant** | Le produit ne peut pas être livré à un client en l'état |
| 🟠 **Critique** | Livrable, mais avec un risque réel d'incident non détecté chez le client |
| 🟡 **Important** | Pas de risque immédiat, mais dette qui coûtera cher à la première évolution |
| 🔵 **Souhaitable** | Gain de valeur, sans urgence |

### Vue d'ensemble

| # | Problème | Catégorie | Niveau |
|---|---|---|---|
| 1.1 | Pas de service headless (le produit est un script interactif) | Intégration | 🔴 Bloquant |
| 1.2 | Aucun transport des détections vers l'extérieur | Intégration | 🔴 Bloquant |
| 1.3 | Aucun contrat d'interface documenté | Intégration | 🔴 Bloquant |
| 3.1 | Coupure du flux vidéo = arrêt définitif et silencieux | Robustesse | 🔴 Bloquant |
| 1.4 | Pas de point de santé (`/health`) | Observabilité | 🟠 Critique |
| 2.1 | Pas de jeu de référence figé par modèle | Non-régression | 🟠 Critique |
| 2.2 | Aucun test de non-régression automatisé | Non-régression | 🟠 Critique |
| 2.3 | Logique métier (lissage, IoU, fusion) non testée | Non-régression | 🟠 Critique |
| 3.2 | Jamais testé en fonctionnement continu (fuite mémoire inconnue) | Robustesse | 🟠 Critique |
| 3.3 | Une exception dans un module emporte tout le pipeline | Robustesse | 🟠 Critique |
| 4.1 | Seuils de confiance non calibrés sur mesure | Justesse | 🟠 Critique |
| 5.1 | Chute et feu déclenchent sur une seule image | Justesse | 🟠 Critique |
| 5.2 | L'historique EPI peut sauter d'une personne à l'autre (**bug réel**) | Justesse | 🟠 Critique |
| 5.3 | Aucune politique de rétention des données sensibles | Conformité | 🟠 Critique |
| 7.4 | Robustesse nuit/contre-jour/intempéries jamais évaluée | Justesse | 🟠 Critique |
| 2.4 | Pas de test d'intégration bout en bout pérennisé | Non-régression | 🟡 Important |
| 3.4 | Journalisation par `print`, non structurée | Observabilité | 🟡 Important |
| 4.2 | Seuils dispersés et en dur dans les modules | Maintenabilité | 🟡 Important |
| 4.3 | Configuration non pilotable par variables d'environnement | Maintenabilité | 🟡 Important |
| 6.1 | Pas d'empaquetage reproductible (Docker) | Livraison | 🟡 Important |
| 6.2 | Pas de registre des modèles (provenance, version) | Livraison | 🟡 Important |
| 6.3 | Procédure de retour arrière non documentée | Livraison | 🟡 Important |
| 7.1 | `smoke` à 39.1% AP@50 | Précision | 🟡 Important |
| 7.2 | `best.pt` : fine-tuning arrêté à 40 époques, classes non-gilet non reprises | Précision | 🔵 Souhaitable |
| 7.3 | `best_gloves.pt` redondant dans la cascade (+15% FPS si retiré) | Performance | 🔵 Souhaitable |
| 8 | Plafond CPU à ~3.8 FPS | Performance | 🔵 Souhaitable |

**Répartition : 4 bloquants, 11 critiques, 8 importants, 3 souhaitables.**

---

## Priorité 1 — Exposer le moteur en service consommable par API

Blocage n°1 pour une livraison client : aujourd'hui le « produit » est un script
qui ouvre une fenêtre `cv2.imshow` et attend la touche `q`. Une plateforme tierce
ne peut rien en consommer.

**1.1 — 🔴 Service headless, mode affichage conservé en option.**
Extraire la boucle de traitement de `unified_surveillance.py` dans un service
sans dépendance à l'affichage ni au clavier, avec arrêt propre sur SIGTERM/SIGINT
(indispensable en conteneur ou sous systemd). L'affichage caméra n'est pas
supprimé : il devient un mode optionnel, destiné aux tests, aux démonstrations
client et au diagnostic visuel. Le drapeau `--no-display` existe déjà — il faut
inverser la logique pour que le headless soit le comportement par défaut.

**1.2 — 🔴 Transport des détections.**
`BusEvenements` produit déjà des `Evenement` datés et structurés (JSONL) : la
brique existe, il manque le transport. Le choix du protocole (REST + webhook,
flux websocket, ou file de messages) doit être arrêté avec l'équipe de la
plateforme consommatrice avant d'écrire le code — c'est la seule décision
extérieure qui bloque ce plan.

**1.3 — 🔴 Contrat d'interface documenté.**
Schéma exact des évènements (type, libellé, boîte, confiance, horodatage,
identifiant de caméra), codes d'erreur, garanties de livraison. C'est ce document
que l'équipe d'en face utilisera ; sans lui, l'intégration se fera par tâtonnement.

**1.4 — 🟠 Point de santé.**
Exposer : flux vidéo vivant ou non, modèles chargés, FPS courant, horodatage de
la dernière détection. Minimum pour qu'un exploitant sache à distance si le
moteur tourne correctement.

## Priorité 2 — Verrouiller la non-régression des modèles

Rien ne garantit aujourd'hui qu'une modification future ne dégrade pas
silencieusement les détections. C'est le risque principal une fois le produit
chez un client : une régression ne se voit pas, elle se subit.

**2.1 — 🟠 Jeu de référence figé** par modèle (les splits `test` existent déjà et
n'ont jamais servi à l'entraînement), avec les métriques actuelles comme valeurs
de référence — celles du tableau d'état ci-dessus.

**2.2 — 🟠 Test de non-régression automatisé** : re-mesure chaque modèle sur son
jeu figé et échoue si l'AP@50 chute de plus d'un seuil convenu (2 points est un
point de départ raisonnable). À lancer avant tout remplacement de `.pt` et avant
toute livraison.

**2.3 — 🟠 Tests unitaires sur la logique métier**, sans GPU, en quelques
secondes : lissage temporel du port d'EPI, association EPI ↔ personne par IoU,
déduplication inter-modèles (`ppe_taxonomy.fusionner`), franchissement de ligne,
anti-répétition du bus d'évènements. Cette logique décide des violations
signalées : une erreur ici est aussi grave qu'une erreur de modèle, et bien plus
facile à introduire.

**2.4 — 🟡 Test d'intégration bout en bout** sur une courte vidéo versionnée,
vérifiant que les évènements attendus sont produits. La méthode a déjà servi en
v2 et a révélé deux vrais bugs (sérialisation JSON des coordonnées numpy, chemin
du ROI porte) — il faut la pérenniser plutôt que la refaire à la main.

## Priorité 3 — Rendre le moteur increvable en fonctionnement continu

Un moteur livré tourne des semaines sans supervision. Le nôtre n'a jamais tourné
plus de quelques secondes.

**3.1 — 🔴 Reconnexion automatique du flux vidéo.**
Aujourd'hui, `cap.read()` qui échoue sort de la boucle et le processus s'arrête.
Sur un flux RTSP réel (coupure réseau, redémarrage de caméra), c'est un arrêt
définitif et silencieux — le client croit le système actif alors qu'il ne
surveille plus rien. Il faut une reconnexion avec temporisation progressive, et
un évènement publié à chaque perte et reprise du flux.

**3.2 — 🟠 Test d'endurance** : plusieurs heures sur une vidéo en boucle, en
surveillant mémoire et FPS. Objectif : détecter une fuite mémoire ou une dérive
de performance avant le client. Rien ne le prouve aujourd'hui — le seul test
réalisé portait sur 8 images.

**3.3 — 🟠 Dégradation propre par module.**
Le principe existe (un modèle absent désactive son module au lieu de planter) ;
il faut l'étendre aux erreurs *en cours d'exécution* : un module qui lève une
exception à répétition doit être désactivé et signalé, sans emporter les cinq
autres.

**3.4 — 🟡 Journalisation structurée** (niveau, module, horodatage) en
remplacement des `print`. C'est ce qui permet de diagnostiquer un incident client
à distance, sans accès à la machine.

## Priorité 4 — Rendre les seuils explicites, mesurés et configurables

Les seuils de confiance décident directement de ce qui est signalé. Ils sont
aujourd'hui dispersés et choisis à la main : `CONF_THRESHOLD = 0.4` global dans
`config.py`, `conf=0.4` en dur dans quatre modules, `conf=0.5` dans un cinquième,
et une table par classe dans `ppe_taxonomy.py`. Aucun n'est justifié par une
mesure.

**4.1 — 🟠 Calibrer chaque seuil sur les courbes précision/rappel** déjà produites
par Ultralytics (`BoxPR_curve.png` existe pour chaque entraînement). Le bon seuil
dépend du coût métier de l'erreur, et ce coût n'est pas symétrique : rater un
ouvrier sans casque (faux négatif) est plus grave que déclencher une vérification
inutile (faux positif). Le seuil doit être choisi en connaissance de ce compromis,
pas par défaut.

**4.2 — 🟡 Centraliser** tous les seuils dans la configuration, aucun en dur dans
les modules.

**4.3 — 🟡 Configuration par variables d'environnement**, avec les valeurs
actuelles comme défauts. Un client doit pouvoir ajuster un seuil ou un chemin de
modèle sans éditer du code Python.

**4.4 — 🟡 Documenter, pour chaque seuil, la mesure qui l'a produit.** Un seuil
sans justification traçable est un seuil que personne n'osera modifier plus tard.

## Priorité 5 — Fiabiliser les détections dans la durée

Un modèle juste image par image peut produire un flux d'évènements incohérent :
scintillement d'état, alertes en rafale, identités qui sautent. C'est ce que le
client percevra comme un « produit pas au point », indépendamment des métriques.

**5.1 — 🟠 Généraliser le lissage temporel.**
Il existe pour les EPI (`HIST_N`/`HIST_K`) et pour la porte, mais **pas pour la
chute ni le feu/fumée** : ces deux modules déclenchent sur une seule image, donc
un faux positif isolé suffit à produire une alerte. Une chute ou un départ de feu
durent plusieurs images — exiger une confirmation sur N images est un gain de
fiabilité sans coût de calcul supplémentaire.

**5.2 — 🟠 Stabiliser l'identité des personnes (bug réel identifié).**
L'association EPI ↔ personne se fait par index de position dans une liste
recalculée à chaque image : si l'ordre des détections change, l'historique de
lissage d'une personne est attribué à une autre — donc une violation peut être
imputée au mauvais individu. Le suivi (`track_id`) est déjà calculé par le module
général ; il faut l'utiliser comme clé d'historique.

**5.3 — 🟠 Politique de rétention des données sensibles.**
Le moteur produit des numéros de plaques (LPR) et peut enregistrer des images. Ce
qui est retenu, combien de temps, et ce qui part dans l'API doit être une décision
explicite et documentée — question contractuelle autant que technique.

## Priorité 6 — Reproductibilité et traçabilité de la livraison

**6.1 — 🟡 Empaquetage conteneurisé** avec versions figées. Un client ne doit pas
avoir à reconstruire un environnement Python à la main. Les deux fichiers
`requirements.txt` divergent déjà sur les versions de `torch` et `numpy`.

**6.2 — 🟡 Registre des modèles** : pour chaque `.pt` livré, tracer provenance
(données d'entraînement, notebook, date, métriques de validation) et version.
L'information existe, mais dispersée entre rapports, historique git et notebooks
Colab.

**6.3 — 🟡 Procédure de retour arrière** documentée et testée : remplacer un
modèle et revenir en arrière doit être une opération sûre et répétable.

## Priorité 7 — Continuer à améliorer la précision (Colab)

**7.1 — 🟡 `smoke` à 39.1%** reste le point faible. Le complément pyro-sdis a
apporté +11.5 points, mais ses images (panaches de feu de forêt vus de loin depuis
une tour de guet) sont éloignées d'une scène de caméra de site. Une source de
fumée plus proche du domaine cible donnerait davantage.

**7.2 — 🔵 Entraînement de `best.pt` plus long.** Le fine-tuning gilet s'est arrêté
à 40 époques. Les autres classes n'ont pas été ré-entraînées sur des annotations
propres, alors que le diagnostic v2 a montré que **toutes** les classes
souffraient de la contamination du jeu de données par lots.

**7.3 — 🔵 Retrait de `best_gloves.pt`** de la cascade. Mesuré : +15% de FPS. Son
seul apport propre est `safety_shoe`, classe pour laquelle aucun jeu de données
local n'existe — si les chaussures de sécurité ne font pas partie du référentiel
de conformité du client, ce modèle est purement redondant.

**7.4 — Robustesse aux conditions réelles** (faible luminosité, contre-jour,
intempéries, occlusion partielle). Jamais évaluée, alors que la cible est une
caméra extérieure. Ce point se scinde en deux, et **seule la première moitié est
prioritaire** :

- **7.4a — 🟠 Mesurer** les modèles actuels en conditions dégradées. Ce n'est pas
  un entraînement : on applique les modèles existants à des images sombres, à
  contre-jour ou partiellement occultées, et on regarde de combien l'AP chute.
  À faire **avant** de promettre quoi que ce soit au client — c'est le type
  d'écart qui, sinon, ne se découvre qu'en exploitation. Réalisable sur cette
  machine, sans Colab.
- **7.4b — 🔵 Ré-entraîner avec augmentation ciblée**, *uniquement si* 7.4a révèle
  une chute significative. Si les modèles tiennent, ce point disparaît.

## Priorité 8 — Serveur et GPU (dernière étape)

🔵 Volontairement repoussé. Le plafond CPU actuel est de ~3.8 FPS. Les leviers
(GPU dédié, matériel embarqué type Jetson, quantification, TensorRT) ne seront
pertinents qu'une fois la cible de déploiement connue et les priorités 1 à 5
traitées. À noter : ONNX a été mesuré en v2 et n'apporte **aucun gain** sur cette
machine — inutile d'y revenir sans nouvelle mesure sur le matériel cible.

---

## Ordre d'exécution recommandé

**Le ré-entraînement est volontairement repoussé en dernière phase.** Les modèles
actuels sont déjà bons (84–98% AP@50 sauf `smoke`) ; ce qui empêche la livraison
n'est pas leur précision mais l'absence d'API, de tests et de robustesse. Ajouter
des époques d'entraînement maintenant n'améliorerait pas le produit là où il est
faible. Les étapes 1 à 4 se font donc **sans aucun Colab**.

**Étape 1 — les 4 bloquants** (1.1, 1.2, 1.3, 3.1). Ce sont eux, et eux seuls, qui
séparent un prototype d'un produit livrable.

**Étape 2 — les critiques liés à la justesse et à la non-régression** (2.1 à 2.3,
5.1, 5.2, 4.1). Ils déterminent si les détections livrées sont dignes de confiance
dans la durée.

**Étape 3 — les critiques restants** (1.4, 3.2, 3.3, 5.3, **7.4a**). Ils
déterminent si un incident chez le client sera détecté et diagnosticable. 7.4a est
inclus ici parce que c'est une *mesure*, pas un entraînement : elle dit si les
modèles tiennent en conditions dégradées, et donc si la phase 5 devra traiter ce
sujet ou non.

**Étape 4 — les importants** (2.4, 3.4, 4.2 à 4.4, 6.1 à 6.3), en accompagnement
de la première livraison réelle.

**Étape 5 — phase d'entraînement, en dernier** (7.1, 7.2, 7.3, **7.4b** si 7.4a l'a
justifié, puis 8). C'est la seule phase qui demande des créneaux Colab. Elle
bénéficie d'arriver après les autres : les tests de non-régression de l'étape 2
seront alors en place, donc chaque nouveau modèle sera validé automatiquement
contre les métriques de référence avant remplacement — ce qui n'était pas le cas
en v2.

## Ce qui bloque, et ce qui ne bloque pas

Contrairement à v2, **aucune priorité ici n'est bloquée par l'absence de GPU**.
Les priorités 1 à 6 sont entièrement réalisables sur cette machine. Seule la
priorité 7 demande des créneaux Colab, selon la procédure déjà éprouvée.

Le seul point qui demande une **décision extérieure** est le choix du protocole
d'API (1.2), à arrêter avec l'équipe de la plateforme consommatrice. Tout le
reste peut démarrer immédiatement.
