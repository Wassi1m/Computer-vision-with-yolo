# Plan d'amélioration v6 — passage en production du moteur de détection

Date : 2026-08-13
Fait suite à `v5_plan_amelioration.md`, qu'il ne remplace pas : v5 traite de la
**qualité des détections**, v6 de la **qualité du service**. Les deux sont
nécessaires et se mènent en parallèle, mais ils répondent à deux questions
différentes — « le moteur voit-il juste ? » d'un côté, « un intégrateur peut-il
brancher ce moteur en production ? » de l'autre.

## Le chiffre qui cadre tout ce plan

Le test d'endurance de 30 minutes (`reports/v3_results/endurance.json`) n'avait
été lu jusqu'ici que pour ce qu'il devait prouver : pas de plantage, pas de
fuite évidente. Il contient pourtant la mesure la plus importante du projet.

| Mesure sur 30 min | Valeur |
|---|---|
| Images traitées | 5 994 |
| **Évènements émis** | **2 869** |
| Cadence moyenne | 5,31 img/s |

**Un évènement toutes les 2,1 images.** Soit, en régime continu :

> **≈ 5 735 évènements par heure, ≈ 137 600 par jour et par caméra.**

Ce n'est plus une extrapolation à partir d'un taux de fausse alarme théorique,
comme dans v5 : c'est le moteur réel, sur une vidéo réelle, mesuré. Aucune
plateforme aval ne consomme cela. **Tant que ce nombre n'a pas chuté de trois
ordres de grandeur, le moteur n'est pas intégrable**, quelle que soit la
qualité de ses modèles.

Et il faut le dire clairement : ce déluge ne vient **pas** d'un défaut des
modèles. Le détecteur de fumée repère 96,7 % des scènes qui en contiennent. Le
défaut est que le moteur émet un évènement par *détection d'image* au lieu d'un
évènement par *fait réel*.

## Périmètre

Inchangé depuis v3 : ce dépôt est le **moteur de détection**. Il ne notifie
personne, n'affiche rien, ne stocke pas d'historique métier. Tout ce qui suit
vise à en faire un composant qu'une plateforme tierce peut brancher, exploiter
et superviser — pas un produit fini.

C'est précisément pourquoi ce plan existe séparément : un moteur peut être
excellent en détection et rester inintégrable, faute de contrat stable, de
livraison fiable et d'identité de caméra.

## Garde-fous

Aux garde-fous de v4 et v5 s'ajoutent trois règles propres à cette étape.

1. **Le schéma d'évènement se fige avant l'intégration, pas après.** Chaque
   champ manquant ajouté après le branchement de la plateforme est une rupture
   de compatibilité chez quelqu'un d'autre. C'est la raison d'être de la
   priorité 1.
2. **Rien n'est « prêt pour la production » sans une mesure de 24 h.** Les
   défauts qui comptent en exploitation — dérive mémoire, descripteurs de
   fichiers, reconnexions RTSP, rotation de logs — sont invisibles sur 30
   minutes. Le test d'endurance actuel est un test de démarrage, pas un test
   d'endurance.
3. **Toute donnée personnelle est traitée avant la mise en service, pas
   après.** Un numéro de plaque en clair dans un webhook est une donnée
   personnelle qui traverse le réseau ; la corriger après coup implique de
   purger l'aval.

---

## Classification

| Niveau | Signification |
|---|---|
| 🔴 **Bloquant** | La mise en production est impossible ou irresponsable sans cela |
| 🟠 **Critique** | Le moteur fonctionne, mais l'intégrateur subit un défaut majeur |
| 🟡 **Important** | Qualité d'exploitation, à faire avant la montée en charge |
| 🔵 **Souhaitable** | Confort, dette technique |

### Vue d'ensemble

| # | Point | Nature | Niveau |
|---|---|---|---|
| 1.1 | `camera_id` absent du schéma d'évènement | Contrat | 🔴 |
| 1.2 | Évènement = détection d'image, pas fait réel (137 600/jour) | Contrat | 🔴 |
| 1.3 | Identité stable d'évènement (`event_id`, corrélation) | Contrat | 🔴 |
| 2.1 | Webhook sans livraison garantie : évènements perdus en silence | Fiabilité | 🔴 |
| 2.2 | Aucune mesure au-delà de 30 min ; pente mémoire non élucidée | Fiabilité | 🔴 |
| 2.3 | Cible de déploiement CPU/GPU toujours non tranchée (v4 §1.1) | Décision | 🔴 |
| 3.1 | Plaques d'immatriculation en clair, sans option de pseudonymisation | Conformité | 🔴 |
| 3.2 | Aucune authentification des appels sortants | Sécurité | 🟠 |
| 4.1 | Corpus vidéo de validation inexistant (v5 §5.2) | Mesure | 🔴 |
| 4.2 | Robustesse nocturne mesurée sur dégradation simulée uniquement | Mesure | 🟠 |
| 4.3 | Scénarios ligne / objet abandonné / foule jamais mesurés (v4 §3.1) | Mesure | 🟠 |
| 5.1 | Pas de supervision exploitable (métriques, journalisation structurée) | Exploitation | 🟠 |
| 5.2 | Configuration éparpillée, pas de validation au démarrage | Exploitation | 🟡 |
| 5.3 | Image Docker non validée sur la cible retenue | Exploitation | 🟡 |

---

## Priorité 1 — Figer le contrat d'évènement (🔴)

**C'est le seul lot qui doit être terminé avant que la plateforme aval ne se
branche.** Tout le reste peut évoluer ensuite sans rupture ; le schéma, non.

**1.1 — `camera_id` dans chaque évènement.**
Le champ n'existe pas (`Evenement` dans `improvements/unified_surveillance.py`
porte `t`, `frame`, `source`, `type`, `libelle`, `conf`, `box`, `extra`), et
`docs/contrat_api.md` §8 le reconnaît : « une seule caméra par processus […] le
champ identifiant de caméra n'existe pas encore ».

Un site réel a dix à cent caméras. Sans cet identifiant, la plateforme reçoit un
flux d'évènements dont elle ne peut pas dire d'où ils viennent — et le corriger
après intégration oblige tous les consommateurs à s'adapter.

À ajouter : `camera_id` (fourni par configuration, jamais deviné) et
`site_id` optionnel. Coût : quelques lignes. Coût si fait après : une migration
chez l'intégrateur.

**1.2 — Un évènement par fait, pas par image.**
C'est l'application directe de la couche de qualification de v5, vue depuis le
contrat. Aujourd'hui chaque image qui contient de la fumée produit un évènement
`feu` ; il en faut **un seul** pour l'épisode, qui naît, dure et se termine.

Le schéma doit donc porter :
- un **état** : `suspicion` / `probable` / `confirmé` / `terminé` ;
- un **début** et une **durée** ;
- les **éléments de preuve** (nombre d'images confirmantes, taux de croissance,
  corroborations), pour que la plateforme applique *sa* politique.

Rétrocompatibilité : conserver le format actuel derrière un drapeau, en le
faisant correspondre à `confirmé`.

**1.3 — Identité et corrélation.**
Un évènement qui dure doit être **identifiable** entre ses mises à jour :
`event_id` stable, plus un `sequence` incrémental. Sans cela, la plateforme ne
peut ni dédupliquer après une reprise réseau, ni relier une fin à son début —
et un rejeu de webhook crée un doublon indiscernable.

## Priorité 2 — Fiabilité d'exploitation (🔴)

**2.1 — Livraison garantie des évènements.**
`SortieWebhook` est un envoi « au mieux » : `self.echecs += 1` sur exception, et
l'évènement est perdu. Le commentaire du code assume ce choix pour ne pas
ralentir le moteur — le raisonnement est juste, la conséquence ne l'est pas :
**un départ de feu détecté pendant un redémarrage de la plateforme disparaît
sans trace.**

Ce qu'il faut, sans renoncer au découplage :
- une **file persistante sur disque** (un simple JSONL avec position de lecture
  suffit) ;
- des **réessais avec temporisation croissante** ;
- une **borne** : au-delà, on abandonne mais on le **signale**, plutôt que de
  perdre en silence ;
- le compteur d'échecs **exposé sur `/health`** — il existe déjà, il n'est pas
  publié.

**2.2 — Endurance réelle sur 24 h.**
Le test actuel dure 30 minutes. Sur cette fenêtre, la mémoire va de 1 065 à
1 148 Mo, avec une pente de régression d'environ **86 Mo/h**. Extrapolée, cette
pente donnerait ~2 Go/jour — mais **30 minutes ne permettent pas de conclure** :
l'oscillation observée est du même ordre que la tendance mesurée.

C'est exactement pourquoi le test long est nécessaire. Il faut trancher entre
« oscillation normale du ramasse-miettes » et « fuite qui tue le processus au
bout de trois jours », et seule une mesure de 24 h le peut. À surveiller aussi :
descripteurs de fichiers, threads du `ThreadPoolExecutor`, comportement sur
coupure RTSP prolongée, taille des journaux.

**2.3 — Trancher CPU ou GPU.**
Question ouverte depuis v4 §1.1, et devenue plus urgente : la couche de
qualification de v5 a besoin de savoir combien d'images par seconde elle peut
espérer, puisque toutes ses règles sont temporelles. Confirmer un panache sur
une fenêtre de 4 images n'a pas le même sens à 5 img/s qu'à 25.

Mesure actuelle : **5,31 img/s** sur CPU. La décision conditionne 1.2, 6.1 de v5,
et le dimensionnement de toute l'intégration. Elle ne peut pas être prise dans ce
dépôt — elle appartient au client et à la plateforme.

## Priorité 3 — Conformité et sécurité (🔴 / 🟠)

**3.1 — Données personnelles : les plaques.**
`docs/contrat_api.md` §8 signale que les évènements `plaque` contiennent les
immatriculations en clair, et §9 laisse la question ouverte. **Une plaque est une
donnée personnelle** ; elle transite aujourd'hui en clair dans un POST HTTP.

Le moteur ne peut pas décider de la politique de rétention — c'est le rôle de la
plateforme — mais il doit **offrir le choix** :
- plaque en clair (défaut actuel, à conserver mais explicite) ;
- empreinte non réversible (`plaque_hash`), pour reconnaître un véhicule déjà vu
  sans jamais transmettre le numéro ;
- présence seule, sans lecture.

Un mode configurable, décidé au déploiement. Sans lui, l'intégrateur n'a aucune
option de conformité.

**3.2 — Authentifier les appels sortants.** 🟠
Le webhook part sans en-tête d'authentification (question ouverte du contrat
§9.3). N'importe qui connaissant l'URL peut injecter de faux évènements dans la
plateforme. Un jeton porteur, ou une signature HMAC de la charge utile, suffit et
coûte peu.

## Priorité 4 — Débloquer la mesure (🔴 / 🟠)

**4.1 — Corpus vidéo annoté.** 🔴
Reporté depuis v5 §5.2, et c'est **la dépendance la plus lente du projet**.
Toutes les règles de v5 et le point 1.2 de ce plan sont temporels : ils sont
littéralement immesurables sur des images fixes. Il faut des séquences annotées
au niveau de l'évènement (« départ de feu de t=12 s à t=90 s »), jour et nuit,
avec et sans évènement.

Quelques dizaines de séquences suffisent pour commencer. **À lancer avant tout le
reste**, puisque c'est le seul lot qui ne dépend que de l'extérieur.

**4.2 — Confirmer la nuit sur de vraies images.** 🟠
Les 36,1 % de détection nocturne et le rejet du prétraitement CLAHE reposent sur
une dégradation **simulée** par assombrissement d'images de jour. Une vraie
image nocturne a un bruit de capteur et un éclairage artificiel ponctuel qu'une
simulation ne reproduit pas. Les conclusions restent les meilleures disponibles,
mais elles portent une réserve à lever avec le corpus 4.1.

**4.3 — Scénarios jamais mesurés.** 🟠
Franchissement de ligne, objet abandonné, comptage de foule : implémentés,
jamais évalués sur un jeu annoté (v4 §3.1). Le moteur les expose comme les
autres, avec la même apparence de fiabilité, sans qu'aucun chiffre ne l'étaye.
Au minimum, **le déclarer dans le contrat d'API** tant que la mesure n'existe
pas — un intégrateur ne doit pas croire ces scénarios validés au même titre que
l'EPI ou le feu.

## Priorité 5 — Exploitabilité (🟠 / 🟡)

**5.1 — Supervision.** 🟠
`/health` existe et expose l'état courant. Il manque ce qui permet de diagnostiquer
à distance : cadence instantanée, latence par module, taille de la file de
sortie, compteur d'échecs de livraison, nombre de reconnexions, mémoire.
Une exposition au format Prometheus est le standard attendu par les
plateformes ; à défaut, un JSON stable suffit.

Journalisation structurée (une ligne JSON par évènement de journal, avec
`camera_id`) plutôt que du texte libre : sans cela, aucune agrégation possible
sur un parc de caméras.

**5.2 — Configuration validée au démarrage.** 🟡
Les réglages se répartissent entre `config.py`, des variables d'environnement et
des arguments de ligne de commande. Un fichier de configuration unique, validé
au démarrage avec un message d'erreur explicite, évite la classe de pannes la
plus coûteuse en production : un moteur qui démarre « normalement » avec un
mauvais seuil ou un modèle absent, et ne signale rien.

**5.3 — Valider l'image Docker sur la cible.** 🟡
Un `Dockerfile` existe. Il n'a pas été validé sur la cible retenue en 2.3 — image
CUDA si GPU, image CPU sinon — ni soumis au test d'endurance de 2.2 depuis le
conteneur. Une image qui fonctionne en développement et échoue en production sur
un pilote GPU est un classique.

---

## Ordre d'exécution recommandé

**Étape 0 — lancer la collecte du corpus vidéo (4.1), aujourd'hui.** C'est la
seule dépendance externe et la plus lente. Tout le reste avance en parallèle.

**Étape 1 — figer le contrat (1.1, 1.3), puis l'authentification (3.2) et le
mode plaque (3.1).** Purement local, rapide, et **impératif avant que la
plateforme ne se branche**. Un jour de travail évite une migration.

**Étape 2 — la livraison garantie (2.1) et le test 24 h (2.2).** Le test tourne
seul pendant qu'on travaille sur autre chose : à lancer tôt.

**Étape 3 — le socle de qualification de v5 (1.1, 1.2, 1.3 de v5) qui alimente
le point 1.2 d'ici.** C'est le lot qui fait tomber les 137 600 évènements
quotidiens, et il n'a besoin d'aucun GPU.

**Étape 4 — trancher CPU/GPU (2.3) avec le client**, puis régler les règles
temporelles sur le corpus arrivé entre-temps.

**Étape 5 — supervision et empaquetage (5.1, 5.2, 5.3)**, une fois le
comportement stabilisé.

## Ce que ce plan ne fait pas

**Il ne demande aucun entraînement.** C'est délibéré, et l'épisode du 12 août le
justifie : un entraînement Kaggle de 2 h 54 a produit un modèle *moins bon sur
les deux axes* (détection 96,7 % → 91,6 %, fausse alarme 2,0 % → 5,3 %), parce
que `lr0` était resté au défaut d'un entraînement depuis zéro. Le modèle en place
n'a pas été remplacé, à raison.

La leçon dépasse ce réglage : **les modèles ne sont pas le facteur limitant.**
Un détecteur qui voit 96,7 % des scènes de fumée n'a pas besoin d'être meilleur ;
il a besoin d'un moteur qui sache quoi faire de ce qu'il voit.

**Il ne construit toujours ni alerte, ni tableau de bord, ni interface.** Le
périmètre reste le moteur de détection et son contrat.

**Il ne traite pas la performance brute** (v4 §1.2, §1.3, v5 §6.1), qui reste
suspendue à la décision CPU/GPU de 2.3.

## Ce qui reste vrai des plans précédents

- **v4** : la décision de déploiement (§1.1), la cadence cible par scénario
  (§1.2) et l'allègement de `best.pt` (§1.3) restent entiers.
- **v5** : la couche de qualification reste **le** chantier technique majeur.
  Ce plan v6 en est le complément côté service, pas le remplaçant.
- **Le prétraitement nocturne (v5 §3.2) reste rejeté**, mesures à l'appui. Ne
  pas le réessayer sans raison nouvelle.
