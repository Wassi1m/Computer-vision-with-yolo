# Plan d'amélioration v7 — ce qui ne dépend plus du calcul

Date : 2026-08-16
Fait suite à `v6_plan_amelioration.md`. Ce plan est court parce qu'il ne
contient que ce qui **ne peut pas être obtenu par un entraînement de plus**.

## Le constat qui clôt la phase d'entraînement

Quatre ré-entraînements en cinq jours ont porté le modèle EPI de **0,0427 à
0,7689 de mAP@50** — douze classes ressuscitées, le casque passé de 32 % à 65 %
de détection, les cônes de 48 % à 98 %. Le cinquième, en cours, vise les deux
dernières classes faibles et rares.

Après lui, **le calcul n'apportera plus rien** : les deux classes qui resteront
en dessous ne sont pas limitées par le nombre d'exemples.

| Classe | Détection | Exemples | Pourquoi elle plafonne |
|---|---|---|---|
| `Hardhat` | 65 % | **28 996** | la mieux pourvue du jeu, et la plus faible |
| `NO-Hardhat` | 72 % | 9 705 | autant que `Safety Cone`, qui sort à 98 % |

Passer `Hardhat` de 2 504 à 10 004 exemples l'a fait progresser de 32 % à 65 %.
Les 18 992 restants du même jeu ne donneront pas autant : ils viennent des mêmes
chantiers, sous les mêmes angles. **Ce qui manque n'est pas la quantité mais la
diversité.**

Ce plan traite donc les deux seules dépendances qui restent, et toutes deux
sortent du dépôt.

---

## 1 — 🔴 Le corpus vidéo de votre site

**C'est le blocage le plus important du projet**, et il l'est depuis le plan v5.

Toute la couche de qualification décide sur une **séquence**, pas sur une image.
Elle est donc *littéralement immesurable* sur des photos. Trois fonctions déjà
livrées n'ont jamais vu une seule vraie séquence :

| Fonction | État | Ce qu'on ignore |
|---|---|---|
| Machine à états des évènements | livrée, 12 tests | le bon réglage de `--seuil-probable` et `--seuil-confirme` |
| Objet abandonné | livrée, 7 tests | son taux de fausse alarme réel |
| Densité de foule | livrée, 12 tests | idem, et la justesse de la calibration en conditions réelles |

La logique est couverte par 31 tests unitaires. **Leur comportement sur le
terrain est inconnu.** Un réglage trop strict rate des évènements, trop laxiste
noie l'aval — et rien ne permet aujourd'hui de trancher.

### Ce qu'il faut, précisément

Quelques **dizaines de séquences**, pas des milliers :

- de 30 secondes à 5 minutes chacune ;
- issues de **vos caméras**, à leur position et leur angle définitifs ;
- annotées **au niveau de l'évènement**, pas de l'image : « départ de feu de
  t=12 s à t=90 s », « sac déposé à t=45 s », « ouvrier sans casque de t=20 s à
  t=35 s » — un simple fichier texte suffit ;
- couvrant **jour et nuit**, et surtout des séquences **sans aucun évènement** :
  ce sont elles qui mesurent le taux de fausse alarme, et elles sont les plus
  faciles à produire.

Une caméra qui filme une journée normale fournit déjà l'essentiel du corpus
négatif.

### Pourquoi c'est le point de départ

Sans ces vidéos, chaque réglage de la couche de qualification est un pari. Avec
elles, il devient une mesure. C'est la différence entre un moteur qu'on livre en
espérant et un moteur qu'on livre en sachant.

---

## 2 — 🟠 Enrichir le casque par des images d'autres chantiers

Seule voie restante pour `Hardhat` et `NO-Hardhat`. Plusieurs jeux publics
existent, tous librement téléchargeables au format YOLO :

| Jeu | Volume | Lien |
|---|---|---|
| **Hard Hat Universe** | 7 036 images | [universe.roboflow.com/universe-datasets/hard-hat-universe-0dy7t](https://universe.roboflow.com/universe-datasets/hard-hat-universe-0dy7t) |
| **Construction Helmet Detection** | 8 083 images | [universe.roboflow.com/construction-helmet/construction-helmet-detection](https://universe.roboflow.com/construction-helmet/construction-helmet-detection) |
| **Safety Helmets** | 6 329 images | [universe.roboflow.com/construction-helmets/safety_helmets](https://universe.roboflow.com/construction-helmets/safety_helmets) |
| Recherche par classe | — | [universe.roboflow.com/search?q=class:hardhat](https://universe.roboflow.com/search?q=class%3Ahardhat) |

Ces jeux ont souvent une classe `head` ou `person` sans casque, qui correspond
directement à `NO-Hardhat` — la classe la plus critique et la plus faible.

### ⚠️ Le piège, et il est mortel

**Ces jeux annotent le casque, et rien d'autre.** Les ajouter tels quels
apprendrait au modèle que la zone du gilet, des gants et du masque est du
**fond** sur ces images.

C'est **exactement le mécanisme qui a détruit ce modèle** : le fine-tuning de
juillet avait tourné sur un jeu ne contenant que du gilet, et le réseau avait
effacé douze classes sur quatorze. Il a fallu cinq jours et quatre
ré-entraînements pour les récupérer.

Trois précautions, non négociables :

1. **Remapper les classes explicitement** avant toute fusion, comme le font déjà
   `improvements/p4_merge_smoke_dataset.py` et `p5_merge_fall_dataset.py`. Une
   table écrite, pas une correspondance devinée sur les noms.
2. **Limiter la proportion d'images mono-concept.** Un tiers du jeu final au
   maximum, pour que le réseau continue de voir des images complètes.
3. **Mesurer les quatorze classes après**, sur `ppe_dataset/test` **et** en
   détection de scène. La mAP seule ne verrait pas l'effondrement : elle est
   restée à 0,88 pendant que douze classes étaient à zéro.

### Espérance réaliste

`Hardhat` de 65 % à 75-80 %, `NO-Hardhat` de 72 % à 80 %. Un gain net, mais qui
ne justifie pas de prendre le risque avant que le point 1 soit traité.

---

## 3 — 🟡 Enrichir `ppe_complement.pt` (gants, lunettes, chaussures)

Même mécanisme que le casque (§2), sur le second modèle de la cascade. Trois
classes plafonnent, et contrairement au casque, **aucun jeu local** ne peut
les enrichir : `ppe_dataset` n'a pas de `safety_shoe`, et ses images `Goggles`
/ `Gloves` sont déjà utilisées à plein par `ppe_detector.pt`.

| Classe | Détection mesurée | Blocage |
|---|---|---|
| `Gloves` | 33 % | aucun jeu local dédié à ce modèle |
| `goggles` | 0 % sur l'échantillon (le modèle détecte la classe ailleurs, cf. `models_calsse.txt`) | idem |
| `safety_shoe` | non mesurable | `ppe_dataset` n'annote aucune chaussure |

### Jeux candidats (Roboflow Universe, format YOLO)

| Classe visée | Jeu | Volume | Lien |
|---|---|---|---|
| Les trois à la fois | **PPEs** — glove/no_glove, goggles/no_goggles, shoes/no_shoes, + helmet/mask/suit | 24 924 images | [universe.roboflow.com/personal-protective-equipment/ppes-kaxsi](https://universe.roboflow.com/personal-protective-equipment/ppes-kaxsi) |
| Gants | **Safety Gloves** — Gloves / NO-Gloves | ~3 373 images (10 459 en version augmentée ×3) | [universe.roboflow.com/roboflow-universe-projects/safety-gloves-xbnf8](https://universe.roboflow.com/roboflow-universe-projects/safety-gloves-xbnf8) |
| Gants + chaussures | **Construction PPE** — Gloves, Safety Boot, Helmet, Safety Vest + négatifs | 8 845 images | [universe.roboflow.com/skcet-g4h72/construction-ppe-rdhzo](https://universe.roboflow.com/skcet-g4h72/construction-ppe-rdhzo) |
| Lunettes | **Safety Goggles v1** — head, Mask, Goggles, eyes_with_goggles, eyes_without_goggles | 1 288 images | [universe.roboflow.com/database-sjrvw/safety-goggles](https://universe.roboflow.com/database-sjrvw/safety-goggles) |
| Lunettes | **Safety Goggles – PPE** — Coverall, Face_Shield, Gloves, Goggles, Mask (déjà multiclasse) | 411 images | [universe.roboflow.com/database-sjrvw/safety-goggles---ppe](https://universe.roboflow.com/database-sjrvw/safety-goggles---ppe) |
| Chaussures | **Safety Shoes dataset** — classes `person`, `safety_shoe` | 1 089 images | [universe.roboflow.com/ahmed-alqulayti/safety-shoes-dataset](https://universe.roboflow.com/ahmed-alqulayti/safety-shoes-dataset) |

**PPEs** (24 924 images) couvre les trois manques d'un coup et est déjà
multiclasse — point de départ naturel. **Safety Shoes dataset** a une
particularité utile : sa classe s'appelle **exactement** `safety_shoe`, comme
`ppe_complement.pt` — aucun nom à deviner sur ce point-là, seule la table de
correspondance formelle reste à écrire pour les autres jeux.

Roboflow bloque la récupération automatique des pages (403) : les volumes et
classes ci-dessus viennent de la recherche, pas d'une lecture directe des
pages. **Vérifier chaque jeu à la main avant usage** — contenu réel, split,
et licence (Roboflow n'impose pas une licence unique à tous ses jeux publics).

### Le même piège qu'au §2, sur un autre modèle

`ppe_complement.pt` n'a que 6 classes et aucune version `NO-`. Le fine-tuner
sur un jeu mono-concept (que des gants, par exemple) reproduirait le
mécanisme qui a effacé 12 classes de `ppe_detector.pt` en juillet — ici sur
les 5 autres classes de ce modèle-ci. Mêmes précautions non négociables
qu'au §2 : remappage explicite (étendre `p2_table_correspondance_epi.py`),
plafond d'un tiers d'images mono-concept, mesure des 6 classes après --
pas seulement celle visée.

Point ouvert, non tranché ici : plusieurs jeux ci-dessus apportent des
classes négatives (`no_glove`, `no_goggles`, `no_shoes`) que
`ppe_complement.pt` n'a jamais eues. Les intégrer changerait son rôle : il
pourrait alors signaler une infraction, pas seulement confirmer un
équipement présent. C'est une décision de conception à trancher avant
l'entraînement, pas un simple ajout de données.

### ⚠️ Remarque : fusion des jeux puis entraînement en un seul passage

Décision prise : les jeux ci-dessus seront téléchargés puis **fusionnés en un
seul jeu**, sur lequel **un seul entraînement** sera lancé -- pas de campagne
itérative comme les quatre ré-entraînements qui ont sauvé `ppe_detector.pt`
en cinq jours.

Cela change ce qui doit être vérifié, et quand : sans itération pour rattraper
une erreur, tout ce qui est normalement corrigé *après coup* doit être
correct *avant* de lancer le run. Concrètement, avant le seul entraînement :

- la **table de correspondance** (remappage des noms de classe entre les
  jeux, extension de `p2_table_correspondance_epi.py`) doit être écrite et
  relue, pas improvisée pendant la fusion ;
- le **plafond d'un tiers d'images mono-concept** doit être respecté dans le
  jeu fusionné final, mesuré, pas supposé -- `PPEs` et `Construction PPE`
  sont déjà multiclasses et abaissent ce risque, mais le mélange avec
  `Safety Gloves` ou `Safety Shoes dataset` (mono-concept) peut le faire
  remonter selon les proportions choisies ;
- la **décision sur les classes négatives** (`no_glove`, `no_goggles`,
  `no_shoes`) doit être prise avant la fusion, pas découverte dedans -- les
  garder change la taxonomie du modèle, les exclure change le contenu du jeu.

Si le résultat est mauvais sur une classe précise, il n'y aura pas de second
essai ciblé : la seule option restante sera de refaire toute la fusion. La
mesure des 6 classes après entraînement (§ ci-dessus) n'est donc pas une
formalité de clôture ici -- c'est le seul moment où l'erreur, s'il y en a
une, sera visible.

### Stratégie retenue pour protéger les 5 autres classes

Six mesures concrètes, dans l'ordre où elles doivent être prises -- toutes
avant le run unique, sauf la dernière :

**0. Référence AVANT, figée** — fait le 2026-08-16, avant tout téléchargement :
`reports/v3_results/ppe_complement_avant.json`, produit par
`tests/mesure_scene_ppe_complement.py --sortie ...`. Seule mesure
comparable après coup, sur les 5 classes mesurables :

| Classe | Détection (avant) |
|---|---|
| `Gloves` | 13 % |
| `Vest` | 80 % |
| `goggles` | 0 % |
| `helmet` | 47 % |
| `mask` | 67 % |
| `safety_shoe` | non mesurable (aucun équivalent local) |

**1. Construire le jeu fusionné pour minimiser le piège du fond.**
`PPEs` et `Construction PPE` (déjà multiclasses) forment le socle : chaque
image y annote plusieurs des 6 classes ensemble, donc peu de zones que le
modèle apprendrait à tort comme « fond ». Les jeux mono-concept
(`Safety Gloves`, `Safety Goggles`, `Safety Shoes dataset`) restent plafonnés
à un tiers du total, comme déjà décidé plus haut.

**2. Écrire la table de correspondance avant la fusion**, pas pendant --
étendre `p2_table_correspondance_epi.py` avec le mapping explicite de
chaque classe de chaque jeu source vers l'une des 6 classes de
`ppe_complement.pt`, ou vers « ignorée ». La décision sur les classes
négatives (`no_glove`, `no_goggles`, `no_shoes` -- cf. point ouvert
ci-dessus) fait partie de cette table, pas un ajout après coup.

**3. Fabriquer un jeu de rappel (rehearsal), puisqu'aucun jeu d'origine
n'existe localement pour ce modèle.** C'est la pièce qui manque le plus par
rapport à la réparation de `ppe_detector.pt` en juillet, qui s'est appuyée
sur `ppe_dataset` (un vrai sur-ensemble local). Ici, à défaut :
faire tourner `ppe_complement.pt` **actuel** sur un lot d'images neutres
(ni issues des jeux Roboflow téléchargés, ni sur-représentant une classe --
par exemple un sous-ensemble de `ppe_dataset/train` non utilisé pour la
mesure) à seuil de confiance élevé (0.5), et garder les détections comme
pseudo-labels sur les 6 classes. Injecter ce lot dans le jeu fusionné, en
proportion suffisante pour peser sur le gradient (au moins autant d'images
que la plus grosse source mono-concept retenue). C'est un substitut fabriqué
au sur-ensemble qui a sauvé l'autre modèle -- imparfait, mais c'est la seule
option en l'absence de données locales.

**4. Reprendre les hyperparamètres déjà validés sur ce projet**, pas en
redécouvrir de nouveaux sur un run qui ne pourra pas être recommencé :
`optimizer="SGD"` explicite (leçon du 12 août -- `lr0` seul est sans effet
sans lui), `lr0=0.001` (leçon du run EPI de fin juillet/début août).
Envisager de geler les premiers blocs du backbone : un fine-tuning partiel
limite par construction l'ampleur du remaniement des poids, donc le risque
d'oubli catastrophique -- au prix d'un gain potentiellement plus faible sur
les 3 classes ciblées. Peu d'épreuves, arrêt anticipé sur un critère qui
regarde les 6 classes, pas seulement la perte d'entraînement.

**5. Écrire le critère de rejet MAINTENANT, avant de voir un seul chiffre
du résultat.** Le 15 août, le critère de rejet du gilet avait été fixé puis
sciemment outrepassé -- une décision assumée, documentée, pas une
découverte a posteriori. Sur un run qui ne pourra pas être recommencé, la
même discipline s'impose : par exemple, rejeter le candidat si une des 5
classes non ciblées perd plus de 15 points de détection de scène par
rapport à `ppe_complement_avant.json` -- en gardant à l'esprit que `Gloves`
est déjà à 13 %, donc avec très peu de marge avant de tomber à zéro.

**6. Mesurer après, avec le même outil, sur les 6 classes** --
`python tests/mesure_scene_ppe_complement.py --sortie reports/v3_results/ppe_complement_apres.json`
-- et comparer classe par classe à `ppe_complement_avant.json`, pas se
limiter à vérifier que les 3 classes ciblées ont progressé.

---

## 4 — Ce qui reste du plan v6

Inchangé, et toujours à faire avant la mise en production :

| Action | Origine | Coût |
|---|---|---|
| Pseudonymisation des plaques | v6 §3.1 | ⚠️ **abandonnée** — l'utilisateur veut la lecture en clair |
| Profils jour / nuit branchés | v5 §3.1, §3.3 | 1 j |
| Référence de fond par caméra | v5 §1.1 | 2 j |
| Signature de croissance de la fumée | v5 §2.1 | 2 j |
| Chute : transition plutôt que posture | v5 §2.2 | 2 j |
| Plaque : vote multi-images | v5 §2.4 | 1 j |
| Supervision, configuration, Docker | v6 §5 | 3 j |
| Décision CPU / GPU | v4 §1.1 | décision client |

Les quatre premières lignes **dépendent du corpus vidéo** : ce sont des règles
temporelles, elles ne se règlent pas sans séquences.

---

## Ordre d'exécution

**1. Lancer la collecte vidéo aujourd'hui.** C'est long, ça ne dépend que de
vous, et ça débloque quatre chantiers.

**2. Pendant ce temps, finir le plan v6** — supervision, configuration, image
Docker. Trois jours, aucune dépendance externe.

**3. Régler la couche de qualification** dès les premières vidéos reçues.

**4. Enrichir le casque, les gants, les lunettes et les chaussures** en
dernier, quand tout le reste est stable et que le garde-fou de non-régression
peut détecter immédiatement une rechute — sur `ppe_detector.pt` (§2) comme sur
`ppe_complement.pt` (§3).

## Ce que ce plan ne fait pas

Il ne demande **aucun entraînement supplémentaire** sur les données actuelles.
Cette voie est épuisée, et l'insistance coûterait plus qu'elle ne rapporterait :
chaque rééquilibrage déplace la capacité du réseau d'une classe vers une autre.

Le modèle voit bien. Ce qui manque désormais tient en une phrase : **des images
de votre monde à vous.**
