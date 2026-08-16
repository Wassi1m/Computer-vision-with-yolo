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

## 3 — Ce qui reste du plan v6

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

**4. Enrichir le casque** en dernier, quand tout le reste est stable et que le
garde-fou de non-régression peut détecter immédiatement une rechute.

## Ce que ce plan ne fait pas

Il ne demande **aucun entraînement supplémentaire** sur les données actuelles.
Cette voie est épuisée, et l'insistance coûterait plus qu'elle ne rapporterait :
chaque rééquilibrage déplace la capacité du réseau d'une classe vers une autre.

Le modèle voit bien. Ce qui manque désormais tient en une phrase : **des images
de votre monde à vous.**
