# Plan v8 — Campagne d'entraînement EPI sur les jeux Roboflow

Date : 2026-08-17
Fait suite à `v7_plan_amelioration.md` §2 et §3, dont il **corrige les
hypothèses** : les six archives ont été téléchargées et inspectées, et trois
d'entre elles ne contiennent pas ce que le plan v7 annonçait.

Objectif : améliorer les classes EPI faibles **sans dégrader une seule classe
en place**. Le principe directeur est celui qui a fonctionné le 17 août pour
`masque_gilet.pt` (mAP 0.9098) — un modèle dédié branché en cascade — par
opposition au fine-tuning global qui a effacé douze classes en juillet.

---

## 1 — Ce que contiennent réellement les archives

Inventaire vérifié le 2026-08-17 (lecture des `data.yaml` dans les `.zip`) :

| Archive | Classes réelles | Images (train/val/test) | Licence |
|---|---|---|---|
| **Hard Hat Universe** v26 | `head`, `helmet`, `hi-viz helmet`, `hi-viz vest`, `person` | 7 037 (4 913 / 1 415 / 709) | **Domaine public** |
| **PPEs** v8 | `glove`, `goggles`, `helmet`, `mask`, `shoes`, `suit` + 6 négatifs | 24 927 (19 420 / 3 571 / 1 936) | CC BY 4.0 |
| **Safety Gloves** v5 | `Gloves`, `NO-Gloves` | 10 462 (9 449 / 677 / 336) | CC BY 4.0 |
| **Construction PPE** v1 | `hat`, `no hat`, `vest`, `no vest` | 1 127 (929 / 134 / 64) | CC BY 4.0 |
| **Safety Shoes Detection** v1 | `Safety-shoes`, `not_safety_shoe`, `safety_shoe` | 415 (363 / 35 / 17) | CC BY 4.0 |
| **Construction Helmet Detection** v2 | `head` **seul** | 7 038 (4 917 / 1 414 / 707) | CC BY 4.0 |

### Les trois écarts avec le plan v7

| Jeu | Annoncé en v7 | Réalité |
|---|---|---|
| Construction Helmet Detection | 8 083 img, casque | 7 038 img, **`head` seul** — aucune distinction casque/sans-casque |
| Construction PPE | 8 845 img, `Gloves` + `Safety Boot` | **1 127 img**, `hat`/`vest` — **ni gants ni chaussures** |
| Safety Shoes | `ahmed-alqulayti`, 1 089 img | autre dépôt, **415 img**, 3 classes pour un concept |

L'avertissement du plan v7 (« vérifier chaque jeu à la main ») était justifié :
les volumes et classes y venaient de la recherche, pas d'une lecture des pages,
que Roboflow bloque en HTTP 403.

---

## 1 bis — Résultats de l'audit (`p12_audit_sources_roboflow.py`, 2026-08-17)

Les trois vérifications bloquantes du §5 ont été exécutées **avant** toute copie
d'image. Deux d'entre elles ont changé la campagne.

| Question | Verdict |
|---|---|
| **A.** `head` = tête nue dans Hard Hat Universe ? | ✅ **Oui.** 3 boîtes sur 4 781 recouvrent un casque (IoU ≥ 0.5), soit 0,06 % — du bruit d'annotation, **filtré** par `p13` plutôt que d'écarter le jeu. Le mappage `head → NO-Hardhat` est sûr. |
| **B.** `Safety-shoes` vs `safety_shoe` ? | 🟡 Les deux sont peuplées (81 et 273 images) — sans objet, voir C. |
| **C.** Volume `shoes` dans PPEs ? | ❌ **21 images**, pas 19 420. |

### ⚠️ Le modèle chaussures : d'abord abandonné à tort, puis rétabli

`PPEs` déclare douze classes ; l'audit en trouve **quatre totalement vides** —
`helmet`, `mask`, `no_helmet`, `no_mask` — malgré la mention « allclasses » de
son intitulé. Le premier comptage a conclu que `shoes` tenait en 51 images et
que le modèle chaussures était mort.

**Ce comptage était faux : il ne portait que sur le split `train`.** Le `valid`
de PPEs contient 348 images `shoes` et 291 `no_shoes`, là où son `train` n'en
compte respectivement que 21 et 30. Totaux réels, tous splits confondus :

| Concept | Images |
|---|---|
| `safety_shoe` | **823** |
| `NO-safety_shoe` | **380** |

C'est petit, mais entraînable pour un modèle dédié à 2 classes partant de
`ppe_detector.pt`. Le modèle **`epi_chaussures.pt` est donc rétabli**, en
troisième kernel.

**Corollaire : les splits d'origine sont inutilisables.** Entraîner sur le
`train` de PPEs reviendrait à ne presque rien apprendre puis à valider sur
l'essentiel des données. `p13_jeux_roboflow.py` met donc toutes les sources en
commun et **redécoupe 80/20 lui-même**, stratifié sur la classe positive et à
graine fixe — 750 images d'entraînement, 188 de validation, 2 370 instances.

**Ce que cela coûte, et qu'il faut assumer :** il n'existe alors **aucun jeu de
test indépendant** pour ce concept — `ppe_dataset` n'annote aucune chaussure.
Le candidat se juge sur son split `val` **et** sur une contre-épreuve de faux
positifs, qui est ici le critère qui décide (§7).

### 🔄 `PPEs` est en réalité un jeu gants + lunettes

```
glove      4 977 img   |   goggles      7 341 img
no_glove   5 995 img   |   no_goggles   6 085 img
```

Avec `Safety Gloves`, cela fait **27 657 images d'entraînement** sur les gants
et lunettes, négatifs compris. Le second kernel change donc de cible :
`epi_chaussures.pt` devient **`epi_gants_lunettes.pt`**, ce qui préserve le
parallélisme prévu au §6.

### ⚠️ `construction_helmet_detection` est écarté

Sa classe unique `head` est présente sur la **totalité** de ses 4 916 images, et
aucune classe casque n'y permet de situer ce que `head` désigne. Contrairement à
Hard Hat Universe, la vérification A y est impossible : le jeu est inutilisable
en l'état.

---

## 2 — 🔴 Le constat qui réoriente la campagne

Le plan v7 §3 visait `Gloves`, `goggles` et `safety_shoe` parce que
`ppe_complement.pt` y est mauvais (13 %, 0 %, non mesurable). **Mais
`ppe_complement.pt` est redondant sur deux de ces trois classes.**

`ppe_detector.pt` couvre déjà les gants et les lunettes, et parmi ses meilleurs
scores :

| Classe | `ppe_detector.pt` | `ppe_complement.pt` |
|---|---|---|
| `Gloves` | **AP 0.932** | 13 % détecté |
| `NO-Gloves` | **AP 0.909** | *(aucune classe négative)* |
| `Goggles` | **AP 0.960** | 0 % détecté |
| `NO-Goggles` | **AP 0.961** | *(aucune classe négative)* |
| `safety_shoe` | *(absente)* | non mesurable |

C'est d'ailleurs ce que dit déjà `p2_table_correspondance_epi.py` : sur cinq de
ses six classes, `ppe_complement.pt` est un doublon qu'on écarte à la fusion.
**`safety_shoe` est son seul apport net.**

Conséquence directe, et elle économise plusieurs heures de GPU :

- **`Safety Gloves` (10 462 images) ne corrige aucun déficit mesuré.** Entraîner
  dessus pour « améliorer les gants » reviendrait à remplacer une classe à
  AP 0.93 par un modèle non éprouvé. Ce jeu passe en réserve.
- **Le seul vrai vide du parc est `safety_shoe`** — aucun modèle, aucun jeu
  local, aucune mesure.
- **Le seul vrai déficit mesuré est le casque** — `Hardhat` 65 %,
  `NO-Hardhat` 72 %.

La campagne se réduit donc à **deux modèles**, pas six jeux à fusionner.

---

## 3 — Architecture cible de la cascade

Aucun fichier `.pt` en place n'est écrasé. Deux **nouveaux** modèles s'ajoutent,
chacun prioritaire sur un lot de classes précis (`improvements/ppe_taxonomy.py`).

| Rang | Modèle | Classes servies | État |
|---|---|---|---|
| 1 | `masque_gilet.pt` | Mask, NO-Mask, Safety Vest, NO-Safety Vest | ✅ en place |
| 2 | **`epi_casque.pt`** | **Hardhat, NO-Hardhat** | 🆕 à entraîner |
| 3 | **`epi_chaussures.pt`** | **safety_shoe, NO-safety_shoe** | 🆕 à entraîner |
| 4 | `ppe_detector.pt` | les 8 autres + **filet de secours** | ✅ inchangé |
| — | `ppe_complement.pt` | — | ⚠️ devient inutile (voir §7) |

**Pourquoi cette forme protège les modèles en place.** Un modèle neuf ne peut
pas oublier : il n'a rien appris avant. `ppe_detector.pt` n'est pas ré-entraîné,
donc ses douze classes saines sont hors d'atteinte par construction. Si un
candidat échoue, on ne le branche pas — et la cascade retombe sur
`ppe_detector.pt`, exactement comme aujourd'hui.

---

## 4 — Tables de correspondance

Écrites **avant** la fusion, comme l'exige le plan v7. Toute classe non listée
est explicitement **ignorée**, jamais devinée sur son nom.

### `epi_casque.pt` — 2 classes

| Source | Classe source | → |
|---|---|---|
| Hard Hat Universe | `helmet`, `hi-viz helmet` | `Hardhat` |
| Hard Hat Universe | `head` | `NO-Hardhat` ⚠️ *à vérifier, cf. §5* |
| Hard Hat Universe | `hi-viz vest`, `person` | ignorées |
| Construction PPE | `hat` | `Hardhat` |
| Construction PPE | `no hat` | `NO-Hardhat` |
| Construction PPE | `vest`, `no vest` | ignorées |
| `ppe_dataset` (local) | `Hardhat` (3), `NO-Hardhat` (8) | idem |

Les trois sources annotent **les deux** classes : aucune image mono-concept,
donc le plafond d'⅓ du plan v7 est sans objet ici.

### `epi_chaussures.pt` — 2 classes

| Source | Classe source | → |
|---|---|---|
| PPEs | `shoes` | `safety_shoe` |
| PPEs | `no_shoes` | `NO-safety_shoe` |
| PPEs | les 10 autres | ignorées (servies ailleurs dans la cascade) |
| Safety Shoes Detection | `safety_shoe`, `Safety-shoes` | `safety_shoe` ⚠️ *doublon à confirmer, cf. §5* |
| Safety Shoes Detection | `not_safety_shoe` | `NO-safety_shoe` |

**Décision actée le 2026-08-17 : les classes négatives sont conservées.** Les
deux nouveaux modèles peuvent donc signaler une infraction, et pas seulement
confirmer un équipement présent — ce que `ppe_complement.pt` ne savait pas
faire. Contrepartie : `docs/contrat_api.md` gagne `NO-safety_shoe` et devra
être mis à jour avant la bascule.

---

## 5 — ⚠️ Deux vérifications bloquantes, avant toute copie d'image

Ces deux points peuvent invalider une table ci-dessus. Ils passent en premier.

**A. Que désigne `head` dans Hard Hat Universe ?** La convention usuelle veut
`head` = tête nue et `helmet` = tête casquée. Si ce jeu annote **toute** tête,
casquée comprise, alors `head → NO-Hardhat` empoisonnerait la classe la plus
critique du parc en lui apprenant que des ouvriers casqués sont des
infractions. Vérification : extraire les images portant à la fois `head` et
`helmet`, et en inspecter une vingtaine à l'œil. Si le doute persiste, le jeu
est écarté et il reste Construction PPE + `ppe_dataset`.

**B. `Safety-shoes` et `safety_shoe` sont-elles la même chose ?** Deux classes
distinctes pour un concept identique dans un jeu de 415 images sent l'artefact
d'annotation. Compter les instances de chacune ; si l'une est quasi vide, elle
est ignorée plutôt que fusionnée.

**C. Combien d'images de PPEs annotent réellement `shoes` ?** Le jeu compte
19 420 images d'entraînement, mais rien ne dit combien portent une chaussure.
C'est ce chiffre — et lui seul — qui décide si `epi_chaussures.pt` est viable.
En dessous de ~1 500 images, le modèle ne vaudra pas la session GPU et la
campagne se réduit au casque.

---

## 6 — Entraînement parallèle sur Kaggle

Kaggle autorise **deux sessions GPU simultanées** par compte (quota ~30 h/semaine).
Les deux modèles étant indépendants, ils tournent **en parallèle**, pas l'un
après l'autre.

| | Kernel 1 | Kernel 2 |
|---|---|---|
| Modèle | `epi_casque.pt` | `epi_chaussures.pt` |
| Classes | 2 | 2 |
| Images (est.) | ~13 000 | ~2 000 à 20 000 *(selon §5-C)* |
| Poids de départ | `ppe_detector.pt` | `ppe_detector.pt` |
| Budget `time=` | 5 h | 5 h |
| Durée attendue | 3 – 4 h | 2 – 5 h |

Les deux réutilisent l'ossature déjà éprouvée de `kaggle/entrainer_masque_gilet.py` :
reprise automatique sur coupure, `save_period=5`, `time=` en garde-fou avant la
limite des 12 h, **`optimizer="SGD"` nommé explicitement** et `lr0=0.001` — les
deux leçons qui ont coûté un run entier le 12 août.

### Transfert des jeux — la contrainte à respecter

> Un transfert de jeu de données ne doit jamais reposer sur une seule session
> réseau. *(Leçon de l'envoi `scp` de 774 Mo cassé après 98 minutes, GPU
> facturé à ne rien faire.)*

Trois mesures, dans cet ordre :

1. **Construire les jeux fusionnés en local d'abord.** On ne téléverse jamais
   les 2 Go d'archives brutes : après filtrage et remappage, `epi_casque` pèse
   ~400 Mo et `epi_chaussures` bien moins. Un échec de transfert ne coûte alors
   que quelques minutes.
2. **Un jeu Kaggle par modèle**, téléversés séparément
   (`kaggle datasets create -p <dossier> --dir-mode zip`). Deux transferts
   moyens échouent moins souvent qu'un gros, et un seul est à reprendre.
3. **Vérifier après téléversement** le nombre de fichiers côté Kaggle avant de
   lancer quoi que ce soit — un envoi tronqué qui passe inaperçu se paie en
   heures de GPU sur un jeu incomplet.

Le split `test` de `ppe_dataset` **n'est jamais téléversé** : il reste local et
intact pour juger les candidats sans qu'ils aient pu le voir — même règle que
`p10_sous_ensemble_epi.py` et `p11_jeu_masque_gilet.py`.

### 🔑 Bloquant : jeton Kaggle absent

`~/.kaggle/kaggle.json` n'existe pas sur cette machine, et aucune variable
`KAGGLE_*` n'est définie. Rien ne peut être téléversé ni lancé sans lui.

```
Kaggle > profil > Settings > API > Create New Token
mkdir -p ~/.kaggle && mv ~/Téléchargements/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

---

## 7 — Critères de rejet, écrits avant de voir un chiffre

Le 15 août, un critère de rejet fixé puis outrepassé a été assumé et documenté.
Ceux-ci sont figés **maintenant**, avant le premier résultat.

**Mesure de référence AVANT** — déjà disponible, aucune à refaire :
`reports/v3_results/scores_par_classe.json` (casque) et
`reports/v3_results/ppe_complement_avant.json` (chaussures).

### `epi_casque.pt`

Mesuré sur `ppe_dataset/test` filtré et remappé — jamais vu à l'entraînement,
même protocole que `masque_gilet.pt`.

| Condition | Seuil |
|---|---|
| ✅ Branché en cascade si | `Hardhat` ≥ **75 %** détecté **et** `NO-Hardhat` ≥ **80 %** |
| ❌ Rejeté si | l'une des deux est **sous** son niveau actuel (65 % / 72 %) |
| 🟡 Zone grise entre les deux | décision explicite et documentée, jamais implicite |

### `epi_chaussures.pt`

Aucune référence locale n'existe : `ppe_dataset` n'annote aucune chaussure. Le
candidat est donc jugé sur le split `test` de ses propres sources, **et** par
une contre-épreuve de faux positifs sur des images sans chaussure visible
(`p1_contre_epreuve_faux_positifs.py`).

| Condition | Seuil |
|---|---|
| ✅ Branché si | AP@50 ≥ **0.60** sur `safety_shoe` **et** faux positifs ≤ **5 %** |
| ❌ Rejeté si | faux positifs > 10 % — une chaussure hallucinée vaut moins que pas de classe du tout |

### Garde-fou commun, non négociable

Après branchement de chaque candidat, **mesurer les 14 classes** de la cascade
complète, pas seulement celles visées — `tests/test_non_regression.py` puis
`improvements/generer_classes.py`. La mAP globale ne verrait pas un
effondrement : elle est restée à 0,88 pendant que douze classes étaient à zéro.

### Sort de `ppe_complement.pt`

Si `epi_chaussures.pt` est accepté, `ppe_complement.pt` n'apporte **plus rien** :
ses cinq autres classes sont des doublons déjà écartés par
`p2_table_correspondance_epi.py`. Il sera alors retiré de la cascade — ce qui
supprime au passage un modèle dont `Gloves` à 13 % et `goggles` à 0 % tiraient
la qualité vers le bas. Le fichier est conservé hors dépôt, pas supprimé.

---

## 8 — Ordre d'exécution

| # | Étape | Dépend de | Durée |
|---|---|---|---|
| 0 | Déposer `~/.kaggle/kaggle.json` | **vous** | 2 min |
| 1 | Extraire les 6 archives (zone déjà `.gitignore`) | — | 10 min |
| 2 | **Audit des 3 vérifications bloquantes (§5)** | 1 | 30 min |
| 3 | Écrire `p12_table_correspondance_roboflow.py` | 2 | — |
| 4 | Construire les 2 jeux fusionnés en local | 3 | 30 min |
| 5 | Téléverser les 2 jeux + scripts sur Kaggle | 0, 4 | 1 h |
| 6 | **Lancer les 2 kernels en parallèle** | 5 | 3 – 5 h |
| 7 | Juger chaque candidat contre le §7 | 6 | 30 min |
| 8 | Brancher les acceptés, régénérer `models_calsse.txt` | 7 | 20 min |

Les étapes 1 à 4 ne consomment aucun GPU et ne touchent aucun modèle en place :
elles sont sans risque et peuvent démarrer dès l'accord donné. **L'étape 2 peut
annuler l'étape 3** si `head` s'avère ambigu — c'est précisément pourquoi elle
passe avant.

---

## 9 — Ce que ce plan ne promet pas

**Il ne rendra pas « toutes les classes précises ».** Neuf des quatorze classes
de `ppe_detector.pt` sont déjà entre 0.83 et 0.96 d'AP, et rien ici ne les
touche — ni en bien ni en mal, c'est le but.

Ce qui progresse : le casque (les deux classes les plus faibles du parc) et
`safety_shoe` (qui n'existait pas). Ce qui ne bouge pas : gants, lunettes,
personne, échelle, cône, chute — déjà bons, et les toucher serait un risque
sans contrepartie.

Et le plan v7 garde le dernier mot sur la priorité réelle : la couche de
qualification décide sur des **séquences**, et reste immesurable sans le corpus
vidéo de votre site. Cette campagne améliore la détection image par image ;
elle ne remplace pas ces vidéos.
