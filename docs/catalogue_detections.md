# Catalogue des détections

Ce que le moteur sait détecter aujourd'hui, classe par classe, avec les
performances mesurées et les limites connues.

Ce document répond à une question précise que se pose tout intégrateur :
**« sur quoi puis-je compter, et à quel point ? »** Il complète
`docs/contrat_api.md` (le format des évènements) et `docs/registre_modeles.md`
(la provenance et l'historique des modèles). En cas de désaccord entre les
trois, `tests/reference_modeles.json` fait foi — c'est le seul fichier vérifié
automatiquement, par `tests/test_non_regression.py`.

Dernière vérification : 2026-08-13, les cinq modèles à 0,0000 d'écart de leur
référence.

---

## Comment lire les chiffres

Deux indicateurs coexistent, et **les confondre conduit à des décisions
fausses**.

| Indicateur | Question à laquelle il répond | Quand s'en servir |
|---|---|---|
| **mAP@50** | « le rectangle est-il au bon endroit ? » | Comparer deux entraînements du même modèle |
| **Détection de scène** | « l'objet est-il signalé, oui ou non ? » | Décider si un modèle est bon pour l'exploitation |

L'écart peut être énorme. Le détecteur de fumée plafonne à **40 % de mAP@50**
sur la classe `smoke`, mais repère **96,7 %** des scènes qui contiennent de la
fumée. La mAP sanctionne la position d'un rectangle autour d'un panache, qui
n'a physiquement pas de contour — une imprécision sans aucune conséquence
opérationnelle.

**Ne jamais juger la valeur d'un modèle sur sa mAP.** Cette confusion a déjà
conduit à croire faible un détecteur qui fonctionne très bien.

---

## Vue d'ensemble

| Domaine | Détecté | Fiabilité | Statut |
|---|---|---|---|
| Feu et fumée | `fire`, `smoke` | 96,7 % des scènes (jour) | ✅ Exploitable |
| EPI — gilet | porté / non porté | 91,7 % / 77,9 % | ✅ Exploitable |
| EPI — casque, masque, lunettes, gants | porté / non porté | 0,42 à 0,96 selon la classe | ✅ Exploitable |
| Chute | `falling` / `stand` | 99,0 % | ✅ Exploitable |
| Plaque d'immatriculation | `plate` | 86,2 % | ✅ Exploitable |
| Porte | ouverte / fermée / entrouverte | non mesurée | ⚠️ À valider |
| Personnes, véhicules, objets courants | 80 classes COCO | héritée de YOLO26 | ✅ Exploitable |
| Franchissement de ligne | calculé, pas détecté | jamais mesurée | ⚠️ Non validé |
| Objet abandonné | calculé, pas détecté | logique testée, terrain non mesuré | ✅ Disponible |
| Foule — effectif | calculé, pas détecté | celle du détecteur de personnes | ✅ Disponible |
| Foule — densité pers/m² | calculé, pas détecté | géométrie exacte **si calibré** | ✅ Disponible |

---

## 1. Feu et fumée

**Modèle** : `surveillance_suite/models/fire_smoke.pt` — YOLO26n, 2 classes.

| Classe | Signification | AP@50 |
|---|---|---|
| `fire` | Flammes visibles | 0,9037 |
| `smoke` | Fumée, panache | 0,4071 |

**Performance réelle** (`tests/mesure_operationnelle.py`, jeu `fire_smoke_v9`,
1 165 images dont 671 positives) :

| Seuil | Scène détectée | Fausse alarme |
|---|---|---|
| **0,10** | **96,7 %** | 2,0 % |
| 0,20 | 92,5 % | 1,6 % |
| 0,30 | 85,2 % | 1,6 % |
| 0,50 | 65,1 % | 1,4 % |

**Limites**
- **La nuit, la détection tombe à 36 %** — le modèle ne perd pas seulement de la
  confiance, il perd le signal. C'est la faiblesse la plus sérieuse du parc.
- Le taux de fausse alarme de 2 % par image devient ingérable en flux continu
  (~43 000 par jour à 25 img/s). La réponse est la couche de qualification du
  plan v5, pas un réglage de seuil.
- Le prétraitement nocturne a été testé et **rejeté** : tous les traitements
  dégradent la détection, car le faible contraste de la fumée *est* le signal.

**Ne détecte pas** : la fumée lointaine comme catégorie distincte. Un modèle à
3 classes (`smoke_distant`) a été entraîné le 2026-08-12 et **rejeté** — moins
bon sur les deux axes.

---

## 2. Équipements de protection individuelle

**Modèle principal** : `ppe_detection/models/ppe_detector.pt` — YOLOv8m, 25,8 M
paramètres, 14 classes. C'est le modèle le plus lourd du parc : il consomme à
lui seul ~78 % du temps de la cascade EPI.

| Classe | Signification |
|---|---|
| `Person` | Personne (support d'ancrage des EPI) |
| `Hardhat` / `NO-Hardhat` | Casque porté / absent |
| `Safety Vest` / `NO-Safety Vest` | Gilet haute visibilité porté / absent |
| `Mask` / `NO-Mask` | Masque porté / absent |
| `Goggles` / `NO-Goggles` | Lunettes portées / absentes |
| `Gloves` / `NO-Gloves` | Gants portés / absents |
| `Safety Cone` | Cône de signalisation |
| `Ladder` | Échelle |
| `Fall-Detected` | Personne au sol (redondant avec le modèle chute) |

**Modèle secondaire** : `ppe_detection/models/ppe_complement.pt` — YOLO26n,
6 classes (`Gloves`, `Vest`, `goggles`, `helmet`, `mask`, `safety_shoe`).
Ajoute `safety_shoe`, absent du modèle principal. Désactivable par
`--sans-gants` : le retirer de la cascade rend ~15 % de cadence.

**Performance mesurée** — les 14 classes, sur les 4 423 images du split `test`
de `ppe_dataset`, jamais vues à l'entraînement (2026-08-15) :

| Classe | AP@50 | | Classe | AP@50 |
|---|---|---|---|---|
| `Ladder` | 0,9619 | | `Safety Cone` | 0,6484 |
| `Goggles` | 0,9598 | | `Hardhat` | 0,5965 |
| `NO-Goggles` | 0,9571 | | `Safety Vest` | 0,5688 |
| `Person` | 0,9524 | | `Mask` | 0,5271 |
| `Gloves` | 0,9014 | | `NO-Mask` | 0,4678 |
| `NO-Gloves` | 0,8268 | | `NO-Hardhat` | 0,4150 |
| `Fall-Detected` | 0,8242 | | `NO-Safety Vest` | 0,1593 |
| | | | **mAP@50** | **0,6976** |

Sur `ppe_vest_clean_14c/val`, seul jeu où le gilet est annoté proprement :
`Safety Vest` **0,9170** et `NO-Safety Vest` **0,7786**.

**Pourquoi deux jeux d'évaluation.** Chacun ne peut juger que ce qu'il annote
correctement. `ppe_dataset` annote les douze autres classes mais son gilet est
incohérent — environ 14 300 de ses 30 765 images laissent la zone du gilet non
annotée, ce qui compte toute détection correcte comme une fausse alarme.
`ppe_vest_clean_14c` fait l'inverse. Ne lire qu'un des deux conduit à une
conclusion fausse, dans un sens ou dans l'autre.

### Ce que ce jeu de données ne permet plus d'améliorer

Mesuré le 2026-08-16 en **détection de scène** — le modèle voit-il l'objet,
indépendamment de la précision du rectangle :

| Classe | Détection | Exemples | Levier restant |
|---|---|---|---|
| `Hardhat` | 65 % | **28 996** | ❌ pas la rareté |
| `NO-Hardhat` | 72 % | 9 705 | ❌ pas la rareté |
| `NO-Mask` | 78 % | 1 577 | ✅ duplication |
| `NO-Safety Vest` | 70 % | 1 435 | ✅ duplication |
| `Mask` | 95 % | 1 989 | — |
| `Safety Cone` | 98 % | 9 843 | — |

Les deux dernières colonnes disent l'essentiel. **`Hardhat` est la classe la
mieux pourvue du jeu et reste la plus faible** : passer de 2 504 à 10 004
exemples l'a fait progresser de 32 % à 65 %, mais les 18 992 restants ne
donneront pas autant. Sa limite n'est pas le nombre d'images mais leur
diversité — le jeu vient de quelques chantiers, sous quelques angles.

**Améliorer le casque demandera d'autres images, pas plus des mêmes** : autres
sites, autres distances, autres conditions d'éclairage. C'est la seule
amélioration EPI qui reste, et elle sort du périmètre de ce dépôt.

Même remarque pour `NO-Hardhat` : reconnaître l'*absence* d'un casque est
intrinsèquement plus difficile que le reconnaître, et ses 9 705 exemples
suffisent en nombre. Toutes les classes négatives du parc sont en dessous de
leurs positives — c'est le défaut structurel de ce jeu.

**Limites**
- **`NO-Safety Vest` a reculé de 7,5 points** lors du ré-entraînement du
  2026-08-15 (0,8534 → 0,7786 sur le jeu gilet). Perte assumée en échange de la
  récupération des douze classes qui étaient à zéro, mais réelle : cette classe
  signale l'infraction, l'erreur y est donc la plus coûteuse. Cause probable :
  elle ne compte que 1 435 instances contre 4 499 pour `Safety Vest`, et souffre
  le plus du partage de capacité avec douze classes supplémentaires. Correction
  prévue par sur-échantillonnage au prochain entraînement.
- `NO-Hardhat` (0,415) et `NO-Mask` (0,468) restent les plus faibles : les
  classes négatives sont partout moins bien apprises que leurs positives.
- Les deux modèles ont des taxonomies différentes (`helmet` vs `Hardhat`,
  `Vest` vs `Safety Vest`) ; la correspondance est faite par
  `improvements/ppe_taxonomy.py`, avec un seuil propre à chaque classe.
- L'ancrage EPI ↔ personne par confinement existe
  (`improvements/qualification.py`) mais **n'est pas encore branché** dans le
  pipeline, qui associe toujours par IoU — une méthode inadaptée à un rapport
  « partie de » (un casque occupe ~3 % d'une personne).

---

## 3. Chute

**Modèle** : `surveillance_suite/models/fall_detector.pt` — YOLO26n, 2 classes.

| Classe | AP@50 |
|---|---|
| `falling` | 0,9923 |
| `stand` | 0,9879 |

C'est le modèle le plus fiable du parc, et le plus robuste aux conditions
dégradées (98 % en faible luminosité).

**Limite majeure, et elle est structurelle** : le modèle juge une **posture sur
une image isolée**. Une personne accroupie, assise ou penchée ressemble à une
personne au sol. Une vraie chute est un **évènement dynamique** — inversion
brutale du rapport hauteur/largeur, puis immobilité prolongée. Sans le « avant »
et le « après », il n'y a pas de chute, seulement une posture.

Les 99 % mesurés portent donc sur la reconnaissance d'une posture, **pas sur la
détection d'une chute réelle**. À ne pas présenter au client comme équivalent.

---

## 4. Plaque d'immatriculation

**Modèle** : `surveillance_suite/models/license_plate.pt` — YOLO26n, 1 classe
`plate` (localisation seule). La lecture des caractères est faite par OCR en
aval (`module_lpr.py`).

**AP@50** : 0,8619

**Limites**
- **Effondrement sous flou de mouvement : 48,6 %** — soit une plaque sur deux
  perdue sur un véhicule en mouvement, ce qui est le cas d'usage principal.
- Basse résolution : 61,9 %. La lecture se fait aujourd'hui sur une seule image
  alors qu'un véhicule qui traverse le champ en offre dix à trente.
- **Donnée personnelle** : le numéro est transmis en clair dans les évènements.
  Aucune option de pseudonymisation n'existe (voir plan v6 §3.1).

---

## 5. Porte

**Modèle** : `surveillance_suite/models/door_classifier.pt` — classificateur
YOLO11n-cls, 3 classes : `Open`, `Closed`, `Semi` (entrouverte).

**Aucune métrique de référence.** Ce modèle n'est pas couvert par
`test_non_regression.py`. Sa fiabilité est inconnue et il ne devrait pas être
présenté comme validé.

---

## 6. Détection générale

**Modèle** : `yolo26n.pt` — les 80 classes COCO standard (personne, voiture,
camion, sac, vélo, etc.), sans ré-entraînement.

Sert de socle : c'est lui qui fournit les personnes et le suivi (`track_id`)
dont dépendent l'ancrage EPI, le franchissement de ligne et le comptage.
Performances héritées du modèle public.

---

## 7. Scénarios calculés (pas détectés)

Ces fonctions ne sont pas des modèles : elles raisonnent sur les sorties de la
détection générale.

**Franchissement de ligne** — `AnalyseurLigne`. Compte les passages d'un
`track_id` de part et d'autre d'une ligne définie en configuration.
**Jamais mesuré** sur un jeu annoté. Fonctionne, mais aucun chiffre ne l'étaye.

**Objet abandonné** — `AnalyseurObjetAbandonne`. Un bagage (`backpack`,
`handbag`, `suitcase`, `bicycle`, `skateboard` — classes COCO déjà connues du
détecteur général) resté immobile, sans personne à proximité, pendant le délai
configuré (`--delai-abandon`, 30 s par défaut).

**Aucun entraînement n'a été nécessaire** : les classes existent dans COCO et le
suivi est déjà calculé. Le coût mesuré est de **0,2 % du temps de traitement**.

Les seuils sont relatifs à la taille de l'objet et non en pixels absolus : un
sac à 5 m et le même sac à 50 m n'occupent pas le même nombre de pixels, et un
rayon fixe n'aurait pas le même sens selon la profondeur de la scène.

Une alerte au plus par objet : sans cela un sac oublié produirait un évènement
par image jusqu'à la fin du flux.

**Limite** : la logique est couverte par sept tests unitaires, mais **jamais
mesurée sur des séquences réelles** — il n'existe pas encore de corpus vidéo
annoté (plan v6 §4.1). Le taux de fausse alarme sur le terrain est inconnu.

**Foule — effectif et densité** — `AnalyseurFoule`. Compte les personnes dont
le point de contact au sol tombe dans la zone surveillée, et exprime une
**densité en personnes par mètre carré** lorsque la caméra est calibrée.

Aucun entraînement non plus : ce qui manquait n'était pas un modèle mais la
**géométrie du sol**. Coût mesuré : **0,1 %** du temps de traitement.

| Mode | Ce qui est publié |
|---|---|
| **Avec `--calibration-sol`** | effectif **et** densité pers/m² exacte |
| Sans calibration | effectif seul ; `densite_pers_m2` vaut `null` |

Le champ vaut `null` et non une estimation : mieux vaut annoncer qu'on ne sait
pas que publier une valeur inventée que l'aval prendrait pour argent comptant.

**Calibration** : `python outils/calibrer_sol.py --source <flux>`, quatre clics
sur les coins d'un rectangle au sol dont on connaît les dimensions. Une fois par
caméra, à l'installation. L'outil re-mesure l'aire après projection et prévient
si l'écart trahit des points mal pointés.

**Tous les seuils sont des paramètres** (`--seuil-densite`, `--seuil-effectif`,
`--zone-foule`), également réglables par variables d'environnement `MOTEUR_*` :
ils dépendent du site et de la réglementation applicable, et n'ont aucune valeur
universelle. Le défaut de 2,0 pers/m² correspond à « 10 personnes pour 5 m² ».

Un seul évènement à la formation de la foule, un à sa dissolution
(`foule_terminee`) — jamais un par image.

**Limites**
- La géométrie est exacte, **le comptage reste celui du détecteur**. En foule
  dense, les occlusions font sous-estimer l'effectif, donc la densité. Au-delà
  d'environ 3-4 pers/m², un modèle de comptage par carte de densité serait plus
  adapté que la détection par boîtes.
- Le rectangle de calibration doit être **posé au sol**. Calibrer sur le sommet
  d'un muret donnerait un plan qui n'est pas celui où marchent les personnes.
- La précision dépend entièrement de celle des clics : un coin pointé à 10 px
  près sur un rectangle de 100 px introduit ~10 % d'erreur sur les distances.
- **Jamais mesuré sur des séquences réelles** (corpus absent, plan v6 §4.1).

> L'approche précédente (`crowd_density_detector_auto.py`, script autonome
> conservé pour référence) estimait l'échelle depuis la hauteur apparente des
> personnes, en supposant 1,70 m. Sans réglage, mais l'hypothèse s'effondre
> précisément en **vue plongeante** — l'angle habituel d'une caméra de comptage :
> vu du dessus, une personne proche et une personne lointaine ont presque la
> même hauteur apparente. Elle ne donnait d'ailleurs que des distances entre
> paires, jamais une densité surfacique.

---

## Robustesse en conditions dégradées

mAP@50 mesurée sous dégradation simulée
(`reports/v3_results/robustesse_conditions_reelles.json`) :

| Modèle | Référence | Faible lum. | Contre-jour | Pluie/brouillard | Flou | Basse rés. |
|---|---|---|---|---|---|---|
| `fall_detector` | 0,991 | 0,981 | 0,984 | 0,993 | 0,992 | 0,991 |
| `ppe_detector` | 0,906 | 0,830 | 0,857 | 0,877 | 0,824 | 0,879 |
| `license_plate` | 0,854 | 0,780 | 0,862 | 0,772 | **0,486** | 0,619 |
| `fire_smoke` | 0,739 | **0,351** | 0,582 | 0,635 | 0,659 | 0,643 |

Deux points noirs : **la fumée en faible luminosité** et **la plaque sous flou
de mouvement**.

Réserve importante : ces dégradations sont **simulées** à partir d'images de
jour. Une vraie image nocturne a un bruit de capteur et un éclairage artificiel
ponctuel qu'une simulation ne reproduit pas. Les chiffres restent les meilleurs
disponibles, mais ils demandent confirmation sur un corpus réel.

---

## Cadence

**5,31 images/seconde** sur CPU (mesure d'endurance sur 30 minutes, 5 994
images). Ce n'est pas un flux temps réel à 25 img/s : les évènements sont
échantillonnés à cette cadence. La cible de déploiement (CPU ou GPU) n'est pas
tranchée — voir le plan v6 §2.3.
