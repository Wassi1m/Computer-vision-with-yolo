# Plan d'amélioration v5 — couche de qualification des détections

Date : 2026-08-11
Fait suite à `v4_plan_amelioration.md` (même dossier), dont les priorités
restent valables mais changent d'ordre : ce plan démontre que le point faible
du moteur n'est **pas** la vision des modèles.

## Le constat qui renverse le diagnostic

En cherchant pourquoi `smoke` plafonnait à 40 % de mAP@50, une mesure a montré
que la question était mal posée. Le même modèle, évalué non plus sur la
précision de ses rectangles mais sur sa capacité à repérer une scène contenant
de la fumée (`tests/mesure_operationnelle.py`, jeu `fire_smoke_v9`, 1 165
images dont 671 positives) :

| Seuil de confiance | Détection de la scène | Fausse alarme | Scènes ratées |
|---|---|---|---|
| **0,10** | **96,7 %** | **2,0 %** | 22 / 671 |
| 0,20 | 92,5 % | 1,6 % | 50 |
| 0,30 | 85,2 % | 1,6 % | 99 |
| 0,50 | 65,1 % | 1,4 % | 234 |

**Le modèle voit la fumée dans 96,7 % des scènes qui en contiennent.** Ce que
la mAP@50 sanctionnait, c'est la position exacte du rectangle autour d'un objet
qui n'a physiquement pas de contour — une imprécision sans aucune conséquence
opérationnelle. Le projet pilotait donc son effort sur un indicateur qui ne
mesurait pas le service rendu.

Le vrai mur est ailleurs, et une seule multiplication le montre :

> 2 % de fausse alarme par image × 25 images/s = **0,5 fausse alerte par
> seconde, soit ~43 000 par jour et par caméra.**

Inutilisable. Et l'échappatoire évidente — monter le seuil — coûte un tiers des
départs de feu dès 0,50. **Aucun réglage de seuil ne résout ce compromis, parce
qu'il est posé au mauvais endroit : sur l'image isolée.**

D'où ce plan : ajouter au moteur une **couche de qualification** qui décide sur
une *séquence* et sur un *contexte*, là où le modèle ne décide que sur un
*pixel*. Le modèle propose, la couche dispose.

## Périmètre

Inchangé depuis v3/v4 : ce dépôt est le moteur de détection. Cette couche n'est
donc **pas** un système d'alerte — elle ne notifie personne, n'affiche rien. Son
rôle est de transformer une suite de détections brutes bruitées en **évènements
qualifiés et gradués**, accompagnés des éléments de preuve, que la plateforme
aval consomme pour appliquer *sa* politique d'alerte.

## Garde-fous

Aux six garde-fous de v4 s'ajoutent quatre règles propres à cette couche, parce
qu'elle introduit un risque nouveau : **un filtre qui supprime des fausses
alarmes supprime toujours aussi des vraies détections.**

1. **Toute règle de qualification se mesure sur les deux axes à la fois** —
   taux de détection *et* taux de fausse alarme. Une règle qui n'est validée que
   sur la baisse des fausses alarmes est un piège : elle peut avoir supprimé
   autant de vrais départs de feu.
2. **Budget de latence explicite par scénario.** Confirmer prend du temps, et ce
   temps n'a pas la même valeur partout (voir §3). Aucune règle temporelle sans
   son budget écrit.
3. **Désactivable à chaud.** Chaque règle derrière un drapeau, la couche entière
   derrière `--sans-qualification`, pour comparer en parallèle et revenir en
   arrière sans redéploiement — comme `--sans-gants` aujourd'hui.
4. **La couche ne remplace jamais le modèle.** Elle ne peut que *rejeter* ou
   *graduer* une détection existante, jamais en inventer une. Un scénario qui
   échoue au niveau du modèle doit être corrigé au niveau du modèle.

---

## Architecture de la couche

```
Flux caméra (fixe)
   │
   ├─ [0] Profil de scène ......... jour / nuit / dégradé  → règle les seuils
   ├─ [1] Prétraitement adaptatif .. CLAHE nuit, uniquement si nécessaire
   ├─ [2] Détecteurs (existants) ... seuil BAS : on privilégie le rappel
   ├─ [3] Vérification contextuelle  sur les candidats seulement, coût quasi nul
   ├─ [4] Suivi + signature temporelle  la séquence tranche ce que l'image ne peut pas
   └─ [5] Évènement gradué → API ... suspicion / probable / confirmé + preuves
```

Le principe directeur : **descendre le seuil du détecteur pour ne rien rater,
puis remonter la précision par le contexte et le temps**, au lieu de chercher
un seuil unique qui n'existe pas.

---

## Classification des actions

| Niveau | Signification |
|---|---|
| 🔴 **Bloquant** | Sans cela, le moteur reste inexploitable en production continue |
| 🟠 **Critique** | Gain majeur, réalisable sans dépendance externe |
| 🟡 **Important** | Gain réel, mais après les deux précédents |
| 🔵 **Souhaitable** | Confort, robustesse à long terme |

### Vue d'ensemble

| # | Action | Module | Niveau |
|---|---|---|---|
| 1.1 | Référence de fond par caméra | socle | 🔴 |
| 1.2 | Suivi des détections entre images | socle | 🔴 |
| 1.3 | Machine à états des évènements gradués | socle | 🔴 |
| 2.1 | Fumée : signature de croissance + différentiel couleur | feu/fumée | 🟠 |
| 2.2 | Chute : signature de transition, pas de posture | chute | 🟠 |
| 2.3 | EPI : ancrage sur la personne + persistance | EPI | 🟠 |
| 2.4 | Plaque : vote multi-images + sélection de la meilleure image | plaque | 🟠 |
| 3.1 | Détection automatique jour / nuit / dégradé | transverse | 🟠 |
| 3.2 | ~~Prétraitement CLAHE conditionnel~~ — mesuré, **rejeté** | transverse | ❌ |
| 3.3 | Jeux de seuils par profil + confirmation renforcée la nuit | transverse | 🟠 |
| 4.1 | Évènements gradués dans le contrat d'API | API | 🟠 |
| 4.2 | Qualité de scène auto-déclarée | API | 🟡 |
| 5.1 | Métriques d'exploitation comme indicateur principal | mesure | 🔴 |
| 5.2 | Corpus vidéo de validation | mesure | 🔴 |
| 6.1 | Vérification haute résolution sur candidat | perf + précision | 🟡 |
| 6.2 | Corroboration entre modules | transverse | 🔵 |

---

## Priorité 1 — Le socle (🔴)

Sans ces trois briques, aucune règle de qualification n'est possible : elles
sont toutes des consommatrices de contexte.

**1.1 — Référence de fond par caméra.**
Une caméra de surveillance est **fixe** : c'est l'atout le plus sous-exploité du
projet. Maintenir une image de référence glissante de la scène vide (médiane sur
plusieurs minutes, mise à jour lente) permet de répondre à « cette zone
a-t-elle changé ? » — question infiniment plus discriminante que « cette zone
ressemble-t-elle à de la fumée ? ». Un mur gris est gris depuis toujours ; une
zone qui *devient* grise est un évènement.
Prévoir une réinitialisation sur changement brutal (caméra déplacée, passage
jour/nuit, éclairage allumé).

**1.2 — Suivi des détections entre images.**
Un tracker par recouvrement de boîtes (IoU) suffit : pas besoin de ré-identification
sophistiquée sur une caméra fixe. Chaque piste conserve son historique — position,
aire, score, âge. C'est le support de toutes les règles temporelles.
Coût négligeable, aucun GPU.

**1.3 — Machine à états des évènements.**
Trois états, avec transitions mesurables :
`suspicion` (candidat vu) → `probable` (persiste et satisfait la signature du
module) → `confirmé` (signature complète, éventuellement corroborée).
Un évènement descend aussi d'état, et se termine. C'est cette machine qui
remplace le booléen « détecté / pas détecté » actuel.

## Priorité 2 — Règles de qualification par module (🟠)

Chaque scénario a une signature temporelle *différente*. C'est pourquoi une
règle générique de type « détecté 5 fois de suite » ne fonctionnerait pour
aucun : elle laisserait passer les faux positifs corrélés (un nuage reste un
nuage pendant 5 images) tout en pénalisant les évènements brefs.

**2.1 — Feu / fumée : croissance et différentiel.**
Le discriminant n'est pas la persistance mais la **croissance depuis une
source** : l'aire augmente, le centre s'élève, la forme se déforme. Un nuage
dérive latéralement sans grandir depuis un point ; un objet gris fixe ne fait
ni l'un ni l'autre.
S'y ajoute la vérification couleur en différentiel contre la référence 1.1 :
saturation qui chute, contraste local écrasé, écarts |R−G| et |G−B| proches de
zéro. Pour le feu, la règle complémentaire `R > G > B` avec saturation élevée
est un signal fort et quasi gratuit.
Budget de latence : **30 s** — un départ de feu n'exige pas la seconde.

**2.2 — Chute : une transition, pas une posture.**
C'est le point le plus mal traité aujourd'hui. Une personne accroupie, assise ou
penchée ressemble à une personne au sol sur une image isolée. La chute est un
**évènement dynamique** : rapport hauteur/largeur qui s'inverse brutalement,
puis **immobilité prolongée**. Sans le « avant » et le « après », il n'y a pas
de chute, seulement une posture.
Budget de latence : **3-5 s** — c'est le scénario le plus contraint, une
personne au sol a besoin d'aide immédiatement. La confirmation doit donc être
courte, quitte à accepter plus de fausses alarmes qu'ailleurs. Ce compromis se
tranche avec le client.

**2.3 — EPI : ancrage sur la personne et persistance.**
Deux règles simples et très rentables :
- **Cohérence géométrique** : un casque se trouve dans le tiers supérieur d'une
  personne détectée, un gilet au torse. Une détection d'EPI sans personne, ou
  mal placée sur elle, est un faux positif — à rejeter sans autre examen.
- **Persistance** : un ouvrier ne perd pas son casque pendant une image isolée.
  Le scintillement image à image est la première source de fausses violations.
  Statuer sur une fenêtre glissante (ex. « non conforme sur 80 % des images des
  5 dernières secondes ») élimine ce bruit.
Budget de latence : **5-10 s**, une non-conformité EPI n'est pas une urgence.

**2.4 — Plaque : vote multi-images et sélection de la meilleure image.**
Aujourd'hui la plaque est lue sur une image, alors qu'un véhicule qui traverse
le champ en offre dix à trente. Deux corrections indépendantes :
- **Sélection de la meilleure image** de la piste (netteté par variance du
  laplacien) au lieu de la première venue. C'est la réponse directe à
  l'effondrement mesuré sous flou de mouvement (48,6 %) — sur dix images il y en
  a presque toujours une nette.
- **Vote caractère par caractère** sur l'ensemble des lectures de la piste. Une
  erreur d'OCR est rarement la même d'une image à l'autre ; la majorité converge.
Budget de latence : la durée de passage du véhicule.

## Priorité 3 — Profils jour / nuit (🟠)

Le projet applique aujourd'hui les mêmes seuils de jour comme de nuit, alors que
la performance des modèles s'effondre en basse lumière. Un réglage unique est
forcément mauvais dans au moins une des deux conditions.

**3.1 — Détection automatique du profil.**
Luminance médiane de l'image et proportion de pixels sombres suffisent à classer
la scène en `jour` / `nuit` / `dégradé` (brouillard, contre-jour). Bascule avec
hystérésis pour éviter les oscillations au crépuscule.

**3.2 — Prétraitement conditionnel : hypothèse testée, ❌ REJETÉE.**
L'idée paraissait solide — une image nocturne est hors de la distribution
d'entraînement, la ramener dans une plage normale devait restaurer une partie de
la performance sans réentraîner. La mesure
(`tests/essai_pretraitement_nuit.py`, 400 images, seuil 0,15) dit le contraire :

| Traitement | Détection | Fausse alarme | Coût/image |
|---|---|---|---|
| **Aucun (nuit brute)** | **36,1 %** | 1,2 % | 0 ms |
| Gamma 0,45 | 28,3 % | 1,2 % | 0,5 ms |
| CLAHE (canal L) | 25,2 % | 0,6 % | 8,8 ms |
| CLAHE + débruitage | 21,7 % | 1,8 % | 802 ms |

**Tous les traitements dégradent la détection.** La raison tient à la nature de
l'objet : le **faible contraste de la fumée est précisément le signal**.
L'égalisation le remonte au niveau du reste de l'image et l'efface ; le gamma
amplifie le bruit du capteur autant que le signal utile. Ce qui vaut pour un
objet rigide mal éclairé ne vaut pas pour un voile semi-transparent.

À ne pas réessayer sans raison nouvelle. Une nuance subsiste toutefois : la
dégradation est ici **simulée** par assombrissement d'images de jour. Une vraie
image nocturne (bruit de capteur réel, éclairage artificiel ponctuel) pourrait
réagir différemment — à reconfirmer sur le corpus 5.2, mais sans en attendre
grand-chose.

Ce résultat négatif a une valeur propre : il oriente l'effort nocturne vers la
seule voie qui reste, la confirmation temporelle (voir encadré ci-dessous).

**3.3 — Jeux de seuils par profil.** Mesuré.
La courbe de seuil en conditions nocturnes
(`tests/mesure_operationnelle.py --degradation nuit`, 500 images) :

| Seuil | Détection | Fausse alarme | À 25 img/s |
|---|---|---|---|
| 0,01 | **63,8 %** | 10,0 % | 2,5 fausses/s |
| 0,03 | 55,6 % | 5,4 % | 1,4 fausse/s |
| 0,10 | 43,0 % | 2,7 % | 0,7 fausse/s |
| 0,15 | 37,3 % | 0,9 % | 0,2 fausse/s |
| *(rappel : jour à 0,10)* | *96,7 %* | *2,0 %* | |

Deux enseignements. D'abord, **abaisser le seuil la nuit récupère beaucoup** :
de 37 % à 64 % de détection, gain gratuit et immédiat. Ensuite, **cela ne suffit
pas** : même au plus permissif, plus d'un tiers des scènes nocturnes ne
produisent aucune détection. De nuit le modèle ne perd pas seulement de la
confiance, il perd le signal — un simple réglage ne le rendra pas.

La règle nocturne est donc : **seuil très bas (0,01-0,03) + confirmation
temporelle stricte**. On assume 5 à 10 % de faux positifs par image — inacceptable
tel quel, à 1,4 fausse alerte par seconde — pour les éliminer ensuite par la
signature de croissance. C'est le seul compromis qui tienne : la précision par
image n'existe pas la nuit, celle par séquence se construit.

Réserve : mesures sur dégradation simulée. Le tiers de scènes invisibles par
image le resterait-il sur 750 images successives, où le bruit du capteur varie ?
Probablement pas, mais seul le corpus 5.2 le dira.

> ### Pourquoi la nuit se joue sur la séquence, pas sur l'image
>
> La même mesure qui invalide 3.2 révèle une faiblesse autrement sérieuse :
> **36,1 % de détection la nuit contre ~95 % de jour.** Le moteur rate près de
> deux scènes de fumée sur trois dans l'obscurité — et cette fois, contrairement
> à l'affaire de la mAP@50, c'est un déficit opérationnel réel.
>
> Ce chiffre ne condamne pourtant pas l'exploitation nocturne, à une condition :
> ne plus décider sur une image. **36 % par image n'est pas 36 % par
> évènement.** Un départ de feu observé 30 s à 25 img/s offre ~750 occasions de
> le voir ; même avec des détections partiellement corrélées, la probabilité
> d'en attraper plusieurs reste élevée.
>
> La couche de qualification n'est donc pas seulement un filtre anti-fausses
> alarmes pour le jour : **elle est ce qui rend la nuit exploitable**, en
> convertissant un rappel faible par image en un rappel élevé par évènement.
> C'est la justification la plus forte de tout ce plan.

## Priorité 4 — Contrat d'API (🟠 / 🟡)

**4.1 — Évènements gradués.**
Le schéma d'évènement expose l'état (`suspicion`/`probable`/`confirmé`) et les
éléments de preuve : durée d'observation, taux de croissance, nombre d'images
confirmantes, corroborations. La plateforme applique alors sa propre politique —
alerter dès `probable` sur un site sensible, attendre `confirmé` ailleurs.
Le moteur cesse de décider à la place de son intégrateur, ce qui est exactement
son rôle.
Rétrocompatibilité : conserver le champ actuel, en le faisant correspondre à
`confirmé`, pour ne pas casser l'intégration existante.

**4.2 — Qualité de scène auto-déclarée.** 🟡
Le moteur doit savoir dire « je vois mal ». Un champ `qualite_scene` (nette /
dégradée / inexploitable) fondé sur luminance, netteté et contraste évite le pire
des cas : produire silencieusement des détections peu fiables que l'aval prend
pour argent comptant. Une caméra sale ou éblouie doit se signaler.

## Priorité 5 — Mesure (🔴)

Ces deux points sont bloquants : **sans eux, aucune règle ci-dessus n'est
réglable**, et le risque de dégrader le rappel en croyant améliorer la précision
est maximal.

**5.1 — Faire des métriques d'exploitation l'indicateur principal.**
`tests/mesure_operationnelle.py` existe désormais. La mAP@50 reste utile pour
comparer deux entraînements du même modèle, mais **elle ne doit plus servir à
juger la valeur du moteur** — elle a déjà conduit à rejeter un modèle sur 3,3
points d'écart mesurés sur 99 instances, et à croire faible un détecteur qui
repère 96,7 % des scènes.

**5.2 — Corpus vidéo de validation.**
Toutes les mesures actuelles portent sur des images isolées. Or **toutes les
règles de ce plan sont temporelles** : elles sont littéralement immesurables sur
des images fixes. Il faut des séquences annotées au niveau de l'évènement
(« départ de feu de t=12 s à t=90 s »), couvrant jour et nuit, avec et sans
évènement. Quelques dizaines de séquences suffisent pour commencer.
C'est la dépendance la plus lourde de ce plan, et la seule qui demande une
contribution extérieure.

## Priorité 6 — Compléments (🟡 / 🔵)

**6.1 — Vérification haute résolution sur candidat.** 🟡
Détecteur en basse résolution sur l'image entière (rapide), puis re-vérification
en haute résolution **uniquement sur le rectangle candidat** (rare, donc peu
coûteux). On obtient la sensibilité du 1280 px au prix du 640 px.
Ce point sert deux objectifs à la fois : il améliore la détection des petits
objets (fumée lointaine, plaque de loin) *et* répond au blocage 1.2 de v4
(« ~5-7 FPS, pas temps réel »), puisqu'il évite de traiter toute l'image en haute
résolution.

**6.2 — Corroboration entre modules.** 🔵
`fire` détecté près d'une `smoke` renforce les deux. Une personne détectée
renforce une chute. Un objet abandonné suppose qu'une personne l'ait déposé.
Ces liens sont déjà disponibles dans le pipeline unifié et gratuits à exploiter.

---

## Ordre d'exécution recommandé

**Étape 1 — débloquer la mesure (5.1, 5.2).** Sans corpus vidéo, tout le reste
se règle à l'aveugle. 5.1 est déjà fait ; 5.2 demande des séquences — à lancer
immédiatement car c'est la dépendance la plus lente.

**Étape 2 — le socle (1.1, 1.2, 1.3), en parallèle de l'étape 1.** Pur code,
sans GPU, sans dépendance externe. Testable sur les quelques vidéos disponibles.

**Étape 3 — les deux règles au meilleur rapport gain/effort (2.4, 2.3).** La
sélection de meilleure image pour les plaques et la persistance EPI sont
simples, à faible risque, et attaquent des faiblesses déjà chiffrées (48,6 % sous
flou ; scintillement EPI).

**Étape 4 — les profils jour/nuit (3.1, 3.3).** 3.2 est déjà tranché : mesuré,
rejeté. Reste à détecter le profil et à adapter seuils et exigence de
confirmation — le seul levier nocturne qui subsiste.

**Étape 5 — les règles temporelles fines (2.1, 2.2).** Ce sont les plus
puissantes mais elles exigent le corpus vidéo de l'étape 1 pour être réglées.

**Étape 6 — l'API (4.1, 4.2)**, une fois les états stabilisés, puis 6.1 et 6.2.

## Ce que ce plan ne fait pas

Il ne remplace pas v4 : la décision CPU/GPU (v4 §1.1), la cadence cible par
scénario (v4 §1.2) et les questions ouvertes du contrat d'API (v4 §2.4) restent
entières, et deviennent même plus urgentes — une couche temporelle a besoin de
savoir combien d'images par seconde elle peut espérer.

Il ne dispense pas non plus d'entraîner : `smoke_distant` (v4 §2.1, en cours) et
les classes EPI jamais retravaillées (v4 §3.2) gagnent toujours à l'être. Mais
l'ordre est désormais clair — **la couche de décision apportera davantage, plus
vite et pour moins cher, que n'importe quel entraînement supplémentaire.**
