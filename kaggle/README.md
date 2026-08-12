# Entraînement sur Kaggle

Remplace le flux GCP (`deploy/00_tout_entrainer.sh`), inutilisable depuis
l'épuisement des crédits — l'API Compute refuse toute opération sans facturation
active, y compris l'arrêt et la suppression des instances.

Kaggle impose trois contraintes que la VM n'avait pas, et qui expliquent la
forme de ce dossier :

| Contrainte Kaggle | Conséquence |
|---|---|
| Session limitée (~12 h), interruptible | L'entraînement doit **reprendre**, pas recommencer |
| `/kaggle/input` en lecture seule | Le jeu dégradé s'écrit dans `/kaggle/working` |
| Seul `/kaggle/working` est conservé | Tout ce qui doit survivre y est écrit |

---

## 1. Téléverser le jeu de données (une seule fois)

Le jeu `fire_smoke_v9` fait 877 Mo. L'interface web convient, mais la ligne de
commande est plus fiable à cette taille.

```bash
pip install kaggle
# Deposer le jeton API dans ~/.kaggle/kaggle.json
#   (Kaggle > votre profil > Settings > API > Create New Token)
chmod 600 ~/.kaggle/kaggle.json

cd "surveillance_suite/data/dataset"
kaggle datasets create -p fire_smoke_v9 --dir-mode zip
```

Un fichier `dataset-metadata.json` est fourni dans ce dossier : le copier dans
`fire_smoke_v9/` avant la commande. Il porte l'identifiant `wassimmay` ; le
remplacer par le vôtre si vous téléversez sous un autre compte, l'`id` devant
correspondre au propriétaire du jeu.

**Mettre à jour le jeu** après modification, sans recréer :

```bash
kaggle datasets version -p fire_smoke_v9 -m "nouvelle version" --dir-mode zip
```

## 2. Téléverser les scripts et les poids de départ

Un second jeu, minuscule, contenant de quoi lancer l'entraînement :

```bash
mkdir -p /tmp/ppe-scripts
cp kaggle/entrainer_kaggle.py improvements/p8_dataset_nuit.py /tmp/ppe-scripts/
cp surveillance_suite/models/fire_smoke.pt /tmp/ppe-scripts/
# adapter dataset-metadata.json (titre : ppe-scripts) puis :
kaggle datasets create -p /tmp/ppe-scripts
```

## 3. Créer le notebook

1. **New Notebook**
2. *Settings* → **Accelerator : GPU T4 x2** (ou P100)
3. *Settings* → **Internet : On** — nécessaire pour `pip install ultralytics`
4. *Add Input* → ajouter les **deux** jeux (`fire-smoke-v9` et `ppe-scripts`)

Puis une seule cellule :

```python
!pip install -q ultralytics
!python "$(find /kaggle/input -name entrainer_kaggle.py | head -1)" --epochs 60 --imgsz 896 --batch 16
```

Le script retrouve seul les jeux attachés (il cherche par motif, pas par chemin
en dur : le nom de montage dépend du titre donné à l'upload). Le `find` de la
cellule sert la même logique pour se localiser lui-même : un chemin
`/kaggle/input/ppe-scripts/...` écrit en dur a échoué au premier essai, le nom
de montage réel ne correspondant pas au titre donné à l'upload.

## 4. Reprendre après une interruption

C'est le point qui change tout par rapport à GCP. Si la session est coupée avant
la fin :

> **Attention : « moins d'époques que demandé » ne veut pas dire « interrompu ».**
> Le run du 2026-08-12 s'est arrêté à 26 époques sur 60 demandées, et il était
> pourtant **terminé** : `EarlyStopping` a constaté 25 époques sans progrès et a
> coupé volontairement. Deux signes distinguent une fin normale d'une coupure de
> session : la validation finale a bien tourné, et le script a imprimé son
> `=== Termine ===`. Une session tuée ne laisse rien imprimer.
>
> Suivre la procédure de reprise ci-dessous sur un run *terminé* ne le
> prolongera pas : Ultralytics refuse par une assertion
> (`trainer.py`, `resume_training`) —
> `training to 60 epochs is finished, nothing to resume`. La raison est que
> `strip_optimizer`, exécuté en fin d'entraînement, inscrit `epoch = -1` dans
> `last.pt`. Le script **plantera** au lieu de reprendre.
>
> Pour pousser un modèle plafonné plus loin, il faut un **nouvel** entraînement
> partant de `best.pt` avec `--patience 0` (désactive l'arrêt anticipé) — pas
> une reprise. Mais si la patience a expiré, c'est justement la preuve que les
> époques supplémentaires n'apportent rien.

1. Ouvrir le notebook, *File* → **Add Input** → onglet **Notebook Output**
2. Sélectionner **la sortie de la session précédente**
3. Relancer la même cellule

Le script détecte `sorties/fumee_robuste/weights/last.pt` dans les entrées,
recopie l'état du run dans `/kaggle/working`, et relance Ultralytics avec
`resume=True`. L'entraînement repart à l'époque suivante, pas de zéro.

`save_period=5` force une sauvegarde toutes les 5 époques : une coupure ne coûte
jamais plus de 5 époques de calcul.

## 5. Récupérer le modèle

En fin de session, `/kaggle/working/sorties/fumee_robuste/weights/best.pt` est
téléchargeable depuis l'onglet **Output** du notebook.

Un `resume_entrainement.json` y résume le nombre d'époques effectuées, la
meilleure époque et son mAP@50.

Ce résumé classe les époques par **fitness** (`0.1×mAP50 + 0.9×mAP50-95`), le
critère qu'Ultralytics utilise lui-même pour choisir `best.pt`. Trier par mAP50
seule — ce que faisait la première version du script — désignait une époque qui
n'était pas celle enregistrée dans `best.pt`, contradiction visible dans le log
du 12 août (« Best results observed at epoch 1 » contre « meilleure_epoque : 24 »).
Le champ `mAP50_max_toutes_epoques` conserve l'information de la meilleure mAP50
brute, à titre indicatif.

### Résultat du run du 2026-08-12

26 époques en 2 h 54 sur T4, arrêt par `EarlyStopping` (voir l'encadré du §4).

| Classe | mAP@50 |
|---|---|
| `fire` | 0.965 |
| `smoke` | 0.836 |
| `smoke_distant` | 0.747 |

Ces chiffres ne sont **pas** comparables à ceux du registre (mesurés sur
`fire_smoke_enriched`, 2 classes) : jeu différent, taxonomie différente. Seule
l'étape 4 ci-dessous tranche. L'intérêt réel du candidat se joue sur
`smoke_distant`, une classe que le modèle en place ne connaît pas du tout.

---

## Avant d'installer le modèle dans le projet

Les garde-fous du plan v4 restent valables, et l'épisode du 11 août montre
pourquoi : un modèle fine-tuné y avait gagné 42 points de robustesse nocturne
tout en perdant 3,3 points sur `smoke` de jour — il a été rejeté, à raison.

```bash
# 1. Sauvegarder le modele en place, hors depot
cp surveillance_suite/models/fire_smoke.pt /tmp/fire_smoke_avant_kaggle.pt

# 2. Installer le candidat
cp ~/Telechargements/best.pt surveillance_suite/models/fire_smoke.pt

# 3. Non-regression : un echec impose de restaurer la sauvegarde
python tests/test_non_regression.py --modele fire_smoke

# 4. Metriques d'exploitation -- celles qui comptent vraiment
python tests/mesure_operationnelle.py \
    --modele surveillance_suite/models/fire_smoke.pt \
    --donnees surveillance_suite/data/dataset/fire_smoke_v9 \
    --classes-fumee 1 2 --classes-modele 1 2
```

**Attention au nombre de classes.** Le jeu `fire_smoke_v9` en compte trois
(`fire`, `smoke`, `smoke_distant`) là où le modèle en place n'en a que deux.
Le repli de `smoke_distant` vers l'évènement `smoke` **est fait** depuis le
2026-08-12, dans `surveillance_suite/modules/module_fire_smoke.py`
(`LABELS_EQUIVALENTS`) : sans lui, la fumée lointaine — justement celle qui
permet la détection la plus précoce — ne déclencherait rien. Le label d'origine
reste exposé sous la clé `label_modele`, pour la couche de qualification du
plan v5.

> ### ⚠️ L'étape 3 n'est PAS un critère de rejet pour un modèle à 3 classes
>
> `test_non_regression.py` valide sur `fire_smoke_enriched` (`nc: 2`) et compare
> `smoke` à 0.4071. Or un modèle à trois classes prédit `smoke_distant` là où ce
> jeu annote `smoke` : l'AP de la classe `smoke` **chute mécaniquement**, sans
> que le modèle soit moins bon. Rejeter sur ce chiffre serait une erreur.
>
> Sur ce test, ne retenir que :
> - **`fire`** (référence 0.9037) — la classe n'a pas bougé, comparaison honnête,
>   et c'est un vrai garde-fou ;
> - **`smoke`** — à lire, jamais à opposer au candidat.
>
> Le verdict appartient à l'étape 4, seule comparable au chiffre près :
> `--classes-modele 1 2` fusionne `smoke` et `smoke_distant` côté modèle, et sur
> l'ancien modèle à 2 classes la classe 2 n'existe simplement pas. Le baseline
> `operationnel_fire_smoke_avant.json` a été mesuré exactement ainsi.

## Ce qu'il ne faut pas juger sur la mAP@50

Mesuré le 11 août : le détecteur feu/fumée repère **96,7 %** des scènes
contenant de la fumée alors que sa mAP@50 sur cette classe plafonne à 40 %. La
mAP sanctionne la position du rectangle autour d'un objet qui n'a pas de contour
net — une imprécision sans conséquence opérationnelle. Comparer deux
entraînements du même modèle par la mAP reste légitime ; en faire le critère de
mise en production ne l'est pas.
