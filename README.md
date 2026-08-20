# Moteur de détection — EPI et sécurité de site

Moteur de vision par ordinateur qui analyse **un** flux vidéo et publie des
évènements : port des EPI, feu et fumée, chute, franchissement de ligne, objet
abandonné, affluence, plaque d'immatriculation, état de porte.

Le rôle de ce dépôt s'arrête à la détection. Les alertes, les tableaux de bord
et l'exploitation métier appartiennent à la plateforme qui consomme ces
évènements, via le contrat décrit dans `docs/contrat_api.md`.

## Point d'entrée

Un seul, quel que soit le module :

```bash
python improvements/unified_surveillance.py --source rtsp://… \
    --camera-id cam-quai-3 --webhook https://… --health-port 8080
```

Les deux applications d'origine (`ppe_detection/` et `surveillance_suite/`)
ouvraient chacune leur propre caméra. Elles ne sont plus des points d'entrée :
ce qui subsiste de `surveillance_suite/` est la bibliothèque de modules que le
moteur unifié charge, et de `ppe_detection/` les poids et les jeux de données.

## Documentation

| Fichier | Contenu |
|---|---|
| `docs/contrat_api.md` | Format des évènements, garanties de livraison, `/health` et `/metrics`. **À lire avant toute intégration.** |
| `docs/exploitation.md` | Déploiement, configuration, surveillance, diagnostic |
| `docs/catalogue_detections.md` | Ce que chaque module détecte, et ce qu'il vaut |
| `docs/registre_modeles.md` | Origine, version et métriques de chaque poids |
| `docs/manifeste_modeles.json` | Empreinte SHA-256 des poids en production |

## Modèles

Les poids `.pt` (142 Mo) ne sont **pas** versionnés dans git. Ce qui fait
référence, c'est `docs/manifeste_modeles.json` et `tests/reference_modeles.json` :
empreintes et métriques, committées même quand le binaire ne l'est pas.

```bash
python outils/verifier_modeles.py --verifier   # les poids en place sont-ils les bons ?
```

## Vérifications

```bash
pytest tests/test_logique_metier.py tests/test_integration.py -q   # rapide, sans modèle
python tests/test_non_regression.py                                # lent, charge les modèles
python improvements/p19_mesure_cascade.py --n 400                  # mesure de la cascade EPI
```

Les deux premiers tournent à chaque poussée (`.github/workflows/tests.yml`).
Les deux derniers sont manuels : ils ont besoin des poids, absents du dépôt.

## Conteneurs

```bash
docker build -t moteur-epi:cpu .                         # CPU
docker build -f Dockerfile.gpu -t moteur-epi:gpu .       # GPU (NVIDIA)
docker run --gpus all -p 8080:8080 moteur-epi:gpu --source rtsp://…
```

L'image CPU épingle `torch+cpu` : sur un serveur GPU elle ignore la carte sans
rien signaler. C'est la raison d'être de `Dockerfile.gpu` et du drapeau
`--verifier-gpu`, qui refuse de démarrer si CUDA est indisponible.
