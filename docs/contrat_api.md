# Contrat d'interface — moteur de détection

Version : 1.0 (2026-08-08)

Ce document décrit ce que le moteur de détection produit et comment le
consommer. Il s'adresse à l'équipe qui développe la plateforme d'alerting
et de supervision.

**Périmètre du moteur** : analyser un flux vidéo et produire des évènements de
détection. Le moteur ne notifie personne, n'affiche aucun tableau de bord et
ne stocke pas d'historique long. Ces fonctions relèvent de la plateforme
consommatrice.

---

## 1. Modes de fonctionnement

| Mode | Commande | Usage |
|---|---|---|
| **Production (headless)** | `python unified_surveillance.py --source rtsp://…` | Par défaut. Aucun affichage, aucune interaction clavier. |
| **Test / démonstration** | `… --display` | Ouvre une fenêtre avec la vidéo annotée. Réservé aux tests et démonstrations. |

L'affichage ne change rien à ce qui est détecté ni à ce qui est émis : il ne
fait que dessiner. Un test visuel et une exécution de production produisent les
mêmes évènements.

## 2. Transport des évènements

Le protocole définitif n'est pas encore arrêté avec la plateforme
consommatrice. Le transport est donc isolé derrière une interface interne
(`Sortie`) : en ajouter un nouveau ne demande aucune modification du moteur.

Disponibles aujourd'hui, cumulables :

| Transport | Activation | Comportement |
|---|---|---|
| **Webhook HTTP** | `--webhook https://…` | Un `POST` JSON par évènement, envoyé de façon asynchrone. |
| **Fichier JSONL** | `--events chemin.jsonl` | Un objet JSON par ligne. Utile en test et comme trace locale de secours. |
| **Console** | (défaut si aucun autre) | Une ligne lisible par évènement. |

**Garanties et limites du webhook**, à connaître avant intégration :

- L'envoi est **asynchrone** : le moteur n'attend pas la réponse et ne ralentit
  pas si le consommateur est lent.
- Il n'y a **ni réessai ni accusé de réception** : un évènement perdu est perdu.
  C'est un choix assumé — la détection ne doit jamais s'arrêter parce que le
  consommateur est indisponible. Si une livraison garantie est nécessaire, il
  faudra passer à une file de messages ; c'est précisément la décision à
  arrêter ensemble.
- L'ordre d'arrivée **n'est pas garanti**. Utiliser le champ `t` pour ordonner.

## 3. Format d'un évènement

Objet JSON, encodé en UTF-8.

```json
{
  "t": 1786209770.4062107,
  "frame": 3,
  "source": "epi",
  "type": "violation_epi",
  "libelle": "Personne 1 — SANS CASQUE !",
  "conf": 0.0,
  "box": [189, 250, 266, 556],
  "extra": {}
}
```

| Champ | Type | Description |
|---|---|---|
| `t` | float | Horodatage Unix (secondes, décimales). **Clé d'ordonnancement.** |
| `frame` | int | Numéro d'image depuis le démarrage. `-1` pour les évènements techniques (flux perdu/repris) qui ne proviennent pas d'une image. |
| `source` | string | Module émetteur : `epi`, `chute`, `feu`, `ligne`, `porte`, `lpr`, `capture`. |
| `type` | string | Nature de l'évènement (voir §4). |
| `libelle` | string | Description lisible, en français. **Destinée à l'affichage, pas à être analysée par programme** — utiliser `type` pour toute logique. |
| `conf` | float | Confiance du modèle, entre 0 et 1. `0.0` quand la notion ne s'applique pas (une violation EPI résulte d'un lissage sur plusieurs images, pas d'une détection unique). |
| `box` | array\|null | Boîte englobante `[x1, y1, x2, y2]` en pixels de l'image analysée. `null` si sans objet (état de porte, évènement technique). |
| `extra` | object | Champs additionnels selon le type. Peut être vide. |

## 4. Types d'évènements

| `source` | `type` | Signification | `box` |
|---|---|---|---|
| `epi` | `violation_epi` | Un EPI obligatoire manque à une personne | Boîte de la personne |
| `chute` | `chute` | Personne au sol détectée | Boîte de la personne |
| `feu` | `fire` | Départ de feu détecté | Boîte de la zone |
| `feu` | `smoke` | Fumée détectée | Boîte de la zone |
| `ligne` | `franchissement` | Un objet suivi a franchi la ligne virtuelle | `null` |
| `objet_abandonne` | `objet_abandonne` | Bagage immobile, sans personne à proximité depuis le délai configuré (30 s par défaut) | Boîte de l'objet |
| `foule` | `foule` | Seuil d'effectif ou de densité franchi dans la zone | `null` |
| `foule` | `foule_terminee` | Retour sous le seuil | `null` |
| `porte` | `porte` | Changement d'état de la porte (ouverte/fermée) | `null` |
| `lpr` | `plaque` | Plaque d'immatriculation lue | Boîte de la plaque |
| `capture` | `flux_perdu` | Le flux vidéo n'est plus lisible | `null` |
| `capture` | `flux_repris` | Le flux vidéo est rétabli | `null` |

**`flux_perdu` et `flux_repris` sont essentiels côté plateforme** : ils
permettent de distinguer « aucune alerte parce que le site est calme » de
« aucune alerte parce que le moteur ne voit plus rien ». Sans les traiter, une
caméra tombée ressemble à un site parfaitement sûr.

## 5. Anti-répétition

Un évènement identique (même `source`, `type` et `libelle`) n'est **pas réémis
avant 3 secondes**. Une personne sans casque pendant une minute produit donc
une vingtaine d'évènements, pas plusieurs centaines.

Conséquence pour la plateforme : la répétition d'un évènement indique que la
condition **persiste**, pas qu'un nouvel incident distinct s'est produit. La
logique de regroupement en « incident » relève de la plateforme.

## 6. Point de santé

Activé par `--health-port 8899`. Un exploitant peut alors interroger :

```
GET /health
```

```json
{
  "modeles": ["general", "ligne", "chute", "feu", "porte", "epi"],
  "flux_connecte": true,
  "reconnexions": 0,
  "frames": 93,
  "fps": 6.23,
  "evenements": 28,
  "derniere_detection": 1786209791.3175383,
  "uptime_s": 23.2,
  "sain": true
}
```

| Code HTTP | Signification |
|---|---|
| `200` | Moteur sain : flux connecté **et** images analysées |
| `503` | Moteur dégradé : flux perdu, ou aucune image analysée |

Le champ `reconnexions` mérite d'être surveillé : une valeur qui augmente
régulièrement signale un flux instable, même si le moteur reste « sain » à
l'instant du sondage.

## 7. Cycle de vie

Le moteur s'arrête proprement sur `SIGTERM` et `SIGINT` : il termine l'image en
cours, libère la caméra, vide les envois en attente puis rend la main. Il est
donc utilisable tel quel sous systemd ou dans un conteneur.

Un flux réseau indisponible **au démarrage** n'empêche pas le service de
démarrer : il entre en boucle de reconnexion (temporisation progressive de 1 s
à 30 s) et publie `flux_repris` dès que la caméra répond. Un fichier vidéo
introuvable, en revanche, est traité comme une erreur de configuration et
provoque un arrêt immédiat.

## 8. Limites connues

À prendre en compte dans le dimensionnement de l'intégration :

- **Cadence** : environ 3,8 images/seconde sur CPU (12 cœurs, sans GPU). Ce
  n'est pas un flux temps réel à 25 images/seconde ; les évènements sont
  échantillonnés à cette cadence.
- **Pas de livraison garantie** sur le webhook (voir §2).
- **Une seule caméra par processus.** Plusieurs caméras demandent plusieurs
  instances, et le champ identifiant de caméra n'existe pas encore dans le
  format d'évènement — à ajouter dès que le multi-caméra sera au programme.
- **Données sensibles** : les évènements `plaque` contiennent des numéros
  d'immatriculation en clair. La politique de rétention côté plateforme doit
  être définie explicitement.

## 9. Questions ouvertes à trancher ensemble

1. **Protocole définitif** : webhook HTTP suffit-il, ou une file de messages
   (livraison garantie, réessais, tampon) est-elle nécessaire ?
2. **Identifiant de caméra / de site** dans les évènements : quel format
   attend la plateforme ?
3. **Authentification** des appels sortants : jeton d'API, mTLS, autre ?
4. **Rétention des plaques** : le moteur doit-il les transmettre en clair, les
   pseudonymiser, ou ne signaler que la présence d'un véhicule ?
