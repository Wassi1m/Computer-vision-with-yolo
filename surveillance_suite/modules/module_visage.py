"""
Module 10 : reconnaissance faciale — détection et signature (embedding).

Ce module ne nomme personne. Il produit une **signature numérique** de chaque
visage vu, et c'est la plateforme qui décide à qui elle appartient. Cette
répartition n'est pas un détail d'implémentation, elle répond à trois
contraintes :

1. **Le trombinoscope reste en un seul endroit.** Un gabarit facial est une
   donnée biométrique. Le répliquer sur chaque hôte GPU multiplierait par autant
   le nombre d'endroits à sécuriser, à purger à l'échéance de rétention et à
   inscrire au registre des accès. La base vectorielle vit donc côté plateforme,
   dans PostgreSQL, et nulle part ailleurs.
2. **Le moteur n'a pas accès à la base.** C'est la règle d'architecture du
   projet : les deux dépôts ne partagent aucun import, ils communiquent par le
   webhook décrit dans `docs/contrat_api.md`.
3. **L'enrôlement et la reconnaissance doivent utiliser le même modèle.** Une
   signature calculée par ArcFace ne se compare pas à une signature calculée par
   autre chose. En publiant le nom et la dimension du modèle dans chaque
   évènement (`modele`, `dim`), la plateforme peut refuser une comparaison entre
   gabarits hétérogènes au lieu de produire un score qui ne veut rien dire.

## Ce qui est réutilisé plutôt que redupliqué

Le module ne détecte **pas** les personnes : `AnalyseurGeneral` l'a déjà fait, à
chaque image, avec son suivi (`persist=True`). On lui emprunte
`ctx["personnes"]` — une liste de `(track_id, box)` — et on ne cherche un visage
que **dans le tiers supérieur** de chaque boîte de personne. Deux gains :

- le détecteur de visages travaille sur une vignette de quelques milliers de
  pixels au lieu de l'image entière ;
- chaque visage trouvé hérite immédiatement du `track_id` de la personne qui le
  porte, ce qui est **la** clé qui permettra de rattacher « sans casque » à un
  employé. Un détecteur de visages lancé sur l'image entière aurait produit des
  visages orphelins, qu'il aurait fallu réassocier aux personnes après coup.

## Économie d'émission

Une signature n'est utile qu'une fois par piste. Réémettre 512 nombres à chaque
image pour la même personne saturerait le webhook sans rien apprendre à
personne. Le module n'émet donc que lorsque la signature **apprend quelque
chose** : première vue d'une piste, ou vue nettement meilleure que la meilleure
déjà envoyée (§`SEUIL_AMELIORATION`). Sur une personne qui traverse le champ en
dix secondes, cela fait deux ou trois émissions, pas trois cents.
"""

from __future__ import annotations

import base64
import logging
import math

import numpy as np

logger = logging.getLogger("surveillance.visage")


#: Nom du paquet de modèles InsightFace. `buffalo_l` = SCRFD-10GF pour la
#: détection + ArcFace R50 (glint360k) pour l'embedding, 512 dimensions.
#: Choisi pour le compromis demandé : c'est le meilleur des paquets InsightFace
#: prêts à l'emploi qui tienne encore dans le budget d'une image partagée avec
#: six autres modèles. `antelopev2` est plus précis mais deux fois plus lourd.
PAQUET_DEFAUT = "buffalo_l"

#: Dimension de la signature ArcFace. Publiée dans chaque évènement : la
#: plateforme refuse de comparer deux gabarits de dimensions différentes.
DIMENSION = 512

#: Fraction haute de la boîte d'une personne dans laquelle chercher un visage.
#: 0,45 et non 0,33 : une personne assise, penchée ou accroupie a la tête plus
#: bas dans sa boîte qu'une personne debout, et manquer son visage coûte plus
#: cher qu'analyser quelques pixels de trop.
FRACTION_TETE = 0.45

#: Marge ajoutée autour de la vignette de tête, en fraction de sa largeur. Un
#: visage coupé au ras du menton dégrade fortement l'alignement d'ArcFace.
MARGE = 0.15

#: En deçà, la vignette est trop petite pour qu'ArcFace produise une signature
#: exploitable. Mesure classique sur SCRFD : sous ~50 px de côté, le taux
#: d'identification s'effondre. On refuse plutôt que de produire une signature
#: qui polluerait la recherche.
TAILLE_MINIMALE = 48

#: Gain de qualité exigé pour réémettre la signature d'une piste déjà envoyée.
SEUIL_AMELIORATION = 0.12


def _nettete(vignette: np.ndarray) -> float:
    """Netteté d'une vignette, entre 0 et 1 (variance du laplacien, normalisée).

    Un visage flou produit une signature instable : elle tombera à mi-chemin
    entre plusieurs employés au lieu de désigner le bon. Mieux vaut ne rien
    émettre. Le seuil de 120 sur la variance du laplacien est la valeur usuelle
    pour un visage en niveaux de gris ; au-delà on considère la netteté acquise.
    """
    import cv2

    gris = cv2.cvtColor(vignette, cv2.COLOR_BGR2GRAY) if vignette.ndim == 3 else vignette
    return float(min(1.0, cv2.Laplacian(gris, cv2.CV_64F).var() / 120.0))


def qualite(vignette: np.ndarray, score_detection: float, pose: float | None = None) -> float:
    """Note de qualité d'un visage, entre 0 et 1.

    Trois facteurs, multipliés parce qu'ils sont éliminatoires et non
    compensatoires : un visage parfaitement net vu de profil intégral ne vaut
    pas mieux qu'un visage de face parfaitement flou. Une moyenne les aurait
    laissés se compenser.

    - `score_detection` : la confiance de SCRFD, qui chute déjà sur un visage
      partiellement masqué ;
    - la netteté, qui distingue un visage figé d'un visage en mouvement rapide ;
    - la taille, plafonnée : au-delà de 112 px (l'entrée d'ArcFace) un visage
      plus grand n'apporte plus rien.
    """
    h, l = vignette.shape[:2]
    taille = min(1.0, min(h, l) / 112.0)
    note = float(score_detection) * _nettete(vignette) * taille
    if pose is not None:
        # `pose` = écart au frontal en degrés. Au-delà de 60° un visage n'est
        # plus identifiable de façon fiable par ArcFace.
        note *= max(0.0, 1.0 - abs(pose) / 60.0)
    return max(0.0, min(1.0, note))


def encoder(signature: np.ndarray) -> str:
    """Sérialise une signature pour le transport JSON.

    En `float16` et non `float32` : la signature est normalisée (norme 1), ses
    composantes tiennent entre -0,2 et 0,2, et `float16` y garde trois chiffres
    significatifs — bien au-delà de ce que la comparaison cosinus exige. Le gain
    est un facteur deux sur une charge utile émise des milliers de fois par jour.
    Le JSON brut aurait coûté huit fois plus que ce base64.
    """
    return base64.b64encode(np.asarray(signature, dtype=np.float16).tobytes()).decode("ascii")


def decoder(texte: str) -> np.ndarray:
    """Inverse d'`encoder`. Utilisé côté plateforme et par les tests."""
    return np.frombuffer(base64.b64decode(texte), dtype=np.float16).astype(np.float32)


class DetecteurVisage:
    """SCRFD (détection) + ArcFace (signature), via InsightFace.

    Même interface que les autres détecteurs de la suite : `.detecter(vignette)`
    renvoie une liste de dicts. L'import d'InsightFace est différé au premier
    usage : le moteur doit pouvoir démarrer, et les six autres modules
    fonctionner, sur une machine où la reconnaissance faciale n'est pas
    installée. C'est la règle du projet — aucune fonctionnalité existante ne doit
    être cassée par une nouvelle.
    """

    def __init__(self, paquet: str = PAQUET_DEFAUT, seuil: float = 0.5, gpu: bool = True):
        self.paquet, self.seuil = paquet, seuil
        self.disponible = False
        self._app = None
        try:
            import onnxruntime
            from insightface.app import FaceAnalysis

            # Sur un serveur CPU, `onnxruntime` n'a pas de fournisseur CUDA :
            # le demander quand même fait écrire un avertissement à chaque
            # démarrage et laisse croire que le GPU sert. On ne garde que ce
            # que l'installation sait réellement faire.
            gpu = gpu and "CUDAExecutionProvider" in onnxruntime.get_available_providers()
            fournisseurs = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"] if gpu
                else ["CPUExecutionProvider"]
            )
            # `allowed_modules` restreint le chargement à ce qu'on utilise :
            # sans cela InsightFace charge aussi l'estimation d'âge et de genre,
            # que ce projet n'exploite pas et n'a aucune raison de calculer sur
            # des personnes filmées — c'est de la donnée sensible produite pour
            # rien.
            self._app = FaceAnalysis(
                name=paquet,
                allowed_modules=["detection", "recognition"],
                providers=fournisseurs,
            )
            self._app.prepare(ctx_id=0 if gpu else -1, det_size=(320, 320))
            self.disponible = True
            logger.info("reconnaissance faciale prête (paquet=%s, gpu=%s)", paquet, gpu)
        except ImportError:
            logger.warning(
                "insightface absent : la reconnaissance faciale est désactivée, "
                "les autres modules continuent. Installer avec "
                "`pip install insightface onnxruntime-gpu` pour l'activer."
            )
        except Exception as e:  # noqa: BLE001 — un échec de chargement ne doit pas tuer le moteur
            logger.error("chargement de la reconnaissance faciale impossible : %s", e)

    def detecter(self, image: np.ndarray) -> list[dict]:
        """Renvoie `[{box, signature, score, pose}]` pour l'image donnée."""
        if not self.disponible:
            return []
        try:
            trouves = self._app.get(image)
        except Exception as e:  # noqa: BLE001
            logger.error("échec de l'analyse d'un visage : %s", e)
            return []

        resultats = []
        for v in trouves:
            if float(v.det_score) < self.seuil:
                continue
            sig = getattr(v, "normed_embedding", None)
            if sig is None:
                continue
            resultats.append({
                "box": tuple(int(c) for c in v.bbox),
                "signature": np.asarray(sig, dtype=np.float32),
                "score": float(v.det_score),
                "pose": float(v.pose[1]) if getattr(v, "pose", None) is not None else None,
            })
        return resultats


def tete_de(boite_personne, forme_image) -> tuple[int, int, int, int] | None:
    """Vignette où chercher le visage d'une personne, ou `None` si trop petite.

    Rogner avant de détecter est ce qui rend ce module compatible avec les six
    autres sur le même flux : SCRFD travaille sur ~150×150 px au lieu de
    1920×1080.
    """
    x1, y1, x2, y2 = (int(v) for v in boite_personne)
    largeur, hauteur = x2 - x1, y2 - y1
    if largeur <= 0 or hauteur <= 0:
        return None

    marge = int(largeur * MARGE)
    hx1 = max(0, x1 - marge)
    hy1 = max(0, y1 - marge)
    hx2 = min(forme_image[1], x2 + marge)
    hy2 = min(forme_image[0], y1 + int(hauteur * FRACTION_TETE) + marge)

    if min(hx2 - hx1, hy2 - hy1) < TAILLE_MINIMALE:
        return None
    return hx1, hy1, hx2, hy2


class SuiviSignatures:
    """Mémoire des signatures déjà émises, par identifiant de suivi.

    Répond à une seule question : « ai-je déjà envoyé, pour cette piste, une
    signature au moins aussi bonne que celle-ci ? ». Si oui, on se tait.

    La purge est indispensable : sur un flux qui tourne des semaines, les
    `track_id` ne sont jamais réutilisés et le dictionnaire croîtrait sans fin.
    Même raisonnement que `AnalyseurEPI._histo` et `ConfirmationTemporelle.purger`.
    """

    def __init__(self, capacite: int = 512):
        self.capacite = capacite
        self._meilleures: dict[int, float] = {}
        self._ordre: list[int] = []

    def merite_emission(self, track_id: int, note: float) -> bool:
        precedente = self._meilleures.get(track_id)
        if precedente is None:
            self._enregistrer(track_id, note)
            return True
        if note >= precedente + SEUIL_AMELIORATION:
            self._meilleures[track_id] = note
            return True
        return False

    def _enregistrer(self, track_id: int, note: float):
        self._meilleures[track_id] = note
        self._ordre.append(track_id)
        while len(self._ordre) > self.capacite:
            del self._meilleures[self._ordre.pop(0)]

    def oublier(self, track_ids_vivants):
        """Retire les pistes qui ne sont plus dans le champ."""
        vivants = set(track_ids_vivants)
        for tid in [t for t in self._ordre if t not in vivants]:
            self._ordre.remove(tid)
            self._meilleures.pop(tid, None)
