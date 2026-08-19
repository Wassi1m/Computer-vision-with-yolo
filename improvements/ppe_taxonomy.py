#!/usr/bin/env python3
"""P2 — Table de correspondance explicite entre les modèles EPI.

Problème traité
---------------
`ppe_dual_model_backup2.py` fait cohabiter `ppe_detector.pt` (M1, 14 classes) et
`ppe_complement.pt` (M2, 6 classes) en associant leurs sorties par simple
recherche de sous-chaîne (`MOTS_CLES_EPI`) puis par IoU avec les personnes.
Trois défauts en découlent :

1. **Aucune vérification sémantique.** La correspondance repose sur le fait que
   `"vest"` est une sous-chaîne de `"Safety Vest"`. Renommer une classe dans un
   ré-entraînement casse silencieusement la fusion. C'est aussi ce qui a fait
   planter la validation croisée des deux modèles (`IndexError`) : rien
   n'exprime que les index de classes des deux modèles ne sont pas comparables.

2. **Un seul seuil par EPI pour deux modèles décalibrés.** Le fichier note
   lui-même que `Hardhat` (M1) plafonne vers 0.55 tandis que `helmet` (M2)
   plafonne vers 0.04 ; appliquer `CONF_EPI["casque"] = 0.15` aux deux revient à
   ignorer complètement M2 sur le casque tout en payant son coût de calcul.

3. **Pas de déduplication inter-modèles.** Un même casque détecté par M1 et M2
   produit deux boîtes, comptées deux fois à l'affichage.

Ce module remplace la correspondance implicite par une table explicite, et
expose une fusion qui déduplique par concept.

Constat structurant issu de la table
------------------------------------
M2 ne couvre **aucun concept absent de M1 sauf `safety_shoe`**, et ne possède
aucune classe négative (pas de `NO-*`) : il ne peut donc jamais déclencher une
alerte de non-conformité, seulement confirmer un port. Si les chaussures de
sécurité ne font pas partie du référentiel de conformité à surveiller, M2 est
redondant et peut être retiré de la cascade — ce qui supprime un tiers des
passes avant par image (cf. priorité 3, vitesse du pipeline).
"""

from dataclasses import dataclass, field
from typing import Iterable

# ── Concepts EPI canoniques ──────────────────────────────────────────────────
# Clé interne stable, indépendante du nom de classe de n'importe quel modèle.
EPI_CANONIQUES = ("casque", "masque", "lunettes", "gilet", "gants", "chaussures")

LIBELLES_FR = {
    "casque":     ("Casque porte",        "SANS CASQUE !"),
    "masque":     ("Masque porte",        "SANS MASQUE !"),
    "lunettes":   ("Lunettes portees",    "SANS LUNETTES !"),
    "gilet":      ("Gilet porte",         "SANS GILET !"),
    "gants":      ("Gants portes",        "SANS GANTS !"),
    "chaussures": ("Chaussures securite", "SANS CHAUSSURES !"),
}

# EPI dont l'absence constitue une violation à signaler.
EPI_OBLIGATOIRES = {
    "casque":     True,
    "masque":     False,   # dépend du contexte du site
    "lunettes":   False,
    "gilet":      True,
    "gants":      False,
    "chaussures": False,
}


@dataclass(frozen=True)
class Correspondance:
    """Ce qu'une classe d'un modèle donné signifie dans le référentiel canonique.

    epi      : concept canonique visé, ou None si la classe ne décrit pas un EPI
               (`Person`, `Ladder`, `Safety Cone`, `Fall-Detected`).
    porte    : True = l'EPI est porté, False = il est absent (classe négative).
               None pour les classes hors EPI.
    conf_min : seuil de confiance propre à cette classe *de ce modèle*. Les deux
               modèles n'étant pas calibrés pareil, le seuil appartient à la
               classe, pas au concept.
    evenement: pour les classes hors EPI, l'évènement métier correspondant.
    """
    epi: str | None
    porte: bool | None
    conf_min: float
    evenement: str | None = None


# ── M1 : best.pt — YOLOv8m, 14 classes ───────────────────────────────────────
# Seul modèle porteur de classes négatives, donc seul capable de déclencher une
# alerte de non-conformité. Référence en cas de conflit avec M2.
M1_NOM = "ppe_detector.pt"
M1 = {
    "Hardhat":        Correspondance("casque",     True,  0.15),
    "NO-Hardhat":     Correspondance("casque",     False, 0.15),
    "Mask":           Correspondance("masque",     True,  0.08),
    "NO-Mask":        Correspondance("masque",     False, 0.08),
    "Goggles":        Correspondance("lunettes",   True,  0.08),
    "NO-Goggles":     Correspondance("lunettes",   False, 0.08),
    "Gloves":         Correspondance("gants",      True,  0.08),
    "NO-Gloves":      Correspondance("gants",      False, 0.08),
    "Safety Vest":    Correspondance("gilet",      True,  0.12),
    "NO-Safety Vest": Correspondance("gilet",      False, 0.12),
    "Person":         Correspondance(None, None, 0.25, evenement="personne"),
    "Fall-Detected":  Correspondance(None, None, 0.25, evenement="chute"),
    "Ladder":         Correspondance(None, None, 0.25, evenement="echelle"),
    "Safety Cone":    Correspondance(None, None, 0.25, evenement="cone"),
}

# ── M2 : ppe_complement.pt — YOLOv8n, 6 classes ─────────────────────────────────
# Aucune classe négative. Scores structurellement bas (observés : `helmet` et
# `goggles` plafonnent vers 0.04), d'où des seuils propres bien plus bas que
# ceux de M1 — les mélanger était l'erreur de calibration d'origine.
M2_NOM = "ppe_complement.pt"
M2 = {
    "helmet":      Correspondance("casque",     True, 0.03),
    "mask":        Correspondance("masque",     True, 0.03),
    "goggles":     Correspondance("lunettes",   True, 0.03),
    "Vest":        Correspondance("gilet",      True, 0.05),
    "Gloves":      Correspondance("gants",      True, 0.05),
    "safety_shoe": Correspondance("chaussures", True, 0.10),
}

# ── M3 : masque_gilet.pt — YOLOv8m, 4 classes dédiées ───────────────────────
# Entraîné le 2026-08-17 uniquement sur les images de `ppe_dataset` qui
# annotent réellement masque ou gilet (improvements/p11_jeu_masque_gilet.py).
# M1 souffre sur ces deux concepts d'un défaut d'ANNOTATION, pas de capacité :
# mesuré sur le sous-ensemble cohérent (p1_eval_par_concept.py), il atteint
# déjà 0.96/0.92/0.89/0.67 d'AP50 -- contre 0.55/0.60/0.58/0.17 publiés sur le
# jeu complet, pollué par les ~29 000 images qui n'annotent ni l'un ni l'autre.
# M3 va plus loin : entraîné sur les 6 059 images cohérentes disponibles (train
# + val, le split test restant local pour le jugement), il atteint sur ce test
# jamais vu 0.97/0.96/0.93/0.77 d'AP50 -- voir
# reports/v3_results/masque_gilet_candidat.json. Il prend donc le pas sur M1
# pour `masque` et `gilet` uniquement ; M1 reste inchangé et sert de filet de
# secours si M3 n'est pas chargé.
M3_NOM = "masque_gilet.pt"
M3 = {
    "Mask":           Correspondance("masque", True,  0.25),
    "NO-Mask":        Correspondance("masque", False, 0.15),
    "Safety Vest":    Correspondance("gilet",   True,  0.25),
    "NO-Safety Vest": Correspondance("gilet",   False, 0.15),
}

# ── M4 : epi_casque.pt — 2 classes dédiées ──────────────────────────────────
# Entraîné le 2026-08-18 (campagne v8) sur Hard Hat Universe + Construction PPE
# + 6 000 images de `ppe_dataset`, avec 25 % d'images de fond. Le casque était
# le plus gros déficit mesuré du parc : M1 n'en détectait que 65 % (`Hardhat`)
# et 72 % (`NO-Hardhat`) des scènes annotées. Le plan v7 avait établi que le
# plafond venait d'un manque de DIVERSITE et non de volume -- 28 996 exemples
# locaux issus des mêmes chantiers n'avaient pas suffi.
# Mesuré sur `ppe_dataset/test` (jamais téléversé) : 99.2 % et 98.3 % de
# détection de scène, AP50 0.927 / 0.963 -- voir
# reports/v3_results/casque_candidat.json. Il prend donc le pas sur M1 pour le
# casque uniquement ; M1 reste le filet de secours s'il n'est pas chargé.
M4_NOM = "epi_casque.pt"
M4 = {
    "Hardhat":    Correspondance("casque", True,  0.25),
    "NO-Hardhat": Correspondance("casque", False, 0.15),
}

# ── M5 : epi_gants_lunettes.pt — 4 classes dédiées ──────────────────────────
# Entraîné le 2026-08-18 sur PPEs v8 + Safety Gloves v5 + 6 000 images de
# `ppe_dataset`, avec 25 % d'images de fond.
# Un premier candidat, entraîné le 2026-08-17 SANS les images de `ppe_dataset`,
# avait été REJETE : il perdait sur trois des quatre classes face à M1. La cause
# n'était pas sa capacité mais un écart de domaine -- il avait appris sur les
# seules sources Roboflow puis avait été jugé sur le corpus local. Les 6 000
# images ajoutées lui donnent les deux domaines, et le candidat du 18 bat
# désormais M1 sur les QUATRE classes : 0.955/0.931/0.966/0.965 contre
# 0.932/0.909/0.960/0.961 (reports/v3_results/gants_candidat.json). Sa détection
# de scène, jamais mesurée auparavant sur ces classes, va de 95 % à 99 %.
M5_NOM = "epi_gants_lunettes.pt"
M5 = {
    "Gloves":      Correspondance("gants",    True,  0.25),
    "NO-Gloves":   Correspondance("gants",    False, 0.15),
    "Goggles":     Correspondance("lunettes", True,  0.25),
    "NO-Goggles":  Correspondance("lunettes", False, 0.15),
}

# ── M6 : epi_chaussures.pt — 2 classes dédiées ──────────────────────────────
# Entraîné le 2026-08-19, au TROISIEME essai. Les deux premiers ont été rejetés
# et le detail de chaque rejet est conserve dans reports/v3_results/ :
# le modele confondait chaussure et chaussure DE SECURITE, faute de sources
# opposant les deux. Trois jeux ajoutes ont corrige cela.
#
# Mesure sur un jeu de test de 851 images reservees AVANT l'entrainement --
# les deux campagnes precedentes n'en avaient aucun et se jugeaient sur un
# proxy : AP50 0.814 / 0.632, precision 0.841 / 0.906.
#
# ⚠️ SEUILS ASYMETRIQUES, ET C'EST DELIBERE.
# Les deux classes n'ont pas la meme gravite en cas d'erreur. `safety_shoe`
# affirme une CONFORMITE : s'y tromper -- prendre une basket de ville pour une
# chaussure de securite, ce que ce modele fait encore -- fabrique une fausse
# conformite, et masque une infraction reelle. `NO-safety_shoe` ne produit
# qu'une fausse alerte, genante mais sans danger. D'ou 0.50 sur la premiere et
# 0.35 sur la seconde : on prefere manquer une conformite que d'en inventer une.
#
# ⚠️ Ce modele DEPEND de l'ancrage a la personne pour etre exploitable.
# Mesure nu, il declenche sur 59 % des images de ppe_dataset (visages, cones,
# carrosseries) ; apres ancrage, sur 1,0 % (improvements/p18_ancrage_chaussures.py).
# La regle qui le sauve est PLAGES_ANATOMIQUES["chaussures"] = (0.55, 1.10)
# dans qualification.py. Lancer le moteur avec --sans-ancrage-epi rendrait ce
# modele inutilisable.
M6_NOM = "epi_chaussures.pt"
M6 = {
    "safety_shoe":    Correspondance("chaussures", True,  0.50),
    "NO-safety_shoe": Correspondance("chaussures", False, 0.35),
}

TABLES = {M1_NOM: M1, M2_NOM: M2, M3_NOM: M3, M4_NOM: M4, M5_NOM: M5, M6_NOM: M6}

# Concepts que M2 apporte et que M1 ne couvre pas.
APPORT_UNIQUE_M2 = sorted(
    {c.epi for c in M2.values() if c.epi} - {c.epi for c in M1.values() if c.epi}
)  # -> ['chaussures']

# Concepts pour lesquels M1 fait autorité (il les couvre et sait dire l'absence).
PRIORITE_M1 = sorted({c.epi for c in M1.values() if c.epi})

# Ordre de priorité PAR CONCEPT, utilisé par `fusionner()`. Chaque modèle dédié
# n'est prioritaire que sur les concepts qu'il couvre ET sur lesquels il a été
# mesuré meilleur que M1, sur un split que l'entraînement n'a jamais vu. Partout
# ailleurs M1 reste la référence, exactement comme avant.
# Les modèles dédiés étant tous optionnels au chargement, l'ordre décrit une
# préférence, pas une dépendance : si l'un manque, la cascade retombe sur le
# suivant.
PRIORITE_MODELE = {
    "casque":     (M4_NOM, M1_NOM, M2_NOM),
    "masque":     (M3_NOM, M1_NOM, M2_NOM),
    "lunettes":   (M5_NOM, M1_NOM, M2_NOM),
    "gilet":      (M3_NOM, M1_NOM, M2_NOM),
    "gants":      (M5_NOM, M1_NOM, M2_NOM),
    # M6 est le seul modele du parc qui sache dire l'ABSENCE de chaussure de
    # securite. M2 reste derriere lui en filet, mais sans classe negative et
    # sans mesure exploitable : son `safety_shoe` n'a jamais pu etre evalue,
    # `ppe_dataset` n'annotant aucune chaussure.
    "chaussures": (M6_NOM, M2_NOM),
}


def verifier_coherence(modeles: dict[str, dict[int, str]]) -> list[str]:
    """Contrôle que les tables correspondent aux modèles réellement chargés.

    `modeles` : {nom_modele: {indice: nom_classe}}, un item par modèle
    effectivement chargé (M2 et M3 sont optionnels dans la cascade).

    À appeler au démarrage : si un modèle est ré-entraîné avec une taxonomie
    différente, on veut une erreur explicite au lancement plutôt qu'une fusion
    silencieusement fausse en production.
    """
    erreurs = []
    for nom_modele, noms in modeles.items():
        table = TABLES.get(nom_modele)
        if table is None:
            erreurs.append(f"{nom_modele}: aucune table de correspondance definie")
            continue
        presentes = set(noms.values())
        attendues = set(table)
        for c in sorted(presentes - attendues):
            erreurs.append(f"{nom_modele}: classe '{c}' présente dans le modèle mais absente de la table")
        for c in sorted(attendues - presentes):
            erreurs.append(f"{nom_modele}: classe '{c}' dans la table mais absente du modèle")
    return erreurs


@dataclass
class DetectionEPI:
    """Une détection ramenée au référentiel canonique."""
    epi: str | None
    porte: bool | None
    evenement: str | None
    conf: float
    box: tuple[int, int, int, int]
    modele: str
    classe_source: str
    libelle: str = field(default="")

    def __post_init__(self):
        if self.epi:
            self.libelle = LIBELLES_FR[self.epi][0 if self.porte else 1]
        else:
            self.libelle = self.evenement or self.classe_source


def traduire(modele_nom: str, classe: str, conf: float, box) -> DetectionEPI | None:
    """Applique la table : renvoie None si la classe est inconnue ou sous le seuil."""
    corr = TABLES.get(modele_nom, {}).get(classe)
    if corr is None or conf < corr.conf_min:
        return None
    return DetectionEPI(corr.epi, corr.porte, corr.evenement, conf, tuple(box), modele_nom, classe)


def _iou(a, b) -> float:
    xa, ya = max(a[0], b[0]), max(a[1], b[1])
    xb, yb = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def fusionner(detections: Iterable[DetectionEPI], iou_seuil: float = 0.5) -> list[DetectionEPI]:
    """Déduplique les détections portant sur le même concept au même endroit.

    Règle de résolution, dans l'ordre :
      1. deux détections du même concept qui se recouvrent (IoU > seuil) sont le
         même objet physique ;
      2. l'ordre de `PRIORITE_MODELE[concept]` tranche entre modèles — M1
         l'emporte partout sauf là où un modèle dédié a été mesuré meilleur :
         M3 sur `masque`/`gilet`, M4 sur `casque`, M5 sur `gants`/`lunettes`
         (voir leurs commentaires respectifs dans ppe_taxonomy.py) ;
      3. à modèle égal, la confiance la plus élevée l'emporte.

    Les détections hors EPI (personne, chute, échelle, cône) traversent sans
    fusion : aucun modèle dédié ne les couvre.
    """
    dets = [d for d in detections if d is not None]
    epi_dets = [d for d in dets if d.epi]
    autres = [d for d in dets if not d.epi]

    def rang(d: DetectionEPI) -> int:
        ordre = PRIORITE_MODELE.get(d.epi, (M1_NOM, M2_NOM))
        return ordre.index(d.modele) if d.modele in ordre else len(ordre)

    epi_dets.sort(key=lambda d: (rang(d), -d.conf))

    gardees: list[DetectionEPI] = []
    for d in epi_dets:
        if any(g.epi == d.epi and _iou(g.box, d.box) > iou_seuil for g in gardees):
            continue
        # M2 n'est conservé que là où il apporte quelque chose : soit un concept
        # que M1 ne couvre pas, soit une zone où M1 n'a rien vu (traité ci-dessus).
        gardees.append(d)
    return gardees + autres


if __name__ == "__main__":
    print("Concepts canoniques :", ", ".join(EPI_CANONIQUES))
    print(f"\n{M1_NOM} : {len(M1)} classes -> concepts {sorted({c.epi for c in M1.values() if c.epi})}")
    print(f"{M2_NOM} : {len(M2)} classes -> concepts {sorted({c.epi for c in M2.values() if c.epi})}")
    print(f"{M3_NOM} : {len(M3)} classes -> concepts {sorted({c.epi for c in M3.values() if c.epi})}")
    print(f"{M4_NOM} : {len(M4)} classes -> concepts {sorted({c.epi for c in M4.values() if c.epi})}")
    print(f"{M5_NOM} : {len(M5)} classes -> concepts {sorted({c.epi for c in M5.values() if c.epi})}")
    print(f"{M6_NOM} : {len(M6)} classes -> concepts {sorted({c.epi for c in M6.values() if c.epi})}")
    print(f"\nApport unique de {M2_NOM} : {APPORT_UNIQUE_M2 or 'aucun'}")
    print(f"Classes negatives dans {M2_NOM} : "
          f"{[c for c, v in M2.items() if v.porte is False] or 'aucune'}")
    print("\n=> Si les chaussures de securite ne sont pas au referentiel, "
          f"{M2_NOM} est entierement redondant et peut sortir de la cascade.")
    print("\nPriorite par concept :")
    for concept, ordre in PRIORITE_MODELE.items():
        print(f"  {concept:<12} {' > '.join(ordre)}")
