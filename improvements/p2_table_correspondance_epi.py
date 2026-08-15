#!/usr/bin/env python3
"""P2 — Table de correspondance de classes entre `ppe_detector.pt` et `ppe_complement.pt`.

Alternative retenue à "ré-entraîner un seul modèle sur l'union des deux
taxonomies" (l'option recommandée par `v2_plan_amelioration.md`) : celle-ci
est **impossible en l'état**, GPU ou pas -- il n'existe aucun dataset local
pour les classes propres à `ppe_complement.pt`. `safety_shoe` en particulier
n'apparaît dans aucune image annotée disponible sur cette machine ; l'union
des taxonomies ne peut donc pas être apprise, seulement supposée.

Cette table permet de fusionner les détections des deux modèles par IoU sans
double-compter les concepts qu'ils partagent sous des noms différents
(`Hardhat`/`helmet`, `Safety Vest`/`Vest`, `Goggles`/`goggles`, `Mask`/`mask`,
`Gloves`/`Gloves`) -- correction du bug documenté en v1 (IndexError lors de la
validation croisée, dû à cette incohérence non résolue.

Seule `safety_shoe` reste un apport net et sans ambiguïté de `ppe_complement.pt` :
aucun équivalent, aucun conflit possible.
"""

CORRESPONDANCE = {
    # classe ppe_complement.pt -> classe ppe_detector.pt équivalente (None si aucun équivalent)
    "Gloves":      "Gloves",
    "Vest":        "Safety Vest",
    "goggles":     "Goggles",
    "helmet":      "Hardhat",
    "mask":        "Mask",
    "safety_shoe": None,  # seul apport net, pas de classe "chaussure" dans ppe_detector.pt
}

# Classes de ppe_detector.pt qui n'ont pas de contrepartie dans ppe_complement.pt : à ne
# détecter que via ppe_detector.pt (pas de choix à faire, pas de fusion nécessaire).
SANS_EQUIVALENT_BEST = [
    "Fall-Detected", "Ladder", "NO-Gloves", "NO-Goggles", "NO-Hardhat",
    "NO-Mask", "NO-Safety Vest", "Person", "Safety Cone",
]


def fusionner_detections(dets_best: list[dict], dets_gloves: list[dict], iou_seuil: float = 0.5) -> list[dict]:
    """Fusionne deux listes de détections {"classe": str, "box": (x1,y1,x2,y2), "conf": float}.

    Règle : si une détection `ppe_complement` correspond (même concept + IoU >= seuil)
    à une détection `best`, on garde celle de `best` (modèle de référence, mieux
    mesuré) et on ignore le doublon. Sinon on l'ajoute -- c'est le cas de
    `safety_shoe`, systématiquement conservée.
    """
    def iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        aire_a = (ax2 - ax1) * (ay2 - ay1)
        aire_b = (bx2 - bx1) * (by2 - by1)
        union = aire_a + aire_b - inter
        return inter / union if union > 0 else 0.0

    resultat = list(dets_best)
    for dg in dets_gloves:
        equiv = CORRESPONDANCE.get(dg["classe"])
        if equiv is None:
            resultat.append(dg)  # safety_shoe : apport net, toujours conservée
            continue
        doublon = any(db["classe"] == equiv and iou(db["box"], dg["box"]) >= iou_seuil for db in dets_best)
        if not doublon:
            resultat.append({**dg, "classe": equiv})
    return resultat


if __name__ == "__main__":
    print("Correspondance ppe_complement.pt -> ppe_detector.pt :")
    for src, dst in CORRESPONDANCE.items():
        print(f"  {src:14} -> {dst or '(aucun équivalent -- apport net)'}")
