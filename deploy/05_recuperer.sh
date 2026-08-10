#!/usr/bin/env bash
# Recupere les poids et les mesures d'un entrainement termine.
#
#     ./deploy/05_recuperer.sh epi
#
# Le modele est depose dans reports/v3_results/ et n'est PAS installe
# automatiquement : il doit d'abord passer le garde-fou de non-regression.
# Installer un modele sans cette verification est exactement ce que le
# dispositif existe pour empecher.

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

# Demarre l'instance si besoin et l'arrete en sortant : elle est facturee
# a l'heure. GARDER_VM=1 laisse la main a l'appelant (orchestrateur).
arreter_en_sortant

CIBLE_JEU="${1:-}"
[[ -n "$CIBLE_JEU" ]] || echec "usage : $0 <epi|chute|feu|plaque>"

case "$CIBLE_JEU" in
    epi)    DESTINATION="ppe_detection/models/best.pt";                 REF_MODELE=ppe_best ;;
    chute)  DESTINATION="surveillance_suite/models/fall_detector.pt";   REF_MODELE=fall_detector ;;
    feu)    DESTINATION="surveillance_suite/models/fire_smoke.pt";      REF_MODELE=fire_smoke ;;
    plaque) DESTINATION="surveillance_suite/models/license_plate.pt";   REF_MODELE=license_plate ;;
    *) echec "jeu inconnu : $CIBLE_JEU" ;;
esac

NOM_RUN="${CIBLE_JEU}_robuste"
SESSION="train_${CIBLE_JEU}"
DOSSIER="$GCP_WORKDIR/sorties/$NOM_RUN"

verifier_connexion

if vm "tmux has-session -t '$SESSION' 2>/dev/null"; then
    alerte "l'entrainement tourne encore : les poids recuperes seront intermediaires"
    read -rp "Continuer quand meme ? [o/N] " reponse
    [[ "$reponse" == "o" ]] || exit 0
fi

vm "[[ -f '$DOSSIER/weights/best.pt' ]]" || echec "aucun poids dans $DOSSIER/weights/"

LOCAL="$RACINE/reports/v3_results/$NOM_RUN"
mkdir -p "$LOCAL"

info "Recuperation des poids et des mesures"
vers_vm "$CIBLE:$DOSSIER/weights/best.pt" "$LOCAL/best.pt"
for f in resultats_p8.json results.csv results.png BoxPR_curve.png confusion_matrix.png args.yaml; do
    scp "${OPTS_SSH[@]}" -q "$CIBLE:$DOSSIER/$f" "$LOCAL/" 2>/dev/null || true
done
scp "${OPTS_SSH[@]}" -q "$CIBLE:$GCP_WORKDIR/sorties/${NOM_RUN}.log" "$LOCAL/" 2>/dev/null || true

succes "recupere dans reports/v3_results/$NOM_RUN/"

if [[ -f "$LOCAL/resultats_p8.json" ]]; then
    echo
    info "Mesures de l'entrainement"
    python3 - "$LOCAL/resultats_p8.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
for phase in ("avant", "apres"):
    if phase not in d:
        continue
    print(f"  {phase} :")
    for contexte, m in d[phase].items():
        print(f"    {contexte:20} mAP@50 = {m['mAP50']:.4f}  {m.get('classes', {})}")
a = d.get("avant", {}).get("conditions_normales", {}).get("mAP50")
b = d.get("apres", {}).get("conditions_normales", {}).get("mAP50")
if a is not None and b is not None:
    print(f"\n  Conditions normales : {a:.4f} -> {b:.4f} ({b - a:+.4f})")
    if b - a < -0.02:
        print("  ATTENTION : le cas nominal s'est degrade de plus de 2 points.")
PY
fi

cat <<EOF

Le modele n'est pas installe : il doit d'abord etre valide.

  1. Mesurer la robustesse gagnee (conditions degradees) :
       cp "reports/v3_results/$NOM_RUN/best.pt" /tmp/candidat.pt
       # puis comparer a reports/v3_results/robustesse_conditions_reelles.json

  2. Sauvegarder le modele actuel, puis installer le candidat :
       cp "$DESTINATION" "/tmp/\$(basename "$DESTINATION" .pt)_avant_p8.pt"
       cp "reports/v3_results/$NOM_RUN/best.pt" "$DESTINATION"

  3. Verifier la non-regression en conditions normales :
       python tests/test_non_regression.py --modele $REF_MODELE

     Echec (code 1) : restaurer la sauvegarde de l'etape 2.
     Succes         : figer la nouvelle reference, puis commiter.
       python tests/test_non_regression.py --modele $REF_MODELE --maj
EOF
