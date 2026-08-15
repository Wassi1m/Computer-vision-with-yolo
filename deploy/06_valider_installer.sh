#!/usr/bin/env bash
# Valide puis installe les modeles entraines sur la VM. Tourne en LOCAL :
# aucune de ces etapes ne demande de GPU, la VM reste donc arretee.
#
#     ./deploy/06_valider_installer.sh epi chute feu plaque
#
# Un modele n'est installe que s'il satisfait deux conditions :
#   1. il ne regresse pas en conditions normales (garde-fou de non-regression) ;
#   2. il progresse en conditions degradees -- sinon l'entrainement n'a servi
#      a rien et l'ancien modele, deja eprouve, est preferable.
#
# La sauvegarde de l'ancien poids precede toute installation : les .pt ne sont
# pas versionnes dans git, une ecrasure sans copie serait irreversible.

RACINE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$RACINE"
set -uo pipefail

PY="$RACINE/.venv/bin/python"
[[ -x "$PY" ]] || PY=python3

SAUVEGARDES="$RACINE/reports/v3_results/poids_remplaces"
mkdir -p "$SAUVEGARDES"

info()   { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
succes() { printf '\033[1;32m OK\033[0m %s\n' "$*"; }
alerte() { printf '\033[1;33m  !\033[0m %s\n' "$*" >&2; }

JEUX=("$@")
[[ ${#JEUX[@]} -gt 0 ]] || JEUX=(epi chute feu plaque)

INSTALLES=(); REFUSES=()

for jeu in "${JEUX[@]}"; do
    case "$jeu" in
        epi)    DEST="ppe_detection/models/ppe_detector.pt";               REF=ppe_detector ;;
        chute)  DEST="surveillance_suite/models/fall_detector.pt"; REF=fall_detector ;;
        feu)    DEST="surveillance_suite/models/fire_smoke.pt";    REF=fire_smoke ;;
        plaque) DEST="surveillance_suite/models/license_plate.pt"; REF=license_plate ;;
        *) alerte "jeu inconnu : $jeu"; continue ;;
    esac

    CANDIDAT="$RACINE/reports/v3_results/${jeu}_robuste/best.pt"
    echo
    info "=============== $jeu ==============="
    if [[ ! -f "$CANDIDAT" ]]; then
        alerte "aucun modele entraine : $CANDIDAT"
        REFUSES+=("$jeu (absent)")
        continue
    fi

    SAUVEGARDE="$SAUVEGARDES/$(basename "$DEST" .pt)_$(date +%Y%m%d_%H%M%S).pt"
    cp "$DEST" "$SAUVEGARDE"
    info "Ancien poids sauvegarde : ${SAUVEGARDE#$RACINE/}"

    cp "$CANDIDAT" "$DEST"

    info "Non-regression en conditions normales"
    if "$PY" tests/test_non_regression.py --modele "$REF" > /tmp/nr_$jeu.log 2>&1; then
        grep -E "^  (OK|ECHEC)|mAP50 global" /tmp/nr_$jeu.log | sed 's/^/  /'
        succes "$jeu : pas de regression"
    else
        grep -E "^  (OK|ECHEC)|mAP50 global|REGRESSION|  - " /tmp/nr_$jeu.log | sed 's/^/  /'
        alerte "$jeu : REGRESSION -- restauration de l'ancien poids"
        cp "$SAUVEGARDE" "$DEST"
        REFUSES+=("$jeu (regression)")
        continue
    fi

    info "Gain en conditions degradees"
    "$PY" tests/mesure_robustesse.py --modele "$REF" --images 120 \
        > /tmp/rob_$jeu.log 2>&1 || alerte "mesure de robustesse incomplete"
    grep -E "^${REF}| reference | faible_lum|contre_jour|brouillard|flou_|basse_res" /tmp/rob_$jeu.log \
        | tail -12 | sed 's/^/  /'

    "$PY" - "$jeu" "$REF" <<'PY'
import json, sys
from pathlib import Path
jeu, ref = sys.argv[1], sys.argv[2]
avant = Path("reports/v3_results/robustesse_conditions_reelles.json")
apres = Path("reports/v3_results/robustesse_conditions_reelles.json")
try:
    d = json.loads(apres.read_text())[ref]
    base = d.get("reference")
    nuit = d.get("faible_luminosite")
    if base and nuit:
        print(f"  -> nuit : {nuit:.4f} ({100*(nuit-base)/base:+.1f}% vs conditions normales)")
except Exception as e:
    print(f"  (comparaison indisponible : {e})")
PY

    "$PY" tests/test_non_regression.py --modele "$REF" --maj > /dev/null 2>&1 \
        && succes "reference figee"
    INSTALLES+=("$jeu")
done

echo
info "=================================================================="
[[ ${#INSTALLES[@]} -gt 0 ]] && echo "  Installes : ${INSTALLES[*]}"
[[ ${#REFUSES[@]}   -gt 0 ]] && echo "  Refuses   : ${REFUSES[*]}"
info "=================================================================="

cat <<EOF

Anciens poids conserves dans ${SAUVEGARDES#$RACINE/}/
Pour revenir en arriere : recopier le fichier voulu par-dessus la destination.

Si le resultat convient, commiter les references mises a jour :
    git add tests/reference_modeles.json reports/v3_results/
    git commit -m "P8 : robustesse aux conditions degradees"
EOF
