#!/usr/bin/env bash
# Suit l'avancement d'un entrainement en cours sur la VM.
#
#     ./deploy/04_suivi.sh epi            # etat instantane
#     ./deploy/04_suivi.sh epi --continu  # rafraichi toutes les 60 s
#     ./deploy/04_suivi.sh epi --journal  # deroule le journal en direct
#
# L'etat instantane est concu pour repondre en un coup d'oeil aux trois seules
# questions qui comptent pendant un entrainement : est-ce que ca tourne encore,
# ou en est-on, et est-ce que ca progresse.

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

CIBLE_JEU="${1:-}"
[[ -n "$CIBLE_JEU" ]] || echec "usage : $0 <epi|chute|feu|plaque> [--continu|--journal]"
MODE="${2:-}"

NOM_RUN="${CIBLE_JEU}_robuste"
SESSION="train_${CIBLE_JEU}"
DOSSIER="$GCP_WORKDIR/sorties/$NOM_RUN"
JOURNAL="$GCP_WORKDIR/sorties/${NOM_RUN}.log"

# Script de consultation : il ne demarre JAMAIS l'instance. Une VM arretee
# signifie qu'aucun entrainement ne tourne -- la demarrer pour « suivre » un
# entrainement inexistant ne ferait que relancer la facturation sans raison.
ETAT="$(etat_vm)"
if [[ "$ETAT" != "RUNNING" ]]; then
    printf '\033[1;33mInstance %s\033[0m -- aucun entrainement en cours.\n\n' "$ETAT"
    echo "Pour lancer un entrainement (demarre l'instance automatiquement) :"
    echo "    ./deploy/03_entrainer.sh $CIBLE_JEU"
    echo "    ./deploy/00_tout_entrainer.sh"
    echo
    echo "Resultats deja recuperes en local :"
    ls -1 "$RACINE/reports/v3_results/${NOM_RUN}/" 2>/dev/null | sed 's/^/    /' \
        || echo "    (aucun)"
    exit 0
fi

vm "echo pret" >/dev/null 2>&1 || echec "instance RUNNING mais SSH injoignable (IP changee ?)"

if [[ "$MODE" == "--journal" ]]; then
    info "Journal en direct (Ctrl-C pour quitter)"
    exec ssh "${OPTS_SSH[@]}" "$CIBLE" "tail -f '$JOURNAL'"
fi

etat() {
    echo "──────────────────────────────────────────────────────────────"
    date '+%H:%M:%S'

    if vm "tmux has-session -t '$SESSION' 2>/dev/null"; then
        printf '\033[1;32mEN COURS\033[0m  session %s\n' "$SESSION"
    else
        printf '\033[1;33mARRETE\033[0m  (termine, ou interrompu)\n'
    fi

    vm "
        if command -v nvidia-smi >/dev/null 2>&1; then
            nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu \
                       --format=csv,noheader,nounits 2>/dev/null \
            | awk -F', ' '{printf \"GPU       %s%% util, %.1f/%.1f Go VRAM, %s C\n\", \$1, \$2/1024, \$3/1024, \$4}'
        fi

        csv='$DOSSIER/results.csv'
        if [[ -f \"\$csv\" ]]; then
            total=\$(grep -c '' \"\$csv\")
            echo \"Epoques   \$((total - 1)) terminees\"
            echo
            head -1 \"\$csv\" | awk -F, '{printf \"  %-7s %-10s %-10s %-10s\n\", \"epoque\", \"mAP50\", \"mAP50-95\", \"box_loss\"}'
            tail -n 3 \"\$csv\" | awk -F, '{printf \"  %-7s %-10.4f %-10.4f %-10.4f\n\", \$1, \$8, \$9, \$3}'
        else
            echo 'Epoques   pas encore de results.csv (chargement du modele / premiere epoque)'
            tail -n 4 '$JOURNAL' 2>/dev/null | sed 's/^/  /'
        fi
    " 2>/dev/null || alerte "lecture de l'etat impossible"
}

if [[ "$MODE" == "--continu" ]]; then
    info "Rafraichissement toutes les 60 s (Ctrl-C pour quitter)"
    while true; do
        etat
        vm "tmux has-session -t '$SESSION' 2>/dev/null" || {
            echo; succes "entrainement termine"
            echo "Recuperation :  ./deploy/05_recuperer.sh $CIBLE_JEU"
            break
        }
        sleep 60
    done
else
    etat
    echo "──────────────────────────────────────────────────────────────"
    echo "Suivi continu :  $0 $CIBLE_JEU --continu"
    echo "Journal direct:  $0 $CIBLE_JEU --journal"
fi
