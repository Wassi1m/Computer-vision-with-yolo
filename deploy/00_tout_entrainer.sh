#!/usr/bin/env bash
# Chaine complete : demarre la VM, prepare, entraine tous les modeles,
# recupere les resultats, puis ARRETE LA VM.
#
#     ./deploy/00_tout_entrainer.sh                 # les 4 modeles
#     ./deploy/00_tout_entrainer.sh epi chute       # une selection
#     ./deploy/00_tout_entrainer.sh --epochs 80 epi
#
# L'instance est arretee en sortie quoi qu'il arrive -- fin normale, erreur ou
# Ctrl-C. C'est le point essentiel : un GPU L4 oublie en marche coute plusieurs
# euros par jour pour rien.
#
# Les entrainements s'enchainent en serie et non en parallele : un seul GPU,
# deux entrainements simultanes se disputeraient la VRAM et seraient tous deux
# ralentis.

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

EPOCHS=60
PROPORTION=0.6
IMGSZ=640
JEUX=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --epochs)     EPOCHS="$2"; shift 2 ;;
        --proportion) PROPORTION="$2"; shift 2 ;;
        --imgsz)      IMGSZ="$2"; shift 2 ;;
        --garder-vm)  export GARDER_VM=1; shift ;;
        -*) echec "option inconnue : $1" ;;
        *)  JEUX+=("$1"); shift ;;
    esac
done
[[ ${#JEUX[@]} -gt 0 ]] || JEUX=(epi chute feu plaque)

DEBUT=$(date +%s)
arreter_en_sortant          # garantit l'arret de la VM en sortie

info "Modeles a entrainer : ${JEUX[*]}  ($EPOCHS epoques, proportion $PROPORTION, imgsz $IMGSZ)"

# ── Phase 0 : deposer les jeux dans le bucket, VM ETEINTE ────────────────────
#
# L'envoi depuis une liaison domestique est long (des dizaines de minutes par
# jeu). Le faire pendant que le GPU tourne revient a payer une L4 a ne rien
# faire : c'est exactement ce qui a coute 98 min de facturation pour un
# transfert qui a fini par casser. On depose donc tout avant d'allumer quoi que
# ce soit -- et comme `televerser_bucket` ignore ce qui est deja en place, cette
# phase est quasi instantanee dans les runs suivants.
if bucket_actif; then
    info "=== Depot des jeux dans le bucket (VM eteinte, aucune facturation) ==="
    for jeu in "${JEUX[@]}"; do
        case "$jeu" in
            epi)    CHEMIN="$RACINE/ppe_detection/data/extracted/ppe_vest_clean_14c" ;;
            chute)  CHEMIN="$RACINE/ppe_detection/data/extracted/fall_detection_enriched" ;;
            feu)    CHEMIN="$RACINE/surveillance_suite/data/dataset/fire_smoke_enriched" ;;
            fumee)  CHEMIN="$RACINE/surveillance_suite/data/dataset/fire_smoke_v9" ;;
            plaque) CHEMIN="$RACINE/surveillance_suite/data/dataset/license_plate_unified" ;;
            *) alerte "jeu inconnu : $jeu"; continue ;;
        esac
        [[ -d "$CHEMIN" ]] || { alerte "$jeu : jeu absent ($CHEMIN)"; continue; }
        NOM="$(basename "$CHEMIN")"
        ARCHIVE="/tmp/${NOM}.tar.gz"
        [[ -f "$ARCHIVE" ]] || { info "Archivage de $NOM"; tar -czhf "$ARCHIVE" -C "$(dirname "$CHEMIN")" "$NOM"; }
        televerser_bucket "$ARCHIVE" "${NOM}.tar.gz"
    done
else
    alerte "GCP_BUCKET non defini dans .env.gcp : les jeux partiront en scp,"
    alerte "sans reprise sur coupure et avec la VM allumee (donc facturee)."
fi

demarrer_vm

info "=== Preparation de la VM ==="
GARDER_VM=1 "$RACINE/deploy/01_preparer_vm.sh" || echec "preparation de la VM echouee"

REUSSIS=(); ECHOUES=()

for jeu in "${JEUX[@]}"; do
    echo
    info "=================================================================="
    info "  $jeu"
    info "=================================================================="

    if ! GARDER_VM=1 "$RACINE/deploy/02_envoyer.sh" "$jeu"; then
        alerte "$jeu : envoi echoue, modele ignore"
        ECHOUES+=("$jeu (envoi)")
        continue
    fi

    if ! GARDER_VM=1 "$RACINE/deploy/03_entrainer.sh" "$jeu" \
            --epochs "$EPOCHS" --proportion "$PROPORTION" --imgsz "$IMGSZ"; then
        alerte "$jeu : lancement echoue, modele ignore"
        ECHOUES+=("$jeu (lancement)")
        continue
    fi

    # Attente de la fin de l'entrainement. On sonde la session tmux plutot que
    # de garder un SSH ouvert : une coupure reseau ne doit pas interrompre
    # l'entrainement ni faire echouer ce script.
    SESSION="train_${jeu}"
    info "Entrainement en cours -- point d'avancement toutes les 5 min"
    DEBUT_RUN=$(date +%s)
    while vm "tmux has-session -t '$SESSION' 2>/dev/null"; do
        sleep 300
        ECOULE=$(( ($(date +%s) - DEBUT_RUN) / 60 ))
        LIGNE=$(vm "csv='$GCP_WORKDIR/sorties/${jeu}_robuste/results.csv'
                    [[ -f \$csv ]] && tail -n1 \$csv | awk -F, '{printf \"epoque %s  mAP50=%.4f\", \$1, \$8}' \
                                   || echo 'demarrage...'" 2>/dev/null)
        echo "    [$jeu] ${ECOULE} min  ${LIGNE}"
    done

    if ! GARDER_VM=1 "$RACINE/deploy/05_recuperer.sh" "$jeu" <<< "o"; then
        alerte "$jeu : recuperation echouee"
        ECHOUES+=("$jeu (recuperation)")
        continue
    fi
    REUSSIS+=("$jeu")
    succes "$jeu termine"
done

echo
info "=================================================================="
DUREE=$(( ($(date +%s) - DEBUT) / 60 ))
echo "  Duree totale : ${DUREE} min"
[[ ${#REUSSIS[@]} -gt 0 ]] && echo "  Reussis : ${REUSSIS[*]}"
[[ ${#ECHOUES[@]} -gt 0 ]] && echo "  Echoues : ${ECHOUES[*]}"
info "=================================================================="

cat <<EOF

Les modeles sont dans reports/v3_results/<jeu>_robuste/ et ne sont PAS installes.
Pour les valider et les installer :

    ./deploy/06_valider_installer.sh ${REUSSIS[*]:-<jeu>}

L'instance va maintenant etre arretee.
EOF
