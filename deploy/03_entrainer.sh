#!/usr/bin/env bash
# Lance l'entrainement de robustesse sur la VM, dans une session tmux detachee.
#
#     ./deploy/03_entrainer.sh epi
#     ./deploy/03_entrainer.sh epi --epochs 80 --proportion 0.8
#     ./deploy/03_entrainer.sh epi --reprendre     # apres une interruption
#
# tmux plutot que nohup : la session reste attachable depuis la VM, ce qui
# permet de voir l'entrainement en direct et de l'interrompre proprement.
# L'entrainement survit a la fermeture de la connexion SSH dans les deux cas.

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

# Demarre l'instance si besoin et l'arrete en sortant : elle est facturee
# a l'heure. GARDER_VM=1 laisse la main a l'appelant (orchestrateur).
arreter_en_sortant

CIBLE_JEU="${1:-}"
[[ -n "$CIBLE_JEU" ]] || echec "usage : $0 <epi|chute|feu|fumee|plaque> [options]"
shift

EPOCHS=60
PROPORTION=0.6
# 640 par defaut, comme les modeles deja en place. A augmenter pour les objets
# petits : la fumee lointaine mesure 19x16 px a 640, contre 27x22 px a 896 --
# c'est la difference entre un objet a la limite du detectable et un objet
# confortablement couvert par la tete P3.
IMGSZ=640
REPRENDRE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --epochs)     EPOCHS="$2"; shift 2 ;;
        --proportion) PROPORTION="$2"; shift 2 ;;
        --imgsz)      IMGSZ="$2"; shift 2 ;;
        --reprendre)  REPRENDRE="--reprendre"; shift ;;
        *) echec "option inconnue : $1" ;;
    esac
done

case "$CIBLE_JEU" in
    epi)    JEU=ppe_vest_clean_14c;      POIDS=best.pt ;;
    chute)  JEU=fall_detection_enriched; POIDS=fall_detector.pt ;;
    feu)    JEU=fire_smoke_enriched;     POIDS=fire_smoke.pt ;;
    fumee)  JEU=fire_smoke_v9;           POIDS=fire_smoke.pt ;;
    plaque) JEU=license_plate_unified;   POIDS=license_plate.pt ;;
    *) echec "jeu inconnu : $CIBLE_JEU" ;;
esac

NOM_RUN="${CIBLE_JEU}_robuste"
SESSION="train_${CIBLE_JEU}"

verifier_connexion

vm "[[ -d '$GCP_WORKDIR/donnees/$JEU' ]]" \
    || echec "$JEU absent de la VM. Lancer d'abord : ./deploy/02_envoyer.sh $CIBLE_JEU"
vm "[[ -f '$GCP_WORKDIR/modeles/$POIDS' ]]" \
    || echec "$POIDS absent de la VM. Lancer d'abord : ./deploy/02_envoyer.sh $CIBLE_JEU"

if vm "tmux has-session -t '$SESSION' 2>/dev/null"; then
    echec "une session '$SESSION' tourne deja sur la VM.
  Suivre    : ./deploy/04_suivi.sh $CIBLE_JEU
  Interrompre : ssh ... 'tmux kill-session -t $SESSION'"
fi

JEU_ROBUSTE="$GCP_WORKDIR/donnees/${JEU}_robuste"

if [[ -z "$REPRENDRE" ]]; then
    info "Construction du jeu enrichi (images degradees)"
    vm "cd '$GCP_WORKDIR' && ./.venv/bin/python scripts/p8_dataset_nuit.py \
            --source 'donnees/$JEU' --proportion $PROPORTION" \
        || echec "construction du jeu echouee"
else
    info "Reprise : le jeu enrichi existant est reutilise"
fi

info "Lancement de l'entrainement (session tmux '$SESSION')"
JOURNAL="$GCP_WORKDIR/sorties/${NOM_RUN}.log"

vm "cd '$GCP_WORKDIR' && tmux new-session -d -s '$SESSION' \
    \"./.venv/bin/python -u scripts/p8_train_nuit.py \
        --donnees '$JEU_ROBUSTE' \
        --poids 'modeles/$POIDS' \
        --nom '$NOM_RUN' \
        --epochs $EPOCHS \
        --imgsz $IMGSZ \
        --batch ${GCP_BATCH:-32} \
        --workers ${GCP_WORKERS:-4} \
        --reference '$GCP_WORKDIR/donnees/$JEU/data.yaml' \
        --sorties '$GCP_WORKDIR/sorties' \
        $REPRENDRE 2>&1 | tee '$JOURNAL'\""

sleep 8
if vm "tmux has-session -t '$SESSION' 2>/dev/null"; then
    succes "entrainement demarre"
    vm "tail -n 15 '$JOURNAL' 2>/dev/null" | sed 's/^/  /' || true
else
    alerte "la session s'est terminee immediatement -- voici le journal :"
    vm "tail -n 40 '$JOURNAL' 2>/dev/null" | sed 's/^/  /'
    echec "demarrage echoue"
fi

echo
echo "Suivi       :  ./deploy/04_suivi.sh $CIBLE_JEU"
echo "Recuperation:  ./deploy/05_recuperer.sh $CIBLE_JEU   (une fois termine)"
