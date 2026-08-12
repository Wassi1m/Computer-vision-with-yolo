#!/usr/bin/env bash
# Envoie sur la VM un jeu de donnees, les poids de depart et les scripts.
#
#     ./deploy/02_envoyer.sh epi          # ppe_vest_clean_14c  + best.pt
#     ./deploy/02_envoyer.sh chute        # fall_detection_...  + fall_detector.pt
#     ./deploy/02_envoyer.sh feu          # fire_smoke_enriched + fire_smoke.pt
#     ./deploy/02_envoyer.sh plaque       # license_plate_...   + license_plate.pt
#
# Les images sont transferees en archive compressee plutot qu'une a une : des
# milliers de petits fichiers en scp passent l'essentiel du temps en aller-retour
# reseau, pas en transfert utile.

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

# Demarre l'instance si besoin et l'arrete en sortant : elle est facturee
# a l'heure. GARDER_VM=1 laisse la main a l'appelant (orchestrateur).
arreter_en_sortant

CIBLE_JEU="${1:-}"
[[ -n "$CIBLE_JEU" ]] || echec "usage : $0 <epi|chute|feu|fumee|plaque>"

case "$CIBLE_JEU" in
    epi)
        DATASET="$RACINE/ppe_detection/data/extracted/ppe_vest_clean_14c"
        POIDS="$RACINE/ppe_detection/models/best.pt" ;;
    chute)
        DATASET="$RACINE/ppe_detection/data/extracted/fall_detection_enriched"
        POIDS="$RACINE/surveillance_suite/models/fall_detector.pt" ;;
    feu)
        DATASET="$RACINE/surveillance_suite/data/dataset/fire_smoke_enriched"
        POIDS="$RACINE/surveillance_suite/models/fire_smoke.pt" ;;
    fumee)
        # P9 : meme domaine que `feu`, mais sur le jeu reconstruit a 3 classes
        # (fire / smoke / smoke_distant). Voir improvements/p9_dataset_fumee.py
        # pour la raison de la separation.
        DATASET="$RACINE/surveillance_suite/data/dataset/fire_smoke_v9"
        POIDS="$RACINE/surveillance_suite/models/fire_smoke.pt" ;;
    plaque)
        DATASET="$RACINE/surveillance_suite/data/dataset/license_plate_unified"
        POIDS="$RACINE/surveillance_suite/models/license_plate.pt" ;;
    *)  echec "jeu inconnu : $CIBLE_JEU (attendus : epi, chute, feu, fumee, plaque)" ;;
esac

[[ -d "$DATASET" ]] || echec "jeu de donnees absent : $DATASET"
[[ -f "$POIDS"   ]] || echec "poids absents : $POIDS"

verifier_connexion
vm "mkdir -p '$GCP_WORKDIR'/{donnees,modeles,scripts,sorties}"

NOM_JEU="$(basename "$DATASET")"

if vm "[[ -d '$GCP_WORKDIR/donnees/$NOM_JEU' ]]"; then
    alerte "$NOM_JEU deja present sur la VM, transfert des images ignore"
    alerte "(supprimer '$GCP_WORKDIR/donnees/$NOM_JEU' sur la VM pour forcer)"
else
    ARCHIVE="/tmp/${NOM_JEU}.tar.gz"
    if [[ ! -f "$ARCHIVE" ]]; then
        info "Archivage de $NOM_JEU"
        # -h : suit les liens symboliques. Les jeux fusionnes (fire_smoke_enriched,
        # fall_detection_enriched) sont construits par symlink pour ne pas dupliquer
        # les images sur le disque local ; sans -h, l'archive ne contiendrait que
        # des liens brises.
        tar -czhf "$ARCHIVE" -C "$(dirname "$DATASET")" "$NOM_JEU"
    else
        info "Archive $ARCHIVE deja presente, reutilisee"
    fi

    if bucket_actif; then
        # Chemin recommande : l'archive transite par le bucket, ce qui rend le
        # transfert reprenable et permet de l'envoyer VM eteinte (voir lib.sh).
        televerser_bucket "$ARCHIVE" "${NOM_JEU}.tar.gz"
        vm_recuperer_bucket "${NOM_JEU}.tar.gz" "$GCP_WORKDIR/donnees"
    else
        alerte "GCP_BUCKET non defini : envoi direct en scp, sans reprise possible"
        alerte "(une coupure reseau fait tout recommencer -- voir lib.sh)"
        info "Envoi ($(du -h "$ARCHIVE" | cut -f1)) -- cela peut prendre plusieurs minutes"
        vers_vm "$ARCHIVE" "$CIBLE:$GCP_WORKDIR/donnees/"
    fi

    vm "cd '$GCP_WORKDIR/donnees' && tar -xzf '${NOM_JEU}.tar.gz' && rm '${NOM_JEU}.tar.gz'"
    rm -f "$ARCHIVE"

    # Le data.yaml porte un chemin absolu propre au poste local : il ne veut
    # rien dire sur la VM. On le reecrit.
    vm "sed -i '1s|^path:.*|path: $GCP_WORKDIR/donnees/$NOM_JEU|' '$GCP_WORKDIR/donnees/$NOM_JEU/data.yaml'"
    succes "$NOM_JEU installe"
fi

info "Envoi des poids $(basename "$POIDS")"
vers_vm "$POIDS" "$CIBLE:$GCP_WORKDIR/modeles/"

info "Envoi des scripts d'entrainement"
vers_vm "$RACINE/improvements/p8_dataset_nuit.py" "$RACINE/improvements/p8_train_nuit.py" \
        "$CIBLE:$GCP_WORKDIR/scripts/"

vm "cat '$GCP_WORKDIR/donnees/$NOM_JEU/data.yaml'" | sed 's/^/  /'
succes "envoi termine"
echo
echo "Suite :  ./deploy/03_entrainer.sh $CIBLE_JEU"
