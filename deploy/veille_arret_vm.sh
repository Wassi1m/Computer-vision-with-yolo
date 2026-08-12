#!/usr/bin/env bash
# Veille d'arret autonome -- s'execute SUR LA VM, pas sur le poste local.
#
# Jusqu'ici, l'arret de l'instance dependait entierement du script
# `00_tout_entrainer.sh` tournant sur le poste de l'utilisateur. Ce couplage a
# deux defauts, tous deux constates :
#
#   - poste eteint, en veille ou hors reseau -> plus personne n'arrete la VM,
#     qui reste facturee. Un dataset envoye sans entrainement lance a ainsi
#     laisse le GPU tourner ~4 h pour rien.
#   - coupure reseau breve -> le `ssh` de surveillance echoue, le script croit
#     l'entrainement termine et arrete la VM, tuant un run de plusieurs heures.
#
# La VM doit donc pouvoir se couper seule. Un `poweroff` invite fait passer
# l'instance GCE en TERMINATED : la facturation vCPU/GPU/RAM s'arrete (seul le
# disque persistant reste factures, quelques centimes par jour).
#
# Cette veille n'est pas le chemin nominal : quand le poste local est vivant, il
# recupere les resultats et arrete la VM lui-meme, bien avant l'expiration du
# delai de grace. C'est un filet, et il ne doit se declencher que si le chemin
# nominal a echoue.
#
#     ./veille_arret_vm.sh train_fumee            # dans un tmux detache
#     ./veille_arret_vm.sh train_fumee 45 24      # grace 45 min, plafond 24 h

set -uo pipefail

SESSION="${1:?usage: $0 <session_tmux> [grace_min] [plafond_h] [attente_max_min]}"
GRACE_MIN="${2:-45}"      # laisse au poste local le temps de rapatrier les poids
PLAFOND_H="${3:-24}"      # arret inconditionnel : protege d'un entrainement fige
ATTENTE_MAX_MIN="${4:-120}"  # delai laisse a l'entrainement pour demarrer

journal() { printf '%s  %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"; }

demarrage=$(date +%s)
depasse_plafond() {
    (( ($(date +%s) - demarrage) > PLAFOND_H * 3600 ))
}

arreter() {
    journal "ARRET de la VM : $1"
    sync
    sudo poweroff
    exit 0
}

journal "veille demarree (session=$SESSION grace=${GRACE_MIN}min plafond=${PLAFOND_H}h)"

# ── 1. Attendre que l'entrainement demarre ───────────────────────────────────
# Si aucune session n'apparait, c'est que le lancement a echoue ou n'a jamais eu
# lieu : laisser la VM allumee dans ce cas est du gaspillage pur.
journal "attente du demarrage de '$SESSION' (max ${ATTENTE_MAX_MIN} min)"
attente=0
while ! tmux has-session -t "$SESSION" 2>/dev/null; do
    sleep 60
    (( attente++ ))
    if (( attente >= ATTENTE_MAX_MIN )); then
        arreter "aucun entrainement lance apres ${ATTENTE_MAX_MIN} min"
    fi
    depasse_plafond && arreter "plafond de ${PLAFOND_H} h atteint"
done
journal "entrainement detecte, surveillance en cours"

# ── 2. Attendre la fin de l'entrainement ─────────────────────────────────────
while tmux has-session -t "$SESSION" 2>/dev/null; do
    sleep 120
    depasse_plafond && arreter "plafond de ${PLAFOND_H} h atteint (entrainement fige ?)"
done
journal "entrainement termine"

# ── 3. Delai de grace pour la recuperation par le poste local ────────────────
# Si le poste est vivant, il arrete la VM pendant ce delai et cette veille
# disparait avec la machine. Sinon, on coupe.
journal "delai de grace de ${GRACE_MIN} min avant arret"
for ((i = 0; i < GRACE_MIN; i++)); do
    sleep 60
    # Un nouvel entrainement relance pendant la grace annule l'arret : on
    # repart en surveillance plutot que de couper sous les pieds de l'operateur.
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        journal "nouvel entrainement detecte, veille relancee"
        exec "$0" "$SESSION" "$GRACE_MIN" "$PLAFOND_H" "$ATTENTE_MAX_MIN"
    fi
done

arreter "entrainement termine et delai de grace ecoule"
