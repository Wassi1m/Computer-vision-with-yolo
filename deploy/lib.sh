
#!/usr/bin/env bash
# Fonctions communes aux scripts de deploiement GCP.
# A sourcer, pas a executer directement.

set -euo pipefail

RACINE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FICHIER="$RACINE/.env.gcp"

if [[ ! -f "$ENV_FICHIER" ]]; then
    echo "Configuration absente : $ENV_FICHIER" >&2
    echo "La creer a partir du modele :  cp .env.gcp.exemple .env.gcp" >&2
    exit 1
fi

# shellcheck disable=SC1090
set -a; source "$ENV_FICHIER"; set +a
GCP_SSH_KEY="${GCP_SSH_KEY/#\$HOME/$HOME}"
GCP_SSH_KEY="${GCP_SSH_KEY/#\~/$HOME}"

CIBLE="${GCP_USER}@${GCP_HOST}"

# BatchMode : echoue franchement plutot que d'attendre une saisie interactive,
# ce qui bloquerait un script lance sans terminal.
OPTS_SSH=(-i "$GCP_SSH_KEY" -o BatchMode=yes -o StrictHostKeyChecking=no
          -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR
          -o ServerAliveInterval=30 -o ConnectTimeout=15)

vm() { ssh "${OPTS_SSH[@]}" "$CIBLE" "$@"; }
vers_vm() { scp "${OPTS_SSH[@]}" -C "$@"; }

info()   { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
succes() { printf '\033[1;32m OK\033[0m %s\n' "$*"; }
alerte() { printf '\033[1;33m  !\033[0m %s\n' "$*" >&2; }
echec()  { printf '\033[1;31mECHEC\033[0m %s\n' "$*" >&2; exit 1; }

# ── Cycle de vie de l'instance ───────────────────────────────────────────────
#
# La VM est facturee tant qu'elle tourne (GPU L4 : plusieurs dizaines de
# centimes par heure). Elle ne doit donc jamais rester allumee sans raison.
# Chaque script la demarre au besoin, et l'arrete en sortant -- y compris en cas
# d'erreur, d'ou le piege sur EXIT installe par `arreter_en_sortant`.
#
# Arreter (et non supprimer) conserve le disque : pilotes, environnement Python
# et jeux de donnees deja transferes survivent d'une session a l'autre.

GCLOUD="${GCLOUD/#\$HOME/$HOME}"
GCLOUD="${GCLOUD:-gcloud}"
GC=("$GCLOUD" compute instances --project "$GCP_PROJET" --zone "$GCP_ZONE")

etat_vm() {
    "$GCLOUD" compute instances describe "$GCP_INSTANCE" \
        --project "$GCP_PROJET" --zone "$GCP_ZONE" \
        --format='value(status)' 2>/dev/null || echo INCONNU
}

ip_externe_vm() {
    "$GCLOUD" compute instances describe "$GCP_INSTANCE" \
        --project "$GCP_PROJET" --zone "$GCP_ZONE" \
        --format='value(networkInterfaces[0].accessConfigs[0].natIP)' 2>/dev/null
}

# Met a jour GCP_HOST dans .env.gcp : l'IP externe est ephemere et change a
# chaque demarrage, une valeur perimee donnerait un echec de connexion obscur.
memoriser_ip() {
    local ip="$1"
    GCP_HOST="$ip"
    CIBLE="${GCP_USER}@${GCP_HOST}"
    sed -i "s|^GCP_HOST=.*|GCP_HOST=$ip|" "$ENV_FICHIER"
}

# Nombre de tentatives de demarrage et attente entre deux essais. Les GPU sont
# une ressource rare : une zone peut etre temporairement saturee
# (ZONE_RESOURCE_POOL_EXHAUSTED) alors que la capacite revient quelques minutes
# plus tard. Echouer au premier refus obligerait a relancer la chaine a la main.
DEMARRAGE_TENTATIVES="${DEMARRAGE_TENTATIVES:-12}"
DEMARRAGE_ATTENTE_S="${DEMARRAGE_ATTENTE_S:-300}"

demarrer_vm() {
    local etat; etat="$(etat_vm)"
    case "$etat" in
        RUNNING) ;;
        TERMINATED|SUSPENDED|STOPPED)
            local sortie tentative=1
            while true; do
                info "Instance $etat -- demarrage (tentative $tentative/$DEMARRAGE_TENTATIVES)"
                if sortie=$("$GCLOUD" compute instances start "$GCP_INSTANCE" \
                        --project "$GCP_PROJET" --zone "$GCP_ZONE" --quiet 2>&1); then
                    break
                fi
                if ! grep -q "ZONE_RESOURCE_POOL_EXHAUSTED\|STOCKOUT" <<<"$sortie"; then
                    echo "$sortie" >&2
                    echec "demarrage de l'instance impossible"
                fi
                if (( tentative >= DEMARRAGE_TENTATIVES )); then
                    echec "aucun GPU disponible dans $GCP_ZONE apres $tentative tentatives.
  Google n'a plus de g2-standard-4 + NVIDIA L4 libre dans cette zone.
  Options :
   - patienter : la capacite se libere en general en quelques dizaines de minutes ;
   - relancer avec une attente plus longue :
       DEMARRAGE_TENTATIVES=24 DEMARRAGE_ATTENTE_S=600 ./deploy/00_tout_entrainer.sh
   - recreer une instance equivalente dans une autre zone (us-central1-b/c,
     us-west4-a, europe-west4-a...) puis mettre a jour GCP_ZONE dans .env.gcp.
     Une VM ne se deplace pas entre zones : il faut en creer une nouvelle."
                fi
                alerte "zone $GCP_ZONE saturee (aucun L4 libre) -- nouvel essai dans $((DEMARRAGE_ATTENTE_S / 60)) min"
                sleep "$DEMARRAGE_ATTENTE_S"
                ((tentative++))
            done
            ;;
        STAGING|PROVISIONING) info "Instance en cours de demarrage ($etat)" ;;
        INCONNU) echec "instance $GCP_INSTANCE introuvable dans $GCP_PROJET / $GCP_ZONE" ;;
        *) alerte "etat inattendu : $etat" ;;
    esac

    local ip; ip="$(ip_externe_vm)"
    [[ -n "$ip" ]] || echec "aucune IP externe sur l'instance"
    [[ "$ip" == "$GCP_HOST" ]] || { info "Nouvelle IP externe : $ip"; memoriser_ip "$ip"; }

    # Le service SSH met quelques dizaines de secondes a repondre apres un
    # demarrage a froid : on attend au lieu d'echouer sur la premiere tentative.
    info "Attente de SSH sur $CIBLE"
    for i in $(seq 1 40); do
        if ssh "${OPTS_SSH[@]}" "$CIBLE" "echo pret" >/dev/null 2>&1; then
            succes "instance joignable"
            return 0
        fi
        sleep 5
    done
    echec "SSH injoignable apres 200 s.
  - la cle $GCP_SSH_KEY est-elle autorisee sur l'instance ?
  - l'instance a-t-elle bien une IP externe ?"
}

arreter_vm() {
    local etat; etat="$(etat_vm)"
    if [[ "$etat" == "TERMINATED" || "$etat" == "STOPPED" ]]; then
        succes "instance deja arretee"
        return 0
    fi
    info "Arret de l'instance (facturation interrompue)"
    "$GCLOUD" compute instances stop "$GCP_INSTANCE" \
        --project "$GCP_PROJET" --zone "$GCP_ZONE" --quiet >/dev/null \
        && succes "instance arretee" \
        || alerte "ARRET ECHOUE -- l'instance continue d'etre facturee.
  Arreter manuellement :
    $GCLOUD compute instances stop $GCP_INSTANCE --project $GCP_PROJET --zone $GCP_ZONE"
}

# A appeler en debut de script pour garantir l'arret quoi qu'il arrive (fin
# normale, erreur, Ctrl-C). Passer GARDER_VM=1 pour laisser tourner -- utile
# entre deux etapes d'une meme chaine, ou pendant un entrainement.
arreter_en_sortant() {
    trap '
        code=$?
        if [[ "${GARDER_VM:-0}" == "1" ]]; then
            alerte "instance laissee en marche (GARDER_VM=1) -- pensez a l arreter"
        else
            arreter_vm
        fi
        exit $code
    ' EXIT INT TERM
}

verifier_connexion() {
    if vm "echo pret" >/dev/null 2>&1; then
        return 0
    fi
    # Echec de connexion : le plus souvent l'instance est arretee, ou son IP a
    # change. Les deux cas se resolvent par demarrer_vm.
    demarrer_vm
}

# ── Transfert des jeux de donnees via Cloud Storage ──────────────────────────
#
# Envoyer un jeu directement en scp depuis le poste local a montre ses limites :
# une archive de 774 Mo sur une liaison domestique a 76 Ko/s demande plus de
# deux heures en une seule session TCP, sans reprise possible. Un envoi a
# effectivement casse a 35 % sur un « Connection reset by peer », perdant 98 min
# de travail et autant de facturation GPU -- car la VM tournait pendant ce
# temps, sans rien faire d'autre qu'attendre des octets.
#
# Passer par un bucket corrige les deux problemes a la fois :
#   - `gcloud storage` fragmente et reprend automatiquement : une coupure ne
#     fait plus repartir de zero ;
#   - l'envoi se fait **VM eteinte**, donc sans facturation ; la VM ne s'allume
#     que pour tirer le fichier depuis le bucket, sur le reseau interne Google,
#     en quelques minutes.
#
# Le bucket doit etre dans la meme region que la VM : le transfert interne est
# alors gratuit et rapide. Sans `GCP_BUCKET` dans .env.gcp, les scripts
# retombent sur le scp direct.

bucket_actif() { [[ -n "${GCP_BUCKET:-}" ]]; }

# Envoie un fichier local vers le bucket, en ignorant l'envoi si l'objet y est
# deja avec la meme taille -- un jeu de donnees ne change pas entre deux runs,
# le re-televerser serait du temps perdu.
televerser_bucket() {
    local fichier="$1" objet="${2:-$(basename "$1")}"
    local taille_locale taille_distante
    taille_locale=$(stat -c '%s' "$fichier")
    taille_distante=$("$GCLOUD" storage ls -l "$GCP_BUCKET/$objet" \
        --project "$GCP_PROJET" 2>/dev/null | awk 'NR==1{print $1}')
    if [[ "$taille_distante" == "$taille_locale" ]]; then
        succes "$objet deja dans le bucket ($(numfmt --to=iec "$taille_locale")), envoi ignore"
        return 0
    fi
    info "Envoi vers $GCP_BUCKET/$objet ($(numfmt --to=iec "$taille_locale")) -- reprenable"
    "$GCLOUD" storage cp "$fichier" "$GCP_BUCKET/$objet" \
        --project "$GCP_PROJET" || echec "envoi vers le bucket echoue"
    succes "$objet depose dans le bucket"
}

# Fait tirer l'objet par la VM. On privilegie `gcloud storage` s'il est present
# (fragmentation, reprise), sinon on retombe sur l'API JSON avec le jeton du
# serveur de metadonnees -- disponible sur toute instance GCE, sans dependance.
vm_recuperer_bucket() {
    local objet="$1" destination="$2"
    info "La VM recupere $objet depuis le bucket (reseau interne Google)"
    vm "
        set -e
        cd '$destination'
        if command -v gcloud >/dev/null 2>&1; then
            gcloud storage cp '$GCP_BUCKET/$objet' . --project '$GCP_PROJET'
        else
            jeton=\$(curl -s -H 'Metadata-Flavor: Google' \
                'http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token' \
                | python3 -c 'import sys,json; print(json.load(sys.stdin)[\"access_token\"])')
            curl -s -f -C - -H \"Authorization: Bearer \$jeton\" -o '$objet' \
                'https://storage.googleapis.com/storage/v1/b/${GCP_BUCKET#gs://}/o/$objet?alt=media'
        fi
    " || echec "recuperation depuis le bucket echouee"
    succes "$objet recupere sur la VM"
}
