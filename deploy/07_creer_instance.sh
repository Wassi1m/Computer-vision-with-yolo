#!/usr/bin/env bash
# Cree une instance d'entrainement dans la premiere zone disposant d'un L4 libre.
#
#     ./deploy/07_creer_instance.sh
#     ./deploy/07_creer_instance.sh europe-west4-a us-central1-b   # zones imposees
#
# Pourquoi creer plutot que redemarrer : une VM n'est pas deplacable d'une zone
# a l'autre. Quand la zone d'origine n'a plus de GPU libre
# (ZONE_RESOURCE_POOL_EXHAUSTED), la seule issue est une nouvelle instance
# ailleurs.
#
# L'image retenue est une Deep Learning VM : pilote NVIDIA, CUDA et PyTorch y
# sont deja installes et coherents entre eux. C'est ce qui evite l'installation
# manuelle du pilote, qui echoue sur Debian 13 (le paquet
# linux-headers-<version> du noyau cloud n'existe pas sous ce nom).
#
# Il n'y a pas d'API publique renseignant la capacite disponible d'une zone :
# la seule facon de savoir est de tenter la creation. C'est ce que fait ce
# script, zone par zone, en s'arretant au premier succes.

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

# Zones europeennes en tete : le transfert des jeux de donnees represente plus
# d'un Go depuis la France, la latence et le debit y sont bien meilleurs.
ZONES_DEFAUT=(
    europe-west4-a europe-west4-b europe-west4-c
    europe-west1-b europe-west1-c
    europe-west2-a europe-west2-b
    europe-west3-a europe-west3-b
    us-central1-a us-central1-b us-central1-c
    us-east1-b us-east1-c us-east1-d
    us-east4-a us-east4-c
    us-west1-a us-west4-a
)
ZONES=("$@")
[[ ${#ZONES[@]} -gt 0 ]] || ZONES=("${ZONES_DEFAUT[@]}")

NOUVELLE_INSTANCE="${NOUVELLE_INSTANCE:-ppe-train-$(date +%Y%m%d-%H%M%S)}"
TYPE_MACHINE="${TYPE_MACHINE:-g2-standard-4}"
IMAGE_FAMILLE="${IMAGE_FAMILLE:-pytorch-2-9-cu129-ubuntu-2204-nvidia-580}"
IMAGE_PROJET=deeplearning-platform-release
TAILLE_DISQUE="${TAILLE_DISQUE:-100GB}"

info "Recherche d'une zone disposant d'un ${TYPE_MACHINE} + NVIDIA L4"
echo "  instance : $NOUVELLE_INSTANCE"
echo "  image    : $IMAGE_FAMILLE (pilote + CUDA + PyTorch preinstalles)"
echo "  zones    : ${#ZONES[@]} a essayer"
echo

CLE_PUBLIQUE="$(cat "${GCP_SSH_KEY}.pub")"
FICHIER_CLES=$(mktemp)
# Format attendu par GCP : "<utilisateur>:<cle>".
echo "${GCP_USER}:${CLE_PUBLIQUE}" > "$FICHIER_CLES"
trap 'rm -f "$FICHIER_CLES"' EXIT

ZONE_TROUVEE=""
for zone in "${ZONES[@]}"; do
    printf '  %-20s ' "$zone"
    if sortie=$("$GCLOUD" compute instances create "$NOUVELLE_INSTANCE" \
            --project="$GCP_PROJET" \
            --zone="$zone" \
            --machine-type="$TYPE_MACHINE" \
            --accelerator="type=nvidia-l4,count=1" \
            --maintenance-policy=TERMINATE \
            --image-family="$IMAGE_FAMILLE" \
            --image-project="$IMAGE_PROJET" \
            --boot-disk-size="$TAILLE_DISQUE" \
            --boot-disk-type=pd-balanced \
            --metadata-from-file=ssh-keys="$FICHIER_CLES" \
            --metadata=install-nvidia-driver=True \
            --scopes=https://www.googleapis.com/auth/cloud-platform \
            --quiet 2>&1); then
        printf '\033[1;32mDISPONIBLE\033[0m\n'
        ZONE_TROUVEE="$zone"
        break
    fi

    if grep -q "ZONE_RESOURCE_POOL_EXHAUSTED\|STOCKOUT\|does not have enough resources" <<<"$sortie"; then
        printf 'saturee\n'
    elif grep -q "QUOTA_EXCEEDED\|Quota .* exceeded" <<<"$sortie"; then
        printf '\033[1;33mquota insuffisant\033[0m\n'
    elif grep -q "already exists" <<<"$sortie"; then
        printf 'instance de ce nom deja presente\n'
        ZONE_TROUVEE="$zone"; break
    else
        printf 'echec\n'
        echo "$sortie" | grep -E "^ERROR|message:" | head -3 | sed 's/^/      /'
    fi
done

[[ -n "$ZONE_TROUVEE" ]] || echec "aucune zone disponible parmi les ${#ZONES[@]} essayees.
  Soit toutes les zones sont saturees (reessayer plus tard),
  soit le quota NVIDIA_L4_GPUS du projet est a 0 en dehors de us-central1.
  Le quota se demande dans : IAM et administration > Quotas > NVIDIA_L4_GPUS"

echo
succes "instance creee dans $ZONE_TROUVEE"

IP=$("$GCLOUD" compute instances describe "$NOUVELLE_INSTANCE" \
        --project="$GCP_PROJET" --zone="$ZONE_TROUVEE" \
        --format='value(networkInterfaces[0].accessConfigs[0].natIP)')

info "Mise a jour de .env.gcp"
sed -i "s|^GCP_INSTANCE=.*|GCP_INSTANCE=$NOUVELLE_INSTANCE|" "$ENV_FICHIER"
sed -i "s|^GCP_ZONE=.*|GCP_ZONE=$ZONE_TROUVEE|" "$ENV_FICHIER"
sed -i "s|^GCP_HOST=.*|GCP_HOST=$IP|" "$ENV_FICHIER"
grep -E "^GCP_(INSTANCE|ZONE|HOST)=" "$ENV_FICHIER" | sed 's/^/  /'

# L'image Deep Learning installe le pilote au premier demarrage : elle n'est pas
# joignable immediatement.
info "Attente de SSH (l'image installe le pilote au premier demarrage)"
CIBLE="${GCP_USER}@${IP}"
for i in $(seq 1 60); do
    if ssh "${OPTS_SSH[@]}" "$CIBLE" "echo pret" >/dev/null 2>&1; then
        succes "instance joignable"
        break
    fi
    sleep 10
    [[ $i -eq 60 ]] && echec "SSH injoignable apres 10 min"
done

info "Verification du GPU"
ssh "${OPTS_SSH[@]}" "$CIBLE" "
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null \
        || echo '  pilote pas encore pret -- il s installe en tache de fond, patienter 2-3 min'
" | sed 's/^/  /'

cat <<EOF

ATTENTION : l'ancienne instance $GCP_INSTANCE existe toujours dans us-central1-a.
Son disque reste facture meme arretee. La supprimer si elle n'est plus utile :
    $GCLOUD compute instances delete instance-20260810-093132 \\
        --project=$GCP_PROJET --zone=us-central1-a --quiet

Suite :
    ./deploy/00_tout_entrainer.sh
EOF
