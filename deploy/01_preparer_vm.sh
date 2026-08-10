#!/usr/bin/env bash
# Prepare la VM GCP pour l'entrainement : pilotes GPU, Python, dependances.
#
# Idempotent : relancable sans risque, chaque etape verifie d'abord si elle a
# deja ete faite. C'est ce qui permet de le rejouer apres un redemarrage de la
# machine sans tout reinstaller.

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

# Demarre l'instance si besoin et l'arrete en sortant : elle est facturee
# a l'heure. GARDER_VM=1 laisse la main a l'appelant (orchestrateur).
arreter_en_sortant

verifier_connexion
succes "connecte a $CIBLE"

info "Etat de la machine"
vm "
    echo \"  OS      : \$(. /etc/os-release && echo \$PRETTY_NAME)\"
    echo \"  CPU     : \$(nproc) coeurs\"
    echo \"  RAM     : \$(free -g | awk '/^Mem:/{print \$2}') Go\"
    echo \"  Disque  : \$(df -h / | awk 'NR==2{print \$4}') libres\"
"

info "Pilote NVIDIA"
if vm "command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1"; then
    vm "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | sed 's/^/  /'"
    succes "GPU deja operationnel"
else
    alerte "pilote absent ou non fonctionnel : installation (plusieurs minutes)"
    # Installation depuis les depots Debian plutot que par le script GCP
    # `install_gpu_driver.py` : celui-ci echoue sur Debian 13 (trixie), ou il
    # cherche un paquet `linux-headers-<version exacte>` qui n'existe pas sous
    # ce nom pour le noyau cloud.
    #
    # Seul le *pilote noyau* est necessaire ici : PyTorch embarque sa propre
    # bibliotheque CUDA, le toolkit complet serait plusieurs Go inutiles.
    vm "
        set -e
        # Le pilote NVIDIA est dans non-free ; les images GCP n'activent que main.
        sudo sed -i 's/^Components: main\$/Components: main contrib non-free non-free-firmware/' \
            /etc/apt/sources.list.d/debian.sources 2>/dev/null || true
        sudo apt-get update -qq
        # Le meta-paquet -cloud-amd64 suit le noyau installe : plus fiable que de
        # deviner le nom exact de la version en cours.
        sudo apt-get install -y -qq build-essential dkms curl \
            linux-headers-cloud-amd64 || sudo apt-get install -y -qq linux-headers-amd64
        sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq nvidia-driver firmware-nvidia-gsp
    " || echec "installation du pilote GPU echouee -- consulter les journaux de la VM"

    # Le module noyau n'est charge qu'apres redemarrage : nvidia-smi resterait
    # muet sans cela, meme installation reussie.
    info "Redemarrage de la VM (chargement du module noyau)"
    vm "sudo systemctl reboot" >/dev/null 2>&1 || true
    sleep 30
    for i in $(seq 1 40); do
        vm "echo pret" >/dev/null 2>&1 && break
        sleep 10
    done

    vm "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | sed 's/^/  /'" \
        || echec "GPU toujours indisponible apres installation et redemarrage.
  Verifier sur la VM :  dkms status  puis  dmesg | grep -i nvidia"
    succes "pilote GPU installe"
fi

info "Environnement Python"
vm "
    set -e
    sudo apt-get install -y -qq python3-venv python3-pip libgl1 libglib2.0-0 ffmpeg tmux >/dev/null 2>&1
    mkdir -p '$GCP_WORKDIR'
    if [[ ! -d '$GCP_WORKDIR/.venv' ]]; then
        python3 -m venv '$GCP_WORKDIR/.venv'
    fi
    '$GCP_WORKDIR/.venv/bin/pip' install --quiet --upgrade pip
"
succes "environnement pret"

info "Dependances d'entrainement (torch CUDA + ultralytics)"
# La variante CUDA de torch est indispensable ici : la variante CPU utilisee en
# local ignorerait purement et simplement le L4.
vm "
    set -e
    '$GCP_WORKDIR/.venv/bin/pip' install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu124
    '$GCP_WORKDIR/.venv/bin/pip' install --quiet ultralytics opencv-python-headless
"

info "Verification : PyTorch voit-il le GPU ?"
vm "'$GCP_WORKDIR/.venv/bin/python' -c \"
import torch
print(f'  torch      : {torch.__version__}')
print(f'  CUDA dispo : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU        : {torch.cuda.get_device_name(0)}')
    print(f'  VRAM       : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} Go')
else:
    raise SystemExit('CUDA indisponible : l entrainement tournerait sur CPU')
\"" || echec "PyTorch ne voit pas le GPU"

succes "VM prete pour l'entrainement"
echo
echo "Suite :  ./deploy/02_envoyer.sh <jeu_de_donnees>"
