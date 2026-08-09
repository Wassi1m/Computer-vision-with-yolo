# Moteur de détection — image de production.
#
# Les modèles (.pt) sont copiés dans l'image plutôt que montés : une image
# donnée correspond ainsi à un jeu de modèles précis et reproductible. Changer
# de modèle produit une nouvelle image, ce qui rend le retour arrière trivial
# (cf. docs/exploitation.md).
#
# IMPORTANT : les .pt ne sont PAS dans le dépôt git (voir .gitignore, 142 Mo,
# trop lourd pour un dépôt de code). Ce build échoue sur un clone frais tant
# que ppe_detection/models/*.pt et surveillance_suite/models/*.pt n'ont pas été
# copiés manuellement dans l'arbre de travail au préalable, depuis l'endroit où
# ils sont conservés (poste local, stockage de modèles).

FROM python:3.12-slim

# libGL et libglib : requis par opencv-python, absents des images slim.
# ffmpeg : décodage des flux RTSP.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Couche de dépendances séparée : elle ne se reconstruit que si les versions
# changent, pas à chaque modification de code.
COPY requirements-service.txt .
RUN pip install --no-cache-dir -r requirements-service.txt

COPY improvements/ ./improvements/
COPY surveillance_suite/ ./surveillance_suite/
COPY ppe_detection/models/ ./ppe_detection/models/
COPY tests/ ./tests/
COPY docs/ ./docs/

# Ultralytics écrit sa configuration et ses téléchargements dans le HOME.
ENV YOLO_CONFIG_DIR=/tmp/ultralytics \
    PYTHONUNBUFFERED=1 \
    MOTEUR_LOG_JSON=1 \
    MOTEUR_HEALTH_PORT=8080

EXPOSE 8080

# Le conteneur est déclaré défaillant si le moteur ne voit plus le flux ou
# n'analyse aucune image — c'est le rôle de /health (503 dans ce cas).
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=4).status==200 else 1)"

WORKDIR /app/improvements

# Headless par défaut : l'affichage n'a pas de sens en conteneur.
ENTRYPOINT ["python", "unified_surveillance.py"]
CMD ["--source", "0"]
