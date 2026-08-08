"""
Configuration centrale du système de surveillance intelligente.
Modifie ces valeurs selon ta caméra / scène.
"""

# --------------------------------------------------------------------------
# Modèles
# --------------------------------------------------------------------------
MODEL_GENERAL = "yolo26n.pt"        # détection générale (personnes, voitures...) COCO
MODEL_POSE = "yolo26n-pose.pt"      # squelette -> détection de chute (secours si MODEL_FALL absent)

# Détecteur de chute dédié (P5, entraîné sur Colab -- voir module_fall.py).
# Utilisé en priorité si présent ; sinon repli automatique sur l'heuristique
# pose (angle du tronc + ratio largeur/hauteur, moins fiable).
MODEL_FALL = "models/fall_detector.pt"      # ou None pour forcer l'heuristique

# Modèle spécialisé fumée/feu : PAS inclus dans YOLO26 de base.
# Télécharge un modèle entraîné (ex: Roboflow "fire-smoke-yolov8") et mets le
# fichier .pt ici. Laisse à None pour désactiver ce module.
MODEL_FIRE_SMOKE = "models/fire_smoke.pt"   # ou None

# Modèle détecteur de plaques d'immatriculation : PAS inclus dans YOLO26 de
# base non plus. Télécharge un modèle "license-plate" (Roboflow Universe a
# plusieurs modèles gratuits) et mets le fichier .pt ici.
MODEL_PLATE = "models/license_plate.pt"     # ou None

# --------------------------------------------------------------------------
# Module 3 : ligne virtuelle (franchissement)
# Deux points (x, y) en pixels qui définissent la ligne sur l'image.
# --------------------------------------------------------------------------
LINE_START = (100, 400)
LINE_END = (900, 400)

# --------------------------------------------------------------------------
# Module 5 : porte (zone à surveiller)
# Rectangle (x1, y1, x2, y2) délimitant la porte dans l'image.
# Laisse à None pour sélectionner le ROI interactivement à la souris au
# lancement (recommandé : plus le rectangle est serré sur la porte, plus
# la détection est fiable).
# --------------------------------------------------------------------------
DOOR_ROI = None
DOOR_DIFF_THRESHOLD = 0.15      # seuil SSIM (0=identique, 1=très différent).
                                 # Redescendu de 0.28 -> 0.15 : maintenant que
                                 # l'occlusion par une personne est filtrée
                                 # séparément (DOOR_OCCLUSION_THRESHOLD), on n'a
                                 # plus besoin d'un seuil aussi élevé. Un seuil
                                 # trop haut empêchait de détecter le vrai
                                 # changement d'état (ton test montrait un score
                                 # de 0.175 pour un changement réel, resté sous
                                 # l'ancien seuil de 0.28).
                                 # Regarde le score affiché à l'écran : note la
                                 # valeur stable porte fermée, puis porte
                                 # ouverte, et place ce seuil entre les deux
                                 # (idéalement au milieu).
DOOR_STABILITY_FRAMES = 8       # 5 -> 8 : plus de frames consécutives requises
                                 # pour confirmer un changement d'état, réduit
                                 # encore les faux positifs liés au bruit
DOOR_CALIBRATION_SAMPLES = 25   # 15 -> 25 : référence plus stable, moyennée
                                 # sur davantage de frames pendant la calibration
DOOR_OCCLUSION_THRESHOLD = 0.15 # si une personne recouvre plus de 15% du ROI,
                                 # on ignore la frame pour la détection porte
                                 # (évite de confondre "personne devant la porte"
                                 # avec "la porte a changé d'état")
                                 # -- ceci ne s'applique qu'à l'heuristique SSIM
                                 # de secours, pas au classifieur entraîné.

# Classifieur porte entraîné (méthode recommandée, voir README) :
DOOR_CLASSIFIER_SMOOTHING = 7    # frames de vote majoritaire (anti-flicker)
DOOR_CLASSIFIER_MIN_CONF = 0.6   # confiance minimale pour prendre en compte une prédiction

# --------------------------------------------------------------------------
# Module 2 : détection de chute
# Une personne est considérée "tombée" si le ratio largeur/hauteur de sa
# boîte dépasse ce seuil (une personne debout est haute et étroite, une
# personne au sol est large et basse) ET si l'angle du tronc (via pose)
# est proche de l'horizontale.
# --------------------------------------------------------------------------
FALL_ASPECT_RATIO_THRESHOLD = 1.4
FALL_ANGLE_THRESHOLD_DEG = 45   # angle du tronc par rapport à la verticale

# --------------------------------------------------------------------------
# Général
# --------------------------------------------------------------------------
CONF_THRESHOLD = 0.4
CAMERA_SOURCE = 0   # 0 = webcam, ou chemin vers un fichier vidéo / URL RTSP
