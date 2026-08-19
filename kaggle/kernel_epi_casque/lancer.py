"""Kernel de lancement : installe ultralytics puis entraine le modele 'casque'.

Scripte plutot que saisi dans une cellule, pour etre pousse et suivi par
l'API Kaggle sans passer par l'interface web -- les deux modeles de la
campagne tournent ainsi en parallele dans deux sessions distinctes.
"""
import glob
import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ultralytics"])

# Recherche par motif : le nom de montage depend du titre donne a l'upload,
# un chemin /kaggle/input/... ecrit en dur a deja fait echouer un lancement.
candidats = glob.glob("/kaggle/input/**/entrainer_epi_cascade.py", recursive=True)
if not candidats:
    raise SystemExit("entrainer_epi_cascade.py introuvable dans /kaggle/input")

code = subprocess.call([sys.executable, candidats[0], "--modele", "casque"])
if code != 0:
    raise SystemExit(code)
