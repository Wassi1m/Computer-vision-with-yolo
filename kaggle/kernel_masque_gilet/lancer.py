"""Kernel de lancement : installe ultralytics puis lance entrainer_masque_gilet.py.

Equivalent de la cellule notebook manuelle habituelle sur ce projet, mais
scripte pour etre pousse et suivi par l'API Kaggle sans intervention web.
"""
import glob
import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ultralytics"])

candidats = glob.glob("/kaggle/input/**/entrainer_masque_gilet.py", recursive=True)
if not candidats:
    raise SystemExit("entrainer_masque_gilet.py introuvable dans /kaggle/input")

code = subprocess.call([sys.executable, candidats[0]])
if code != 0:
    raise SystemExit(code)
