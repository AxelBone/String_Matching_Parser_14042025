import os
from pathlib import Path
import sys

# === Ajouter shared/ au sys.path si besoin (optionnel selon ton projet) ===
sys.path.append(str(Path(__file__).resolve().parents[2] / 'shared'))

# === Chemin local du projet (cosine_idf/calculating_cosine_similarities) ===
base_path = Path(__file__).resolve().parent

# === Métadonnées de l’analyse ===
NOM_ANALYSE = "calcul_mat_sim_entre_termes_HPO"
AUTEURS = "Axel"
OBJECTIF = "Calculer une matrice de poids Lin et WP entre termes HPO."
ETAPES = ""
DATASET_NAME = "SHEPHERD"

# === Fichier d'entrée local (copie dataset dans cosine_idf/data/) ===
LOCAL_DATA_FOLDER = base_path.parent / 'data'
LOCAL_EHR_HPO_PATH = LOCAL_DATA_FOLDER / 'subset500_simulated_patients_SHEPHERD_updated_2025hpo.csv'

# === Dossier de sortie local ===
OUTPUT_FOLDER = base_path.parent / 'from_similarity_to_clustering' / 'output'
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
