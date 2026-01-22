import os 
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
print(base_path)

# Métadonnées de l'analyse
NOM_ANALYSE = "Analyse_Rennes_binary_approach"
AUTEURS = "Axel"
OBJECTIF = "Segmentation de patients par similarité sémantique"
ETAPES = "Distribution, distance, PCA, clustering, visualisation"

# Contrôles
INPUT_IS_BINARY_MATRIX = True
PLOTTING_SIM_DIST = False
DIST_TRANSFO = False
STAND_TRANSFO = False
EXECUTE_PCA = True
STOP_AFTER_METRICS = False
N_COMPONENTS_PCA = 50
N_CLUSTERS = 17
KMEDOIDS_MAX_K = 1
MIN_FREQ = 0.05
P_THRESHOLD = 0.1

# Fichiers d'entrée
EHR_HPO_PATH = os.path.join(base_path, 'data', 'subset500_simulated_patients_SHEPHERD_updated_2025hpo.csv')
SIM_MAT_PATH = os.path.join(base_path, 'data', 'cosine_similarity_wIDF_SHEPHERD.csv') # A CHANGER SURTOUT DANS LE MAIN
HPO_ONTOLOGY_PATH = os.path.join(base_path, 'data', 'hpoterms_list_v2025_01_16.txt')

# Fichiers de sortie
OUTPUT_FOLDER = os.path.join(base_path, 'output')

