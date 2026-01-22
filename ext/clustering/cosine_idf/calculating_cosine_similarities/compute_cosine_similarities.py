# === Imports ===
import sys
from pathlib import Path
import json
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# === Configuration des chemins ===
CURRENT_DIR = Path(__file__).resolve()
ROOT_DIR = CURRENT_DIR.parents[3]
PROJECT_DIR = CURRENT_DIR.parents[1]
SHARED_DIR = ROOT_DIR / "shared"
sys.path.extend([str(PROJECT_DIR), str(ROOT_DIR), str(SHARED_DIR)])

# === Imports locaux ===
from config import *

# === Utilitaires ===
def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def compute_idf_weighted_cosine_similarity(ehr_hpo_df: pd.DataFrame, output_path: str) -> pd.DataFrame:
    """
    Calcule la similarité cosinus pondérée par l'IDF entre patients et sauvegarde la matrice.

    Paramètres :
    - ehr_hpo_df : DataFrame (patients x termes HPO) avec 0/1 ou fréquences
    - output_path : chemin de sauvegarde de la matrice de similarité (format .csv)

    Retour :
    - similarity_df_weighted : DataFrame avec les similarités pondérées (patients x patients)
    """
    # Étape 1 : Calcul de l'IDF
    n_patients = ehr_hpo_df.shape[0]
    term_occurrences = (ehr_hpo_df > 0).sum(axis=0)
    idf = np.log(n_patients / (term_occurrences + 1e-20))  # éviter log(0)

    # Étape 2 : Pondération par IDF
    ehr_hpo_weighted = ehr_hpo_df * idf

    # Étape 3 : Calcul de la similarité cosinus pondérée
    similarity_matrix = cosine_similarity(ehr_hpo_weighted.values)

    # Étape 4 : Conversion en DataFrame avec index/colonnes patients
    similarity_df_weighted = pd.DataFrame(similarity_matrix, 
                                          index=ehr_hpo_df.index, 
                                          columns=ehr_hpo_df.index)

    # Étape 5 : Sauvegarde
    similarity_df_weighted.to_csv(output_path)
    print(f"Matrice de similarité sauvegardée dans : {output_path}")

    return similarity_df_weighted

def prepare_analysis():
    """
    Crée un sous-dossier d'analyse (daté) dans PROJECT_DIR/output
    et y écrit un fichier metadata.json minimal.
    """
    date_str = datetime.now().strftime("%Y%m%d")
    subfolder_name = f"{NOM_ANALYSE}_{date_str}"
    output_path = PROJECT_DIR / "output" / subfolder_name
    output_path.mkdir(parents=True, exist_ok=True)

    metadata = {
        "nom_analyse": NOM_ANALYSE,
        "date": date_str,
        "auteurs": [a.strip() for a in AUTEURS.split(",")],
        "objectif": OBJECTIF,
        "etapes": ETAPES,
        "dataset": DATASET_NAME,
        "fichiers_utilises": {
            "ehr_hpo": str(LOCAL_EHR_HPO_PATH.resolve()),
        },
    }

    with open(output_path / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    return output_path


# === COSINE CLASSIQUE ===
def compute_cosine_classic(ehr_hpo: pd.DataFrame, output_path: Path) -> None:
    """
    Calcule la similarité cosinus classique sur la matrice EHR-HPO binaire / pondérée.

    - ehr_hpo : DataFrame index = patients, colonnes = HPO, valeurs = 0/1 ou poids
    - output_path : dossier de sortie (Path)
    """
    log("▶️ Cosine classique")
    cosine_sim_mat = cosine_similarity(ehr_hpo.values)
    cosine_df = pd.DataFrame(
        cosine_sim_mat,
        index=ehr_hpo.index,
        columns=ehr_hpo.index,
    )
    out_file = output_path / f"cosine_similarity_{DATASET_NAME}.csv"
    cosine_df.to_csv(out_file)
    log(f"💾 Cosine classique sauvegardé dans : {out_file}")
    del cosine_df, cosine_sim_mat


# === COSINE + IDF ===
def compute_cosine_idf(ehr_hpo: pd.DataFrame, output_path: Path) -> None:
    """
    Calcule la similarité cosinus après pondération IDF des colonnes HPO.
    """
    log("▶️ Cosine avec pondération IDF")
    out_file = output_path / f"cosine_similarity_wIDF_{DATASET_NAME}.csv"
    compute_idf_weighted_cosine_similarity(
        ehr_hpo,
        output_path=out_file,
    )
    log(f"💾 Cosine IDF sauvegardé dans : {out_file}")


# === MAIN ===
def main():
    log("🚀 Début de l’analyse (cosine + cosine IDF uniquement)")
    log("Chargement des fichiers...")

    ehr_hpo = pd.read_csv(LOCAL_EHR_HPO_PATH, index_col=0)
    log(f"✔️ Shape EHR-HPO: {ehr_hpo.shape}")

    output_path = prepare_analysis()

    # Calcul des deux métriques uniquement
    compute_cosine_classic(ehr_hpo, output_path)
    compute_cosine_idf(ehr_hpo, output_path)

    log("✅ Analyse terminée (cosine & cosine IDF).")


# === LANCEMENT ===
if __name__ == "__main__":
    main()

