import sys
import os
import pandas as pd
import json
from datetime import datetime
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn_extra.cluster import KMedoids
import logging
from pathlib import Path
import random

# Ajouter le chemin du dossier parent (from_similarity_to_clustering)
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import *
from utils import explo_clustering, plot_similarity_distribution_from_mat, evaluate_kmedoids_clustering, viz_clustering

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = os.path.join(OUTPUT_FOLDER, f"clustering_analysis_{timestamp}.log")

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

def prepare_analysis():
    date_str = datetime.now().strftime("%Y%m%d")
    subfolder_name = f"{NOM_ANALYSE}_{date_str}"
    output_path = os.path.join("output", subfolder_name)
    os.makedirs(output_path, exist_ok=True)

    metadata = {
        "nom_analyse": NOM_ANALYSE,
        "date": date_str,
        "auteurs": [a.strip() for a in AUTEURS.split(",")],
        "objectif": OBJECTIF,
        "etapes": ETAPES,
        "config": {
            "INPUT_IS_BINARY_MATRIX": INPUT_IS_BINARY_MATRIX,
            "PLOTTING_SIM_DIST": PLOTTING_SIM_DIST,
            "DIST_TRANSFO":DIST_TRANSFO,
            "STANDARDISATION_TRANSFO":STAND_TRANSFO,
            "EXECUTE_PCA": EXECUTE_PCA,
            "STOP_AFTER_METRICS": STOP_AFTER_METRICS,
            "N_COMPONENTS_PCA": N_COMPONENTS_PCA,
            "N_CLUSTERS": N_CLUSTERS,
            "KMEDOIDS_MAX_K": KMEDOIDS_MAX_K,
            "MIN_FREQ": MIN_FREQ,
            "P_THRESHOLD": P_THRESHOLD,
            "EHR_HPO_PATH": EHR_HPO_PATH,
            "SIM_MAT_PATH": SIM_MAT_PATH,
            "HPO_ONTOLOGY_PATH": HPO_ONTOLOGY_PATH,
            "OUTPUT_FOLDER": OUTPUT_FOLDER
        },
        "fichiers_utilises": {
            "ehr_hpo": os.path.abspath(EHR_HPO_PATH),
            "hpo": os.path.abspath(HPO_ONTOLOGY_PATH),
            "sim_mat": os.path.abspath(SIM_MAT_PATH)
        }
    }
    return metadata, output_path


def main(sim_df, ehr_hpo, hpoterms):
    """
    0. Metadata
    1. Plot distrib similarity 
    2. Tansformation en distance
    3. Standardisation des distances
    4. PCA (50dim) 
    5. Métriques Kmedoids
    6. Kmedoid 
    7. Kmedoid Viz (Umap, Tsne)
    8. Cluster summary
    9. Pie chart
    10. cluster-level summary
    11. pour chaque cluster
        - tableau formaté
        - proportion inside vs outside
        - enrichment ratio log
    """

    ###### METADONNEES DE LETUDE ##################
    logging.info("Début de l'analyse de clustering")

    metadata, output_path = prepare_analysis()
    logging.info("Préparation des métadonnées terminée")

    metadata["dimensions_fichiers"] = {
        "ehr_hpo": ehr_hpo.shape,
        "sim_mat": sim_df.shape
    }

    with open(os.path.join(output_path, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)
    logging.info("Sauvegarde des métadonnées terminée")

    ##########################################
    # 1. Similarity distribution

    if INPUT_IS_BINARY_MATRIX:
        logging.info("Données binaires détectées : saut de la transformation en distance et standardisation")
        matrix_for_clustering = sim_df.values  

    else:     
        if PLOTTING_SIM_DIST:
            logging.info("Étape 1 : Distribution des similarités")
            plot_similarity_distribution_from_mat.plot_similarity_distribution_from_matrix(sim_df, output_path=f"{output_path}/plot_similarity_distribution.png", title="Similarity Distribution")


        ##########################################
        # 2. Transformation en distance
        logging.info("Étape 2 : Transformation des similarités en distances")
        dist_df = 1 - sim_df

        ##########################################
        # 3. Standardisation des données
        if STAND_TRANSFO :
            logging.info("Étape 3 : Standardisation de la matrice de distance")

            # Option : Standardisation (facultative, selon les besoins)
            mean_distance = np.mean(dist_df)
            std_distance = np.std(dist_df)

            # Standardisation de la matrice de distance
            standardized_dist_matrix = (dist_df - mean_distance) / std_distance

            # Lorsque les CR n'ont rien en commun, la distance peut être très grande (considérée infinie par la standardisation)
            # Il faut donc transformer les inf en grande valeur
            standardized_dist_matrix = np.where(np.isinf(standardized_dist_matrix), 1e10, standardized_dist_matrix)

            # Décalage des valeurs si nécessaire (pour garantir des distances positives)
            min_value = np.min(standardized_dist_matrix)
            if min_value < 0:
                shifted_dist_matrix = standardized_dist_matrix + abs(min_value)
            else:
                shifted_dist_matrix = standardized_dist_matrix

            # Mettre à zéro la diagonale (car la distance à soi-même doit être zéro)
            np.fill_diagonal(shifted_dist_matrix, 0)

        else:
            shifted_dist_matrix = dist_df.values
            
        matrix_for_clustering = shifted_dist_matrix


    ##########################################
    # 4. PCA (50 dim)
    # rajouter param entrée contrôle
    
    if EXECUTE_PCA:
        logging.info(f"Étape 4 : PCA avec {N_COMPONENTS_PCA} composantes")
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(matrix_for_clustering)
        pca = PCA(n_components=N_COMPONENTS_PCA, random_state=123)
        pca_result = pca.fit_transform(scaled_data)
    
    else:
        logging.info("PCA non exécutée")

    ##########################################
    # 5. Métriques Kmedoids

    logging.info(f"Étape 5 : Évaluation KMedoids jusqu'à {KMEDOIDS_MAX_K} clusters")

    if EXECUTE_PCA:
        metric = "euclidean" # Utilisation de la métrique de euclidienne pour PCA
        silhouette_scores, dbi_scores, inertia_scores = evaluate_kmedoids_clustering.evaluate_kmedoids_clustering(pca_result, KMEDOIDS_MAX_K, metric)
    else:
        metric="precomputed" # Utilisation de la métrique pré-calculée pour les matrices de distance
        silhouette_scores, dbi_scores, inertia_scores = evaluate_kmedoids_clustering.evaluate_kmedoids_clustering(matrix_for_clustering, KMEDOIDS_MAX_K, metric)


    ks = list(range(2,KMEDOIDS_MAX_K+1)) 

    evaluate_kmedoids_clustering.plot_metric(
            x_values=ks,
            y_values=silhouette_scores,
            title="Silhouette Score - KMedoids",
            y_label="Silhouette Score",
            file_path_base=f"{output_path}/silhouette_score",
            color="green"
        )

    # Plot et sauvegarde du Davies-Bouldin Index
    evaluate_kmedoids_clustering.plot_metric(
        x_values=ks,
        y_values=dbi_scores,
        title="Davies-Bouldin Index - KMedoids",
        y_label="Davies-Bouldin Index",
        file_path_base=f"{output_path}/davies_bouldin_index",
        color="red"
    )

    # Plot et sauvegarde de l'Elbow Method (WCSS)
    evaluate_kmedoids_clustering.plot_metric(
        x_values=ks,
        y_values=inertia_scores,
        title="Elbow Method - KMedoids",
        y_label="WCSS (Inertia)",
        file_path_base=f"{output_path}/elbow_method",
        color="blue"
    )

    logging.info(f"Étape 6 : KMedoids final avec {N_CLUSTERS} clusters ({'PCA' if EXECUTE_PCA else 'distance'} utilisée)")

    if STOP_AFTER_METRICS:
        logging.info("Arrêt après calcul des métrqiues")
        return

    ##########################################
    # 6. Kmedoid visulisation

    logging.info("Étape 7 : Visualisation des clusters (UMAP, t-SNE, PCA)")

    determined_n_clusters = N_CLUSTERS
    suffix = "euclidean" if EXECUTE_PCA else "cosine"
    
    if EXECUTE_PCA:
        kmedoid = KMedoids(n_clusters=determined_n_clusters, metric="euclidean", random_state=123)
        kmedoid_result = kmedoid.fit(pca_result)
        target_df = pca_result
    else: 
        kmedoid = KMedoids(n_clusters=determined_n_clusters, metric="precomputed", random_state=123)
        kmedoid_result = kmedoid.fit(matrix_for_clustering)
        target_df = matrix_for_clustering

    labels = kmedoid_result.labels_
    viz_clustering.apply_umap(target_df, labels,f"{output_path}/umap_2D_{suffix}")
    viz_clustering.apply_tsne(target_df, labels,f"{output_path}/tsne_2D_{suffix}")
    viz_clustering.apply_pca(target_df, labels,f"{output_path}/pca_2D_{suffix}")


    ##########################################
    # 7. Cluster summary

    logging.info("Étape 8 : Génération du résumé des clusters")

    df_for_clustering = pd.DataFrame(matrix_for_clustering, index=ehr_hpo.index, columns=ehr_hpo.columns)
    global_c_sum = explo_clustering.cluster_size_summary(labels, ehr_hpo, df_for_clustering,output_path=f"{output_path}/cluster_summary.csv")

    # Conserver uniquement les clusters non vides
    non_empty_clusters = global_c_sum[global_c_sum["Nb EHR in Cluster"] > 0]["Cluster"].tolist()
    print(non_empty_clusters)


    ##########################################
    # 8. Pie chart

    logging.info("Étape 9 : Génération du pie chart")
    explo_clustering.plot_cluster_pie_chart(cluster_table=global_c_sum, output_folder=f"{output_path}", n_clusters=determined_n_clusters, title="pie_chart" )


    ##########################################
    # 9. cluster-level summary

    logging.info("Étape 10 : Analyse des caractéristiques par cluster")

    c_level_sum = explo_clustering.cluster_summary_table_global_approach(labels, ehr_hpo, hpoterms, P_THRESHOLD, MIN_FREQ)

    ##########################################
    # 10. ehr level summary table

    logging.info("Étape 11 : Table de résumé par patient")
    explo_clustering.ehr_level_summary_table_with_enrichment(labels, ehr_hpo, c_level_sum, output_path=f"{output_path}/ehr_level_summary_table.csv")

    ##########################################
    # 10. pour chaque cluster
    #     - tableau formaté
    #     - proportion inside vs outside
    #     - enrichment ratio log

    logging.info("Étape 12 : Analyse détaillée par cluster")

    for cluster_id in non_empty_clusters:
        logging.info(f"Traitement du cluster {cluster_id}")

        ### Dossier de sortie
        os.makedirs(f"{output_path}/cluster_{cluster_id}", exist_ok=True)

        explo_clustering.format_summary_table_proportion(c_level_sum, cluster_id, output_path=f"{output_path}/cluster_{cluster_id}/formatted_table.csv")
        explo_clustering.volcano_plot_cluster_enrichment(c_level_sum, cluster_id, output_path=f"{output_path}/cluster_{cluster_id}/enrichment")
        try:

            explo_clustering.compare_cluster_vs_noncluster_plot(c_level_sum, cluster_id, output_path=f"{output_path}/cluster_{cluster_id}/prop")
        except Exception as e:
            logging.error(f"Erreur lors du plot du cluster {cluster_id} : {e}")


    logging.info("Analyse complète terminée")

if __name__=="__main__":
    ## ENTRIES
    # hpoterms
    # ehr_hpo
    # sim_mat

    try:
        ### Problème revoir l'éxecution complète avec filtrage sur CP
        sim_df = pd.read_csv(SIM_MAT_PATH, index_col=0)
        ehr_hpo = pd.read_csv(EHR_HPO_PATH, index_col=0)
        ehr_hpo_subset = ehr_hpo.loc[sim_df.index]

        logging.info("Etape de filtration : - retrait CP")
        logging.info(f"Dimension avant filtration : {ehr_hpo_subset.shape}")
              
        # retrait CP
        if "HP:0100021" in ehr_hpo_subset.columns: # petit check de sécu
            ehr_hpo_subset = ehr_hpo_subset.drop(columns=["HP:0100021"])
            print(ehr_hpo_subset.shape)

        ## A SUPPRIMER SI PAS BINARY MATRIX
        if "HP:0100021" in sim_df.columns: # petit check de sécu
            sim_df = sim_df.drop(columns=["HP:0100021"])
            print(sim_df.shape)

        logging.info(f"Dimension après filtration : {ehr_hpo_subset.shape}")
        logging.info(f"Dimension de la matrice : {sim_df.shape}")

        ### SECTION TIRAGE ALEATOIRE POUR ESSAI AVEC MOINS DE DONNEES
        # nb_patients = 100 
        # if sim_df.shape[0] > nb_patients:
        #     sampled_indices = random.sample(list(sim_df.index), nb_patients)
        #     sim_df = sim_df.loc[sampled_indices]
        #     ehr_hpo_subset = ehr_hpo_subset.loc[sampled_indices]
        #     logging.info(f"Dimensions après tirage aléatoire : {sim_df.shape}, {ehr_hpo_subset.shape}")
        # else:
        #     logging.warning("Pas assez de patients pour effectuer un sous-échantillonnage.")

        # Chargement ontologie
        hpoterms = pd.read_csv(HPO_ONTOLOGY_PATH, sep="\t", header=0)
        # print(hpoterms)

        # Vérification des dimensions
        if sim_df.shape[0] != ehr_hpo_subset.shape[0]:
            raise ValueError("Mismatch entre sim_df et ehr_hpo : même nombre de patients attendu.")

        # Execution de la fonction principale
        main(sim_df, ehr_hpo_subset, hpoterms)
    except Exception as e:
        logging.exception(f"Erreur lors de l'exécution : {e}")
        raise