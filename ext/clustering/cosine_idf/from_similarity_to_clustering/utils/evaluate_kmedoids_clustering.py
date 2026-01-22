import argparse
import os
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn_extra.cluster import KMedoids
import plotly.graph_objects as go

def evaluate_kmedoids_clustering(distance_matrix, k_max, metric):
    """
    Évalue les performances du clustering K-Medoids pour différentes valeurs de K et le choix du nb de clusters.

    Calcule les trois métriques suivantes pour chaque nombre de clusters K :
    - Silhouette Score
    - Davies-Bouldin Index (DBI)
    - Inertia (WCSS, pour la méthode du coude) (choix du cluster)

    Paramètres
    ----------
    distance_matrix : np.ndarray
        Matrice carrée (n_samples x n_samples) de distances ou de données PCA.
    k_max : int
        Nombre maximal de clusters à tester (K varie de 2 à k_max).
    metric : str
        Type de métrique utilisé par KMedoids ("precomputed", "euclidean", etc.).

    Retour
    ------
    tuple of lists
        (silhouette_scores, dbi_scores, inertia_scores), chacun contenant une valeur par K testé.
    """
    silhouette_scores = []
    dbi_scores = []
    inertia_scores = []
    
    for k in range(2, k_max + 1):
        kmedoids = KMedoids(n_clusters=k, metric=metric, random_state=123)
        labels = kmedoids.fit_predict(distance_matrix)

        # Silhouette Score (Using 1 - Jaccard distance because it's a similarity score)
        silhouette_avg = silhouette_score(distance_matrix, labels, metric=metric)
        silhouette_scores.append(silhouette_avg)

        # Davies-Bouldin Index (works directly with the distance matrix)
        dbi_score = davies_bouldin_score(distance_matrix, labels)
        dbi_scores.append(dbi_score)

        # Inertia (WCSS) for Elbow Method
        inertia_scores.append(kmedoids.inertia_)

    return silhouette_scores, dbi_scores, inertia_scores


def plot_metric(x_values, y_values, title, y_label, file_path_base, color):
    """
    Affiche et sauvegarde une courbe (ligne + points) d'une métrique de clustering en fonction de K.

    Génère un fichier `.png` et un fichier `.html` interactif.

    Paramètres
    ----------
    x_values : list of int
        Valeurs de K (nombre de clusters).
    y_values : list of float
        Valeurs de la métrique correspondante (silhouette, DBI, inertia...).
    title : str
        Titre du graphique.
    y_label : str
        Label de l'axe Y.
    file_path_base : str
        Chemin de base pour sauvegarder les fichiers (sans extension).
    color : str
        Couleur du tracé (ex: "blue", "green", "red").
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x_values,
        y=y_values,
        mode='markers+lines',
        marker=dict(color=color, size=8),
        name=y_label
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Number of Clusters",
        yaxis_title=y_label,
        template="plotly_white",
        height=600,
        width=800,
        showlegend=True
    )

    # Sauvegarde PNG
    fig.write_image(f"{file_path_base}.png")

    # Sauvegarde HTML
    fig.write_html(f"{file_path_base}.html")


def main_metrics(distance_matrix, k_max, output_path):
    """
    Fonction principale pour l’évaluation des clusters sur un intervalle de valeurs K.

    Exécute :
    - le clustering K-Medoids sur plusieurs K (2 → k_max)
    - le calcul des métriques de qualité
    - l’exportation des graphiques correspondants dans le dossier "metrics/"

    Paramètres
    ----------
    distance_matrix : pd.DataFrame
        Matrice carrée de distances entre patients (index et colonnes = IDs).
    k_max : int
        Nombre maximal de clusters à tester.
    output_path : str
        Dossier dans lequel enregistrer les fichiers de sortie.
    """
    matrix = distance_matrix.values
    metric_dir = os.path.join(output_path, "metrics")
    os.makedirs(metric_dir, exist_ok=True)

    silhouette_scores, dbi_scores, inertia_scores = evaluate_kmedoids_clustering(matrix, k_max)
    ks = list(range(2, k_max + 1))

    # Silhouette
    plot_metric(
        x_values=ks,
        y_values=silhouette_scores,
        title="Silhouette Score - KMedoids",
        y_label="Silhouette Score",
        file_path_base=f"{metric_dir}/silhouette_score",
        color="green"
    )

    # Davies-Bouldin
    plot_metric(
        x_values=ks,
        y_values=dbi_scores,
        title="Davies-Bouldin Index - KMedoids",
        y_label="Davies-Bouldin Index",
        file_path_base=f"{metric_dir}/davies_bouldin_index",
        color="red"
    )

    # Elbow (Inertia)
    plot_metric(
        x_values=ks,
        y_values=inertia_scores,
        title="Elbow Method - KMedoids",
        y_label="WCSS (Inertia)",
        file_path_base=f"{metric_dir}/elbow_method",
        color="blue"
    )
