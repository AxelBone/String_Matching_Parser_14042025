import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap
import numpy as np

def plot_clusters_2d(
    reduced_data,
    cluster_labels,
    method_name,
    param_suffix,
    output_path,
    color_values=None,
    color_name=None,
):
    """
    Génère une visualisation 2D des individus à partir de données projetées (PCA, t-SNE ou UMAP).

    Deux modes de coloration :
    - par cluster (par défaut, avec `cluster_labels`)
    - par variable catégorielle (si `color_values` est fourni)

    Paramètres
    ----------
    reduced_data : np.ndarray
        Données réduites à 2 dimensions (shape = [n_samples, 2]).
    cluster_labels : array-like
        Étiquettes de cluster pour chaque point (utilisées si color_values est None).
    method_name : str
        Nom de la méthode de réduction (ex : 'PCA', 'tSNE', 'UMAP').
    param_suffix : str
        Suffixe décrivant les paramètres utilisés (servira pour le nom de fichier).
    output_path : str
        Dossier de sauvegarde du graphique PNG.
    color_values : array-like, optionnel
        Variable catégorielle pour colorer les points (ex: variant_presence, n_variant, pathogenic_cat).
        Doit être de même longueur que reduced_data.
    color_name : str, optionnel
        Nom de la variable utilisée pour la coloration (pour le titre et le nom de fichier).

    Sortie
    ------
    Enregistre un fichier PNG dans le dossier `output_path`.
    """

    os.makedirs(output_path, exist_ok=True)

    plt.figure(figsize=(8, 6))

    # Sécurité : conversion en array numpy
    cluster_labels = np.array(cluster_labels)

    if color_values is not None:
        color_values = np.array(color_values)
        if len(color_values) != reduced_data.shape[0]:
            raise ValueError(
                f"color_values length ({len(color_values)}) != n_samples ({reduced_data.shape[0]})"
            )

        unique_cats = np.unique(color_values)
        for cat in unique_cats:
            mask = color_values == cat
            plt.scatter(
                reduced_data[mask, 0],
                reduced_data[mask, 1],
                label=f"{color_name} = {cat}",
                s=30,
            )
        legend_title = color_name if color_name is not None else "Category"
        plt.legend(title=legend_title)
        color_suffix = f"colorby_{color_name}"
        title_color_part = f"Coloré par {color_name}"
    else:
        # Mode par défaut : couleur = cluster
        unique_labels = np.unique(cluster_labels)
        for label in unique_labels:
            plt.scatter(
                reduced_data[cluster_labels == label, 0],
                reduced_data[cluster_labels == label, 1],
                label=f"Cluster {label}",
                s=30,
            )
        plt.legend(title="Clusters")
        color_suffix = "colorby_cluster"
        title_color_part = "Coloré par cluster"

    plt.title(f"{method_name} - {param_suffix}\n{title_color_part}")
    plt.tight_layout()

    file_name = f"{method_name}_{param_suffix}_{color_suffix}.png"
    plt.savefig(os.path.join(output_path, file_name))
    plt.close()


def apply_pca(X, labels, output_path, color_values=None, color_name=None):
    """
    Applique plusieurs configurations de PCA à des données et génère des visualisations 2D.

    Paramètres
    ----------
    X : np.ndarray
        Matrice des données originales (shape = [n_samples, n_features]).
    labels : array-like
        Étiquettes de cluster pour chaque point.
    output_path : str
        Dossier de sortie pour les graphes PNG.
    color_values : array-like, optionnel
        Variable catégorielle pour la coloration des points.
    color_name : str, optionnel
        Nom de la variable de coloration.
    """
    configs = [
        {"n_components": 2, "whiten": False, "svd_solver": "auto"},
        {"n_components": 2, "whiten": True, "svd_solver": "auto"},
        {"n_components": 2, "whiten": True, "svd_solver": "randomized"},
    ]

    for i, params in enumerate(configs):
        pca = PCA(**params)
        reduced = pca.fit_transform(X)
        suffix = f"ncomp{params['n_components']}_whiten{params['whiten']}_svd{params['svd_solver']}"
        plot_clusters_2d(
            reduced,
            labels,
            "PCA",
            suffix,
            output_path,
            color_values=color_values,
            color_name=color_name,
        )

def apply_tsne(X, labels, output_path, color_values=None, color_name=None):
    """
    Applique différentes configurations de t-SNE aux données et génère des graphes 2D.

    Paramètres
    ----------
    X : np.ndarray
        Matrice des données originales (shape = [n_samples, n_features]).
    labels : array-like
        Étiquettes de cluster pour chaque point.
    output_path : str
        Dossier de sauvegarde des visualisations PNG.
    color_values : array-like, optionnel
        Variable catégorielle pour la coloration des points.
    color_name : str, optionnel
        Nom de la variable de coloration.
    """
    configs = [
        {"n_components": 2, "perplexity": 30, "learning_rate": 200, "n_iter": 1000, "metric": "cosine"},
        {"n_components": 2, "perplexity": 10, "learning_rate": 500, "n_iter": 1500, "metric": "cosine"},
        {"n_components": 2, "perplexity": 50, "learning_rate": 100, "n_iter": 2000, "metric": "cosine"},
        {"n_components": 2, "perplexity": 50, "learning_rate": 100, "n_iter": 2000, "metric": "euclidean"},
        {"n_components": 2, "perplexity": 50, "learning_rate": 100, "n_iter": 2000, "metric": "precomputed"},
    ]

    n_samples = X.shape[0]

    for i, params in enumerate(configs):
        if params["perplexity"] >= n_samples:
            print(f"[SKIPPED] t-SNE config {i}: perplexity={params['perplexity']} >= n_samples={n_samples}")
            continue

        try:
            tsne = TSNE(**params, random_state=42)
            reduced = tsne.fit_transform(X)
            suffix = f"perp{params['perplexity']}_lr{params['learning_rate']}_metric{params['metric']}"
            plot_clusters_2d(
                reduced,
                labels,
                "tSNE",
                suffix,
                output_path,
                color_values=color_values,
                color_name=color_name,
            )
        except Exception as e:
            print(f"[ERROR] t-SNE config {i} failed: {e}")



def apply_umap(X, labels, output_path, color_values=None, color_name=None):
    """
    Applique UMAP avec différentes configurations pour réduire les données à 2D
    et visualise les points (colorés par cluster ou par variable catégorielle).

    Paramètres
    ----------
    X : np.ndarray
        Données à projeter (shape = [n_samples, n_features]).
    labels : array-like
        Étiquettes de cluster.
    output_path : str
        Dossier de sortie des fichiers PNG.
    color_values : array-like, optionnel
        Variable catégorielle pour la coloration des points.
    color_name : str, optionnel
        Nom de la variable de coloration.
    """
    configs = [
        {"n_components": 2, "n_neighbors": 15, "min_dist": 0.1, "metric": "cosine"},
        {"n_components": 2, "n_neighbors": 5, "min_dist": 0.9, "metric": "cosine"},
        {"n_components": 2, "n_neighbors": 50, "min_dist": 0.01, "metric": "cosine"},
    ]

    for i, params in enumerate(configs):
        reducer = umap.UMAP(**params, random_state=42)
        reduced = reducer.fit_transform(X)
        suffix = f"nn{params['n_neighbors']}_mindist{params['min_dist']}_metric{params['metric']}"
        plot_clusters_2d(
            reduced,
            labels,
            "UMAP",
            suffix,
            output_path,
            color_values=color_values,
            color_name=color_name,
        )
