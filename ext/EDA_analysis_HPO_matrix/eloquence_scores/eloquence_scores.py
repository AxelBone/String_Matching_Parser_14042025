#!/usr/bin/env python
import os
from pathlib import Path
from functools import reduce
from itertools import combinations

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

import helpers  # ton module helpers existant

# =========================
# CONFIG
# =========================
EHR_MATRIX_PATH = Path("data/ohe_20kRennes_EHR_2025_03_10.csv")
HPO_OBO_PATH = Path("data/hpo_v2025_01_16.obo")
OUTPUT_DIR = Path("output")
OUTPUT_CSV = OUTPUT_DIR / "eloquence_scores_dataframe.csv"
DROP_COLUMN = "HP:0100021"  # petite colonne à drop si présente

ROOT_HPO_TERM = "HP:0000118"  # Phenotypic abnormality

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# 1. Chargement de la matrice EHR
# =========================
def load_ehr_matrix(path: Path, drop_col: str | None = None) -> pd.DataFrame:
    ehr_mat = pd.read_csv(path, index_col=0)
    if drop_col and drop_col in ehr_mat.columns:
        ehr_mat = ehr_mat.drop(columns=[drop_col])
    return ehr_mat


# =========================
# 2. Quantité : nb de HPO par EHR
# =========================
def convert_ehr_to_number_of_hpo(ehr: pd.DataFrame) -> pd.DataFrame:
    number_hpo_df = pd.DataFrame(index=ehr.index, columns=["nb_hpo"])
    number_hpo_df["nb_hpo"] = ehr.sum(axis=1)
    return number_hpo_df


# =========================
# 3. Spécificité : profondeur moyenne des HPO
# =========================
def build_hpo_depth_dict(obo_path: Path, root_term: str) -> dict:
    """Construit un dict HPO -> profondeur (distance à la racine) dans l'ontologie."""
    hpo_graph = helpers.load_ontology(str(obo_path))
    pheno_subgraph = helpers.subset_ontology_by_term(hpo_graph, root_term)
    depth_dict = helpers.create_node_to_root_distance_dict(pheno_subgraph)
    return depth_dict, pheno_subgraph


def convert_ehr_to_mean_depth_hpo(ehr: pd.DataFrame, hpo_depth_dict: dict) -> pd.DataFrame:
    """
    Convert an EHR matrix with binary HPO annotations to mean HPO depth values.
    """
    mean_depth_df = pd.DataFrame(index=ehr.index, columns=["mean_hpo_depth"], dtype=float)

    for ehr_id in ehr.index:
        row = ehr.loc[ehr_id]
        present_hpo_terms = row[row == 1].index.tolist()

        if not present_hpo_terms:
            mean_depth_df.loc[ehr_id, "mean_hpo_depth"] = np.nan
            continue

        depths = [
            hpo_depth_dict[hpo]
            for hpo in present_hpo_terms
            if hpo in hpo_depth_dict
        ]

        if depths:
            mean_depth_df.loc[ehr_id, "mean_hpo_depth"] = float(np.mean(depths))
        else:
            mean_depth_df.loc[ehr_id, "mean_hpo_depth"] = np.nan

    return mean_depth_df


# =========================
# 4. Diversité : moyenne des shortest paths entre termes HPO
# =========================
def compute_all_pairs_shortest_paths(pheno_subgraph: nx.MultiDiGraph) -> dict:
    undirected_graph = pheno_subgraph.to_undirected()
    all_shortest_paths = dict(nx.all_pairs_shortest_path_length(undirected_graph))
    return all_shortest_paths


def weighted_mean_shortest_path_per_ehr(
    ehr_mat: pd.DataFrame, all_shortest_paths: dict
) -> pd.DataFrame:
    """
    Pour chaque EHR : moyenne des distances (shortest path) entre paires de HPO présents.
    """
    weighted_mean_distances = []

    for patient_id, row in ehr_mat.iterrows():
        terms = row[row == 1].index.tolist()

        if len(terms) < 2:
            weighted_mean_distances.append(np.nan)
            continue

        distances = []
        for term1, term2 in combinations(terms, 2):
            if term2 in all_shortest_paths.get(term1, {}):
                distances.append(all_shortest_paths[term1][term2])

        if distances:
            weighted_mean_distances.append(float(np.mean(distances)))
        else:
            weighted_mean_distances.append(np.nan)

    return pd.DataFrame(
        {"weighted_mean_shortest_path": weighted_mean_distances},
        index=ehr_mat.index,
    )


# =========================
# 5. IC : information content moyen par EHR
# =========================
def convert_ehr_to_mean_ic(ehr: pd.DataFrame) -> pd.DataFrame:
    """
    IC(hpo) = -log2(freq(hpo)), puis moyenne des IC des termes présents par EHR.
    """
    term_counts = ehr.sum(axis=0)
    term_freq = term_counts / len(ehr)

    ic_scores = -np.log2(term_freq)
    ic_scores.replace([np.inf, -np.inf], 0, inplace=True)

    ehr_ic_matrix = ehr * ic_scores

    ehr_richness_scores = pd.DataFrame(index=ehr.index)
    ehr_term_counts = ehr.sum(axis=1).replace(0, np.nan)
    ehr_richness_scores["ic_score"] = ehr_ic_matrix.sum(axis=1) / ehr_term_counts

    return ehr_richness_scores


# =========================
# 6. Fusion des métriques
# =========================
def build_super_df(ehr_mat: pd.DataFrame, depth_dict: dict, all_shortest_paths: dict) -> pd.DataFrame:
    metrics_df = convert_ehr_to_number_of_hpo(ehr_mat)
    ehr_depth = convert_ehr_to_mean_depth_hpo(ehr_mat, depth_dict)
    diversity_df = weighted_mean_shortest_path_per_ehr(ehr_mat, all_shortest_paths)
    ehr_richness = convert_ehr_to_mean_ic(ehr_mat)

    dfs = [metrics_df, ehr_depth, diversity_df, ehr_richness]
    super_df = reduce(lambda left, right: left.join(right), dfs)
    return super_df


# =========================
# 7. Boxplots (PNG)
# =========================
def plot_boxplots(super_df: pd.DataFrame, outdir: Path):
    """
    - 1 boxplot par colonne (4 PNG)
    - 1 figure avec les 4 boxplots côte à côte (1 PNG)
    """
    metrics = ["nb_hpo", "mean_hpo_depth", "weighted_mean_shortest_path", "ic_score"]

    # Filtrer aux colonnes présentes
    metrics = [m for m in metrics if m in super_df.columns]

    # Boxplot individuel par métrique
    for metric in metrics:
        data = super_df[metric].dropna()

        if data.empty:
            continue

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.boxplot(data, vert=True, labels=[metric])
        ax.set_title(f"Distribution de {metric}")
        ax.set_ylabel(metric)
        plt.tight_layout()
        fig.savefig(outdir / f"boxplot_{metric}.png", dpi=300)
        plt.close(fig)

    # Figure combinée côte à côte
    if metrics:
        fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 5), sharey=False)

        # axes peut être un Axes ou un ndarray
        if len(metrics) == 1:
            axes = [axes]

        for ax, metric in zip(axes, metrics):
            data = super_df[metric].dropna()
            if data.empty:
                continue
            ax.boxplot(data, vert=True)
            ax.set_title(metric)
            ax.set_xticks([])

        fig.suptitle("Distribution des métriques HPO par patient", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(outdir / "boxplots_all_metrics.png", dpi=300)
        plt.close(fig)


# =========================
# main
# =========================
def main():
    print(f"Loading EHR matrix from: {EHR_MATRIX_PATH}")
    ehr_mat = load_ehr_matrix(EHR_MATRIX_PATH, DROP_COLUMN)

    print(f"Loading ontology from: {HPO_OBO_PATH}")
    depth_dict, pheno_subgraph = build_hpo_depth_dict(HPO_OBO_PATH, ROOT_HPO_TERM)

    print("Computing all-pairs shortest paths (can be long the first time)...")
    all_shortest_paths = compute_all_pairs_shortest_paths(pheno_subgraph)

    print("Building metrics dataframe...")
    super_df = build_super_df(ehr_mat, depth_dict, all_shortest_paths)

    # Forcer les colonnes numériques
    for col in super_df.columns:
        super_df[col] = pd.to_numeric(super_df[col], errors="coerce")

    print(f"Saving metrics table to CSV: {OUTPUT_CSV}")
    super_df.to_csv(OUTPUT_CSV, sep=";")

    print("Generating boxplots...")
    plot_boxplots(super_df, OUTPUT_DIR)

    print("Done.")


if __name__ == "__main__":
    main()
