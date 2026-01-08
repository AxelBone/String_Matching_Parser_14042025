import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from typing import Mapping, Optional


def compute_top_hpo_by_branch(
    ehr_hpo: pd.DataFrame,
    hpors: nx.DiGraph,
    root_id: str = "HP:0000118",
    top_n: int = 20,
) -> pd.DataFrame:
    """
    À partir d'une matrice EHR × HPO (binaire) et d'un graphe HPO orienté (racine -> feuilles),
    calcule les HPO les plus fréquents et leur branche principale dans l'ontologie.

    Parameters
    ----------
    ehr_hpo : pd.DataFrame
        Matrix patients × HPO terms (0/1)
    hpors : nx.DiGraph
        Ontology graph (racine -> feuilles), contenant un attribut 'name' sur les noeuds
    root_id : str
        HPO ID de la racine (par ex. 'HP:0000118')
    top_n : int
        Nombre de HPO les plus fréquents à garder

    Returns
    -------
    top_counts : pd.DataFrame
        Colonnes :
        - hpo_id
        - count
        - hpo_name
        - main_branch_id
        - main_branch_name
    """

    # Compter occurrences par HPO, trier, garder top_n
    counts = ehr_hpo.sum(axis=0).sort_values(ascending=False)
    counts = counts.groupby(counts.index).sum()  # au cas où duplicats
    top_counts = counts.nlargest(top_n).reset_index()
    top_counts.columns = ["hpo_id", "count"]

    # Mapping ID -> nom
    id2name = nx.get_node_attributes(hpors, "name")

    def get_path_to_root(term_id: str) -> list[str]:
        """
        Remonte de term_id vers la racine en suivant les prédécesseurs.
        hpors est supposé orienté racine -> feuilles,
        donc les parents sont les predecessors dans hpors.
        """
        path = [term_id]
        current = term_id
        while True:
            parents = list(hpors.predecessors(current))
            if not parents:
                break
            current = parents[0]
            path.append(current)
        return path

    def get_main_branch(term_id: str) -> str:
        """
        Renvoie l'ID de la branche principale : le premier enfant direct de root_id
        sur le chemin root -> ... -> term_id.
        Si le terme n'est pas sous root_id, retourne 'Other'.
        """
        path = get_path_to_root(term_id)
        if root_id in path:
            idx = path.index(root_id)
            # Si root_id n'est pas la feuille la plus profonde : l'enfant juste en dessous
            if idx > 0:
                return path[idx - 1]
        return "Other"

    # Ajouter les noms et branches
    top_counts["hpo_name"] = top_counts["hpo_id"].map(lambda h: id2name.get(h, h))
    top_counts = top_counts.sort_values(by="count", ascending=False)
    top_counts["hpo_name"] = pd.Categorical(
        top_counts["hpo_name"],
        categories=top_counts["hpo_name"],
        ordered=True,
    )

    top_counts["main_branch_id"] = top_counts["hpo_id"].map(get_main_branch)
    top_counts["main_branch_name"] = top_counts["main_branch_id"].map(
        lambda h: id2name.get(h, h)
    )

    return top_counts


def plot_top_hpo_bar(
    top_counts: pd.DataFrame,
    branch_color_mapping: Mapping[str, str],
    ax: Optional[plt.Axes] = None,
    title: str = "",
    annotate: bool = True,
    savepath: Optional[str] = None,
):
    """
    Trace un barplot horizontal des HPO les plus fréquents,
    chaque barre colorée selon la branche principale.

    Parameters
    ----------
    top_counts : pd.DataFrame
        Résultat de compute_top_hpo_by_branch (hpo_name, count, main_branch_id, main_branch_name)
    branch_color_mapping : dict
        Mapping {main_branch_id: color}, les couleurs sont définies dans le notebook
    ax : matplotlib Axes, optional
        Axe sur lequel tracer (sinon une nouvelle figure est créée)
    title : str
        Titre du graphique
    annotate : bool
        Si True, écrit la valeur de count au bout de chaque barre
    savepath : str, optional
        Si fourni, sauvegarde le graphique en PNG à ce chemin

    Returns
    -------
    ax : matplotlib Axes
    """

    # Créer un axe si nécessaire
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 0.4 * len(top_counts)))
        created_fig = True

    # Garder l'ordre décroissant
    df = top_counts.sort_values("count", ascending=False).copy()
    df["hpo_name"] = pd.Categorical(df["hpo_name"], categories=df["hpo_name"], ordered=True)

    # Positions Y
    y_pos = np.arange(len(df))

    # Couleurs : définies à partir du mapping (fallback en gris)
    colors = df["main_branch_id"].map(branch_color_mapping).fillna("lightgrey")

    # Barres horizontales
    ax.barh(y_pos, df["count"].values, color=colors)

    # Labels Y = noms des HPO
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["hpo_name"])
    ax.invert_yaxis()  # top = HPO le plus fréquent

    ax.set_xlabel("Number of occurrences")
    ax.set_ylabel("Symptoms")
    ax.set_title(title)

    # Annoter les valeurs
    if annotate:
        for i, (val, name) in enumerate(zip(df["count"].values, df["hpo_name"])):
            ax.annotate(
                f"{int(val)}",
                xy=(val, i),
                xytext=(3, 0),
                textcoords="offset points",
                va="center",
                ha="left",
                fontsize=9,
            )


    plt.tight_layout()

    if savepath is not None and created_fig:
        plt.savefig(savepath, dpi=300)

    return ax
