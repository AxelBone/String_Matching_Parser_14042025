import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def plot_similarity_distribution_from_matrix(
    sim_matrix,
    output_path: str,
    title: str = "Similarity Distribution"
):
    """
    Affiche et sauvegarde la distribution des similarités d'une matrice donnée.
    
    Arguments :
        sim_matrix : pd.DataFrame ou np.ndarray carré, sans traitement préalable
        output_path : chemin de sortie du plot .png
        title : titre de la figure
    """
    n = sim_matrix.shape[0]

    # Suppression diagonale, extraction des paires i < j
    sim_values = sim_matrix.where(np.triu(np.ones(sim_matrix.shape), k=1).astype(bool)).stack().values
    # np.ones() créé une matrice de 1 de la même forme que sim_matrix
    # np.triu(,k=1) conserve la partie supérieure triangulaire k exclut la diagonale
    # .astype(bool) transforme les 1 en true
    # where(mask) garde les valeurs de la matrice là où le masque vaut true et met NaN ailleurs
    # .stack() empile les lignes en supprimant les NaN => donne une série 1D.
    # .values() convertit la série pandas en un np.array

    # Métriques
    N = n * (n - 1) // 2
    n0 = np.sum(sim_values > 0)
    coverage = n0 / N

    # Affichage
    plt.figure(figsize=(8, 5))
    sns.histplot(sim_values, bins=40, kde=False, color='steelblue')

    # Titre avec stats
    plt.title(f"{title}\nTotal combinations: {N:,} | n₀: {n0:,} | Coverage: {coverage:.1%}", fontsize=12)
    plt.xlabel("Similarity value")
    plt.ylabel("Count")
    plt.grid(True)

    # Sauvegarde
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"✅ Distribution saved to: {output_path}")
