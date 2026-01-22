import numpy as np
import pandas as pd
from scipy.special import logit
import os
from scipy.stats import fisher_exact, zscore
from statsmodels.stats.multitest import multipletests
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

def cluster_size_summary(labels, ehr_df, distances_df, output_path=None):
    """
    Calcule un résumé statistique global pour chaque cluster identifié.

    Pour chaque cluster :
    - Nombre d’EHRs
    - Pourcentage du total
    - Distance intra-cluster moyenne
    - Distance inter-cluster moyenne (globale)
    - Ratio intra/inter

    Paramètres
    ----------
    labels : array-like
        Étiquettes de cluster pour chaque EHR.
    ehr_df : pd.DataFrame
        Matrice binaire EHR x HPO utilisée pour l’analyse.
    distances_df : pd.DataFrame
        Matrice carrée des distances entre EHRs (même ordre que labels).
    output_path : str or Path, optional
        Chemin pour sauvegarder le résumé au format CSV.

    Retour
    ------
    summary_df : pd.DataFrame
        Tableau contenant une ligne par cluster avec les statistiques associées.
    """

    total_ehrs = len(ehr_df)
    summary_data = []

    def intra_cluster_distance(distances_df, labels):
        intra_distances = {}
        for cluster_id in np.unique(labels):
            cluster_indices = np.where(labels == cluster_id)[0]
            cluster_distances = [
                distances_df.iloc[i, j]
                for i in cluster_indices
                for j in cluster_indices if i != j
            ]
            if cluster_distances:
                intra_distances[cluster_id] = np.mean(cluster_distances)
        return intra_distances

    intra_distances = intra_cluster_distance(distances_df, labels)

    def inter_cluster_distance(distances_df, labels):
        inter_distances = [
            distances_df.iloc[i, j]
            for i in range(len(labels))
            for j in range(i + 1, len(labels))
            if labels[i] != labels[j]
        ]
        return np.mean(inter_distances) if inter_distances else np.nan

    inter_distance = inter_cluster_distance(distances_df, labels)

    for cluster_id in np.unique(labels):
        cluster_size = np.sum(labels == cluster_id)
        size_ratio = round((cluster_size / total_ehrs) * 100, 2)

        intra_dist = intra_distances.get(cluster_id, np.nan)
        ratio_intra_inter = intra_dist / inter_distance if inter_distance != 0 else np.nan

        summary_data.append({
            'Cluster': cluster_id,
            'Nb EHR in Cluster': cluster_size,
            'Size Ratio (%)': size_ratio,
            'Intra-cluster Distance': round(intra_dist, 4) if not np.isnan(intra_dist) else np.nan,
            'Inter-cluster Distance': round(inter_distance, 4) if not np.isnan(inter_distance) else np.nan,
            'Intra/Inter Distance Ratio': round(ratio_intra_inter, 4) if not np.isnan(ratio_intra_inter) else np.nan
        })

    summary_df = pd.DataFrame(summary_data)

    # Sauvegarde optionnelle en CSV
    if output_path:
        summary_df.to_csv(output_path, index=False)
        print(f"Résumé sauvegardé sous : {output_path}")

    return summary_df



def plot_cluster_pie_chart(cluster_table, output_folder, n_clusters, random_colors=None, title=None, show=True):
    """
    Génére un camembert interactif illustrant la répartition des EHRs par cluster.

    Paramètres
    ----------
    cluster_table : pd.DataFrame
        Doit contenir au minimum les colonnes 'Cluster' et 'Nb EHR in Cluster'.
    output_folder : str
        Dossier dans lequel enregistrer le graphique interactif.
    n_clusters : int
        Nombre de clusters utilisés (affiché dans le titre).
    random_colors : list of str, optional
        Liste de couleurs personnalisées compatibles avec Plotly.
    title : str, optional
        Titre personnalisé du graphique.
    show : bool
        Affiche ou non le graphique dans l’environnement d’exécution.

    Sauvegarde
    ---------
    Fichier HTML interactif `camembert_<n_clusters>.html` dans le dossier spécifié.
    """

    if title is None:
        title = f"Répartition des EHR par cluster (k={n_clusters})"

    fig = px.pie(
        cluster_table,
        names="Cluster",
        values="Nb EHR in Cluster",
        title=title,
        color="Cluster",
        color_discrete_sequence=random_colors
    )

    # Affichage interactif dans le notebook ou interface
    if show:
        fig.show()

    # Sauvegarde du graphique
    output_path = os.path.join(output_folder, f"camembert_{n_clusters}.html")
    fig.write_html(output_path)

    print(f"[INFO] Graphique sauvegardé : {output_path}")


def format_summary_table_proportion(cluster_summary_df, cluster_id, 
                                     pval_threshold=0.05, 
                                     proportion_threshold=0.1, 
                                     sort_by='Proportion', 
                                     top_n=None, 
                                     output_path=None):
    """
    Formate et filtre le tableau des résultats d'enrichissement pour un cluster donné.

    Applique des seuils sur la p-value et la proportion, et prépare le tableau pour affichage ou export.

    Paramètres
    ----------
    cluster_summary_df : pd.DataFrame
        Résultats d'enrichissement globaux (issue de `cluster_summary_table_global_approach`).
    cluster_id : int
        ID du cluster à analyser.
    pval_threshold : float, default=0.05
        Seuil de significativité (corrigé ou brut).
    proportion_threshold : float, default=0.1
        Seuil minimal de proportion dans le cluster.
    sort_by : str, default='Proportion'
        Colonne utilisée pour le tri (ex: 'Proportion', 'Enrichment_num').
    top_n : int, optional
        Nombre de lignes à conserver (None = tous).
    output_path : str, optional
        Si fourni, le tableau est sauvegardé en CSV.

    Retour
    ------
    summary_table : pd.DataFrame
        Tableau formaté et trié des termes enrichis.
    """

    cluster_data = cluster_summary_df[cluster_summary_df['Cluster'] == cluster_id].copy()

    if 'Fisher P-value' not in cluster_data.columns:
        raise KeyError("'Fisher P-value' column missing from the DataFrame.")
    
    pval_col = 'Corrected P-value' if 'Corrected P-value' in cluster_data.columns else 'Fisher P-value'
    cluster_data['Significatif'] = cluster_data[pval_col] < pval_threshold

    cluster_data = cluster_data[cluster_data['Proportion in Cluster'] >= proportion_threshold]

    summary_table = cluster_data[[
        'HPO Name', 'HPO Code', 'Proportion in Cluster', 'Enrichment Ratio', pval_col, 'Significatif'
    ]].copy()

    summary_table.rename(columns={
        'HPO Name': 'HPO Name',
        'HPO Code': 'Code',
        'Proportion in Cluster': 'Proportion',
        'Enrichment Ratio': 'Enrichment_num',
        pval_col: 'p-value_num'
    }, inplace=True)

    summary_table['Z-score'] = np.nan
    if 'Z-score' in cluster_data.columns:
        summary_table['Z-score'] = cluster_data['Z-score']

    summary_table = summary_table.sort_values(
        by=[sort_by, 'Enrichment_num', 'p-value_num'], 
        ascending=[False, False, False]
    )

    if top_n:
        summary_table = summary_table.head(top_n)

    summary_table['Label'] = range(1, len(summary_table) + 1)

    summary_table['Enrichment'] = summary_table['Enrichment_num'].apply(lambda x: f"{x:.2f}")
    summary_table['p-value'] = summary_table['p-value_num'].apply(lambda x: f"{x:.1e}")
    summary_table['Z-score'] = summary_table['Z-score'].apply(lambda x: f"{x:.2f}" if not np.isnan(x) else "N/A")

    summary_table = summary_table[[
        'Label', 'HPO Name', 'Code', 'Proportion', 'Enrichment', 'p-value', 'Z-score', 'Significatif'
    ]]

    # Sauvegarde CSV si chemin précisé
    if output_path:
        summary_table.to_csv(output_path, index=False)
        print(f"Tableau formaté sauvegardé dans : {output_path}")

    return summary_table


def ehr_level_summary_table_with_enrichment(labels, ehr_df, enriched_terms_df,
                                            enrichment_threshold=0.1,
                                            coverage_threshold=0.1,
                                            output_path=None):
    """
    Produit un tableau récapitulatif pour chaque EHR en évaluant l’adhérence au profil enrichi de son cluster.

    Pour chaque EHR :
    - Nombre de termes HPO
    - Termes enrichis utilisés
    - Ratio de couverture et score d’enrichissement
    - Drapeau d’outlier si seuils non atteints

    Paramètres
    ----------
    labels : array-like
        Étiquettes de cluster pour chaque EHR.
    ehr_df : pd.DataFrame
        Matrice binaire EHR x HPO.
    enriched_terms_df : pd.DataFrame
        Terme enrichis par cluster (issue de `cluster_summary_table_global_approach`).
    enrichment_threshold : float
        Seuil minimum pour le score d’enrichissement (used_enriched / nb_terms).
    coverage_threshold : float
        Seuil minimum de couverture (nb enriched utilisés / nb enrichis du cluster).
    output_path : str, optional
        Chemin de sauvegarde au format CSV.

    Retour
    ------
    ehr_summary_df : pd.DataFrame
        Tableau par patient avec ses scores et indicateurs d’adhérence au cluster.
    """

    enriched_terms_by_cluster = enriched_terms_df.groupby('Cluster')['HPO Code'].apply(set).to_dict()

    ehr_summary = []

    for idx, (i, row) in enumerate(ehr_df.iterrows()):
        ehr_id = i
        cluster_id = labels[idx]
        cluster_terms = enriched_terms_by_cluster.get(cluster_id, set())
        cluster_size = np.sum(labels == cluster_id)

        ehr_terms = set(row.index[row == 1])
        intersect_terms = ehr_terms & cluster_terms

        nb_hpo_terms = len(ehr_terms)
        nb_enriched_terms_used = len(intersect_terms)
        cluster_coverage_pct = nb_enriched_terms_used / len(cluster_terms) if cluster_terms else 0
        cluster_enrichment_score = nb_enriched_terms_used / nb_hpo_terms if nb_hpo_terms else 0

        outlier_flag = int(
            cluster_enrichment_score < enrichment_threshold or
            cluster_coverage_pct < coverage_threshold
        )
        outlier_hpos = list(ehr_terms - cluster_terms)

        ehr_summary.append({
            'EHR_id': ehr_id,
            'Cluster assignment': cluster_id,
            'Cluster size': cluster_size,
            'Nb_HPO_terms': nb_hpo_terms,
            'Nb enriched terms used': nb_enriched_terms_used,
            'Used enriched terms ratio': round(nb_enriched_terms_used / len(cluster_terms), 4) if cluster_terms else 0,
            'Cluster_coverage_Pct': round(cluster_coverage_pct, 4),
            'Cluster_enrichment_score': round(cluster_enrichment_score, 4),
            'Nb outlier terms': len(outlier_hpos),
            'Outlier ratio': round(len(outlier_hpos) / nb_hpo_terms, 4) if nb_hpo_terms else 0,
            'Outlier flag': outlier_flag,
            'Matching enriched HPO terms': list(intersect_terms),
            'Outlier HPO terms': outlier_hpos
        })

    ehr_summary_df = pd.DataFrame(ehr_summary)

    if output_path:
        ehr_summary_df.to_csv(output_path, index=False)
        print(f"Résumé EHR sauvegardé dans : {output_path}")

    return ehr_summary_df


def cluster_summary_table_global_approach(labels, ehr_df, hpo_df, p_threshold=0.05, min_freq_cluster=0.1, output_path=None):
    """
    Effectue une analyse d’enrichissement HPO pour chaque cluster comparé au reste de la population.

    Méthodologie :
    - Test de Fisher pour chaque terme HPO
    - Z-score basé sur la différence de fréquence et son écart type
    - Correction FDR-BH sur les p-values
    - Filtrage par fréquence minimale dans le cluster et significativité

    Paramètres
    ----------
    labels : array-like
        Étiquettes de cluster pour chaque EHR.
    ehr_df : pd.DataFrame
        Matrice binaire des patients (EHRs) x termes HPO.
    hpo_df : pd.DataFrame
        Données de l'ontologie avec colonnes ["hpo_code", "hpo_name"].
    p_threshold : float
        Seuil de p-value corrigée pour considérer un terme significatif.
    min_freq_cluster : float
        Fréquence minimale dans un cluster pour tester un terme.
    output_path : str, optional
        Chemin de sauvegarde du fichier CSV.

    Retour
    ------
    results_df : pd.DataFrame
        Résultats de l’enrichissement pour tous les clusters (filtré).
    """

    total_ehrs = len(ehr_df)
    term_counts_total = ehr_df.sum(axis=0)
    term_freq_global = term_counts_total / total_ehrs
    hpo_name_map = hpo_df.set_index("hpo_code")["hpo_name"].to_dict()
    unique_clusters = np.unique(labels)

    all_results = []

    for cluster_id in unique_clusters:
        cluster_indices = np.where(labels == cluster_id)[0]
        ehr_cluster = ehr_df.iloc[cluster_indices]
        cluster_size = len(ehr_cluster)

        if cluster_size == 0:
            continue

        term_counts_cluster = ehr_cluster.sum(axis=0)
        term_freq_cluster = term_counts_cluster / cluster_size

        for term_code in ehr_df.columns:
            a = term_counts_cluster[term_code]
            c = term_counts_total[term_code]
            b = cluster_size - a
            d = total_ehrs - c

            if a == 0 or c == 0:
                continue

            freq_cluster_term = term_freq_cluster[term_code]
            freq_global_term = term_freq_global[term_code]

            if freq_cluster_term < min_freq_cluster:
                continue

            contingency_table = [[a, b], [c, d]]
            _, p_value = fisher_exact(contingency_table)

            enrichment_ratio = freq_cluster_term / freq_global_term if freq_global_term > 0 else np.nan
            cluster_ratio = a / c if c > 0 else np.nan
            diff = freq_cluster_term - freq_global_term
            pooled_std = np.sqrt(freq_global_term * (1 - freq_global_term) / cluster_size)
            z_score = diff / pooled_std if pooled_std > 0 else np.nan

            all_results.append({
                "Cluster": cluster_id,
                "HPO Code": term_code,
                "HPO Name": hpo_name_map.get(term_code, "Unknown"),
                "Nb EHR in Cluster": cluster_size,
                "Occurrences in Cluster": a,
                "Proportion in Cluster": freq_cluster_term,
                "Occurrences Totals": c,
                "Frq Totals": freq_global_term,
                "Cluster Ratio": cluster_ratio,
                "Enrichment Ratio": enrichment_ratio,
                "Fisher P-value": p_value,
                "Z-score": z_score
            })

    results_df = pd.DataFrame(all_results)

    if not results_df.empty:
        corrected = multipletests(results_df["Fisher P-value"], method='fdr_bh')
        results_df["Corrected P-value"] = corrected[1]
        results_df["Significant"] = results_df["Corrected P-value"] < p_threshold

        # Filtrage et tri
        results_df = results_df[results_df["Significant"]]
        results_df = results_df.sort_values(
            by=["Cluster", "Proportion in Cluster", "Corrected P-value"],
            ascending=[True, False, True]
        ).reset_index(drop=True)

    # Export CSV si demandé
    if output_path:
        results_df.to_csv(output_path, index=False)
        print(f"Tableau résumé sauvegardé dans : {output_path}")

    return results_df

def volcano_plot_cluster_enrichment(
    cluster_summary_df,
    cluster_id,
    pval_threshold=0.05,
    enrichment_threshold=1.5,
    output_path=None
):
    """
    Crée un volcano plot interactif pour visualiser les termes enrichis dans un cluster.

    Axes :
    - x : log2(enrichment ratio)
    - y : -log10(p-value)

    Met en évidence les termes statistiquement enrichis au-delà d’un seuil.

    Paramètres
    ----------
    cluster_summary_df : pd.DataFrame
        Résultats d’enrichissement (cf. `cluster_summary_table_global_approach`).
    cluster_id : int
        ID du cluster à visualiser.
    pval_threshold : float
        Seuil de p-value pour la significativité.
    enrichment_threshold : float
        Seuil minimal de ratio d’enrichissement pour être "significatif".
    output_path : str, optional
        Préfixe du fichier (HTML + PNG) à sauvegarder.

    Sortie
    ------
    Deux fichiers sauvegardés si output_path est donné :
    - Volcano plot interactif (.html)
    - Version statique PNG (.png)
    """

    data = cluster_summary_df[cluster_summary_df['Cluster'] == cluster_id].copy()

    data['log2Enrichment'] = np.log2(data['Enrichment Ratio'].replace(0, np.nan))
    data['-log10P'] = -np.log10(data['Fisher P-value'])

    data['Significatif'] = np.where(
        (data['Fisher P-value'] < pval_threshold) & (data['Enrichment Ratio'] > enrichment_threshold),
        'Significatif',
        'Non significatif'
    )

    data['Size'] = data['Occurrences Totals'] + 1
    data['Label'] = data['HPO Name']

    fig = px.scatter(
        data,
        x='log2Enrichment',
        y='-log10P',
        color='Significatif',
        size='Size',
        text='Label',
        hover_data={
            'HPO Name': True,
            'HPO Code': True,
            'Nb EHR in Cluster': True,
            'Frq Totals': True,
            'Enrichment Ratio': ':.3f',
            'Fisher P-value': ':.2e',
            'log2Enrichment': ':.2f',
            '-log10P': ':.2f',
            'Size': False
        },
        color_discrete_map={'Significatif': 'red', 'Non significatif': 'grey'},
        labels={
            'log2Enrichment': 'log2(Enrichment Ratio)',
            '-log10P': '-log10(p-value)',
            'Significatif': 'Statut'
        },
        title=f'Volcano Plot interactif — Cluster {cluster_id}'
    )

    fig.add_hline(
        y=-np.log10(pval_threshold),
        line_dash="dash",
        line_color="blue",
        line_width=3,
        annotation_text=f"p = {pval_threshold}",
        annotation_position="top left"
    )

    fig.add_vline(
        x=np.log2(enrichment_threshold),
        line_dash="dash",
        line_color="green",
        line_width=3,
        annotation_text=f"Enrichment > {enrichment_threshold}",
        annotation_position="bottom right"
    )

    fig.update_traces(
        marker=dict(
            line=dict(width=1, color='DarkSlateGrey')
        ),
        textposition='top center',
        textfont_size=10,
        texttemplate="%{text}"
    )

    fig.update_layout(
        legend_title_text='',
        template='simple_white',
        showlegend=True,
        margin=dict(l=50, r=50, t=50, b=50)
    )

    # Sauvegarde HTML interactif
    if output_path:
        html_path = output_path + ".html"
        pio.write_html(fig, file=html_path, auto_open=False)
        print(f"Volcano plot interactif sauvegardé dans : {html_path}")

        # Sauvegarde PNG statique
        png_path = output_path + ".png"
        fig.write_image(png_path)
        print(f"Volcano plot PNG sauvegardé dans : {png_path}")

def compare_cluster_vs_noncluster_plot(
    cluster_summary_df,
    cluster_id,
    highlight_significant=True,
    output_path=None
):
    """
    Affiche un scatter plot comparant les fréquences des termes HPO dans le cluster vs hors-cluster.

    Chaque point représente un terme HPO.
    - Axe X : fréquence hors cluster
    - Axe Y : fréquence dans le cluster
    - La diagonale x = y indique absence d'enrichissement

    Paramètres
    ----------
    cluster_summary_df : pd.DataFrame
        Résultats d’enrichissement par cluster.
    cluster_id : int
        Identifiant du cluster à visualiser.
    highlight_significant : bool, default=True
        Coloration des points significatifs en rouge si True.
    output_path : str, optional
        Chemin (sans extension) pour sauvegarder le graphique (HTML et PNG).

    Sortie
    ------
    Deux fichiers sauvegardés si output_path est fourni :
    - HTML interactif
    - PNG statique
    """
    
    df = cluster_summary_df[cluster_summary_df["Cluster"] == cluster_id].copy()

    # Taille totale estimée
    total_ehr = df["Occurrences Totals"].iloc[0] / df["Frq Totals"].iloc[0]
    cluster_size = df["Nb EHR in Cluster"].iloc[0]
    noncluster_size = total_ehr - cluster_size

    # Fréquence hors cluster
    df["Freq Hors Cluster"] = (df["Occurrences Totals"] - df["Occurrences in Cluster"]) / noncluster_size

    # Couleur selon significativité
    if highlight_significant:
        df["Statut"] = np.where(df["Significant"], "Significatif", "Non significatif")
        color = "Statut"
        color_map = {"Significatif": "red", "Non significatif": "grey"}
    else:
        color = None
        color_map = None

    # Base scatter plot
    fig = px.scatter(
        df,
        x="Freq Hors Cluster",
        y="Proportion in Cluster",
        color=color,
        text="HPO Name",
        size="Occurrences Totals",
        color_discrete_map=color_map,
        hover_data={
            "HPO Name": True,
            "HPO Code": True,
            "Occurrences in Cluster": True,
            "Proportion in Cluster": ':.3f',
            "Freq Hors Cluster": ':.3f',
            "Fisher P-value": ':.2e',
            "Corrected P-value": ':.2e',
            "Z-score": ':.2f'
        },
        labels={
            "Freq Hors Cluster": "Fréquence hors cluster",
            "Proportion in Cluster": "Fréquence dans le cluster"
        },
        title=f"Fréquence Cluster vs. Hors Cluster — Cluster {cluster_id}"
    )

    # Définir les bornes pour tracer la diagonale
    all_freqs = pd.concat([df["Freq Hors Cluster"], df["Proportion in Cluster"]])
    min_val = all_freqs.min() * 0.95
    max_val = all_freqs.max() * 1.05

    # Ajout de la diagonale comme trace manuelle
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            line=dict(color='black', dash='dash', width=2),
            name='x = y',
            showlegend=True
        )
    )

    fig.update_traces(
        textposition="top center",
        textfont_size=10,
        marker=dict(line=dict(width=1, color="DarkSlateGrey"))
    )

    fig.update_layout(
        template="simple_white",
        legend_title_text='',
        margin=dict(l=40, r=40, t=50, b=40),
        height=600
    )

    # Sauvegarde HTML interactif et PNG statique si output_path est fourni
    if output_path:
        # Sauvegarde en HTML
        html_path = output_path + ".html"
        pio.write_html(fig, file=html_path, auto_open=False)
        print(f"Scatter plot interactif sauvegardé dans : {html_path}")

        # Sauvegarde en PNG
        png_path = output_path + ".png"
        fig.write_image(png_path)
        print(f"Scatter plot PNG sauvegardé dans : {png_path}")