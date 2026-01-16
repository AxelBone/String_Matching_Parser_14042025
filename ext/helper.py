import ast
import pandas as pd

def normalize_sentence_list(x):
    # Cas déjà bon : liste de strings
    if isinstance(x, list) and all(isinstance(s, str) for s in x):
        # Vérif rapide : si la première string ressemble à une liste sérialisée, on traite
        if len(x) == 1 and x[0].strip().startswith("[") and x[0].strip().endswith("]"):
            try:
                inner = ast.literal_eval(x[0])
                if isinstance(inner, list):
                    return [str(s) for s in inner]
            except Exception:
                return x
        return x

    # Cas liste avec 1 élément qui est une string-représentation de liste
    if isinstance(x, list) and len(x) == 1 and isinstance(x[0], str):
        s = x[0].strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                inner = ast.literal_eval(s)
                if isinstance(inner, list):
                    return [str(e) for e in inner]
            except Exception:
                return []

    # Cas string directement : "[...]" -> liste
    if isinstance(x, str):
        s = x.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                inner = ast.literal_eval(s)
                if isinstance(inner, list):
                    return [str(e) for e in inner]
            except Exception:
                return []
        else:
            # une phrase seule -> on la met dans une liste
            return [s]

    # NaN, None, autres
    if pd.isna(x):
        return []

    # fallback
    return []

df["CLEAN_FR_SPLIT"] = df["CLEAN_FR_SPLIT"].apply(normalize_sentence_list)


df["n_sentences"] = df["CLEAN_FR_SPLIT"].apply(
    lambda x: len(x) if isinstance(x, list) else 0
)

df["n_sentences"].describe()


idx = 0  # ou un autre index
print("Sentences list:", df["CLEAN_FR_SPLIT"].iloc[idx])
print("n_sentences:", df["n_sentences"].iloc[idx])



### Visu
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 5))

# Couleur demandée
violin_color = "#08519c"
box_edge_color = "#000000"  # bordure boxplot noire
box_face_color = "#ffffff"  # intérieur blanc

# Violin plot
sns.violinplot(
    data=df_doc,
    x="n_words",
    inner=None,
    color=violin_color,
    alpha=0.5  # transparence
)

# Boxplot superposé
sns.boxplot(
    data=df_doc,
    x="n_words",
    width=0.2,
    fliersize=2,
    linewidth=1.2,
    boxprops=dict(facecolor=box_face_color, edgecolor=box_edge_color),
    medianprops=dict(color=box_edge_color),
    whiskerprops=dict(color=box_edge_color),
    capprops=dict(color=box_edge_color),
)

plt.title("Distribution du nombre de mots par document")
plt.xlabel("Nombre de mots")
plt.tight_layout()
plt.show()


#####

plt.figure(figsize=(7, 5))
sns.scatterplot(
    data=df_doc,
    x='n_words',
    y='n_hpo',
    alpha=0.3
)
sns.regplot(
    data=df_doc,
    x='n_words',
    y='n_hpo',
    scatter=False,
    color='red'
)

plt.title("Nombre de mots vs nombre de HPO (document) + trend")
plt.xlabel("Nombre de mots")
plt.ylabel("Nombre de HPO")
plt.tight_layout()
plt.show()

#### Régression + scatter
import numpy as np
from sklearn.linear_model import LinearRegression

X = df_doc['n_words'].values.reshape(-1, 1)
y = df_doc['n_hpo'].values

model = LinearRegression().fit(X, y)

slope = model.coef_[0]
intercept = model.intercept_
r2 = model.score(X, y)

plt.figure(figsize=(7, 5))
sns.scatterplot(x=df_doc['n_words'], y=df_doc['n_hpo'], alpha=0.3)
plt.plot(df_doc['n_words'], model.predict(X), color='red')

plt.text(
    0.05, 0.95,
    f'y = {slope:.3f} x + {intercept:.2f}\nR² = {r2:.3f}',
    transform=plt.gca().transAxes,
    va='top'
)

plt.title("n_words vs n_hpo (régression linéaire)")
plt.xlabel("Nombre de mots")
plt.ylabel("Nombre de HPO")
plt.tight_layout()
plt.show()


from scipy.stats import spearmanr
rho, p = spearmanr(df_doc['n_words'], df_doc['n_hpo'])
print(f"Spearman ρ = {rho:.3f}, p = {p:.3e}")
