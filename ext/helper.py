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


######


plt.figure(figsize=(7, 5))

sns.kdeplot(
    data=df_kde,
    x='n_sentences',
    y='n_hpo',
    levels=10,        # nombre de courbes
    thresh=0.01,      # seuil minimal de densité
    fill=False,       # uniquement les lignes, pas de remplissage
    color='#08519c',  # bleu foncé pour rester cohérent
    linewidths=1.2
)

plt.title("Densité 2D n_sentences vs n_hpo")
plt.xlabel("Nombre de phrases")
plt.ylabel("Nombre de HPO")
plt.tight_layout()
plt.show()



#### Variante avec fond

plt.figure(figsize=(7, 5))

sns.kdeplot(
    data=df_kde,
    x='n_sentences',
    y='n_hpo',
    fill=True,        # remplissage
    cmap='Blues',     # dégradé bleu
    thresh=0.01,
    levels=30
)

plt.title("Densité 2D n_sentences vs n_hpo (fond)")
plt.xlabel("Nombre de phrases")
plt.ylabel("Nombre de HPO")
plt.tight_layout()
plt.show()


### Combianison manuelle
plt.figure(figsize=(7, 5))

# 1) Fond de densité
sns.kdeplot(
    data=df_kde,
    x='n_sentences',
    y='n_hpo',
    fill=True,
    cmap='Blues',
    thresh=0.01,
    levels=30
)

# 2) Scatter par-dessus
sns.scatterplot(
    data=df_doc,
    x='n_sentences',
    y='n_hpo',
    alpha=0.2,
    color='black',
    s=10
)

# 3) Régression linéaire (si tu veux sur n_sentences vs n_hpo)
sns.regplot(
    data=df_doc,
    x='n_sentences',
    y='n_hpo',
    scatter=False,
    color='red'
)

plt.title("n_sentences vs n_hpo : densité + scatter + régression")
plt.xlabel("Nombre de phrases")
plt.ylabel("Nombre de HPO")
plt.tight_layout()
plt.show()


plt.figure(figsize=(7, 5))

sns.kdeplot(
    data=df_kde,
    x='n_sentences',
    y='n_hpo',
    levels=10,
    thresh=0.01,
    fill=False,
    color='#08519c',
    linewidths=1.2
)

plt.axis('off')  # si tu veux juste la courbe, sans axes
plt.tight_layout()
plt.savefig("density_nsent_nhpo.png", dpi=300, transparent=True)
plt.close()


######
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(7,5))
sns.kdeplot(
    df_doc["n_hpo"].dropna(),
    fill=True,
    color="#08519c",
    linewidth=2,
    alpha=0.6
)
plt.title("Courbe de densité du nombre de HPO / document")
plt.xlabel("n_hpo")
plt.tight_layout()
plt.show()


#### densité 
plt.figure(figsize=(7,5))
sns.kdeplot(
    df_doc["n_sentences"].dropna(),
    fill=True,
    color="#08519c",
    linewidth=2,
    alpha=0.6
)
plt.title("Courbe de densité du nombre de phrases / document")
plt.xlabel("n_sentences")
plt.tight_layout()
plt.show()


#### Densité ratio
df_doc["hpo_per_sentence"] = df_doc.apply(
    lambda r: r["n_hpo"]/r["n_sentences"] if r["n_sentences"]>0 else 0,
    axis=1
)
plt.figure(figsize=(7,5))
sns.kdeplot(
    df_doc["hpo_per_sentence"].dropna(),
    fill=True,
    color="#08519c",
    linewidth=2,
    alpha=0.6
)
plt.title("Densité phénotypique : HPO par phrase")
plt.xlabel("HPO / phrase")
plt.tight_layout()
plt.show()



#### 
plt.figure(figsize=(7,5))
sns.kdeplot(
    df_doc["n_hpo"].dropna(),
    fill=True,
    color="#08519c",
    linewidth=2,
    alpha=0.4
)
plt.grid(alpha=0.2)
plt.title("Densité du nombre de HPO par document")
plt.xlabel("n_hpo")
plt.tight_layout()
plt.show()



#### 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# On filtre proprement
df_plot = df_doc[['n_sentences', 'n_hpo', 'hpo_per_sentences']].dropna()

# On évite les infinities éventuelles
df_plot = df_plot.replace([np.inf, -np.inf], np.nan).dropna()

# On scale la taille des bulles pour la lisibilité
# (évite que tout soit minuscule)
size_factor = 200  # ⇐ ajustable
bubble_sizes = (df_plot['hpo_per_sentences'] + 1e-6) * size_factor

plt.figure(figsize=(8, 6))

# Scatter enrichi
plt.scatter(
    df_plot['n_sentences'],
    df_plot['n_hpo'],
    s=bubble_sizes,
    alpha=0.35,
    c="#08519c",       # bleu clinique
    edgecolor="black",
    linewidth=0.3
)

# Trend (régression linéaire simple)
sns.regplot(
    data=df_plot,
    x='n_sentences',
    y='n_hpo',
    scatter=False,
    color='red',
    line_kws={'linewidth': 2}
)

plt.xlabel("Nombre de phrases par document (n_sentences)")
plt.ylabel("Nombre de HPO par document (n_hpo)")
plt.title("Relation phrases ↔ HPO (taille = HPO / phrase)")

plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()


##### 
import pandas as pd
import numpy as np
import calplot

# s'assurer que CREATED_AT est datetime
df_doc['CREATED_AT'] = pd.to_datetime(df_doc['CREATED_AT'])

# total HPO par jour
events_hpo = (
    df_doc.groupby(df_doc['CREATED_AT'].dt.date)['n_hpo']
          .sum()
)

# conversion en Series datetime index
events_hpo = pd.Series(events_hpo.values, index=pd.to_datetime(events_hpo.index))

calplot.calplot(
    events_hpo,
    how='sum',
    cmap='Blues',
    suptitle='Total HPO par jour',
)


#### 
events_density = (
    df_doc.groupby(df_doc['CREATED_AT'].dt.date)['hpo_per_sentence']
          .mean()
)

events_density = pd.Series(events_density.values, index=pd.to_datetime(events_density.index))

calplot.calplot(
    events_density,
    how='sum',
    cmap='Purples',
    suptitle='Densité clinique (HPO/phrase) moyenne par jour'
)



##### TEmporal analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1) S'assurer que CREATED_AT est bien en datetime
df_doc['CREATED_AT'] = pd.to_datetime(df_doc['CREATED_AT'])

# 2) Fonctions utilitaires
def build_time_series(df, value_col, freq='M', agg='mean'):
    """
    df        : df_doc
    value_col : colonne à tracer ('n_words', 'n_hpo', 'n_sentences', ...)
    freq      : 'M' = mois, 'W' = semaine, 'Y' = année
    agg       : 'mean' ou 'sum'
    """
    ts = (
        df.set_index('CREATED_AT')[value_col]
          .resample(freq)
    )
    if agg == 'sum':
        ts = ts.sum()
    else:
        ts = ts.mean()
    return ts.dropna()

def plot_time_trend(ts, title, ylabel, smooth_window=12):
    """
    ts            : Series temporelle index datetime
    smooth_window : fenêtre du lissage (en points, ex: 12 mois)
    """
    # série lissée par moyenne glissante
    trend = ts.rolling(window=smooth_window, center=True, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(12, 5))

    # barres d'erreur éventuelles ? (optionnel, on ne les met pas ici)

    # points noirs (valeurs observées)
    ax.plot(ts.index, ts.values, 'o', color='black', markersize=3, label='Valeur mensuelle')

    # ligne bleue reliant les points
    ax.plot(ts.index, ts.values, color='blue', linewidth=1, label='Moyenne par période')

    # courbe rouge lissée
    ax.plot(trend.index, trend.values, color='red', linewidth=2.5, label=f'Moyenne glissante ({smooth_window} périodes)')

    ax.set_title(title, fontsize=18)
    ax.set_xlabel("Temps")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plt.show()

# On suppose que tu as déjà df_doc['n_words']
ts_words = build_time_series(df_doc, value_col='n_words', freq='M', agg='mean')

plot_time_trend(
    ts_words,
    title="Longueur moyenne des documents (mots) au cours du temps",
    ylabel="Nombre moyen de mots par document (mensuel)",
    smooth_window=12  # lissage sur 12 mois
)


# df_doc['n_hpo'] déjà calculé comme len(HPO code)
ts_hpo = build_time_series(df_doc, value_col='n_hpo', freq='M', agg='mean')

plot_time_trend(
    ts_hpo,
    title="Nombre moyen de HPO par document au cours du temps",
    ylabel="HPO moyens par document (mensuel)",
    smooth_window=12
)

ts_sent = build_time_series(df_doc, value_col='n_sentences', freq='M', agg='mean')

plot_time_trend(
    ts_sent,
    title="Nombre moyen de phrases par document au cours du temps",
    ylabel="Phrases moyennes par document (mensuel)",
    smooth_window=12
)

####### 
import pandas as pd

df_doc['CREATED_AT'] = pd.to_datetime(df_doc['CREATED_AT'])

# Exploser le dataset pour avoir une ligne par (date, HPO)
df_exp = df_doc[['CREATED_AT', 'HPO code']].explode('HPO code').dropna()

# 1) Total HPO par jour
hpo_total = df_exp.groupby(df_exp['CREATED_AT'].dt.date)['HPO code'].count()
hpo_total.index = pd.to_datetime(hpo_total.index)

# 2) Nouveaux HPO par jour
seen = set()
records = []

for d, subset in df_exp.groupby(df_exp['CREATED_AT'].dt.date)['HPO code']:
    subset = set(subset.dropna())
    new = subset - seen
    records.append((d, len(new)))
    seen |= subset

hpo_new = pd.Series({pd.to_datetime(d): n for d, n in records})


import matplotlib.pyplot as plt

fig, ax1 = plt.subplots(figsize=(12,5))

# Total HPO
ax1.plot(hpo_total.index, hpo_total.values, color='blue', alpha=0.6, label='Total HPO/jour')
ax1.set_ylabel("Total HPO", color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Nouveaux HPO (2e axe)
ax2 = ax1.twinx()
ax2.plot(hpo_new.index, hpo_new.values, color='red', linewidth=2, label='Nouveaux HPO/jour')
ax2.set_ylabel("Nouveaux HPO", color='red')
ax2.tick_params(axis='y', labelcolor='red')

fig.suptitle("Flux phénotypique clinique : total vs nouveaux HPO par jour", fontsize=16)
plt.tight_layout()
plt.show()


##### 
import pandas as pd

df_doc['CREATED_AT'] = pd.to_datetime(df_doc['CREATED_AT'])

# explode pour avoir un HPO par ligne
df_exp = df_doc[['CREATED_AT', 'HPO code']].explode('HPO code').dropna()

# total HPO par jour
hpo_total = df_exp.groupby(df_exp['CREATED_AT'].dt.date)['HPO code'].count()
hpo_total.index = pd.to_datetime(hpo_total.index)

# nouveaux HPO par jour
seen = set()
records = []
for d, subset in df_exp.groupby(df_exp['CREATED_AT'].dt.date)['HPO code']:
    subset = set(subset)
    new = subset - seen
    records.append((d, len(new)))
    seen |= subset

hpo_new = pd.Series({pd.to_datetime(d): n for d, n in records})

# lissage 30 jours
win = 30

total_smooth = hpo_total.rolling(win, center=True, min_periods=1).mean()
new_smooth   = hpo_new.rolling(win, center=True, min_periods=1).mean()


import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))

plt.plot(total_smooth.index, total_smooth.values,
         color='blue', label=f'Total HPO (lissé {win}j)', linewidth=2)

plt.plot(new_smooth.index, new_smooth.values,
         color='red', label=f'Nouveaux HPO (lissé {win}j)', linewidth=2)

plt.title("Flux phénotypique clinique dans le temps")
plt.xlabel("Temps")
plt.ylabel("Nombre de HPO par jour (lissé)")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


fig, ax1 = plt.subplots(figsize=(12,5))

ax1.plot(total_smooth.index, total_smooth.values, color='blue', linewidth=2)
ax1.set_ylabel("Total HPO/j", color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

ax2 = ax1.twinx()
ax2.plot(new_smooth.index, new_smooth.values, color='red', linewidth=2)
ax2.set_ylabel("Nouveaux HPO/j", color='red')
ax2.tick_params(axis='y', labelcolor='red')

plt.title("Total vs Nouveaux HPO (lissés 30 jours)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()



#### 
hpo_total_cum = hpo_total.cumsum()
hpo_new_cum   = hpo_new.cumsum()


import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))

plt.plot(hpo_total_cum.index, hpo_total_cum.values,
         color='blue', linewidth=2, label='HPO totaux cumulés')

plt.plot(hpo_new_cum.index, hpo_new_cum.values,
         color='red', linewidth=2, label='HPO nouveaux cumulés')

plt.title("Évolution cumulative des HPO produits vs. découverts")
plt.xlabel("Temps")
plt.ylabel("Nombre cumulatif de HPO")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()



### si très décorrélés
fig, ax1 = plt.subplots(figsize=(12,5))

ax1.plot(hpo_total_cum.index, hpo_total_cum.values, color='blue', linewidth=2)
ax1.set_ylabel("HPO totaux cumulés", color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

ax2 = ax1.twinx()
ax2.plot(hpo_new_cum.index, hpo_new_cum.values, color='red', linewidth=2)
ax2.set_ylabel("HPO nouveaux cumulés", color='red')
ax2.tick_params(axis='y', labelcolor='red')

plt.title("Évolution cumulative du phénotype")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
