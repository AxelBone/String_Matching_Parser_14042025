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
