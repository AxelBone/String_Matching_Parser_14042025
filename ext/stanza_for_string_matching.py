import pandas as pd
import stanza

# -------------------------------------------------------------------
# Paramètres
# -------------------------------------------------------------------
INPUT_CSV = "input.csv"
OUTPUT_CSV = "output.csv"
TEXT_COL = "CLEAN_FR_TEXT"
NEW_COL = "CLEAN_FR_SPLIT"

# Tous les combien de textes on affiche la progression
PRINT_EVERY = 100
# -------------------------------------------------------------------


def init_pipeline():
    """
    Initialise le pipeline Stanza pour le français.
    Lancer une fois séparément: stanza.download('fr')
    """
    # Une seule fois sur ta machine (puis commenter) :
    # stanza.download('fr')

    nlp = stanza.Pipeline(
        lang="fr",
        processors="tokenize",
        use_gpu=False
    )
    return nlp


def split_sentences(nlp, text):
    """
    Applique Stanza à un texte et renvoie une LISTE de phrases.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    doc = nlp(text)
    sentences = [" ".join(w.text for w in sent.words) for sent in doc.sentences]
    return sentences


def main():
    # 1) Charger le CSV
    df = pd.read_csv(INPUT_CSV, encoding="utf-8")

    if TEXT_COL not in df.columns:
        raise ValueError(f"La colonne '{TEXT_COL}' n'existe pas dans le fichier CSV.")

    # 2) Récupérer la colonne en vecteur (liste Python)
    texts = df[TEXT_COL].tolist()
    total = len(texts)

    print(f"Nombre total de documents à traiter : {total}")

    # 3) Initialiser Stanza une seule fois
    nlp = init_pipeline()

    # 4) Boucle texte par texte
    results = []
    for i, text in enumerate(texts, start=1):
        sentences = split_sentences(nlp, text)
        results.append(sentences)

        # Affichage de la progression
        if i % PRINT_EVERY == 0 or i == total:
            restant = total - i
            print(f"Traités : {i}/{total} (restant : {restant})")

    # 5) Repose le vecteur dans le DataFrame
    df[NEW_COL] = results

    # 6) Sauvegarde
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"Fichier sauvegardé dans {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
