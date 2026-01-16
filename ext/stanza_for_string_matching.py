import pandas as pd
import stanza

# -------------------------------------------------------------------
# 1) Paramètres à adapter
# -------------------------------------------------------------------
INPUT_CSV = "input.csv"      # chemin de ton csv d'entrée
OUTPUT_CSV = "output.csv"    # chemin de sauvegarde du csv de sortie
TEXT_COL = "CLEAN_FR_TEXT"   # nom de la colonne texte d'entrée
NEW_COL = "CLEAN_FR_SPLIT"   # nom de la nouvelle colonne de phrases
SENT_SEP = " ||| "           # séparateur entre phrases dans la nouvelle colonne
# -------------------------------------------------------------------


def init_pipeline():
    """
    Initialise le pipeline Stanza pour le français.
    Lance stanza.download('fr') une seule fois avant la première utilisation.
    """
    # À faire une seule fois sur ta machine (sinon commente cette ligne) :
    # stanza.download('fr')
    nlp = stanza.Pipeline(
        lang="fr",
        processors="tokenize",
        use_gpu=False  # passe à True si tu as une GPU correctement configurée
    )
    return nlp


def split_sentences_factory(nlp):
    """
    Ferme le pipeline dans une closure pour éviter de le recréer à chaque appel.
    Retourne une fonction split_sentences(text).
    """
    def split_sentences(text):
        # Gérer les NaN ou valeurs vides
        if not isinstance(text, str) or not text.strip():
            return text

        doc = nlp(text)
        # On reconstruit chaque phrase à partir des tokens
        sentences = [" ".join([w.text for w in sent.words]) for sent in doc.sentences]

        # On renvoie un seul string avec les phrases séparées par SENT_SEP
        return SENT_SEP.join(sentences)

    return split_sentences


def main():
    # 1) Charger le CSV
    df = pd.read_csv(INPUT_CSV, encoding="utf-8")

    if TEXT_COL not in df.columns:
        raise ValueError(f"La colonne '{TEXT_COL}' n'existe pas dans le fichier CSV.")

    # 2) Initialiser Stanza
    nlp = init_pipeline()
    split_sentences = split_sentences_factory(nlp)

    # 3) Appliquer Stanza à chaque document de CLEAN_FR_TEXT
    df[NEW_COL] = df[TEXT_COL].apply(split_sentences)

    # 4) Sauvegarder le DataFrame
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"Fichier sauvegardé dans : {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
