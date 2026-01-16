import pandas as pd
import spacy

INPUT_CSV = "input.csv"
OUTPUT_CSV = "output.csv"
TEXT_COL = "CLEAN_FR_TEXT"
NEW_COL = "CLEAN_FR_SPLIT"
PRINT_EVERY = 100

def init_pipeline():
    # charge le modèle français
    nlp = spacy.load("fr_core_news_sm", disable=["tagger", "parser", "ner", "lemmatizer"])
    # On réactive juste ce qu’il faut pour les phrases
    nlp.enable_pipe("senter") if "senter" in nlp.pipe_names else None
    return nlp

def split_sentences(nlp, text):
    if not isinstance(text, str) or not text.strip():
        return []
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]

def main():
    df = pd.read_csv(INPUT_CSV, encoding="utf-8")
    if TEXT_COL not in df.columns:
        raise ValueError(f"Colonne manquante : {TEXT_COL}")

    texts = df[TEXT_COL].tolist()
    total = len(texts)

    print(f"Nombre total de documents : {total}")
    nlp = init_pipeline()

    results = []
    for i, text in enumerate(texts, start=1):
        results.append(split_sentences(nlp, text))
        if i % PRINT_EVERY == 0 or i == total:
            restant = total - i
            print(f"Traités : {i}/{total} (restant : {restant})")

    df[NEW_COL] = results
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"Sauvegardé dans {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
