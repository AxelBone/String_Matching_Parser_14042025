import os
import spacy
import json
from transformers import MarianTokenizer, AutoModelForSeq2SeqLM

# Load model and tokenizer
nlp = spacy.load("en_core_web_sm")

mname = 'Helsinki-NLP/opus-mt-fr-en'
tokenizer = MarianTokenizer.from_pretrained(mname, local_files_only=True)
model = AutoModelForSeq2SeqLM.from_pretrained(mname)

data_path = "../data/FR_simulatedCR_2025_02_14/"
output_path = "../output/translated_chunked_reports.json"
files = os.listdir(data_path)
extension = ".txt"

original_reports = {}
translated_reports = {}

for file in files:
    if file.endswith(extension):
        filename = file.replace(extension, "")

        with open(os.path.join(data_path, file), 'r', encoding='utf8', errors='ignore') as f:
            doc = nlp(f.read())

        # phrases originales
        text = [sent.text for sent in doc.sents]
        original_reports[filename] = text

        # phrases traduites
        full_translation = []
        for sentence in text:
            input_ids = tokenizer.encode(sentence, return_tensors="pt")
            outputs = model.generate(input_ids)
            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
            full_translation.append(decoded)

        translated_reports[filename] = full_translation


with open(output_path, "w", encoding="utf-8") as fh:
    json.dump(translated_reports, fh, indent=2, ensure_ascii=False)
