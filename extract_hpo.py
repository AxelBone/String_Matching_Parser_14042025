import os
from application.Global_objects import HPO_TERMS,AC_OBJ
import numpy as np
import pandas as pd
import re
import csv
from nltk import ngrams,RegexpTokenizer
from nltk.tokenize.util import regexp_span_tokenize
import json
import application.ParserSM as psm

# Directory containing the files
input_directory = "/Users/axel/Documents/Axel/stages/IaDiag_stageM2/docs_transferes/Moussa/StringMatching_14042022/0_data/reports/"
output_directory = "/Users/axel/Documents/Axel/stages/IaDiag_stageM2/docs_transferes/Moussa/StringMatching_14042022/0_data/new_reports/"
output_directory_empty = "/Users/axel/Documents/Axel/stages/IaDiag_stageM2/docs_transferes/Moussa/StringMatching_14042022/0_data/empty_reports/"
output_directory_exceptions = "/Users/axel/Documents/Axel/stages/IaDiag_stageM2/docs_transferes/Moussa/StringMatching_14042022/0_data/exceptions_reports/"

# Assure que les dossiers existent
os.makedirs(output_directory, exist_ok=True)
os.makedirs(output_directory_empty, exist_ok=True)
os.makedirs(output_directory_exceptions, exist_ok=True)

for filename in os.listdir(input_directory):
    print("File Name:", filename)
    filepath = os.path.join(input_directory, filename)

    if os.path.isfile(filepath):
        f_name, _ = os.path.splitext(filename)

        json_file_path = os.path.join(output_directory, f"ann_{f_name}.json")
        empty_file_path = os.path.join(output_directory_empty, f"{f_name}.txt")
        ex_file_path = os.path.join(output_directory_exceptions, f"{f_name}.txt")

        with open(filepath, "r", encoding="utf-8") as file:
            file_contents = file.read()

        try:
            x, y = psm.Extract_HPO_Fr_StringMaching(file_contents)

            annotations = []
            for _, row in x.iterrows():
                annotation = {
                    "start": int(row["start"]),
                    "length": int(row["length"]),
                    "sentence": row["phrase"],
                    "negated": False,
                    "concerned_person": "Patient",
                    "mult_CS": False,
                    "hpoAnnotation": [
                        {
                            "hpoId": row["HPO_ID"],
                            "hpoName": row["phrase"],
                            "parser": "SM",
                            "rating": 3,
                            "ratingInit": 3
                        }
                    ]
                }
                annotations.append(annotation)

            if annotations:
                with open(json_file_path, "w", encoding="utf-8") as json_file:
                    json.dump(annotations, json_file, indent=2, ensure_ascii=False)
            else:
                with open(empty_file_path, "w", encoding="utf-8") as out:
                    out.write(file_contents)

        except Exception as e:
            print(f"[ERROR] {filename}: {e}")
            with open(ex_file_path, "w", encoding="utf-8") as out:
                out.write(file_contents)
                   
  