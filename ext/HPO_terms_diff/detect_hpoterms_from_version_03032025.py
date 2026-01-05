import os
import json 


# Lister les fichiers dans le dossier
# pour chaque fichier, le lire, et vérifier que le terme n'est pas présent dans l'ensemble

recent_hpoterms = set()
with open("ext/HPO_terms_diff/diff_between_hpo_03032025_and_14042022.txt") as fh:
    for line in fh.readlines():
        recent_hpoterms.add(line.strip())

folder_path = "0_data/new_reports/"

detected_terms_in_recent_version = set()

for file in os.listdir(folder_path):
    path_file = folder_path + file
    with open(path_file, "r") as fh:
        annotated = json.load(fh)

    for annot in annotated:
        clinical_term = annot["hpoAnnotation"][0]["hpoId"][0]

        if clinical_term in recent_hpoterms:
            detected_terms_in_recent_version.add(clinical_term)
            print(f"Warming: Term {clinical_term} has been detected.")


    break
    
