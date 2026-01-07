import os
import json

input_path = "0_data/new_reports/"
output_path = "output/formated_for_gene_prio_tool/"

files = os.listdir(input_path)
reports_hpo = {}

for file in files:
    if file.endswith(".json"):
        filename = file.replace(".json", "")
        file_path = input_path + file
        
        with open(file_path, "r") as fh:
            data = json.load(fh)

            hpoId_list = set([hpo["hpoId"][0] for annot in data for hpo in annot["hpoAnnotation"]])
        
        reports_hpo[filename] = list(hpoId_list)
    

with open(output_path + "concated_hpo_for_prio_dataset.json", "w") as fh:
    json.dump(reports_hpo, fh, ensure_ascii=False, indent=2)
