import os
import json
import subprocess

INPUT_DIR = "input"
OUTPUT_DIR = "output"
HPO_2025_FILE = "HPO_terms_2025.txt"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Charger la liste HPO_terms_2025 ===
with open(HPO_2025_FILE, "r", encoding="utf-8") as f:
    hpo_2025_terms = {
        term.strip()
        for term in f.read().split(",")
        if term.strip()
    }

print(f"Loaded {len(hpo_2025_terms)} HPO terms from HPO_terms_2025")

for filename in os.listdir(INPUT_DIR):
    file_path = os.path.join(INPUT_DIR, filename)

    if not os.path.isfile(file_path):
        continue

    # 1) Charger le JSON
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)  # data doit être une liste de dicts

    # 2) Extraire les HP:xxxx (en ignorant negated=true et HPO_terms_2025)
    hpo_terms = []
    for item in data:
        if item.get("negated") is True:
            continue

        for ann in item.get("hpoAnnotation", []):
            for hpo_id in ann.get("hpoId", []):
                if (
                    isinstance(hpo_id, str)
                    and hpo_id.startswith("HP:")
                    and hpo_id not in hpo_2025_terms
                ):
                    hpo_terms.append(hpo_id)

    # Dédupliquer en gardant l'ordre
    seen = set()
    hpo_terms = [x for x in hpo_terms if not (x in seen or seen.add(x))]

    if not hpo_terms:
        print(f"Skipping {filename} (no HPO terms after filtering)")
        continue

    hpo_list = ",".join(hpo_terms)

    # 3) Nom du fichier de sortie
    stem = filename.rsplit(".", 1)[0]
    output_file = os.path.join(OUTPUT_DIR, stem + "_match.tsv")

    # 4) Lancer phenogenius_cli.py
    cmd = [
        "python",
        "phenogenius_cli.py",
        "--result_file", output_file,
        "--hpo_list", hpo_list
    ]

    print(
        f"Running on {filename} "
        f"({len(hpo_terms)} HPO after filtering)"
    )
    subprocess.run(cmd, check=True)

print("Done.")
