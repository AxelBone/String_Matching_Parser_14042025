import os
import subprocess

INPUT_DIR = "input"
OUTPUT_DIR = "output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

for filename in os.listdir(INPUT_DIR):
    file_path = os.path.join(INPUT_DIR, filename)

    if not os.path.isfile(file_path):
        continue

    # lire les termes du fichier
    with open(file_path, "r") as f:
        terms = [line.strip() for line in f if line.strip()]

    if not terms:
        print(f"Skipping {filename} (no terms)")
        continue

    hpo_list = ",".join(terms)
    output_file = os.path.join(
        OUTPUT_DIR,
        filename.rsplit(".", 1)[0] + "_match.tsv"
    )

    cmd = [
        "python",
        "phenogenius_cli.py",
        "--result_file", output_file,
        "--hpo_list", hpo_list
    ]

    print(f"Running on {filename}")
    subprocess.run(cmd, check=True)

print("Done.")
