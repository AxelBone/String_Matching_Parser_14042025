import os
import shutil

OUTPUT_DIR = "output_genes_predicted"
INPUT_FILENAME = "output_file.associated_gene_list"
OUTPUT_SUFFIX = "_match.tsv"

def process_output_dir():
    count = 0
    missing = 0

    for entry in os.listdir(OUTPUT_DIR):
        entry_path = os.path.join(OUTPUT_DIR, entry)

        # On ne garde que les sous-dossiers
        if not os.path.isdir(entry_path):
            continue

        # Exemple: entry = "mygene2_67_output.txt"
        base_name = entry
        if base_name.endswith("_output.txt"):
            base_name = base_name.replace("_output.txt", "")
        else:
            # si jamais le nom ne suit pas exactement la norme
            # on retire le suffixe du dossier si il existe
            for suffix in ("_output", "_out", "_txt"):
                if base_name.endswith(suffix):
                    base_name = base_name[:-len(suffix)]

        src_file = os.path.join(entry_path, INPUT_FILENAME)

        if not os.path.isfile(src_file):
            print(f"[WARN] Missing {INPUT_FILENAME} in {entry}")
            missing += 1
            continue

        dest_file = os.path.join(
            OUTPUT_DIR, f"{base_name}{OUTPUT_SUFFIX}"
        )

        shutil.copyfile(src_file, dest_file)
        count += 1
        print(f"[OK] {src_file} → {dest_file}")

    print(f"\n=== Summary ===")
    print(f"Processed: {count}")
    print(f"Missing:   {missing}")

if __name__ == "__main__":
    process_output_dir()
