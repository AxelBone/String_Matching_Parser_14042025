import json
import argparse
from pathlib import Path
import sys

def main():
    parser = argparse.ArgumentParser(
        description="Concatène les fichiers JSON pour toutes les clés"
    )
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--mapping", default="mapping.json", type=Path)
    parser.add_argument("--output-dir", default="output_files", type=Path)

    parser.add_argument(
        "--file-prefix",
        default="",
        help="Préfixe à ajouter aux fichiers du mapping (ex: 'ann_')"
    )
    parser.add_argument(
        "--file-suffix",
        default=".json",
        help="Suffixe/extension des fichiers (ex: '.json')"
    )

    args = parser.parse_args()

    if not args.mapping.exists():
        sys.exit(f"❌ Mapping introuvable : {args.mapping}")
    if not args.input_dir.exists():
        sys.exit(f"❌ Dossier d'entrée introuvable : {args.input_dir}")

    with open(args.mapping, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for key, file_list in mapping.items():
        concatenated = []
        found_any_file = False

        for filename in file_list:
            effective_name = filename.strip()

            if args.file_prefix and not effective_name.startswith(args.file_prefix):
                effective_name = args.file_prefix + effective_name

            if args.file_suffix and not effective_name.endswith(args.file_suffix):
                effective_name = effective_name + args.file_suffix

            file_path = args.input_dir / effective_name

            if not file_path.exists():
                print(f"⚠️ [{key}] Fichier manquant : {file_path}")
                continue

            found_any_file = True

            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, list):
                print(f"❌ [{key}] {effective_name} n'est pas une liste JSON — ignoré")
                continue

            concatenated.extend(data)

        # 🚫 Aucun fichier trouvé ou aucune donnée
        if not found_any_file or not concatenated:
            print(f"⚠️ [{key}] Aucun fichier valide détecté → fichier non créé")
            continue

        # ✅ Écriture uniquement si contenu
        output_file = args.output_dir / f"{key}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(concatenated, f, indent=2, ensure_ascii=False)

        print(f"✅ {key} → {output_file} ({len(concatenated)} entrées)")

    print("🎉 Terminé")


if __name__ == "__main__":
    main()
