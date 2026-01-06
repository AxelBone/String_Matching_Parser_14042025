import json
import argparse
from pathlib import Path
import sys

def main():
    parser = argparse.ArgumentParser(
        description="Concatène les fichiers JSON pour toutes les clés"
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Dossier contenant les fichiers JSON source"
    )
    parser.add_argument(
        "--mapping",
        default="mapping.json",
        type=Path,
        help="Fichier mapping clé -> fichiers (par défaut: mapping.json)"
    )
    parser.add_argument(
        "--output-dir",
        default="output_files",
        type=Path,
        help="Dossier de sortie (par défaut: output_files)"
    )
    parser.add_argument(
        "--prefix",
        default="",
        help="Préfixe à ajouter devant chaque nom de fichier du mapping (ex: 'ann_'). Par défaut: vide."
    )

    args = parser.parse_args()

    # Vérifications
    if not args.mapping.exists():
        sys.exit(f"❌ Mapping introuvable : {args.mapping}")

    if not args.input_dir.exists():
        sys.exit(f"❌ Dossier d'entrée introuvable : {args.input_dir}")

    # Chargement mapping
    with open(args.mapping, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Traitement de toutes les clés
    for key, file_list in mapping.items():
        concatenated = []

        for filename in file_list:
            # Applique le préfixe seulement si nécessaire
            # (évite ann_ann_... si le mapping contient déjà ann_)
            effective_name = filename
            if args.prefix and not filename.startswith(args.prefix):
                effective_name = args.prefix + filename

            file_path = args.input_dir / effective_name

            if not file_path.exists():
                print(f"⚠️ [{key}] Fichier manquant : {file_path}")
                continue

            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, list):
                print(f"❌ [{key}] {effective_name} n'est pas une liste JSON — ignoré")
                continue

            concatenated.extend(data)

        # Écriture du fichier de la clé
        output_file = args.output_dir / f"{key}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(concatenated, f, indent=2, ensure_ascii=False)

        print(f"✅ {key} → {output_file} ({len(concatenated)} entrées)")

    print("🎉 Traitement terminé pour toutes les clés")


if __name__ == "__main__":
    main()
