import json
import argparse
from pathlib import Path
import sys
import re


def main():
    parser = argparse.ArgumentParser(
        description="Concatène des fichiers texte ou JSON (string) pour toutes les clés"
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
        help="Suffixe/extension des fichiers (ex: '.json', '.txt')"
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

            # On lit tout le contenu brut
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            if not content.strip():
                print(f"⚠️ [{key}] {effective_name} est vide — ignoré")
                continue

            # On tente de parser en JSON. Si ça échoue, on traite comme texte brut
            try:
                data = json.loads(content)
                is_json = True
            except json.JSONDecodeError:
                is_json = False

            if not is_json:
                # 📄 Cas 1 : fichier texte simple
                text = content
                if text:
                    concatenated.append(text)
                else:
                    print(f"⚠️ [{key}] {effective_name} après nettoyage est vide — ignoré")
                continue

            # 📄 Cas 2 : JSON chargé avec succès
            # - si c'est une string → on la prend comme texte
            # - si c'est autre chose → on ignore (pour rester simple)
            if isinstance(data, str):
                text = data
                if text:
                    concatenated.append(text)
                else:
                    print(f"⚠️ [{key}] {effective_name} (JSON string) vide après nettoyage — ignoré")
            else:
                print(f"❌ [{key}] {effective_name} est JSON mais pas une string en racine — ignoré")
                continue

        # 🚫 Aucun fichier trouvé ou aucune donnée
        if not found_any_file or not concatenated:
            print(f"⚠️ [{key}] Aucun fichier valide détecté → fichier non créé")
            continue

        # ✅ Écriture : liste de textes nettoyés
        output_file = args.output_dir / f"{key}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(concatenated, f, indent=2, ensure_ascii=False)

        print(f"✅ {key} → {output_file} ({len(concatenated)} entrées)")

    print("🎉 Terminé")


if __name__ == "__main__":
    main()
