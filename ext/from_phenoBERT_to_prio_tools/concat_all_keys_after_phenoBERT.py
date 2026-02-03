import json
import argparse
from pathlib import Path
import sys
import re
from typing import Any, Dict, List, Optional


LINE_RE = re.compile(
    r"^\s*(\d+)\s+(\d+)\s+(.+?)\s+(HP:\d+)\s+([0-9]*\.?[0-9]+)\s*$"
)
# Groupes :
# 1) int
# 2) int
# 3) label (peut contenir espaces) -> non-greedy
# 4) HP:xxxxxxx
# 5) score float


def parse_json_list(path: Path) -> List[Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("n'est pas une liste JSON")
    return data


def parse_text_lines(path: Path) -> List[Dict[str, Any]]:
    """
    Parse un fichier texte dont chaque ligne ressemble à:
    435 465 hydrocephaly    HP:0000238  1.00
    Retourne une liste de dicts.
    """
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            m = LINE_RE.match(line)
            if not m:
                # Si tu veux être strict, remplace print par raise
                print(f"⚠️ Ligne non reconnue ({path.name}:{lineno}) → ignorée: {line}")
                continue

            out.append({
                "col1": int(m.group(1)),
                "col2": int(m.group(2)),
                "label": m.group(3).strip(),
                "hp_id": m.group(4),
                "score": float(m.group(5)),
                "source_file": path.name,
                "line": lineno
            })
    return out


def parse_file(path: Path) -> List[Any]:
    """
    Choisit le parsing selon l'extension:
    - .json -> JSON liste
    - sinon -> texte lignes
    """
    if path.suffix.lower() == ".json":
        return parse_json_list(path)
    return parse_text_lines(path)


def main():
    parser = argparse.ArgumentParser(
        description="Concatène des fichiers par clé (JSON liste ou texte ligne)"
    )
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--mapping", default="mapping.json", type=Path)
    parser.add_argument("--output-dir", default="output_files", type=Path)

    parser.add_argument("--file-prefix", default="", help="Préfixe à ajouter aux fichiers du mapping")
    parser.add_argument("--file-suffix", default="", help="Suffixe/extension par défaut si manquant (ex: '.json', '.txt')")

    # Optionnel: forcer le type au lieu de détecter par extension
    parser.add_argument("--force-type", choices=["auto", "json", "text"], default="auto")

    args = parser.parse_args()

    if not args.mapping.exists():
        sys.exit(f"❌ Mapping introuvable : {args.mapping}")
    if not args.input_dir.exists():
        sys.exit(f"❌ Dossier d'entrée introuvable : {args.input_dir}")

    with open(args.mapping, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for key, file_list in mapping.items():
        concatenated: List[Any] = []
        found_any_file = False

        for filename in file_list:
            effective_name = filename.strip()

            if args.file_prefix and not effective_name.startswith(args.file_prefix):
                effective_name = args.file_prefix + effective_name

            # Si un suffixe est fourni et que le nom n'a pas déjà une extension/suffixe
            if args.file_suffix and not effective_name.endswith(args.file_suffix):
                effective_name = effective_name + args.file_suffix

            file_path = args.input_dir / effective_name

            if not file_path.exists():
                print(f"⚠️ [{key}] Fichier manquant : {file_path}")
                continue

            found_any_file = True

            try:
                if args.force_type == "json":
                    data = parse_json_list(file_path)
                elif args.force_type == "text":
                    data = parse_text_lines(file_path)
                else:
                    data = parse_file(file_path)

            except Exception as e:
                print(f"❌ [{key}] Erreur parsing {effective_name} → ignoré ({e})")
                continue

            if not isinstance(data, list):
                print(f"❌ [{key}] {effective_name} n'a pas produit une liste — ignoré")
                continue

            concatenated.extend(data)

        if not found_any_file or not concatenated:
            print(f"⚠️ [{key}] Aucun fichier valide détecté → fichier non créé")
            continue

        output_file = args.output_dir / f"{key}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(concatenated, f, indent=2, ensure_ascii=False)

        print(f"✅ {key} → {output_file} ({len(concatenated)} entrées)")

    print("🎉 Terminé")


if __name__ == "__main__":
    main()
