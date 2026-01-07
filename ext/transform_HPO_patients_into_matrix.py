#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import pandas as pd


def iter_hpo_ids_from_item(item: dict) -> Iterable[str]:
    """
    Extrait les HPO IDs d'un item, en supportant la structure:
    item["hpoAnnotation"] = [ { "hpoId": ["HP:....", ...], ... }, ... ]
    """
    ann = item.get("hpoAnnotation", [])
    if not isinstance(ann, list):
        return []
    for a in ann:
        if not isinstance(a, dict):
            continue
        hpo_ids = a.get("hpoId", [])
        if isinstance(hpo_ids, list):
            for h in hpo_ids:
                if isinstance(h, str) and h.startswith("HP:"):
                    yield h
        elif isinstance(hpo_ids, str) and hpo_ids.startswith("HP:"):
            yield hpo_ids


def hpo_set_from_file(path: Path, include_negated: bool) -> Set[str]:
    """
    Charge un fichier JSON et retourne l'ensemble des HPO ids présents.
    Si include_negated == False, on ignore les items avec negated == True.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise ValueError(f"Impossible de lire/parse {path.name}: {e}") from e

    if not isinstance(data, list):
        raise ValueError(f"{path.name}: attendu une liste JSON (array) à la racine.")

    out: Set[str] = set()
    for item in data:
        if not isinstance(item, dict):
            continue
        if not include_negated and item.get("negated") is True:
            continue
        for hpo in iter_hpo_ids_from_item(item):
            out.add(hpo)
    return out


def collect_files(input_dir: Path, pattern: str, recursive: bool) -> List[Path]:
    if recursive:
        return sorted(input_dir.rglob(pattern))
    return sorted(input_dir.glob(pattern))


def build_binary_matrix(
    files: List[Path],
    include_negated: bool,
    use_stem_as_key: bool,
    drop_empty: bool,
) -> pd.DataFrame:
    """
    Retourne un DataFrame binaire avec index=fichiers et colonnes=HPO ids.
    """
    file_to_hpos: Dict[str, Set[str]] = {}
    all_hpos: Set[str] = set()

    for f in files:
        key = f.stem if use_stem_as_key else f.name
        hpos = hpo_set_from_file(f, include_negated=include_negated)
        if drop_empty and not hpos:
            continue
        file_to_hpos[key] = hpos
        all_hpos |= hpos

    cols = sorted(all_hpos)
    idx = sorted(file_to_hpos.keys())

    # Matrice 0/1
    df = pd.DataFrame(0, index=idx, columns=cols, dtype="int8")
    for key, hpos in file_to_hpos.items():
        for h in hpos:
            df.at[key, h] = 1

    df.index.name = "file"
    return df


def main() -> int:
    p = argparse.ArgumentParser(
        description="Transforme des fichiers JSON d'annotations HPO en matrice binaire fichier x HPO."
    )
    p.add_argument("--input-dir", required=True, help="Dossier contenant les JSON.")
    p.add_argument("--pattern", default="*.json", help="Pattern des fichiers (défaut: *.json).")
    p.add_argument("--recursive", action="store_true", help="Recherche récursive dans les sous-dossiers.")
    p.add_argument("--out", required=True, help="Chemin de sortie (.csv ou .parquet ou .xlsx).")
    p.add_argument("--include-negated", action="store_true",
                   help="Inclure aussi les items avec negated=true (défaut: ignorés).")
    p.add_argument("--use-stem", action="store_true",
                   help="Utiliser le nom sans extension comme clé de ligne (défaut: nom complet du fichier).")
    p.add_argument("--drop-empty", action="store_true",
                   help="Ne pas inclure les fichiers sans aucun HPO.")
    args = p.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"input-dir invalide: {input_dir}", file=sys.stderr)
        return 2

    files = collect_files(input_dir, args.pattern, args.recursive)
    if not files:
        print("Aucun fichier trouvé.", file=sys.stderr)
        return 1

    df = build_binary_matrix(
        files=files,
        include_negated=args.include_negated,
        use_stem_as_key=args.use_stem,
        drop_empty=args.drop_empty,
    )

    out_path = Path(args.out)
    suffix = out_path.suffix.lower()

    if suffix == ".csv":
        df.to_csv(out_path, encoding="utf-8")
    elif suffix == ".parquet":
        df.to_parquet(out_path, index=True)
    elif suffix in (".xlsx", ".xls"):
        df.to_excel(out_path, index=True)
    else:
        print("Extension de sortie non supportée. Utilise .csv / .parquet / .xlsx", file=sys.stderr)
        return 2

    print(f"OK: {len(df)} fichiers x {len(df.columns)} HPO -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
