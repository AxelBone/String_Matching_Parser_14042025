#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd


def load_root_list(path: Path) -> List[dict]:
    """Charge un JSON et renvoie une liste d'items à la racine."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise ValueError(f"Impossible de lire/parse {path.name}: {e}") from e

    if not isinstance(data, list):
        raise ValueError(f"{path.name}: attendu une liste JSON (array) à la racine.")
    return [x for x in data if isinstance(x, dict)]


def iter_hp_ids_from_items(items: List[dict], hp_key: str = "hp_id") -> Iterable[str]:
    """
    Extrait les HP IDs depuis une liste d'items de la forme:
      { "hp_id": "HP:....", "label": "...", "score": 1, "source_file": "...", ... }
    """
    for it in items:
        hp = it.get(hp_key)
        if isinstance(hp, str) and hp.startswith("HP:"):
            yield hp


def hpo_set_from_file(path: Path) -> Set[str]:
    """Retourne l'ensemble des HP ids présents dans ce JSON."""
    items = load_root_list(path)
    return set(iter_hp_ids_from_items(items))


def collect_files(input_dir: Path, pattern: str, recursive: bool) -> List[Path]:
    return sorted(input_dir.rglob(pattern) if recursive else input_dir.glob(pattern))


def build_binary_matrix(
    files: List[Path],
    use_stem_as_key: bool,
    drop_empty: bool,
) -> pd.DataFrame:
    """
    DataFrame binaire index=fichiers JSON, colonnes=HP ids.
    """
    file_to_hpos: Dict[str, Set[str]] = {}
    all_hpos: Set[str] = set()

    for f in files:
        key = f.stem if use_stem_as_key else f.name
        hpos = hpo_set_from_file(f)
        if drop_empty and not hpos:
            continue
        file_to_hpos[key] = hpos
        all_hpos |= hpos

    cols = sorted(all_hpos)
    idx = sorted(file_to_hpos.keys())

    df = pd.DataFrame(0, index=idx, columns=cols, dtype="int8")
    for key, hpos in file_to_hpos.items():
        for h in hpos:
            df.at[key, h] = 1

    df.index.name = "file"
    return df


def main() -> int:
    p = argparse.ArgumentParser(
        description="Transforme des fichiers JSON (format hp_id/source_file) en matrice binaire fichier x HPO."
    )
    p.add_argument("--input-dir", required=True, help="Dossier contenant les JSON.")
    p.add_argument("--pattern", default="*.json", help="Pattern des fichiers (défaut: *.json).")
    p.add_argument("--recursive", action="store_true", help="Recherche récursive dans les sous-dossiers.")
    p.add_argument("--out", required=True, help="Chemin de sortie (.csv ou .parquet ou .xlsx).")
    p.add_argument("--use-stem", action="store_true",
                   help="Utiliser le nom sans extension comme clé de ligne (défaut: nom complet du fichier).")
    p.add_argument("--drop-empty", action="store_true",
                   help="Ne pas inclure les fichiers sans aucun HP:....")
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
