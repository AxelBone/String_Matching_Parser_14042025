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
        raise ValueError(f"Impossible de lire/parse {path}: {e}") from e

    if not isinstance(data, list):
        raise ValueError(f"{path.name}: attendu une liste JSON (array) à la racine.")
    return [x for x in data if isinstance(x, dict)]


def iter_records(
    items: List[dict],
    hp_key: str = "hp_id",
    file_key: str = "source_file",
    score_key: str = "score",
) -> Iterable[Tuple[str, str, Optional[float]]]:
    """
    Yield (source_file, hp_id, score).
    - hp_id doit ressembler à HP:....
    - score est optionnel (None si absent ou non numérique)
    """
    for it in items:
        src = it.get(file_key)
        hp = it.get(hp_key)

        if not isinstance(src, str) or not src:
            continue
        if not isinstance(hp, str) or not hp.startswith("HP:"):
            continue

        sc = it.get(score_key, None)
        score_val: Optional[float] = None
        if isinstance(sc, (int, float)):
            score_val = float(sc)
        elif isinstance(sc, str):
            try:
                score_val = float(sc)
            except Exception:
                score_val = None

        yield src, hp, score_val


def build_matrix(
    json_path: Path,
    value_mode: str = "binary",   # "binary" ou "score"
    agg: str = "max",             # "max", "sum", "first"
    use_stem_for_source: bool = False,
) -> pd.DataFrame:
    """
    Construit une matrice source_file x HP:....
    - value_mode="binary" => 1 si présent
    - value_mode="score"  => valeur = score (agrégée si doublons)
    """
    items = load_root_list(json_path)

    # Accumulate
    all_hpos: Set[str] = set()
    all_sources: Set[str] = set()

    # map[(source, hp)] = aggregated_value
    cell: Dict[Tuple[str, str], float] = {}

    def normalize_source(s: str) -> str:
        if not use_stem_for_source:
            return s
        # s peut être "xxx.txt" -> "xxx"
        return Path(s).stem

    for src, hp, score_val in iter_records(items):
        srcn = normalize_source(src)
        all_sources.add(srcn)
        all_hpos.add(hp)

        if value_mode == "binary":
            v = 1.0
        else:
            # mode score: si pas de score, on met 1 par défaut (modifiable si tu veux)
            v = 1.0 if score_val is None else float(score_val)

        key = (srcn, hp)
        if key not in cell:
            cell[key] = v
        else:
            if agg == "max":
                cell[key] = max(cell[key], v)
            elif agg == "sum":
                cell[key] = cell[key] + v
            elif agg == "first":
                pass
            else:
                raise ValueError(f"Agrégation inconnue: {agg}")

    sources = sorted(all_sources)
    hpos = sorted(all_hpos)

    # DataFrame
    dtype = "float32" if value_mode == "score" or agg == "sum" else "int8"
    df = pd.DataFrame(0, index=sources, columns=hpos)

    for (src, hp), v in cell.items():
        df.at[src, hp] = v

    if dtype == "int8":
        df = df.astype("int8")
    else:
        df = df.astype("float32")

    df.index.name = "source_file"
    return df


def main() -> int:
    p = argparse.ArgumentParser(
        description="Transforme un JSON (liste d'items avec hp_id/source_file) en matrice source_file x HPO."
    )
    p.add_argument("--input-json", required=True, help="Chemin du JSON d'entrée (format de l'image).")
    p.add_argument("--out", required=True, help="Chemin de sortie (.csv / .parquet / .xlsx).")

    p.add_argument("--value", choices=["binary", "score"], default="binary",
                   help="Valeur de cellule: binary=1 si présent, score=utilise le champ score (défaut: binary).")
    p.add_argument("--agg", choices=["max", "sum", "first"], default="max",
                   help="Agrégation si doublons (source_file, hp_id) (défaut: max).")
    p.add_argument("--use-stem-source", action="store_true",
                   help="Utiliser le nom sans extension pour source_file (ex: abc.txt -> abc).")

    args = p.parse_args()

    in_path = Path(args.input_json)
    if not in_path.exists() or not in_path.is_file():
        print(f"input-json invalide: {in_path}", file=sys.stderr)
        return 2

    df = build_matrix(
        json_path=in_path,
        value_mode=args.value,
        agg=args.agg,
        use_stem_for_source=args.use_stem_source,
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

    print(f"OK: {len(df)} source_file x {len(df.columns)} HPO -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
