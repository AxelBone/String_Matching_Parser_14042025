#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import yaml



def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))

    if not isinstance(cfg, dict):
        raise ValueError("Config YAML invalide (racine doit être un objet).")

    if "analysis" not in cfg:
        raise ValueError("Config invalide: section 'analysis' manquante.")

    if "runs" not in cfg or not isinstance(cfg["runs"], list) or len(cfg["runs"]) == 0:
        raise ValueError("Config invalide: section 'runs' manquante ou vide.")

    # Vérifications minimales par run
    for i, run in enumerate(cfg["runs"]):
        if "run_id" not in run:
            raise ValueError(f"run[{i}] invalide: 'run_id' manquant.")
        if "relpath_pattern" not in run:
            raise ValueError(f"run[{i}] invalide: 'relpath_pattern' manquant.")
        if "schema" not in run:
            raise ValueError(f"run[{i}] invalide: 'schema' manquant.")
        for k in ["gene_symbol_col", "rank_col", "score_col"]:
            if k not in run["schema"]:
                raise ValueError(
                    f"run[{i}] schema invalide: clé '{k}' manquante."
                )

    return cfg

def read_genetic_table(path: Path) -> pd.DataFrame:
    # auto-detect sep (csv/tsv) via suffix, sinon fallback virgule
    suf = path.suffix.lower()
    if suf in [".tsv", ".txt"]:
        df = pd.read_csv(path, sep="\t", dtype=str)
    else:
        df = pd.read_csv(path, sep=",", dtype=str)
    return df

def enforce_monogenic(df: pd.DataFrame, id_col: str, gene_col: str) -> pd.DataFrame:
    """
    Garde seulement les patients qui ont exactement 1 gène causal (gene_col) non-null.
    Si plusieurs lignes par patient mais même gène -> OK.
    Si plusieurs gènes différents -> exclu.
    """
    d = df.copy()
    d[gene_col] = d[gene_col].astype(str).where(d[gene_col].notna(), None)
    gset = (
        d.dropna(subset=[gene_col])
         .groupby(id_col)[gene_col]
         .apply(lambda s: sorted(set([x for x in s if x and x != "nan"])))
    )
    mono_ids = set(gset[gset.apply(len) == 1].index.tolist())
    return d[d[id_col].isin(mono_ids)].copy()


def find_report_path(report_root: Path, relpattern: str, patient_id: str) -> Path:
    rel = relpattern.format(patient_id=patient_id)
    return report_root / rel



def read_report_generic(path: Path, file_cfg: dict) -> pd.DataFrame:
    sep = file_cfg.get("sep", "\t")
    header = 0 if file_cfg.get("header", True) else None
    df = pd.read_csv(path, sep=sep, header=header, dtype=str)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def normalize_report_columns(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    """
    Normalise les colonnes du report selon le schema :
      - gene_symbol_col, rank_col, score_col
      - extra_cols: {standard_name: source_col}
    Retourne un df avec colonnes standardisées :
      gene_symbol, rank, score, + colonnes extra (noms standards)
    """
    required_keys = ["gene_symbol_col", "rank_col", "score_col"]
    for k in required_keys:
        if k not in schema:
            raise ValueError(f"Schema invalide: clé manquante '{k}'")

    mapping = {
        schema["gene_symbol_col"]: "gene_symbol",
        schema["rank_col"]: "rank",
        schema["score_col"]: "score",
    }

    extra = schema.get("extra_cols", {}) or {}
    for std_name, source_col in extra.items():
        mapping[source_col] = std_name

    missing = [src for src in mapping.keys() if src not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans report: {missing}")

    out = df.rename(columns=mapping).copy()

    keep = ["gene_symbol", "rank", "score"] + list(extra.keys())
    return out[keep]


def extract_gene_row(report_df: pd.DataFrame, gene_symbol: str) -> Tuple[Optional[pd.Series], bool]:
    """
    Retourne (row, duplicated_flag).
    - Si gene présent 1 fois -> row, False
    - Si gene présent plusieurs fois -> première occurrence (ordre fichier), True
    - Si absent -> (None, False)
    """
    print("Gene symbol", gene_symbol)

    matches = report_df[report_df["gene_symbol"] == gene_symbol]
    print("matches?", matches)

    if matches.shape[0] == 0:
        return None, False
    duplicated = matches.shape[0] > 1
    row = matches.iloc[0]
    return row, duplicated


def build_analysis_table(
    genetic_df: pd.DataFrame,
    cfg: Dict[str, Any],
    report_root: Path,
) -> pd.DataFrame:
    """
    Construit une table d'analyse 'long format' :
      1 ligne = 1 (ligne génétique) x 1 run (outil/version/extraction)
    Pour chaque run (défini dans cfg["runs"]), le script cherche le report patient,
    et extrait rank/score (+ champs extra) du gène causal (gene_symbol).
    """
    analysis_cfg = cfg["analysis"]
    runs = cfg["runs"]

    id_col = str(analysis_cfg["id_col"])
    gene_col = str(analysis_cfg["gene_col"])
    keep_only_monogenic = bool(analysis_cfg.get("keep_only_monogenic", True))

    # 1) Filtre monogénique si demandé
    base = genetic_df.copy()
    if keep_only_monogenic:
        base = enforce_monogenic(base, id_col=id_col, gene_col=gene_col)

    print("Genetic rows:", len(genetic_df), "Base rows after monogenic filter:", len(base))
    print("Unique patients in base:", base[id_col].nunique() if len(base) else 0)


    # 2) Boucle sur lignes génétiques x runs
    out_rows: List[Dict[str, Any]] = []

    for _, g_row in base.iterrows():
        patient_id = str(g_row[id_col])
        truth_gene = str(g_row[gene_col])

        for run in runs:
            run_id = run.get("run_id", pd.NA)
            prio_tool = run.get("prio_tool", pd.NA)
            hpo_version = run.get("hpo_version", pd.NA)
            extraction_method = run.get("extraction_method", pd.NA)

            relpattern = run.get("relpath_pattern")
            if not relpattern:
                raise ValueError(f"run_id={run_id}: relpath_pattern manquant")

            file_cfg = run.get("file", {}) or {}
            schema = run.get("schema", {}) or {}

            # Quelles colonnes extra doit-on récupérer pour ce run ?
            extra_cols_map = schema.get("extra_cols", {}) or {}
            extra_std_cols = list(extra_cols_map.keys())

            report_path = find_report_path(report_root, relpattern, patient_id)

            # Ligne de sortie : on recopie toute la ligne génétique + méta run
            row_out: Dict[str, Any] = {c: g_row.get(c) for c in base.columns}
            row_out.update({
                "run_id": run_id,
                "prio_tool": prio_tool,
                "hpo_version": hpo_version,
                "extraction_method": extraction_method,
                "report_path": str(report_path),
            })

            # Valeurs par défaut
            row_out.update({
                "report_found": False,
                "report_read_error": False,
                "report_read_error_msg": pd.NA,
                "gene_found_in_report": False,
                "gene_not_found_flag": False,
                "gene_duplicated_in_report": False,
                "rank": pd.NA,
                "score": pd.NA,
            })
            for c in extra_std_cols:
                row_out[c] = pd.NA

            # 3) Report absent -> NA/False et on passe
            print("Looking for:", report_path)
            if not report_path.exists():
                out_rows.append(row_out)
                continue

            row_out["report_found"] = True

            # 4) Lecture + normalisation
            try:
                rep_raw = read_report_generic(report_path, file_cfg=file_cfg)
                rep = normalize_report_columns(rep_raw, schema=schema)
            except Exception as e:
                row_out["report_read_error"] = True
                row_out["report_read_error_msg"] = str(e)
                out_rows.append(row_out)
                continue

            # 5) Extraction ligne du gène causal
            gene_row, duplicated_flag = extract_gene_row(rep, truth_gene)
            row_out["gene_duplicated_in_report"] = duplicated_flag

            if gene_row is None:
                # Report présent mais gène non trouvé
                row_out["gene_found_in_report"] = False
                row_out["gene_not_found_flag"] = True
                out_rows.append(row_out)
                continue

            # Gène trouvé
            row_out["gene_found_in_report"] = True
            row_out["gene_not_found_flag"] = False

            row_out["rank"] = gene_row.get("rank", pd.NA)
            row_out["score"] = gene_row.get("score", pd.NA)
            for c in extra_std_cols:
                row_out[c] = gene_row.get(c, pd.NA)

            out_rows.append(row_out)

    out = pd.DataFrame(out_rows)

    # 6) Casts numériques
    out["rank"] = pd.to_numeric(out["rank"], errors="coerce").astype("Int64")
    out["score"] = pd.to_numeric(out["score"], errors="coerce")

    # 7) Indicateurs Top-N (rank NA => False)
    out["top1"] = out["rank"].eq(1)
    out["top3"] = out["rank"].le(3).fillna(False)
    out["top5"] = out["rank"].le(5).fillna(False)
    out["top10"] = out["rank"].le(10).fillna(False)

    return out

def main() -> int:
    ap = argparse.ArgumentParser(description="Construit un tableau d'analyse prio (monogéniques) à partir des reports.")
    ap.add_argument("--genetic-table", required=True, help="CSV/TSV contenant la table génétique (avec ID_PAT_ETUDE).")
    ap.add_argument("--report-root", required=True, help="Dossier racine contenant les fichiers de sortie de prio.")
    ap.add_argument("--config", required=True, help="Config YAML décrivant les runs et patterns de chemins.")
    ap.add_argument("--out", default=None, help="Chemin de sortie (override config.analysis.output_path).")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))

    genetic_path = Path(args.genetic_table)
    report_root = Path(args.report_root)

    gdf = read_genetic_table(genetic_path)

    out_df = build_analysis_table(gdf, cfg, report_root)

    out_path = Path(args.out) if args.out else Path(cfg["analysis"].get("output_path", "analysis_table.csv"))
    fmt = (cfg["analysis"].get("output_format", "csv") or "csv").lower()

    if out_path.suffix.lower() == ".parquet" or fmt == "parquet":
        out_df.to_parquet(out_path, index=False)
    else:
        out_df.to_csv(out_path, index=False, encoding="utf-8")

    print(f"OK -> {out_path} | rows={len(out_df)} cols={len(out_df.columns)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
