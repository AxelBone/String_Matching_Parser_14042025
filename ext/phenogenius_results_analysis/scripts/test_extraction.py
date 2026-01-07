def extract_gene_row(report_df: pd.DataFrame, gene_symbol: str) -> Tuple[Optional[pd.Series], bool]:
    """
    Retourne (row, duplicated_flag).
    - Si gene présent 1 fois -> row, False
    - Si gene présent plusieurs fois -> première occurrence (ordre fichier), True
    - Si absent -> (None, False)
    """
    matches = report_df[report_df["gene_symbol"] == gene_symbol]
    if matches.shape[0] == 0:
        return None, False
    duplicated = matches.shape[0] > 1
    row = matches.iloc[0]
    return row, duplicated