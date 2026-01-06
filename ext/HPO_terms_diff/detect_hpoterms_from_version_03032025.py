from pathlib import Path
import json

terms_file = Path("ext/HPO_terms_diff/diff_between_hpo_03032025_and_14042022.txt")
folder = Path("0_data/new_reports")

# Charge la liste de termes récents
recent_hpoterms = set()
with terms_file.open("r", encoding="utf-8") as fh:
    for line in fh:
        term = line.strip()
        if term:
            recent_hpoterms.add(term)

detected_terms = set()
files_scanned = 0
files_failed = 0

for path_file in folder.glob("*.json"):
    files_scanned += 1
    try:
        with path_file.open("r", encoding="utf-8") as fh:
            annotated = json.load(fh)
    except Exception as e:
        files_failed += 1
        print(f"⚠️ [{path_file.name}] Lecture/JSON impossible: {e}")
        continue

    if not isinstance(annotated, list):
        print(f"⚠️ [{path_file.name}] Format inattendu: JSON racine n'est pas une liste")
        continue

    for annot in annotated:
        # On sécurise la structure attendue
        hpo_annotations = (annot or {}).get("hpoAnnotation") or []
        for ha in hpo_annotations:
            hpo_ids = (ha or {}).get("hpoId") or []
            for hpo_id in hpo_ids:
                if hpo_id in recent_hpoterms:
                    if hpo_id not in detected_terms:
                        print(f"Warning: Term {hpo_id} detected in {path_file.name}")
                    detected_terms.add(hpo_id)

print("\n=== Résumé ===")
print(f"Fichiers scannés : {files_scanned}")
print(f"Fichiers en erreur: {files_failed}")
print(f"Termes récents détectés: {len(detected_terms)}")
if detected_terms:
    print("Liste:", ", ".join(sorted(detected_terms)))
