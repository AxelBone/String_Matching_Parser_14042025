import json
from pathlib import Path
import csv

INPUT_JSON = Path("output/translated_chunked_reports.json")  # adapte si besoin
OUTPUT_CSV = Path("output/sentences_table.csv")
OUTPUT_JSONL = Path("output/sentences_table.jsonl")

START_AT_0 = True  # mets False pour commencer à 1


def build_rows(docs: dict):
    rows = []
    for doc_id, spans in docs.items():
        if spans is None:
            continue
        if not isinstance(spans, list):
            # si jamais un doc n'est pas une liste, on skip proprement
            continue

        for i, span in enumerate(spans):
            if span is None:
                span = ""
            span = str(span).strip()

            sentence_id = i if START_AT_0 else i + 1
            rows.append({
                "doc_id": doc_id,
                "span": span,
                "sentence_id": sentence_id
            })
    return rows


def main():
    # Lecture JSON
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        docs = json.load(f)

    rows = build_rows(docs)

    # Écriture CSV
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["doc_id", "span", "sentence_id"])
        writer.writeheader()
        writer.writerows(rows)

    # Écriture JSONL (pratique pour gros volumes)
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[OK] CSV -> {OUTPUT_CSV}")
    print(f"[OK] JSONL -> {OUTPUT_JSONL}")
    print(f"[OK] Lignes: {len(rows)}")


if __name__ == "__main__":
    main()
