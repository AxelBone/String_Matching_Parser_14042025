import os
import pandas as pd

# INPUT PATHS
output_prediction = "output_genes_predicted"
causal_gene_dir = "input_causal_gene_ENTREZID"  # Optional if already loaded
output_rank_evaluation_dir = "output_rank_evaluation"
method_name = "Phen2Gene"

# If needed, load gene_dict (format: {patient_id: [causal_gene]})

results = []

for fname in os.listdir(causal_gene_dir):
    if not fname.endswith(".txt"):
        continue

    patient_id = os.path.splitext(fname)[0]
    causal_path = os.path.join(causal_gene_dir, fname)

    with open(causal_path, "r") as f:
        causal_genes = [line.strip() for line in f if line.strip()]
    
    if not causal_genes:
        continue

    pred_path = os.path.join(output_prediction, f"{patient_id}_output.txt", "output_file.associated_gene_list")
    if not os.path.exists(pred_path):
        continue

    try:
        df_pred = pd.read_csv(pred_path, sep='\t')
        ranked_genes = df_pred['ID'].astype(str).str.strip().tolist()
    except Exception as e:
        print(f"Error reading {pred_path}: {e}")
        continue

    ranks = []
    for gene in causal_genes:
        try:
            rank = ranked_genes.index(gene) +1
            ranks.append(rank)
        except ValueError:
            print(f"Error gene not in list {patient_id}")


    avg_rank = sum(ranks) / len(ranks) if ranks else None

    results.append({
        "patient_id": patient_id,
        "method": method_name,
        "causal_genes": causal_genes,
        "found_ranks": ranks,
        "avg_rank": avg_rank
    })

df = pd.DataFrame(results)

df.to_csv(f"{output_rank_evaluation_dir}/rank_evaluation.csv", index=False)