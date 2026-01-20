import random
import pandas as pd
from faker import Faker

fake = Faker()

runs = ["phenogenius_hpo2022_sm", "phen2gene_hpo_2022_sm"]

genes = [
    ("FLNA", "HGNC:3754"),
    ("MECP2", "HGNC:6990"),
    ("UBE3A", "HGNC:12496"),
    ("KMT2D", "HGNC:7133"),
    ("COL1A1", "HGNC:2197"),
]

acmg_class = ["Pathogenic", "Likely_pathogenic", "VUS"]
clinvar_class = ["Pathogenic", "Likely_pathogenic"]

criteria_map = {
    "Pathogenic": ["PVS1", "PM2", "PS3", "PM5"],
    "Likely_pathogenic": ["PS2", "PM1", "PM2"],
    "VUS": []
}

# --- Nouveau composant important ---
# dictionnaire global pour garantir la stabilité HPO_code -> HPO_name
HPO_DICT = {}

def get_random_hpo(n=1):
    codes = []
    for _ in range(n):
        # random code
        code = f"HP:{random.randint(1000000, 9999999)}"
        
        # si nouveau code → assigner un lorem name
        if code not in HPO_DICT:
            HPO_DICT[code] = fake.sentence(nb_words=3).replace(".", "")
        
        codes.append(code)
    return codes

def generate_row(i, run):
    gene, hgnc = random.choice(genes)

    # nb HPO
    n_hpo = random.randint(5, 15)
    hpo_codes = get_random_hpo(n_hpo)

    # on génère les deux listes
    hpo_ids = [{code: round(random.uniform(0.2, 0.9), 2)} for code in hpo_codes]
    hpo_names = [{HPO_DICT[code]: v[next(iter(v))]} for code, v in zip(hpo_codes, hpo_ids)]

    acmg = random.choice(acmg_class)
    clinvar = random.choice(clinvar_class)

    return {
        "ID_PAT_ETUDE": f"report{i}",
        "IPP": f"IPP{str(i).zfill(6)}",
        "IPP_clef": fake.bothify(text="????"),
        "key_for_chaining": f"report{i}",
        "transcript_from_ehop": f"NM_{random.randint(100000,999999)}.{random.randint(1,10)}",
        "transcript": lambda v: v,
        "transcript_no_version": lambda v: v.split(".")[0],
        "chr": random.choice(list(range(1, 23)) + ["X", "Y"]),
        "pos": random.randint(1_000_000, 200_000_000),
        "ref": random.choice(["A","T","G","C"]),
        "alt": random.choice(["A","T","G","C"]),
        "var_id": f"var{i}",
        "transcript_nm": f"NM_{random.randint(100000,999999)}",
        "gene_symbol": gene,
        "gene_hgnc_id": hgnc,
        "acmg_classification": acmg,
        "acmg_criteria": ",".join(criteria_map[acmg][:2]),
        "clinvar_classification": clinvar,
        "clinvar_review_status": random.choice([
            "reviewed_by_expert_panel",
            "criteria_provided_single_submitter",
            "criteria_provided_multiple_submitters_no_conflicts"
        ]),
        "run_id": run,
        "prio_tool": run.split("_")[0],
        "hpo_version": 2022,
        "extraction_method": "string_matching",
        "report_path": f"ext/phenogenius_results_analysis/data/fake_outputs/report{i}.tsv",
        "report_found": True,
        "report_read_error": False,
        "report_read_error_msg": "",
        "gene_found_in_report": True,
        "gene_not_found_flag": False,
        "gene_duplicated_in_report": False,
        "rank": random.randint(1,500),
        "score": round(random.uniform(1,10), 2),
        "hpo_implicated": str(hpo_ids),
        "hpo_description_implicated": str(hpo_names),
        "phenotype_specificity": random.choice([
            "A - the reported phenotype is highly specific",
            "B - the reported phenotype is consistent",
            "C - the phenotype is reported with limited association",
        ]),
        "top1": random.choice([True, False]),
        "top3": random.choice([True, False]),
        "top5": random.choice([True, False]),
        "top10": random.choice([True, False]),
    }

def generate_dataset(n=50):
    rows = []
    for i in range(1, n+1):
        for run in runs:
            rows.append(generate_row(i, run))
    return pd.DataFrame(rows)

df = generate_dataset(50)
df.to_csv("synthetic_variants.csv", index=False)
print(df.head())
