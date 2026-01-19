import pandas as pd
import numpy as np

np.random.seed(42)

# ---------------------------
# Paramètres de simulation
# ---------------------------
n_patients = 1000
# ID "lisibles" type P001, P002, ...
patient_ids = [f"P{str(i).zfill(3)}" for i in range(1, n_patients + 1)]

# Petit pool de codes HPO factices
POSSIBLE_HPO_CODES = [f"HP:{i:07d}" for i in range(1, 2000)]

rows = []

for idx, pid in enumerate(patient_ids, start=1):
    # ID numérique pour le hex
    patient_int_id = idx
    # hex sans préfixe 0x, sur 8 caractères (à adapter)
    patient_hex_id = format(patient_int_id, "08x")

    # ============================
    # 1) PARTIE TEXTE / HPO (agrégée patient)
    # ============================
    n_documents = np.random.randint(1, 20)
    n_visits = np.random.randint(1, min(7, n_documents) + 1)

    first_doc_date = pd.Timestamp("2018-01-01") + pd.to_timedelta(
        np.random.randint(0, 365 * 3), unit="D"
    )
    followup_days = np.random.randint(0, 365 * 3)
    last_doc_date = first_doc_date + pd.to_timedelta(followup_days, unit="D")

    total_words = int(np.random.randint(1_000, 20_000))
    mean_words_per_doc = total_words / n_documents
    max_words_doc = int(mean_words_per_doc * (1.0 + np.random.rand()))
    total_sentences = int(np.random.randint(100, 3_000))
    mean_avg_sentence_len_words = total_words / max(total_sentences, 1)

    # ---- Nouvelle logique HPO : full list + uniques ----
    # Nombre de codes HPO DISTINCTS
    n_hpo_unique = int(np.random.randint(0, 40))

    if n_hpo_unique > 0:
        # Nombre total de mentions HPO (avec répétitions)
        n_hpo_full_list = int(np.random.randint(n_hpo_unique, n_hpo_unique + 300))

        # Sélection de codes uniques
        hpo_unique_list = list(
            np.random.choice(POSSIBLE_HPO_CODES, size=n_hpo_unique, replace=False)
        )

        # Construction de la full list :
        #  - on s'assure qu'au moins une occurrence de chaque code unique est présente
        #  - puis on ajoute des mentions supplémentaires
        hpo_full_list = hpo_unique_list.copy()
        extra_mentions = n_hpo_full_list - n_hpo_unique
        if extra_mentions > 0:
            hpo_full_list.extend(
                list(
                    np.random.choice(
                        hpo_unique_list, size=extra_mentions, replace=True
                    )
                )
            )
        # Sanity check
        n_hpo_full_list = len(hpo_full_list)
        n_hpo_unique = len(set(hpo_full_list))
    else:
        n_hpo_full_list = 0
        hpo_full_list = []
        hpo_unique_list = []

    # Métriques dérivées (cohérentes avec ce que tu avais dans patient_agg)
    hpo_total_per_doc = n_hpo_full_list / max(n_documents, 1)
    hpo_unique_per_doc = n_hpo_unique / max(n_documents, 1)

    hpo_per_sentence = n_hpo_full_list / max(total_sentences, 1)
    hpo_per_1k_words = (n_hpo_full_list / max(total_words, 1)) * 1000

    # (Optionnel) alias pour compatibilité avec l'ancien schéma
    total_hpo_mentions = n_hpo_full_list
    n_hpo_codes = n_hpo_unique

    n_document_types = int(np.random.randint(1, 6))
    n_units = int(np.random.randint(1, 4))
    n_authors = int(np.random.randint(1, 10))

    age_first = int(np.random.randint(0, 90))
    age_last = int(min(99, age_first + np.random.randint(0, 15)))

    docs_per_year = n_documents / max(followup_days / 365.25, 1)

    # ============================
    # 2) PARTIE VARIANTS (listes)
    # ============================
    n_variants = int(np.random.randint(0, 6))  # 0 à 5 variants

    transcript_from_ehop = [f"TX_{pid}_{i}" for i in range(n_variants)]
    transcript = [f"TR_{pid}_{i}.1" for i in range(n_variants)]
    transcript_no_version = [t.split(".")[0] for t in transcript]

    chr_list = list(np.random.choice(list("123456789XY"), size=n_variants))
    pos_list = list(np.random.randint(1, 1_000_000, size=n_variants))
    ref_list = list(np.random.choice(list("ACGT"), size=n_variants))
    alt_list = list(np.random.choice(list("ACGT"), size=n_variants))

    var_id = [f"{pid}_VAR_{i}" for i in range(n_variants)]
    transcript_nm = [f"NM_{100000 + i}" for i in range(n_variants)]

    gene_symbol = [f"GENE{g}" for g in np.random.randint(1, 50, size=n_variants)]
    gene_hgnc_id = [f"HGNC:{2000 + i}" for i in range(n_variants)]

    acmg_classification = list(
        np.random.choice(["P", "LP", "VUS", "LB", "B"], size=n_variants)
    )
    acmg_criteria = [
        list(
            np.random.choice(
                ["PVS1", "PS1", "PM2", "PP3", "BP1", "BP4"],
                size=np.random.randint(1, 3),
                replace=False,
            )
        )
        for _ in range(n_variants)
    ]

    clinvar_classification = list(
        np.random.choice(
            ["Pathogenic", "Likely_pathogenic", "VUS", "Benign", "Likely_benign"],
            size=n_variants,
        )
    )
    clinvar_review_status = list(
        np.random.choice(
            ["no_assertion", "single", "multi", "reviewed"],
            size=n_variants,
        )
    )

    patient_label = np.random.choice(["monogenic", "polygenic", "no_variants"])

    # ============================
    # 3) CONSTRUCTION D’UNE LIGNE PATIENT
    # ============================
    row = dict(
        PATIENT_ID_HEX=patient_hex_id,

        # --- variables texte/HPO ---
        n_documents=n_documents,
        n_visits=n_visits,
        first_doc_date=first_doc_date,
        last_doc_date=last_doc_date,
        total_words=total_words,
        mean_words_per_doc=mean_words_per_doc,
        max_words_doc=max_words_doc,
        total_sentences=total_sentences,
        mean_avg_sentence_len_words=mean_avg_sentence_len_words,

        # nouvelles colonnes HPO basées sur les full lists
        n_hpo_full_list=n_hpo_full_list,
        n_hpo_unique=n_hpo_unique,
        hpo_total_per_doc=hpo_total_per_doc,
        hpo_unique_per_doc=hpo_unique_per_doc,
        hpo_per_sentence=hpo_per_sentence,
        hpo_per_1k_words=hpo_per_1k_words,

        # (optionnel) alias pour rester compatible avec ton ancien schéma
        total_hpo_mentions=total_hpo_mentions,
        n_hpo_codes=n_hpo_codes,

        # listes HPO, si tu veux mimer patient_agg
        HPO_full_list=hpo_full_list,
        HPO_unique_list=hpo_unique_list,

        n_document_types=n_document_types,
        n_units=n_units,
        n_authors=n_authors,
        patient_age_first=age_first,
        patient_age_last=age_last,
        followup_days=followup_days,
        docs_per_year=docs_per_year,

        # --- variables variantes (listes) ---
        n_variants=n_variants,
        transcript_from_ehop=transcript_from_ehop,
        transcript=transcript,
        transcript_no_version=transcript_no_version,
        chr=chr_list,
        pos=pos_list,
        ref=ref_list,
        alt=alt_list,
        var_id=var_id,
        transcript_nm=transcript_nm,
        gene_symbol=gene_symbol,
        gene_hgnc_id=gene_hgnc_id,
        acmg_classification=acmg_classification,
        acmg_criteria=acmg_criteria,
        clinvar_classification=clinvar_classification,
        clinvar_review_status=clinvar_review_status,

        patient_label=patient_label,
    )

    rows.append(row)

# ---------------------------
# DataFrame final : 1 ligne = 1 patient
# ---------------------------
final_patient_df = pd.DataFrame(rows)

final_patient_df.to_csv("simulated_patient_level_df.csv", sep=";", index=False)
