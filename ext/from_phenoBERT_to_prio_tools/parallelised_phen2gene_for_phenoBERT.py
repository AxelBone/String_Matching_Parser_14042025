import os
import json
import subprocess
import time
import logging
import shutil
from logging.handlers import RotatingFileHandler

# === CONFIGURATION ===
INPUT_DIR = "input"          # dossier avec les <key>.json (anciens OU nouveaux concaténés)
OUTPUT_DIR = "output_genes_predicted"
HPO_2025_FILE = "HPO_terms_2025.txt"
SELECTED_KEYS_FILE = "selected_keys.txt"  # 1 clé par ligne, sans extension

PHEN2GENE_SCRIPT = "../Phen2Gene/phen2gene.py"
WEIGHT_METHOD = "sk"

# Optionnel: filtrage par score (nouveau format concaténé)
MIN_SCORE = None  # ex: 0.5 ; None = pas de filtrage

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "pipeline_phen2gene.log")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# === Logging (console + fichier rotatif) ===
logger = logging.getLogger("pipeline_phen2gene")
logger.setLevel(logging.INFO)
logger.handlers.clear()

fmt = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(fmt)
logger.addHandler(ch)

fh = RotatingFileHandler(LOG_FILE, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
fh.setLevel(logging.INFO)
fh.setFormatter(fmt)
logger.addHandler(fh)

# === Charger la liste HPO_terms_2025 ===
with open(HPO_2025_FILE, "r", encoding="utf-8") as f:
    hpo_2025_terms = {
        term.strip()
        for term in f.read().split(",")
        if term.strip()
    }
logger.info(f"Loaded {len(hpo_2025_terms)} HPO terms from {HPO_2025_FILE}")

# === Charger la liste des fichiers à analyser via les clés ===
selected_filenames = []
with open(SELECTED_KEYS_FILE, "r", encoding="utf-8") as f:
    for line in f:
        key = line.strip()
        if not key:
            continue
        selected_filenames.append(key + ".json")

logger.info(f"Loaded {len(selected_filenames)} keys from {SELECTED_KEYS_FILE}")


def extract_hpo_terms(data, hpo_2025_terms):
    """
    Supporte 2 formats:
    - Ancien: list[dict] avec {negated, hpoAnnotation:[{hpoId:[...]}]}
    - Nouveau concat: list[dict] avec {hp_id, score, ...}

    Retourne une liste HPO dédupliquée (ordre conservé).
    """
    hpo_terms = []

    for item in data:
        if not isinstance(item, dict):
            continue

        # --- Nouveau format concaténé ---
        if "hp_id" in item:
            hp = item.get("hp_id")
            if not (isinstance(hp, str) and hp.startswith("HP:")):
                continue
            if hp in hpo_2025_terms:
                continue

            if MIN_SCORE is not None:
                score = item.get("score", None)
                try:
                    score_val = float(score)
                except (TypeError, ValueError):
                    continue
                if score_val < MIN_SCORE:
                    continue

            hpo_terms.append(hp)
            continue

        # --- Ancien format ---
        if item.get("negated") is True:
            continue

        for ann in item.get("hpoAnnotation", []):
            if not isinstance(ann, dict):
                continue
            for hpo_id in ann.get("hpoId", []):
                if (
                    isinstance(hpo_id, str)
                    and hpo_id.startswith("HP:")
                    and hpo_id not in hpo_2025_terms
                ):
                    hpo_terms.append(hpo_id)

    # Dédupliquer en gardant l'ordre
    seen = set()
    return [x for x in hpo_terms if not (x in seen or seen.add(x))]


missing = []
processed = 0
skipped_no_hpo = 0
failed = 0

t0_total = time.perf_counter()

for i, filename in enumerate(selected_filenames, start=1):
    file_path = os.path.join(INPUT_DIR, filename)

    if not os.path.isfile(file_path):
        missing.append(filename)
        logger.warning(f"[{i}/{len(selected_filenames)}] Missing file: {file_path}")
        continue

    t0_sample = time.perf_counter()
    stem = filename.rsplit(".", 1)[0]

    # Fichier HPO d'entrée pour Phen2Gene
    hpo_input_file = os.path.join(OUTPUT_DIR, stem + "_hpo.txt")
    # Dossier temporaire où Phen2Gene va écrire output_file.associated_gene_list
    tmp_output_dir = os.path.join(OUTPUT_DIR, stem + "_p2g_output")
    # Fichier final
    final_output_file = os.path.join(OUTPUT_DIR, stem + "_match.tsv")

    try:
        # 1) Charger le JSON
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            logger.error(f"[{i}/{len(selected_filenames)}] {filename} is not a JSON list -> skipped")
            continue

        # 2) Extraire les HPO (auto selon format)
        hpo_terms = extract_hpo_terms(data, hpo_2025_terms)

        if not hpo_terms:
            dt = time.perf_counter() - t0_sample
            skipped_no_hpo += 1
            logger.info(
                f"[{i}/{len(selected_filenames)}] Skipping {filename} "
                f"(no HPO terms after filtering) | {dt:.3f}s"
            )
            continue

        # 3) Écrire la liste d'HPO pour Phen2Gene (un ID par ligne)
        with open(hpo_input_file, "w", encoding="utf-8") as hf:
            for hpo in hpo_terms:
                hf.write(hpo + "\n")

        # 4) Lancer Phen2Gene
        os.makedirs(tmp_output_dir, exist_ok=True)

        cmd = [
            "python",
            PHEN2GENE_SCRIPT,
            "-f", hpo_input_file,
            "-w", WEIGHT_METHOD,
            "-out", tmp_output_dir,
        ]

        logger.info(
            f"[{i}/{len(selected_filenames)}] Running Phen2Gene on {filename} "
            f"({len(hpo_terms)} HPO after filtering)"
        )

        result = subprocess.run(cmd, check=False, text=True, capture_output=True)
        dt = time.perf_counter() - t0_sample

        if result.returncode != 0:
            failed += 1
            logger.error(
                f"[{i}/{len(selected_filenames)}] FAILED {filename} | "
                f"returncode={result.returncode} | {dt:.3f}s"
            )
            if result.stdout:
                logger.error(f"[{filename}] STDOUT:\n{result.stdout.strip()}")
            if result.stderr:
                logger.error(f"[{filename}] STDERR:\n{result.stderr.strip()}")
            continue

        # 5) Récupérer output_file.associated_gene_list
        src_file = os.path.join(tmp_output_dir, "output_file.associated_gene_list")

        if not os.path.isfile(src_file):
            failed += 1
            logger.error(
                f"[{i}/{len(selected_filenames)}] FAILED {filename} | "
                f"Phen2Gene output file not found: {src_file} | {dt:.3f}s"
            )
            if result.stdout:
                logger.error(f"[{filename}] STDOUT:\n{result.stdout.strip()}")
            if result.stderr:
                logger.error(f"[{filename}] STDERR:\n{result.stderr.strip()}")
            continue

        if os.path.exists(final_output_file):
            logger.warning(f"Overwriting existing file: {final_output_file}")
            os.remove(final_output_file)

        shutil.move(src_file, final_output_file)

        processed += 1
        logger.info(
            f"[{i}/{len(selected_filenames)}] Done {filename} | "
            f"final_output={final_output_file} | {dt:.3f}s"
        )

        if result.stdout:
            logger.info(f"[{filename}] STDOUT:\n{result.stdout.strip()}")
        if result.stderr:
            logger.warning(f"[{filename}] STDERR:\n{result.stderr.strip()}")

        # 6) Nettoyage
        try:
            if os.path.exists(hpo_input_file):
                os.remove(hpo_input_file)
            if os.path.isdir(tmp_output_dir):
                shutil.rmtree(tmp_output_dir)
        except Exception as cleanup_err:
            logger.warning(f"[{i}/{len(selected_filenames)}] Cleanup warning for {filename}: {cleanup_err}")

    except Exception as e:
        dt = time.perf_counter() - t0_sample
        failed += 1
        logger.exception(
            f"[{i}/{len(selected_filenames)}] EXCEPTION on {filename} "
            f"| {dt:.3f}s | {e}"
        )

t_total = time.perf_counter() - t0_total

logger.info("=== Phen2Gene Summary ===")
logger.info(f"Total listed: {len(selected_filenames)}")
logger.info(f"Processed: {processed}")
logger.info(f"Skipped (no HPO): {skipped_no_hpo}")
logger.info(f"Missing files: {len(missing)}")
logger.info(f"Failed: {failed}")
logger.info(f"Total time: {t_total:.3f}s")

if missing:
    logger.warning(f"Missing files (first 20): {missing[:20]}")
