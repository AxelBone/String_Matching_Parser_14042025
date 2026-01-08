import os
import json
import subprocess
import time
import logging
from logging.handlers import RotatingFileHandler

# === CONFIGURATION ===
INPUT_DIR = "input_hpo_test"          # JSON d'annotation HPO (un fichier par patient)
OUTPUT_DIR = "output_p2g"    # dossier de sortie pour Phen2Gene
# HPO_2025_FILE = "HPO_terms_2025.txt"
SELECTED_KEYS_FILE = "selected_keys.txt"  # fichier contenant une clé par ligne

PHEN2GENE_SCRIPT = "../Phen2Gene/phen2gene.py"  # chemin vers phen2gene.py
WEIGHT_METHOD = "sk"  # ex: "sk", "equal", etc.

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

# Console
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(fmt)
logger.addHandler(ch)

# Fichier (rotation)
fh = RotatingFileHandler(LOG_FILE, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
fh.setLevel(logging.INFO)
fh.setFormatter(fmt)
logger.addHandler(fh)

# === Charger la liste HPO_terms_2025 (même logique que pour phenogenius) ===
with open(HPO_2025_FILE, "r", encoding="utf-8") as f:
    hpo_2025_terms = {
        term.strip()
        for term in f.read().split(",")
        if term.strip()
    }
logger.info(f"Loaded {len(hpo_2025_terms)} HPO terms from {HPO_2025_FILE}")

# === Charger la liste des fichiers à analyser via les clés ===
with open(SELECTED_KEYS_FILE, "r", encoding="utf-8") as f:
    selected_filenames = []
    for line in f:
        key = line.strip()
        if not key:
            continue
        selected_filenames.append(key + ".json")

logger.info(f"Loaded {len(selected_filenames)} keys from {SELECTED_KEYS_FILE}")

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

    # fichier de sortie phen2gene
    output_file = os.path.join(OUTPUT_DIR, stem + "_phen2gene.tsv")
    # fichier temporaire contenant la liste d'HPO pour phen2gene
    hpo_input_file = os.path.join(OUTPUT_DIR, stem + "_hpo.txt")

    try:
        # 1) Charger le JSON
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)  # data = liste de dicts (annotations)

        # 2) Extraire les HP:xxxx (en ignorant negated=true et HPO_terms_2025)
        hpo_terms = []
        for item in data:
            if item.get("negated") is True:
                continue

            for ann in item.get("hpoAnnotation", []):
                for hpo_id in ann.get("hpoId", []):
                    if (
                        isinstance(hpo_id, str)
                        and hpo_id.startswith("HP:")
                        and hpo_id not in hpo_2025_terms
                    ):
                        hpo_terms.append(hpo_id)

        # Dédupliquer en gardant l'ordre
        seen = set()
        hpo_terms = [x for x in hpo_terms if not (x in seen or seen.add(x))]

        if not hpo_terms:
            dt = time.perf_counter() - t0_sample
            skipped_no_hpo += 1
            logger.info(
                f"[{i}/{len(selected_filenames)}] Skipping {filename} "
                f"(no HPO terms after filtering) | {dt:.3f}s"
            )
            continue

        # 3) Écrire la liste d'HPO dans un fichier texte (un ID par ligne)
        with open(hpo_input_file, "w", encoding="utf-8") as hf:
            for hpo in hpo_terms:
                hf.write(hpo + "\n")

        # 4) Lancer phen2gene.py
        cmd = [
            "python",
            PHEN2GENE_SCRIPT,
            "-f", hpo_input_file,
            "-w", WEIGHT_METHOD,
            "-out", output_file,
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

        processed += 1
        logger.info(
            f"[{i}/{len(selected_filenames)}] Done {filename} | "
            f"hpo_input={hpo_input_file} | output={output_file} | {dt:.3f}s"
        )

        if result.stdout:
            logger.info(f"[{filename}] STDOUT:\n{result.stdout.strip()}")
        if result.stderr:
            logger.warning(f"[{filename}] STDERR:\n{result.stderr.strip()}")

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
