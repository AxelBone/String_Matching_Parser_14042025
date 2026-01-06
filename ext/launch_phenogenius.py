import os
import json
import subprocess
import time
import logging
from logging.handlers import RotatingFileHandler

INPUT_DIR = "input"
OUTPUT_DIR = "output"
HPO_2025_FILE = "HPO_terms_2025.txt"
SELECTED_KEYS_FILE = "selected_keys.txt"  # fichier contenant une clé par ligne

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "pipeline.log")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# === Logging (console + fichier rotatif) ===
logger = logging.getLogger("pipeline")
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

# === Charger la liste HPO_terms_2025 ===
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
    output_file = os.path.join(OUTPUT_DIR, stem + "_match.tsv")

    try:
        # 1) Charger le JSON
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)  # data doit être une liste de dicts

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

        hpo_list = ",".join(hpo_terms)

        # 4) Lancer phenogenius_cli.py
        cmd = [
            "python",
            "phenogenius_cli.py",
            "--result_file", output_file,
            "--hpo_list", hpo_list
        ]

        logger.info(
            f"[{i}/{len(selected_filenames)}] Running on {filename} "
            f"({len(hpo_terms)} HPO after filtering)"
        )

        # capture stdout/stderr pour les logs
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
            f"output={output_file} | {dt:.3f}s"
        )

        # log optionnel des sorties
        if result.stdout:
            logger.info(f"[{filename}] STDOUT:\n{result.stdout.strip()}")
        if result.stderr:
            logger.warning(f"[{filename}] STDERR:\n{result.stderr.strip()}")

    except Exception as e:
        dt = time.perf_counter() - t0_sample
        failed += 1
        logger.exception(f"[{i}/{len(selected_filenames)}] EXCEPTION on {filename} | {dt:.3f}s | {e}")

t_total = time.perf_counter() - t0_total

logger.info("=== Summary ===")
logger.info(f"Total listed: {len(selected_filenames)}")
logger.info(f"Processed: {processed}")
logger.info(f"Skipped (no HPO): {skipped_no_hpo}")
logger.info(f"Missing files: {len(missing)}")
logger.info(f"Failed: {failed}")
logger.info(f"Total time: {t_total:.3f}s")

if missing:
    logger.warning(f"Missing files (first 20): {missing[:20]}")
