import os
import re
import pandas as pd

# ====== CONFIG ======
output_dir = "/chemin/vers/dossier_sortie"  # <-- modifie ici
id_col = "DOCUMENT_ID"
text_col = "CR_TEXT_ENG"
ext = ".txt"  # extension des fichiers créés
# ====================

def safe_filename(name: str) -> str:
    """
    Nettoie le nom de fichier pour éviter caractères interdits / bizarres.
    """
    name = str(name).strip()
    # Remplace tout ce qui n'est pas alphanum, underscore, tiret, point par underscore
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    # Evite les noms vides
    return name if name else "UNKNOWN"

# Crée le dossier s'il n'existe pas
os.makedirs(output_dir, exist_ok=True)

# Vérifs colonnes
missing = [c for c in (id_col, text_col) if c not in df.columns]
if missing:
    raise ValueError(f"Colonnes manquantes dans df: {missing}. Colonnes disponibles: {list(df.columns)}")

created = 0
skipped = 0

for i, row in df.iterrows():
    doc_id = row[id_col]
    text = row[text_col]

    # Gérer les valeurs manquantes
    if pd.isna(doc_id):
        skipped += 1
        continue

    filename = safe_filename(doc_id) + ext
    filepath = os.path.join(output_dir, filename)

    # Convertit NaN en chaîne vide si besoin
    content = "" if pd.isna(text) else str(text)

    # Écriture (utf-8)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

    created += 1

print(f"✅ Fichiers créés: {created}")
print(f"⚠️ Lignes ignorées (DOCUMENT_ID manquant): {skipped}")
print(f"📁 Dossier: {output_dir}")
