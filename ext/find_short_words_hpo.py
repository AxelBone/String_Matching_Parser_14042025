#!/usr/bin/env python3
import argparse
import sys
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Trouve toutes les lignes dont la colonne de gauche (libellé) "
            "a une longueur < N caractères, et indique si ces libellés "
            "ont des synonymes (autres libellés partageant le même code HPO)."
        )
    )
    parser.add_argument(
        "-n", "--max-length", type=int, required=True,
        help="Longueur maximale du libellé (strictement inférieure)."
    )
    parser.add_argument(
        "-i", "--input", type=str, required=True,
        help="Chemin du fichier d'entrée (TSV/texte)."
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Fichier log de sortie (optionnel)."
    )
    args = parser.parse_args()

    max_len = args.max_length
    input_file = args.input
    output_file = args.output

    # Lecture du fichier
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Erreur de lecture du fichier d'entrée : {e}")
        sys.exit(1)

    # On réinitialise le log si besoin
    if output_file:
        open(output_file, "w", encoding="utf-8").close()

    def log(msg: str):
        if output_file:
            with open(output_file, "a", encoding="utf-8") as lf:
                lf.write(msg + "\n")
        else:
            print(msg)

    # Première passe : on stocke toutes les lignes parsées
    # et on construit code -> ensemble de libellés (pour détecter les synonymes)
    entries = []  # (line_num, label, code)
    code_to_labels = defaultdict(set)

    for line_num, line in enumerate(lines, start=1):
        line = line.rstrip("\n")

        if "\t" in line:
            left, right = line.split("\t", 1)
            label = left.strip()
            code = right.strip()
        else:
            label = line.strip()
            code = ""

        if not label:
            continue

        entries.append((line_num, label, code))

        if code:
            code_to_labels[code].add(label)

    # Deuxième passe : on identifie les libellés courts et on regarde leurs synonymes
    # found = (line_num, label_len, label, code, nb_synonymes, liste_synonymes)
    found = []
    affected_terms = set()  # (label, code)
    with_synonyms = 0
    without_synonyms = 0

    for line_num, label, code in entries:
        label_len = len(label)

        if label_len < max_len:
            # Calcul des synonymes : autres libellés avec le même code
            synonyms = []
            if code and code in code_to_labels:
                synonyms = sorted(
                    lbl for lbl in code_to_labels[code]
                    if lbl != label
                )

            nb_syn = len(synonyms)
            if nb_syn > 0:
                with_synonyms += 1
            else:
                without_synonyms += 1

            found.append((line_num, label_len, label, code, nb_syn, synonyms))
            affected_terms.add((label, code))

            # Log détaillé par ligne
            log_line = (
                f"[Ligne {line_num}] libellé court (len={label_len}) : "
                f"'{label}'\t{code} | synonymes: {nb_syn}"
            )
            if nb_syn > 0:
                syn_str = " | ".join(synonyms)
                log_line += f" | liste synonymes: {syn_str}"

            log(log_line)

    # Construction d'un résumé par terme (label + code uniq) avec nb_synonymes
    # On prend pour chaque (label, code) le nb_synonymes correspondant
    term_to_nb_syn = {}
    for _, _, label, code, nb_syn, _ in found:
        key = (label, code)
        # si plusieurs lignes identiques, on garde le max (au cas où)
        term_to_nb_syn[key] = max(nb_syn, term_to_nb_syn.get(key, 0))

    # Résumé
    summary_lines = []
    summary_lines.append("\n=== Résumé ===")
    summary_lines.append(f"Seuil : longueur du libellé < {max_len} caractères")
    summary_lines.append(f"Nombre de libellés courts trouvés : {len(found)}")
    summary_lines.append(f"Nombre de lignes/termes concernés : {len(affected_terms)}")
    summary_lines.append(f"Libellés courts avec au moins un synonyme : {with_synonyms}")
    summary_lines.append(f"Libellés courts sans synonyme : {without_synonyms}")
    summary_lines.append("")
    summary_lines.append("Termes courts (label, code, nombre de synonymes) :")
    summary_lines.append("label\tcode\tnb_synonymes")

    for (label, code), nb_syn in sorted(term_to_nb_syn.items(), key=lambda x: (x[0][1], x[0][0])):
        if code:
            summary_lines.append(f"{label}\t{code}\t{nb_syn}")
        else:
            summary_lines.append(f"{label}\t\t{nb_syn}")

    summary_text = "\n".join(summary_lines)
    log(summary_text)
    print(summary_text)


if __name__ == "__main__":
    main()
