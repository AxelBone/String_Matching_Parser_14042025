
def read_hpo_file(path):
    hpo = {}
    with open(path, "r") as fh:
        for line in fh.readlines():
            term =line.strip().split("\t")[0]
            id = line.strip().split("\t")[1]

            if id not in hpo:
                hpo[id] = term

    return hpo




if __name__=="__main__":
    hpo1_path = "hpoterms03032025.txt"
    hpo2_path = "hpoterms14042022.txt"

    hpo1 = read_hpo_file(hpo1_path)
    hpo2 = read_hpo_file(hpo2_path)

    hpo1_id = set(hpo1.keys())
    hpo2_id = set(hpo2.keys())

    diff = hpo1_id - hpo2_id

    with open("diff_between_hpo_03032025_and_14042022.txt", "w") as fh:
        for term in diff:
            fh.write(f"{term}\n")