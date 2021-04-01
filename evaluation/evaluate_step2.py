import csv
import random

def create_silver_labels(file_path, k, l):
    """
    Create silver labels from label counts
    :param file_path: path to result from eval_step_two
    :param k: SUPPORTS-label decision boundary
    :param l: REFUTES-label decision boundary
    :return: None
    """
    with open(file_path + "allCountries.txt", 'r') as cf:
        countries = [line.strip() for line in cf.readlines()]
    with open(file_path + "label_counts_evid_pair.csv", "r") as f:
        reader = csv.reader(f)
        with open("results/" + "/countries/evalRes_" + str(k) + "_" + str(l) + ".csv", "w+", newline='') as rf:
            csv.writer(rf).writerow(["adjective", "Step 2 label", "FC label", "rand label"])
        with open("results/" + "/ethnicities/evalRes_" + str(k) + "_" + str(l) + ".csv", "w+", newline='') as rf:
            csv.writer(rf).writerow(["adjective", "Step 2 label", "FC label", "rand label"])
        header = next(reader)
        for row in reader:
            print(row)
            country, adj1, adj2, pos_sup, neg_sup, neut_sup, pos_ref, neg_ref = row
            if country in countries:
                res_path = "results/countries/evalRes_"
            else:
                res_path = "results/ethnicities/evalRes_"
            if country == "NoPrem":
                continue

            pos_sup = int(pos_sup)
            neg_sup = int(neg_sup)
            pos_ref = int(pos_ref)
            neg_ref = int(neg_ref)

            # classes: 0 neutral corr, 1 negative corr, 2 positive corr
            class_label1 = 0
            class_label2 = 0

            # classify with decision boundaries
            if pos_sup > neg_sup + k:
                class_label1 = 2
            else:
                if pos_ref > neg_ref + l:
                    class_label1 = 1
            if neg_ref > pos_ref + l:
                class_label2 = 1
            else:
                if neg_sup > pos_sup + k:
                    class_label2 = 2
            rand_label1 = random.randint(1, 2)
            rand_label2 = random.randint(1, 2)
            with open(res_path + str(k) + "_" + str(l) + ".csv", "a", newline='') as rf:
                writer = csv.writer(rf)
                writer.writerow([country + "_" + adj1, "2", class_label1, rand_label1])
                writer.writerow([country + "_" + adj2, "1", class_label2, rand_label2])


def sort_adjective_pairs(data_path):
    res_dic = {}
    with open(data_path + "label_counts_evid_pair_asc.csv", 'r') as af:
        reader = csv.reader(af)
        header = next(reader)
        for line in reader:
            country = line[0]
            res_dic[country] = [line[1], "#dummy#", line[6], "#dummy#",
                                line[8], line[7], "#dummy#"]
    with open(data_path + "label_counts_evid_pair_desc.csv", 'r') as df:
        reader = csv.reader(df)
        header = next(reader)
        for line in reader:
            country = line[0]
            try:
                res_dic[country][1] = line[1]
            except KeyError:
                continue
            res_dic[country][3] = line[6]
            res_dic[country][6] = line[7]
    with open("setup/label_counts_evid_pair.csv", 'w+', newline='') as rf:
        writer = csv.writer(rf)
        writer.writerow(["country", "adj1", "adj2", "pos_sup", "neg_sup", "neut_sup", "pos_ref", "neg_ref"])
        for key, value in res_dic.items():
            if "#dummy#" in value:
                continue
            writer.writerow([key] + value)

def main():
    #sort_adjective_pairs("path_to/label_counts/from_step_3/")
    create_silver_labels("setup/", 20, 10)

if __name__== "__main__":
    main()

