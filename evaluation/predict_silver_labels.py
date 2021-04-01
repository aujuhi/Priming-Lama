from transformers import BertTokenizerFast
from transformers import BertForSequenceClassification
from predict import replace_cntr
import csv
import os
import random


model_path = "./fever-factchecker-bert-base-uncased"
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased", do_lower_case=True)
target_names = ["REFUTES", "NOTENOUGHINFO", "SUPPORTS"]
model = BertForSequenceClassification.from_pretrained(model_path, num_labels=len(target_names))



def get_prediction(claim_evid_tup):
    # tokenize claim evidence tupel
    inputs = tokenizer.batch_encode_plus(claim_evid_tup, truncation=True, padding=True, max_length=512,
                                         return_tensors='pt')
    # infer predictions
    outputs = model(**inputs)
    probs = outputs[0].softmax(1)
    return target_names[probs.argmax()]


def eval_step_one(masked_data_path, priming_path, res_path):
    header = ["sentence", "Target adjective", "neutral adjective", "label"]
    with open(masked_data_path, 'r', encoding="utf8") as f:
        data = [line.strip() for line in f.readlines()]
    with open(priming_path, 'r') as f2:
        lines = [line.strip().split("|") for line in f2.readlines()]
    for country, adj1, adj2, label in lines:
        with open(res_path + country + "_" + adj1 + "_" + adj2 + ".csv", "w+", newline='') as rf:
            writer = csv.writer(rf)
            writer.writerow(header)
        for sentence in data:
            evidence_count = replace_cntr(sentence, country)
            print(evidence_count)
            if "country" in masked_data_path:
                evidence_adj = replace_cntr(sentence, "the " + adj1 + " country")
                print(evidence_adj)
                evidence_adj_neut = replace_cntr(sentence, "the " + adj2 + " country")
                print(evidence_adj_neut)
            else:
                evidence_adj = replace_cntr(sentence, adj1)
                evidence_adj_neut = replace_cntr(sentence, adj2)
            res_row = [sentence, get_prediction([(evidence_count, evidence_adj)]),
                       get_prediction([(evidence_count, evidence_adj_neut)]), label]
            print(res_row)
            with open(res_path + country + "_" + adj1 + "_" + adj2 + ".csv", "a", newline='') as rf:
                writer = csv.writer(rf)
                writer.writerow(res_row)


def eval_step_two():
    header = ["country", "adj1 ", "adj2", "pos_sup", "neg_sup", "neut_sup", "pos_ref", "neg_ref"]
    res_path = "/res_path/"
    with open(res_path, 'w+', newline='') as rf:
        csv.writer(rf).writerow(header)
    with open("/allCountries.txt", 'r') as cf:
        countries = [line.strip() for line in cf.readlines()]
    with open("/evidence_validation_sentences.txt", 'r') as cmask:
        sentences = [line.strip() for line in cmask.readlines()]
    with open("/premis_path.txt", 'r') as pf:
        for line in pf.readlines():
            splitted = line.split("|")
            country = splitted[0]
            adj1 = splitted[1]
            adj2 = splitted[2]
            if country in countries:
                adjective1 = "the " + adj1 + " country"
                adjective2 = "the " + adj2 + " country"
            else:
                adjective1 = adj1
                adjective2 = adj2
            count1 = [0, 0, 0]  # "REFUTES", "NOTENOUGHINFO", "SUPPORTS"
            count2 = [0, 0, 0]
            for sentence in sentences:
                evidence_count = replace_cntr(sentence, country)
                print(evidence_count)
                evidence_adj1 = replace_cntr(sentence, adjective1)
                evidence_adj2 = replace_cntr(sentence, adjective2)
                count1[target_names.index(get_prediction([(evidence_count, evidence_adj1)]))] += 1
                count2[target_names.index(get_prediction([(evidence_count, evidence_adj2)]))] += 1
            row = [country, adj1, adj2, count1[2], count2[2], " ", count1[0], count2[0]]
            with open(res_path, 'a', newline='') as rf:
                csv.writer(rf).writerow(row)

def adapt_step1_pred_csv(csv_file_name):
    val_path = r"C:/Users/Anke/Documents/Thesis/data/validation/step1/"
    rows = []
    if "_c_" in csv_file_name:
        pos_prem_file = "ordered_prems_pos_country.csv"
        neg_prem_file = "ordered_prems_neg_country.csv"
    else:
        pos_prem_file = "ordered_prems_pos_ethnicity.csv"
        neg_prem_file = "ordered_prems_neg_ethnicity.csv"
    prem_dic_pos = {}
    with open(val_path + "/setup/" + pos_prem_file, encoding='utf8') as premf_pos:
        reader = csv.reader(premf_pos)
        for line in reader:
            prem_dic_pos[line[0].strip()] = line[1:]
    prem_dic_neg = {}
    with open(val_path + "/setup/" + neg_prem_file, encoding='utf8') as premf_neg:
        reader = csv.reader(premf_neg)
        for line in reader:
            prem_dic_neg[line[0].strip()] = line[1:]
    with open(csv_file_name, 'r') as f:
        reader = csv.reader(f)
        header = next(reader) + ["random_ver2", "Step_1_label_ver2"]
        for line in reader:
            triplet = line[0].split("_")
            country = triplet[0].strip()
            if country not in ["India", "Korea", "Japan", "Argentina", "Chile", "Belgium", "Georgia", "France",
                               "Norway", "Iceland", "Peru", "Colombia", "Iran", "Panama", "Afghanistan", "Czech",
                               "Bosnia", "EU", "Morocco", "Philippines",
                               "Bhutan"] and country not in ["French", "Vietnamese", "Irish", "Japanese", "Greeks",
                                                             "Italians", "English", "Somalis", "Turks",
                                                             "Germans", "Jews", "Arabs", "Finns", "Belarusians",
                                                             "Hungarians", "Assyrians", "Romanians", "Scots",
                                                             "Portuguese", "Tibetans"]:
                continue
            adj1 = triplet[1].strip()
            adj2 = triplet[2].strip()
            l2 = 0
            j = min(50, len(prem_dic_pos[country]) - 1)
            k = min(20, len(prem_dic_neg[country]) - 1)
            if adj1 in prem_dic_pos[country][:j] or adj2 in prem_dic_neg[country][:k]:
                l2 = 1
            if random.randint(0, 1) or random.randint(0, 1):
                r2 = 1
            else:
                r2 = 0
            row = line + [r2, l2]
            rows.append(row)
    res_file = csv_file_name.split(".")[0] + "_" + "ver2.csv"
    with open(res_file, "w+", newline='') as rf:
        writer = csv.writer(rf)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)


def eval_results_step2(file_path, res_path, k, l):
    res = []
    with open(r"C:\Users\Anke\Documents\Thesis\eval_res\experimentCountries\allCountries.txt", 'r') as cf:
        countries = [line.strip() for line in cf.readlines()]
    with open(file_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        '''
        with open(res_path + country1/evalRes"
                  + "_" + str(k) + "_" + str(l) + ".csv", "w+", newline='') as rf:
            csv.writer(rf).writerow(["adjective", "Step 2 label", "FC label", "rand label"])
        with open(res_path + ethnicity1/evalRes"
                  + "_" + str(k) + "_" + str(l) + ".csv", "w+", newline='') as rf:
            csv.writer(rf).writerow(["adjective", "Step 2 label", "FC label", "rand label"])
        '''
        for row in reader:

            print(row)
            country, adj1, adj2, pos_sup, neg_sup, neut_sup, pos_ref, neg_ref = row
            if country in countries:
                res_path = r"C:\Users\Anke\Documents\Thesis\data\validation\step2\labeled\country1\evalRes"
            else:
                res_path = r"C:\Users\Anke\Documents\Thesis\data\validation\step2\labeled\ethnicity1\evalRes"
            if country == "NoPrem":
                continue

            pos_sup = int(pos_sup)
            neg_sup = int(neg_sup)
            pos_ref = int(pos_ref)
            neg_ref = int(neg_ref)
            # classes: 0 neutral corr, 1 negative corr, 2 positive corr
            class_label1 = 0
            class_label2 = 0
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
            rand_label1 = random.randint(0, 2)
            rand_label2 = random.randint(0, 2)
            with open(res_path + "_" + str(k) + "_" + str(l) + ".csv", "a", newline='') as rf:
                writer = csv.writer(rf)
                writer.writerow([country + "_" + adj1, "2", class_label1, rand_label1])
                writer.writerow([country + "_" + adj2, "1", class_label2, rand_label2])


def count_eval_res(file_path, a):
    if "Countries" in file_path:
        res_name = file_path + "eval_counts_c_" + str(a) + ".csv"
    if "Ethni" in file_path:
        res_name = file_path + "eval_counts_e_" + str(a) + ".csv"
    if "step2" in file_path:
        res_name = file_path + "eval_counts_" + str(a) + ".csv"
    with open(res_name, "w+", newline='') as rf:
        csv.writer(rf).writerow(
            ["Country_adj_triplet", "Diff_Support", "Diff_Refute", "Step_1_label", "New_label", "Rand_label"])
    for f_path in (file_path + "label1_20/", file_path + "label2/"):
        for file in os.listdir(f_path):
            if "eval" in file or os.path.getsize(f_path + file) < 23000:
                continue
            df = pd.read_csv(f_path + file)
            print(f_path + file)
            print(df)
            new_df = df.drop(df.columns[0], axis=1)
            new_df = new_df.drop(df.columns[3], axis=1)
            print(new_df)
            cntg_table = new_df.apply(pd.value_counts)
            cntg_table = cntg_table.fillna(0)
            print(cntg_table)
            support_index = list(cntg_table.index).index("SUPPORTS")
            refute_index = list(cntg_table.index).index("REFUTES")
            diff_sup = cntg_table.iloc[support_index, 0] - cntg_table.iloc[support_index, 1]
            diff_ref = cntg_table.iloc[refute_index, 0] - cntg_table.iloc[refute_index, 1]
            if "label1" in f_path:
                old_label = 1
            else:
                old_label = 0
            if diff_sup >= a:
                new_label = 1
            else:
                new_label = 0
            random_label = random.randint(0, 1)
            col = [file.split(".")[0], diff_sup, diff_ref, old_label, new_label, random_label]
            with open(res_name, "a", newline='') as rf:
                csv.writer(rf).writerow(col)
