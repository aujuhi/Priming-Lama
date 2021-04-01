from transformers import BertTokenizerFast
from transformers import BertForSequenceClassification
import csv
import pandas as pd
import os
import re
import random


model_path = "./fever-factchecker-bert-base-uncased"
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased", do_lower_case=True)
target_names = ["REFUTES", "NOTENOUGHINFO", "SUPPORTS"]
model = BertForSequenceClassification.from_pretrained(model_path, num_labels=len(target_names))


########get silver label from FC#######################


def get_prediction(claim_evid_tup):
    """
    get Fact Checking label
    :param claim_evid_tup: Array with tupel of claim and evidence
    :return: Fact Checking label
    """
    # tokenize claim evidence tupel
    inputs = tokenizer.batch_encode_plus(claim_evid_tup, truncation=True, padding=True, max_length=512,
                                         return_tensors='pt')
    # infer predictions
    outputs = model(**inputs)
    probs = outputs[0].softmax(1)
    return target_names[probs.argmax()]


def eval_step_one(masked_data_path, priming_path, res_path):
    """
    Give evaluation data to BERT and save predicted labels
    :param masked_data_path: path to evaluation sentences
    :param priming_path: path to wordpairs to be inserted
    :param res_path: path to result folder
    :return: None
    """
    header = ["sentence", "Target adjective", "neutral adjective", "label"]
    with open(masked_data_path, 'r', encoding="utf8") as f:
        data = [line.strip() for line in f.readlines()]
    with open(priming_path, 'r') as f2:
        lines = [line.strip().split("|") for line in f2.readlines()]
    for country, adj1, adj2, label in lines:
        if "country" in masked_data_path:
            res_path_2 = "countries/" + res_path
        else:
            res_path_2 = "ethnicities/" + res_path
        with open(res_path_2 + country + "_" + adj1 + "_" + adj2 + ".csv", "w+", newline='') as rf:
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
                       get_prediction([(evidence_count, evidence_adj_neut)]), label]# This label is outdated and has to
                                                                                    #  be adapted before computing the
                                                                                    # metrics for Step 1
            print(res_row)
            with open(res_path_2 + country + "_" + adj1 + "_" + adj2 + ".csv", "a", newline='') as rf:
                writer = csv.writer(rf)
                writer.writerow(res_row)


def replace_cntr(text, new_val, target_val = "###TARGET###"):
    """
    replace indicator for later insertion in sentence
    :param text: the evidence sentence with location indicator
    :param new_val: country or adjective to insert at location
    :param target_val: the location indicator string
    :return: sentence with replaced word
    """
    if re.search("a " + target_val, text) or re.search("an " + target_val, text):
        # change a or an if country starts with vocal
        if new_val[0].lower() in ["a", "e", "i", "o", "u"]:
            re.sub("a " + target_val, "an " + new_val, text)
            re.sub("an " + target_val, "an " + new_val, text)
        else:
            re.sub("a " + target_val, "a " + new_val, text)
            re.sub("an " + target_val, "a " + new_val, text)
    res1 = re.sub(target_val, new_val, text)
    # Make replacements at beginning uppercase
    if not res1[0].isupper():
        sent_list =  [word for word in res1.split()]
        sent_list[0] = sent_list[0].title()
        res2 = ' '.join(sent_list)
    else:
        # remove double whitespaces
        res2 = re.sub(' +', ' ', res1)
    return res2


def count_eval_res(file_path, a):
    """
    Count the returned Fact Checking labels from Step 1 and compute silver-labels
    :param file_path: path to validation data
    :param a: decision boundary
    :return: None
    """
    if "countr" in file_path:
        res_name = file_path + "eval_counts_c_" + str(a) + ".csv"
    if "ethni" in file_path:
        res_name = file_path + "eval_counts_e_" + str(a) + ".csv"
    with open(res_name, "w+", newline='') as rf:
        csv.writer(rf).writerow(
            ["Country_adj_triplet", "Diff_Support", "Diff_Refute", "Step_1_label", "New_label", "Rand_label"])
    for file in os.listdir(file_path):
        if "eval" in file:
            continue
        df = pd.read_csv(file_path + file)
        print(file_path + file)
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
        old_label = 0 # old label set in adapt_step1_pred_csv
        if diff_sup >= a:
            new_label = 1
        else:
            new_label = 0
        random_label = random.randint(0, 1)
        col = [file.split(".")[0], diff_sup, diff_ref, old_label, new_label, random_label]
        with open(res_name, "a", newline='') as rf:
            csv.writer(rf).writerow(col)



################## Get classification from Step 1 ##########

def adapt_step1_pred_csv(csv_file_name, val_path):
    """
    adapt the outdated label from Step 1 and the baseline
    :param csv_file_name: result file from eval_step_one
    :param val_path: path to validation data to get ranked list of positive and negative primes
    :return: None
    """
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
            # use only excerpt of countries / ethnics
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

            # label adjective as correlated if it is in the positive or negative wordlist
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


def get_pos_and_neg_dict(csv_path, val_path):
    """
    load the top rank changers from Step 1
    :param csv_path: path to results from Step 1 i.e. to the ranked lists
    :param val_path: path to validation data i.e. to the result file
    :return: dictionary with geo words as keys and list of ranked primes as values
    """
    if "Countr" in csv_path:
        pos_prem_file = "ordered_prems_pos_country.csv"
        neg_prem_file = "ordered_prems_neg_country.csv"
    else:
        pos_prem_file = "ordered_prems_pos_ethnicity.csv"
        neg_prem_file = "ordered_prems_neg_ethnicity.csv"
    prem_dic_pos = {}
    with open(val_path + pos_prem_file) as premf_pos:
        reader = csv.reader(premf_pos)
        for line in reader:
            prem_dic_pos[line[0].strip()] = line[1:]
    prem_dic_neg = {}
    with open(val_path + neg_prem_file) as premf_neg:
        reader = csv.reader(premf_neg)
        for line in reader:
            prem_dic_neg[line[0].strip()] = line[1:]
    return prem_dic_pos, prem_dic_neg

def main():
    eval_step_one("setup/masked_evids_country.txt", "setup/val_step1_country_pairs.txt", "results/")
    count_eval_res("results/countries/", 10)
    adapt_step1_pred_csv("results/eval_counts_c_10.csv", ".")

if __name__== "__main__":
    main()

