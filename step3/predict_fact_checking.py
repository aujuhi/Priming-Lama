import os
import pandas as pd
from transformers import BertTokenizerFast
from transformers import BertForSequenceClassification
import csv
import re


model_name = "fever-factchecker-bert-base-uncased_subset_99"
model_path = "./" + model_name
tokenizer = BertTokenizerFast.from_pretrained(model_name, do_lower_case=True)
target_names = ["REFUTES", "NOTENOUGHINFO", "SUPPORTS"]
model = BertForSequenceClassification.from_pretrained(model_path, num_labels=len(target_names))
model.eval()

def get_prediction(claim_evid_tup):
    """
    get Fact Checking label
    :param claim_evid_tup: Array with tupel of claim and evidence
    :return: Fact Checking label
    """
    # tokenize claim evidence tupel
    inputs = tokenizer.batch_encode_plus(claim_evid_tup, truncation=True, padding=True, max_length=512, return_tensors='pt')
    #infer predictions
    outputs = model(**inputs)
    probs = outputs[0].softmax(1)
    return target_names[probs.argmax()]


def replace_adj(claim, adjectives):
    """
    replace indicator for later insertion in sentence with adjective or adjectives
    :param claim: the evidence sentence with location indicator
    :param adjectives: adjective or list of adjectives to insert
    :return: sentence with replaced adjective
    """
    res = claim + "."[:-1]
    for adj in adjectives:
        if adj == adjectives[-1]:
            res = res.replace("###ADJ", adj + " and ")
        else:
            res = res.replace("###ADJ", adj + ", ###ADJ")
    return res


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


def create_trials_evid_pairs(country_file_path, masked_countr_path, masked_ethn_path, rank_list):
    """
    insert adjectives or geo words and predict results
    :param country_file_path: path to all countries
    :param rank_list: list of strongly correlated adjectives
    :return: None
    """
    # get ethnicity adjective pairs with a strong correlation
    pairs_c = []
    pairs_e = []
    with open(country_file_path, "r", encoding="UTF8") as cf:
        countries = [e.strip() for e in cf.readlines()]

    if "ascending" in rank_list:
        asc_or_desc = "evid_pair_asc/"
    else:
        asc_or_desc = "evid_pair_desc/"

    with open(rank_list, "r") as csv_f:
        reader = csv.reader(csv_f)
        for line in reader:
            if not line:
                continue
            ethnicity = line[0]
            if ethnicity in countries:
                pairs_c.append([ethnicity, line[1].strip()])
            else:
                pairs_e.append([ethnicity, line[1].strip()])

    # replace singular ethnicities in fever evidences
    with open(masked_countr_path, 'r', encoding="utf8") as fevFiles:
        evidences_c = [line.strip() for line in fevFiles.readlines() if line]
    with open(masked_ethn_path, 'r', encoding="utf8") as fevFiles:
        evidences_e = [line.strip() for line in fevFiles.readlines() if line]

    replace_and_predict(asc_or_desc, evidences_c, pairs_c)
    replace_and_predict(asc_or_desc, evidences_e, pairs_e)


def replace_and_predict(asc_or_desc, evids, pairs):
    """
    replace words and predict with Fact Checker and write to csv
    :param asc_or_desc: determiner if list of pos or neg correlated adjectives used
    :param masked_evids: evidence to insert countries or ethnicities /adjectives
    :param pairs: geo word and adjective
    :return: None
    """
    header = ["Masked evidence", "Claim with corr adj, evidence with ethn", "Claim with ethn, evidence with corr adj",
              "Claim with neutral adj, evidence with ethn", "Claim with ethn, evidence with neutral adj", ]
    for count, adj in pairs:
        res_filename = asc_or_desc + count + "_" + adj + ".csv"
        with open(res_filename, 'w+', encoding='utf8', newline='') as rf:
            writer = csv.writer(rf)
            writer.writerow(header)
        for evid in evids:
            evidence_count = replace_cntr(evid, count)
            evidence_adj = replace_cntr(evid, "the " + adj + " country")
            print(evidence_count)
            print(evidence_adj)
            res_row = [evid, get_prediction([(evidence_adj, evidence_count)]),
                       get_prediction([(evidence_count, evidence_adj)]),
                       0,  # originally predicitons with neutral adjectives
                       0]  # were conducted here but this is outdated
            with open(res_filename, 'a', encoding='utf8', newline="") as rf:
                writer = csv.writer(rf)
                writer.writerow(res_row)


def contingency_table_evid_pairs(res_path, trial):
    """
    load results and count class labels for each geo word adjective pair
    :param res_path: path to result file
    :param trial: positive or negative and ethnicity or country
    :return: None
    """
    cols = ["Claim with corr adj, evidence with ethn",
            "Claim with neutral adj, evidence with ethn",
            "Claim with ethn, evidence with corr adj",
            "Claim with ethn, evidence with neutral adj"]
    with open(res_path + "label_counts_" + trial + ".csv", "w+", newline="") as rf:
        writer = csv.writer(rf)
        writer.writerow(["Ethnicity", "Adjective"] + ["No. Support adj_eth", "No. Refute adj_eth",
                                                      "No. Support neut_eth", "No. Refute neut_eth",
                                                      "No. Support eth_adj", "No. Refute eth_adj",
                                                      "No. Support eth_neut", "No. Refute eth_neut"])
    file_path = res_path + trial + "/"

    for file_name in os.listdir(file_path):
        if "README" in file_name:
            continue
        print(file_name)
        df = pd.read_csv(file_path + file_name)
        newdf = df.drop(df.columns[0], axis=1)
        cntg_table = newdf.apply(pd.value_counts)
        cntg_table = cntg_table.fillna(0)
        counts = [cntg_table.loc["SUPPORTS", cols[0]], cntg_table.loc["REFUTES", cols[0]],
                  cntg_table.loc["SUPPORTS", cols[1]], cntg_table.loc["REFUTES", cols[1]],
                  cntg_table.loc["SUPPORTS", cols[2]], cntg_table.loc["REFUTES", cols[2]],
                  cntg_table.loc["SUPPORTS", cols[3]], cntg_table.loc["REFUTES", cols[3]]]
        row = [file_name.split("_")[0], file_name.split("_")[1].split(".")[0]] + counts
        with open(res_path + "label_counts_" + trial + ".csv", "a", newline="") as rf:
            writer = csv.writer(rf)
            writer.writerow(row)


def main():
    create_trials_evid_pairs("allCountries.txt", "fever/masked_evids_country.txt", "fever/masked_evids_ethnicity.txt",
                             "top_ascending_ranks_example.csv")
    contingency_table_evid_pairs("results/", "evid_pair_desc")

if __name__== "__main__":
    main()
