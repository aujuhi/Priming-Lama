import pandas as pd
from collections import Counter
from itertools import chain
import json
import numpy as np
import csv
import os
import re

def readJson(jsonPath, prem_name):
    """
    read result files from predictions of BERT
    :param jsonPath: path to prediction files
    :param prem_name: name of premis used for prediction
    :return: list of Sentences and list of ranked list of predictions by BERT per sentence
    """
    jPath = jsonPath + "result_" + prem_name + ".json"
    f = open(jPath)
    predictions = []
    try:
        data_dict = json.load(f)
    except json.decoder.JSONDecodeError:
        print("Skipped file ", jPath)
        return (None, None)

    sentences = data_dict.keys()
    for sent in list(sentences):
        predictions.append(data_dict[sent])
    return list(sentences), predictions


def readjsonAndWritecsv(json_path, data_path, category):
    """
    reads the predictions from BERT and filters the rank lists for countries/ethnicities
    :param json_path: Path to predictions from BERT
    :param data_path: path to list of countries/ethnic and result
    :param category: Name of file i.e. country or ethnicity
    :return: None
    """
    if "ethnic" in category:
        with open(data_path + category, "r", encoding="utf8") as cf:
            categories = []
            same_ethn = []
            for c in cf.readlines():
                if not c.strip():
                    categories.append(same_ethn)
                    same_ethn = []
                else:
                    same_ethn.append(c.strip())
            categories.append(same_ethn)
    else:
        with open(data_path + category, "r") as cf:
            categories = [c.strip() for c in cf.readlines()]
    premisses = [prem.split("_")[-1].split(".")[0]
                 for prem in os.listdir(json_path)
                 if prem.split(".")[-1] == "json"]
    first_Data = True
    regex = re.compile('[^a-zA-Z]')
    preds_step2 = []
    for pre in premisses:
        sents, preds = readJson(json_path, pre)
        if not sents:
            continue
        result = [regex.sub('', pre)]
        if "ethnic" in category:
            filteredResults = filterResults_Ethn(preds, categories)
        else:
            filteredResults = filterResults(preds, categories)
        header = filteredResults.keys()
        for k in header:
            result.append(filteredResults[k])
        if first_Data:
            with open(data_path + "sorted_result.csv", 'w+', encoding="utf8", newline="") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(["Premisses"] + list(header))
                writer.writerow(result)
                first_Data = False
        else:
            with open(data_path + "sorted_result.csv", 'a', newline="") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(result)


def highscore_diff_to_no_prime(sorted_results_path):
    """
    highscore list of priming words for each country/ethnic, ordered by rank difference
    :param sorted_results_path: path to sorted predictions of BERT
    :return: top_changers: dictionary with countries/ethnic as key and primes ranked by rankdifference as value
             counted_changer: dictionary with primes as key and number of occurence in top changers as value
             rank_diff_all: rank difference of every prem and country/ethnic
    """
    path_f = sorted_results_path + "sorted_result.csv"
    top_changers = {}
    top_changers_neg = {}
    rank_diff_all = {}
    df = pd.read_csv(path_f)
    header = df.keys()[1:]
    categories = []
    for h in header:
        if "_rank" in h:
            categories.append(h)
    prems = df["Premisses"]
    noPrem_ind = list(prems).index("NoPrem")
    for cat in categories:
        ranks = list(df[cat])
        rank_diff = [(prems[i], ranks[i] - ranks[noPrem_ind]) for i in range(len(prems))]
        rank_diff_all[cat] = rank_diff
        sorted_rank_diff = sorted(rank_diff, key=getKey, reverse=False)
        top_changers[cat] = [tup[0] for tup in
                             sorted_rank_diff if tup[0] != "NoPrem" and tup[
                                 1] < 0]  #  premisses with highest rank difference compared to rank of
        # country without priming key:Country, Value: Premisses ordered
        sorted_rank_diff_neg = sorted(rank_diff, key=getKey, reverse=True)
        top_changers_neg[cat] = [tup[0] for tup in
                             sorted_rank_diff if tup[0] != "NoPrem" and tup[
                                 1] > 0]
    counted_changers = Counter(chain.from_iterable(top_changers.values()))  # Count how often a prime is in the
    # top 10 primes of different countries.
    # Key: Prime,
    # Value: Count and list of countries
    for prime, count in counted_changers.items():
        for country_key, top_primes in top_changers.items():
            if prime in top_primes:
                counted_changers[prime] = str(counted_changers[prime]) + " " + country_key

    #print top changers to csv
    with open("ordered_prems_pos_country.csv", 'w+') as pf:
        writer = csv.writer(pf)
        for key, value in top_changers:
            writer.writerow([key] + value)
    with open("ordered_prems_neg_country.csv", 'w+') as nf:
        writer = csv.writer(nf)
        for key, value in top_changers_neg:
            writer.writerow([key] + value)

    return top_changers, counted_changers, rank_diff_all


def getKey(item):
    return item[1]

########################### Countries ###########################

def filterResults(predictions, categories):
    """
    calculate averaged rank for every country and prime pair
    :param predictions: list of ranked lists of predictions by BERT for every sentence
    :param categories: list of targeted countries
    :return: dictionary of countries as key and averaged rank as value
    """
    dic = {}
    for pred in predictions:
        for rank, p in enumerate(pred):
            word = p[0]
            perpl = np.exp(p[1])
            if word in categories:
                wordrank = word + "_rank"
                if word in dic.keys():
                    dic[word].append(perpl)
                    dic[wordrank].append(rank + 1)
                else:
                    dic[word] = [perpl]
                    dic[wordrank] = [rank + 1]
    for key, val in dic.items():
        dic[key] = np.mean(val)
    return dic

########################### Ethnicities ###########################

def filterResults_Ethn(predictions, categories_multi_dim):
    """
    calculate averaged rank with different grammatical forms of ethnicity cummulated
    :param predictions: list of ranked lists of predictions by BERT for every sentence
    :param categories_multi_dim: different grammatical forms of ethnicity per ethnics
    :return: dictionary of ethnicities as key and averaged rank as value
    """
    dic = {}
    for pred in predictions:
        for rank, p in enumerate(pred):
            word = p[0]
            print("try: ", word)
            perpl = np.exp(p[1])
            for cat in categories_multi_dim:
                if word in cat:
                    print("ethnic: ", word)
                    wordrank = cat[0] + "_rank"
                    if cat[0] in dic.keys():
                        dic[cat[0]].append(perpl)
                        dic[wordrank].append(rank + 1)
                    else:
                        dic[cat[0]] = [perpl]
                        dic[wordrank] = [rank + 1]
    for key, val in dic.items():
        dic[key] = np.mean(val)
    return dic


def main():
    readjsonAndWritecsv("example_data/example_lama_probe/", "example_data/", "allCountries.txt")
    top_changers, counted_changers, rank_diff = highscore_diff_to_no_prime("example_data/")


if __name__== "__main__":
    main()

