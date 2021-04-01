import os
import re
import csv
import json
import pandas as pd
import numpy as np


def readJson(jsonPath, prem_name):
    """
    read result files from predictions of BERT
    :param jsonPath: path to prediction files
    :param prem_name: name of premis used for prediction
    :return: list of Sentences and list of ranked list of predictions by BERT per sentence
    """
    jPath = jsonPath + prem_name + ".json"
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


def readjsonAndWritecsv(jsonPath, res_csv_name, category):
    """
    read json, calculate rank changes and save to csv
    :param jsonPath: path to prediction files
    :param res_csv_name: path to result file
    :param category: the targeted adjective
    :return: None
    """
    premisses = [prem.split("_")[-1].split(".")[0]
                 for prem in os.listdir(jsonPath)
                 if prem.split(".")[-1] == "json"]
    regex = re.compile('[^a-zA-Z]')
    preds_step2 = []
    for pre in premisses:
        sents, preds = readJson(jsonPath, pre)
        if not sents:
            continue
        result = [regex.sub('', pre)]
        tups = []
        for i in range(min(len(sents), 200)):
            tups.append(filterRankPair(preds[i], category[0], category[1]))
        preds_step2.append(tups)
    with open(jsonPath + "sorted/" + res_csv_name, "w", newline="") as f:
        writer = csv.writer(f)
        header = [" "] + premisses
        writer.writerow(header)
    cols = []
    for i in range(min(len(sents), 200)):
        col = [sents[i]]
        for tups in preds_step2:
            tup = tups[i]
            if not tup:
                print(category[0], " or ", category[1], "Not predicted at all, sorry!")
                continue
            col.append(tup)
        cols.append(col)
    with open(jsonPath + "sorted/" + res_csv_name, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(cols)


def filterRankPair(prediction, word1, word2):
    """
    filter targeted adjectives from list of prediction
    :param prediction: returned predictions from LAMA-probing
    :param word1: first targeted adjective
    :param word2: antonym of targeted adjective
    :return: rank of both adjectives to write to csv
    """
    pred = [tup[0] for tup in prediction]
    print("Pred:", pred)
    print(word1, word2)
    try:
        rank1 = pred.index(word1) + 1
    except ValueError:
        rank1 = 10000
    try:
        rank2 = pred.index(word2) + 1
    except ValueError:
        rank2 = 10000
    rank_str = str(rank1) + "|" + str(rank2)
    return rank_str


def rank_diff_closed(jsonPath):
    """
    calculate averaged rank difference for country/ethnic prime per adjective
    :param jsonPath: path to filtered file of predictions
    :return: None
    """
    cols = {}
    csv_files = [f for f in os.listdir(jsonPath) if "sorted_result_" in f]
    with open(jsonPath + csv_files[0], "r") as csv_f:
        reader_header = csv.reader(csv_f)
        header = next(reader_header)
    with open(jsonPath + "rank_diff_all.csv", "w+", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
    for csv_file in csv_files:
        words = [csv_file.split("_")[-2], csv_file.split("_")[-1].split(".")[0]]
        dic = {}
        with open(jsonPath  + csv_file, "r") as csv_f:
            reader = csv.reader(csv_f)
            current_header = next(reader)

            # calculate rank difference
            noprem_index = current_header.index("NoPrem")
            for r in reader:
                sentence = r[0]
                rank_noprem = (int(r[noprem_index].split("|")[0]), int(r[noprem_index].split("|")[1]))
                for i, rank_pair in enumerate(r[1:]):
                    ranks = (int(rank_pair.split("|")[0]), int(rank_pair.split("|")[1]))
                    if int(ranks[0]) == 10000 or int(ranks[1]) == 10000:
                        print("Skipped sentence ", sentence)
                        break
                    if current_header[i + 1] in dic.keys():
                        dic[current_header[i + 1]].append(
                            (int(rank_noprem[0]) - int(ranks[0]), int(rank_noprem[1]) - int(ranks[1])))
                    else:
                        dic[current_header[i + 1]] = [(int(rank_noprem[0]) - int(ranks[0]),
                                                       int(rank_noprem[1]) - int(ranks[1]))]
        dic1 = {}
        dic2 = {}
        for key, value in dic.items():
            sum1 = 0
            sum2 = 0
            for pair in value:
                sum1 += pair[0]
                sum2 += pair[1]
            dic1[key] = (sum1, len(value))
            dic2[key] = (sum2, len(value))

        for i, country in enumerate(header[1:]):
            if words[0] in cols.keys():
                try:
                    cols[words[0]][0][i] += dic1[country][0]
                    cols[words[0]][1][i] += dic1[country][1]
                except KeyError:
                    continue
            else:
                try:
                    cols[words[0]] = [np.zeros(len(header) - 1), np.zeros(len(header) - 1)]
                    cols[words[0]][0][i] += dic1[country][0]
                    cols[words[0]][1][i] += dic1[country][1]
                except KeyError:
                    print(country, "not in file ", csv_file)
                    continue
            if words[1] in cols.keys():
                try:
                    cols[words[1]][0][i] += dic2[country][0]
                    cols[words[1]][1][i] += dic2[country][1]
                except KeyError:
                    continue
            else:
                try:
                    cols[words[1]] = [np.zeros(len(header) - 1), np.zeros(len(header) - 1)]
                    cols[words[1]][0][i] += dic2[country][0]
                    cols[words[1]][1][i] += dic2[country][1]
                except KeyError:
                    continue

    # calculate the mean
    for adj, sum_len_pair in cols.items():
        mean_list = []
        for k in range(len(sum_len_pair[0])):
            if sum_len_pair[0][k] != 0.0:
                mean_list.append(sum_len_pair[0][k] / sum_len_pair[1][k])
            else:
                mean_list.append(0.0)
        cols[adj] = [adj] + mean_list
        with open(jsonPath + "rank_diff_all.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(cols[adj])


def readjsonAndWritecsv(jsonPath, res_csv_path, category):
    categories = category
    premisses = [prem.split("_")[-1].split(".")[0]
                 for prem in os.listdir(jsonPath)
                 if prem.split(".")[-1] == "json"]
    preds_step2 = []
    for pre in premisses:
        sents, preds = readJson(jsonPath, pre)
        if not sents:
            continue
        tups = []
        for i in range(min(len(sents), 200)):
            tups.append(filterRankPair(preds[i], categories[0], categories[1]))
        preds_step2.append(tups)
    with open(res_csv_path + "sorted_result_" + categories[0] + "_" + categories[1] + ".csv", "w", newline="") as f:
        writer = csv.writer(f)
        header = [" "] + premisses
        writer.writerow(header)
    cols = []
    for i in range(min(len(sents), 200)):
        col = [sents[i]]
        for tups in preds_step2:
            tup = tups[i]
            if not tup:
                print(categories[0], " or ", categories[1], "Not predicted at all, sorry!")
                continue
            col.append(tup)
        cols.append(col)
    with open(res_csv_path + "sorted_result_" + categories[0] + "_" + categories[1] + ".csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(cols)


def sort_results_closed(jsonPath):
    """
    create sorted result from BERTs predicitons
    :param jsonPath: path to BERTs predicitons
    :return: None
    """
    for json_path in os.listdir(jsonPath):
        if "sorted" in json_path:
            continue
        cat1 = json_path.split("_")[0].strip()
        cat2 = json_path.split("_")[1].strip()
        print(cat1, cat2)
        readjsonAndWritecsv(
            jsonPath + json_path + "/",
            jsonPath,
            [cat1, cat2])


def create_word_list_columnwise(data_path):
    """
    find top adjective prime correlations
    :param jsonPath: path to previous result files
    :return: list of top positive correlated adjectives per prime, list of top negative correlated adjectives per prime
    """
    tup_list_per_country = {}
    tup_list_per_country_asc = {}
    tup_list_per_country_dec = {}
    #df_z = pd.read_csv(jsonPath + "plots/z_werte.csv", index_col=0)
    df = pd.read_csv(data_path + "/rank_diff_all.csv", index_col=0)
    df = df.T
    column_names = df.columns.values.tolist()
    row_names = list(df.index)

    # iterate trough z-values and filter outlier
    for col in column_names:
        tup_per_adj = []
        for row in row_names:
            #z = df_z.at[row, col]
            rank_diff = df.at[row, col]
            #if 3.0 > z > -3.0:
            tup_per_adj.append((row, rank_diff))

        # get ranked countries for every adjective
        tup_per_adj = sorted(tup_per_adj, key=getKey, reverse=True)

        # append adjective and rank to every country
        for i, tup in enumerate(tup_per_adj):
            if tup[0] in tup_list_per_country:
                tup_list_per_country[tup[0]].append([col, i+1, tup[1]])
            else:
                tup_list_per_country[tup[0]] = [[col, i+1, tup[1]]]

    # only return top pos and neg correlated adjectives per country
    for key in tup_list_per_country.keys():
        tup_list_per_country_asc[key] = [tup[0] for tup in sorted(tup_list_per_country[key], key=getKey, reverse=False)
                                         if tup[2] > 0]
        tup_list_per_country_dec[key] = [tup[0] for tup in sorted(tup_list_per_country[key], key=getKey, reverse=True)
                                         if tup[2] < 0]

    with open("ordered_prems_pos_step2.csv", 'w+') as pf:
        writer = csv.writer(pf)
        for key, value in tup_list_per_country_asc:
            writer.writerow([key] + value)
    with open("ordered_prems_neg_step2.csv", 'w+') as nf:
        writer = csv.writer(nf)
        for key, value in tup_list_per_country_dec:
            writer.writerow([key] + value)
    return tup_list_per_country_asc, tup_list_per_country_dec


def getKey(item):
    return item[1]


def main():
    sort_results_closed("example_data_2/example_predictions/")
    rank_diff_closed("example_data_2/example_predictions/")

if __name__== "__main__":
    main()
