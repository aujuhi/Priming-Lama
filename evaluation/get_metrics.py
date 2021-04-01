import pandas as pd
import os
import csv
import random
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


def get_scores_step1(validation_path, result_path):
    """
    calculate metrics with sklearn
    :param validation_path: path to labeled pairs
    :param result_path: path to resultfile
    :return: None
    """
    res = []
    df = pd.read_csv(validation_path)
    label_vers = ["Step_1_label", "Step_1_label_ver2"]
    rand_labels = []
    for i in range(len(list(df["Step_1_label"]))):
        rand_labels.append(1)
    for label in label_vers:
        scores = [precision_score(df["New_label"], df[label]),
                  accuracy_score(df["New_label"], df[label]),
                  recall_score(df["New_label"], df[label]),
                  f1_score(df["New_label"], df[label])]
        scores_base = [precision_score(df["New_label"], rand_labels),
                       accuracy_score(df["New_label"], rand_labels),
                       recall_score(df["New_label"], rand_labels),
                       f1_score(df["New_label"], rand_labels)]
        res.append(
            [label, scores[0], scores_base[0], scores[1], scores_base[1], scores[2], scores_base[2],
             scores[3], scores_base[3]])
    with open(result_path, "w+",
          newline='') as rf:
        writer = csv.writer(rf)
        writer.writerow(
            ["Label version", "precision", "baseline precision", "accuracy", "baseline accuracy", "recall",
             "baseline recall", "F1", "baseline F1"])
        for row in res:
            writer.writerow(row)


def get_scores_step2(data_path, res_path):
    """
    calculate metrics with sklearn
    :param data_path: Path to evalRes files from evaluate_step2
    :param res_path: result path
    :return: None
    """
    print(data_path)
    res = []
    for res_file in os.listdir(data_path):
        if "README" in res_file:
            continue
        trial = res_file.split(".")[0].split("_")[-2] + " " + res_file.split(".")[0].split("_")[-1]
        df = pd.read_csv(data_path + "/" + res_file)
        scores = [precision_score(df["FC label"], df["Step 2 label"], average='micro', labels=[1, 2]),
                  accuracy_score(df["FC label"], df["Step 2 label"]),
                  recall_score(df["FC label"], df["Step 2 label"], average='micro', labels=[1, 2]),
                  f1_score(df["FC label"], df["Step 2 label"], average='micro', labels=[1, 2])]
        print(scores)
        rand_labels = []
        for i in range(len(list(df["rand label"]))):
            rand_labels.append(1)
        scores_base = [precision_score(df["FC label"], rand_labels, average='micro', labels=[1, 2]),
                       accuracy_score(df["FC label"], rand_labels),
                       recall_score(df["FC label"], rand_labels, average='micro', labels=[1, 2]),
                       f1_score(df["FC label"], rand_labels, average='micro', labels=[1, 2])]
        res.append([trial, scores[0], scores_base[0], scores[1], scores_base[1], scores[2], scores_base[2],
                    scores[3], scores_base[3]])
    with open(res_path, "w+", newline='') as rf:
        writer = csv.writer(rf)
        writer.writerow(["Trial", "precision", "baseline precision", "accuracy", "baseline accuracy", "recall",
                         "baseline recall", "F1", "baseline F1"])
        for row in res:
            writer.writerow(row)


get_scores_step1("results/eval_counts_c_10_ver2.csv",
                 "results/metrics_step_1.csv")
get_scores_step2("results/countries/", "results/metrics_step_2.csv")