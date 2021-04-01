# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This file is part of the LAMA-probing project, the link is given in the README.md
#

from lama.modules import build_model_by_name
import lama.evaluation_metrics as evaluation_metrics
import os
from nltk.tokenize.treebank import TreebankWordDetokenizer
import random

data_path = "/ukp-storage-1/unger/data/"
trial_1_c = "/ukp-storage-1/unger/trial_results/experimentCountries/"
trial_1_e = "/ukp-storage-1/unger/trial_results/experimentEthnicities/"
trial_2 = "/ukp-storage-1/unger/trial_results/closed_trial/"
twd = TreebankWordDetokenizer()

class Args_Stud(object):
    bert_model_dir = 'C:/Users/Anke/PycharmProjects/LAMA/pre-trained_language_models/bert/cased_L-24_H-1024_A-16'
    bert_model_name = 'bert-base-cased'
    bert_vocab_name = 'vocab.txt'
    interactive = False
    max_sentence_length = 100
    models = 'bert'
    models_names = ['bert']
    split_sentence = False


def maskPairs(pair_in):
    foldernames = os.listdir(data_path + "data_wiki_processed_4/")
    X = {}
    found = 0
    with open(trial_1_c + "allCountries.txt", "r", encoding="utf8") as ca:
        cues = [c.strip().lower() for c in ca.readlines() if c]
    while found < 30000:
        folder = data_path + "data_wiki_processed_4/" + foldernames[random.randint(0, len(foldernames) - 1)] + "/"
        found += 1
        articles = [a for a in os.listdir(folder)]
        if not articles:
            continue
        article_name = folder + articles[random.randint(0, len(articles) - 1)]
        with open(article_name, "r", encoding="utf8") as article:
            try:
                sentences = [line.strip().split(" ") for line in article.readlines()]
            except UnicodeDecodeError:
                print("Decode Error in file " + article_name)
                continue
        for sent in sentences:
            pair, sent_str = masking(sent, pair_in, cues)
            if pair and sent_str not in X:
                pair.sort()
                pair = pair[0] + "_" + pair[1]
                if pair in X.keys():
                    if sent_str not in X[pair]:
                        X[pair].append(sent_str)
                        found = 0
                else:
                    X[pair] = [sent_str]
                    found = 0
    for key in X.keys():
        with open(trial_2 + "masked_sents_" + key + ".txt",
              'w+') as fp:
            for sen in X[key]:
                fp.write(sen + "\n")
    print("Finish:", X)
    return X


def masking(sentence, pair, cues, bert):
    delim = sentence[-1][-1]
    sentence[-1] = sentence[-1].replace(delim, "")
    for target in pair:
        if twd.detokenize(sentence).find(target) != -1:
            masked, syn, ant = checkSent(sentence, pair, target, cues, delim, bert)
            if masked:
                print("Ergebnis:", syn, ant)
                return [syn, ant], masked
    return False, False

def checkSent(sentence, pair, target, cues, delim, bert):
    for word in sentence:
        if word in cues:
            print("Rejected because ", word, " is in sentence.")
            return False, False, False
    try:
        sentence[sentence.index(target)] = "[MASK]"
    except ValueError:
        return False, False, False
    sent_str = twd.detokenize(sentence) + delim
    data = lama(sent_str, bert)
    data = [d[0] for d in data][:300]
    with open(trial_2 + "priming/syn_lists/" + pair[0] + ".txt", "r") as syn_f:
        lines_1 = syn_f.readlines()
    syns = [l.strip() for l in lines_1]
    with open(trial_2 + "priming/syn_lists/" + pair[1] + ".txt", "r") as ant_f:
        lines_2 = ant_f.readlines()
    ants = [l.strip() for l in lines_2]
    syns = [pair[0]] + syns
    ants = [pair[1]] + ants
    syn_ind = []
    ant_ind = []
    for s in syns:
        if s in data:
            syn_ind.append(data.index(s))
    for a in ants:
        if a in data:
            ant_ind.append(data.index(a))
    if syn_ind and ant_ind:
        return sent_str, data[min(syn_ind)], data[min(ant_ind)]
    return False, False, False


def lama(sent, bert):
    data = []
    filtered_log_probs_list, [token_ids], [masked_indices] = bert.get_batch_generation([[sent]], try_cuda=True)
    # rank over the subset of the vocab (if defined) for the SINGLE masked tokens
    if masked_indices and len(masked_indices) > 0:
        MRR, P_AT_X, experiment_result, return_msg = evaluation_metrics.get_ranking(filtered_log_probs_list[0],
                                                                                    masked_indices, bert.vocab,
                                                                                    index_list=None)
        res = experiment_result["topk"]
        for r in res:
            data.append((r["token_word_form"],r["log_prob"]))
    return data


def main():
    with open(trial_2 + "priming/antonyme_adj.txt", "r") as ah:
        pairs = [(p.split()[0].strip(), p.split()[1].strip()) for p in ah.readlines()]
    print(pairs)
    args_stud = Args_Stud()
    bert = build_model_by_name("bert", args_stud)
    for pair in pairs:
        maskPairs(pair, bert)
    return


if __name__ == "__main__":
    main()