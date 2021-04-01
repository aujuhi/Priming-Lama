# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# This file is part of the LAMA-probing project, the link is given in the README.md
#

from lama.modules import build_model_by_name
from lama.utils import print_sentence_predictions
import lama.evaluation_metrics as evaluation_metrics
import json
import os
import time

class Args_Stud(object):
    bert_model_dir = './bert/cased_L-24_H-1024_A-16'
    bert_model_name = 'bert-base-cased'
    bert_vocab_name = 'vocab.txt'
    interactive = False
    max_sentence_length = 100
    models = 'bert'
    models_names = ['bert']
    split_sentence = False


def main():
    args_stud = Args_Stud()
    bert = build_model_by_name("bert", args_stud)
    vocab_subset = None
    f = open('./LAMA/lama/collected_paths.json', )
    path_s = json.load(f)
    sent_path_ = path_s['sent2eval']
    prem_path = path_s['premis2eval']
    res_path_ = path_s["res_file"]
    paths = os.listdir(sent_path_)
    for path in paths:
        sent_path = sent_path_+ path
        res_path = res_path_ + path.split(".")[0].split("_")[-2] + "_" + path.split(".")[0].split("_")[-2] + "/"
        os.makedirs(res_path, exist_ok=True)
        with open(sent_path, "r", encoding="utf8") as sf:
            sentences = [s.rstrip for s in sf.readlines()]
        print(sentences)
        with open(prem_path, "r") as pf:
            premisses = [p.rstrip() for p in pf.readlines()]
        data = {}
        for s in sentences:
            data[s] = []
            original_log_probs_list, [token_ids], [masked_indices] = bert.get_batch_generation([[s]], try_cuda=True)
            index_list = None
            if vocab_subset is not None:
                # filter log_probs
                filter_logprob_indices, index_list = bert.init_indices_for_filter_logprobs(vocab_subset)
                filtered_log_probs_list = bert.filter_logprobs(original_log_probs_list, filter_logprob_indices)
            else:
                filtered_log_probs_list = original_log_probs_list

            # rank over the subset of the vocab (if defined) for the SINGLE masked tokens
            if masked_indices and len(masked_indices) > 0:
                MRR, P_AT_X, experiment_result, return_msg = evaluation_metrics.get_ranking(filtered_log_probs_list[0],
                                                                                            masked_indices, bert.vocab,
                                                                                            index_list=index_list)
                res = experiment_result["topk"]
                for r in res:
                    data[s].append((r["token_word_form"], r["log_prob"]))
        with open(res_path + "NoPrem.json", "w+",
                  encoding="utf-8") as f:
            json.dump(data, f)
        for pre in premisses:
            for s in sentences:
                data[s] = []
                sentence = [str(pre) + "? " + s]
                original_log_probs_list, [token_ids], [masked_indices] = bert.get_batch_generation([sentence], try_cuda=False)
                index_list = None
                if vocab_subset is not None:
                    # filter log_probs
                    filter_logprob_indices, index_list = bert.init_indices_for_filter_logprobs(vocab_subset)
                    filtered_log_probs_list = bert.filter_logprobs(original_log_probs_list, filter_logprob_indices)
                else:
                    filtered_log_probs_list = original_log_probs_list

                # rank over the subset of the vocab (if defined) for the SINGLE masked tokens
                if masked_indices and len(masked_indices) > 0:
                    MRR, P_AT_X, experiment_result, return_msg = evaluation_metrics.get_ranking(filtered_log_probs_list[0], masked_indices, bert.vocab, index_list=index_list)
                    res = experiment_result["topk"]
                    for r in res:
                        data[s].append((r["token_word_form"], r["log_prob"]))
            with open(res_path + pre + ".json", "w+",
                  encoding="utf-8") as f:
                    json.dump(data, f)

if __name__ == '__main__':
    main()
