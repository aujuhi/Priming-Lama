import json
import re
import numpy.random as rand
import os
import spacy


nlp = spacy.load("en_core_web_sm")

def find_evid_pairs(data_path, nei_file, train_file, country_file_path, prems_full_hist_path):
    """
    find evidences containing countries or adjectives and write to file
    :param data_path:
    :param country_file_path:
    :param prems_full_hist_path:
    :return: None
    """
    c_mani_set = []
    e_mani_set = []
    stop = False
    # load country- and ethnicitywords
    with open(country_file_path, "r", encoding="UTF8") as cf:
        countries = [c.strip() for c in cf.readlines()]
    # get fever files with NEI
    with open(nei_file, 'r') as fevFiles:
        lines = fevFiles.readlines()
    # get all premisses
    with open(prems_full_hist_path, 'r', encoding='UTF8') as premsFile:
        prems = [p.strip() for p in premsFile.readlines()]

    # get all ids from train set to not use them here
    with open(train_file, "r", encoding="utf8") as train_f:
        train_lines = train_f.readlines()
    ids = [json.loads(train_line)['annotation_id'] for train_line in train_lines]
    # only use evidence, which are not part of the train set
    doc_files = os.listdir(data_path + "docs/")
    rand.shuffle(doc_files)
    for file in doc_files:
        if file in ids:
            continue
        with open(data_path + "docs/" + file, 'r', encoding="utf8") as docFile:
            try:
                sentences = [sentence.strip() for sentence in docFile.readlines()]
            except FileNotFoundError:
                continue
        for sentence in sentences:
            print(sentence)
            if len(c_mani_set) <= 300:
                countr = check_sent_count(sentence, countries, prems)
                if countr:
                    c_mani_set.append(countr)
            if len(e_mani_set) <= 300:
                ethn = check_sent_ethn(sentence, countries)
                if ethn:
                    e_mani_set.append(ethn)
                    break
    #        if len(e_mani_set) >= 300 and len(c_mani_set) >= 300:
            if len(e_mani_set) >= 300:
                stop = True
                break
        print(e_mani_set)
        if stop:
            break
    with open(data_path + "masked_evids_country.txt", "w+", encoding="utf8") as cf:
        for line in c_mani_set:
            cf.write(line + "\n")
    with open(data_path + "masked_evids_ethnicity.txt", "w+", encoding="utf8") as ef:
        for line in e_mani_set:
            ef.write(line + "\n")


def check_sent_count(sent, word_list, prems):
    """
    pick sentences that meet country requirements
    :param sent: sentence to check
    :param word_list: list of countries
    :param prems: premises from previous steps to avoid
    :return: adapted sentence if requirements met, else False
    """
    tagged_sent = nlp(sent)
    new_sent = False
    # only use short sentences
    if len(tagged_sent) > 25:
        print("Too long")
        return False
    for i, token in enumerate(tagged_sent):
        # avoid NEs except countries
        if i > 0 and token.is_alpha and not token.text[0].islower():
            # don't use sents containing previous premisses
            if token.text in prems:
                return False
            # only one country per sentence
            if new_sent:
                print("More than one country: ", sent)
                return False
            # find and replace country
            for word in word_list:
                if word == token.text:
                    new_sent = re.sub(word, " ###TARGET### ", sent)
                    break
            if not new_sent:
                print("NE: ", token)
                return False
        # no adjectives in country sentences
        if token.pos_ == "ADJ":
            print("Adjective in country sent: ", sent)
            return False
    if new_sent:
        print("GOT ONE: ", new_sent)
        return new_sent
    else:
        print("No country")
        return False


def check_sent_ethn(sent, count_and_ethn):
    """
    pick sentences that meet country requirements
    :param sent: sentence to check
    :param count_and_ethn: premises from previous steps to avoid
    :return: adapted sentence if requirements met, else False
    """
    tagged_sent = nlp(sent)
    new_sent = False
    # only use short sentences
    if len(tagged_sent) > 15:
        return False
    for i, token in enumerate(tagged_sent):
        if i == 0:
            if not token.is_alpha:
                print("Began with punctuation: ", sent)
                return False
        if i == len(tagged_sent) - 1:
            if token.text not in ["!", ".", "?"]:
                print("No punctuation at end: ", sent)
        # avoid NEs
        if i > 0 and token.is_alpha and not token.text[0].islower():
            print("NE: ", token)
            return False
        #avoid other countries or ethnicities
        if i == 0:
            if token.text in count_and_ethn:
                return False
        # only sentences with one Adjective
        if new_sent and token.pos_ == "ADJ":
            print("more than one adj: ", sent)
            return False
        # exclude words like ordinals
        exclude = ["one", "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "few", "many", "such"]
        # replace adjective with target mask
        if token.pos_ == "ADJ" and token.text not in exclude:
            new_sent = re.sub(token.text, " ###TARGET### ", sent)
    if new_sent:
        print("GOT ONE: ", new_sent)
        return new_sent
    else:
        return False


def main():
    find_evid_pairs("fever/", "fever/train30000.jsonl", "fever/train99.jsonl", "allCountries.txt", "primes.txt")

if __name__== "__main__":
    main()

