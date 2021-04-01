import os
import random
from collections import Counter
import json
from nltk.stem.porter import PorterStemmer
from nltk.tokenize.treebank import TreebankWordDetokenizer


twd = TreebankWordDetokenizer()


def mask_sentences(data_path, replace_list, clues, res_filename):
    """
    iterate through articles and mask geographical words in wiki sentences
    :param data_path: path to retrieved wiki sentences
    :param replace_list: tokens to replace, i.e. ethnics or countries
    :param clues: words related to words of replace list, capitals etc
    :param res_filename: path to result file
    :return: list of masked sentences
    """
    foldernames = os.listdir(data_path)
    good_sents = dict.fromkeys(['X', 'Y'])
    X = []
    Y = []
    found = 0
    while len(X) < 50 and found < 10000:
        folder = data_path + foldernames[random.randint(0, len(foldernames) - 1)] + "/"
        print(folder)
        found += 1
        articles = [a for a in os.listdir(folder) if "oneNE" in a]
        if not articles:
            continue
        # randomly pick articles to check for mentionings of
        article_name = folder + articles[random.randint(0, len(articles) - 1)]
        with open(article_name, "r", encoding="utf8") as article:
            try:
                sentences = [line.strip().split(" ") for line in article.readlines()]
            except UnicodeDecodeError:
                print("Decode Error in file " + article_name)
                continue
        for sent in sentences:
            rep, sent_str = masking(sent, replace_list, clues)
            if rep and sent_str not in X:
                print("Good sentence: ", sent_str)
                found = 0
                # the masked sentenc
                X.append(sent_str)
                # the country/ethnic masked out
                Y.append(rep)
                cnt = Counter(Y)
                if cnt[rep] > 25:
                    replace_list.remove(rep)
                    print("More than 25 mentions of ", rep)
                print(X)
        if len(X) == len(Y):
            good_sents['X'] = X
            good_sents['Y'] = Y
        else:
            print("Error. Some annotations are missing.")
            break
        print(good_sents)
        with open(res_filename + "masked_sentences_" + str(len(X) % 3) + ".json", 'w+') as fp:
            json.dump(good_sents, fp)
    with open(res_filename + "masked_sentences.json",
              'w+') as fp:
        json.dump(good_sents, fp)
    print("Finish:", good_sents)
    return good_sents


def masking(sentence, replace_list, clues):
    """
    replace country or ethnic with BERTs [MASK] token
    :param sentence: sentence from retrieved wiki sentences
    :param replace_list: list of countries or ethnics
    :param clues: words related to words of replace list, capitals etc
    :return: replaced countr/ethnic and masked sentence or False if no geo term contained
    """
    delim = sentence[-1][-1]
    sentence[-1] = sentence[-1].replace(delim, "")
    sent_lower = [w.lower() for w in sentence]
    for rep in replace_list:
        if twd.detokenize(sentence).find(" " + rep + " ") != -1 \
                and checkSent(sent_lower, replace_list, clues, rep):
            sentence[sentence.index(rep)] = "[MASK]"
            sent_str = twd.detokenize(sentence) + delim
            print(rep, sent_str)
            return rep, sent_str
    return False, False


def checkSent(sent_lower, reps, clue, current):
    """
    check if sentence meets requirements
    :param sent_lower: lowercased sentence from wiki article
    :param reps: countries or ethnics to ensure not two countries are in sentence
    :param clue: capitals etc
    :param current: current country/ethnic to be replaced
    :return: True if sent meets requirement, else False
    """
    stemmer = PorterStemmer()
    stemmed_sent = [stemmer.stem(w.lower()) for w in sent_lower]
    current = current.lower()
    words = [w.lower() for w in reps if w.lower() != current]
    stemmed_words = [stemmer.stem(w) for w in words]
    words += stemmed_words
    if clue:
        clue2 = [w.lower() for w in clue if w.lower() != current]
        stemmed_clue = [stemmer.stem(w) for w in clue2]
        stemmed_clue += clue2
        words += stemmed_clue
    for word in words:
        for i in range(len(sent_lower)):
            if i != sent_lower.index(current):
                if word and (word in stemmed_sent[i] or word in sent_lower[i]):
                    print("Wrong word: ", word)
                    return False
    return True


def remove_sents_with_articles(sent_path, max_no_one_country):
    """
    Remove resulting sentences containing ''a'' or ''an'' before country or ethnic or ethnics occuring to often
    :param sent_path: resulting masked sentences
    :param max_no_one_country: how often countries occured and were masked
    :return: None
    """
    with open(sent_path, "r") as sf:
        sf_json = json.load(sf)
        sentences = sf_json["X"]
        gt = sf_json["Y"]
        filtered_sent = []
        filtered_gt = []
        count = dict.fromkeys(gt)
    for i, sent in enumerate(sentences):
        if " a [MASK]" in sent or " an [MASK]" in sent:
            continue
        if count[gt[i]] is None or count[gt[i]] < max_no_one_country:
            filtered_sent.append(sent)
            filtered_gt.append(gt[i])
            count = Counter(filtered_gt)
            print(count)
    sf_json["X"] = filtered_sent
    sf_json["Y"] = filtered_gt
    print(sf_json)
    res_filename = sent_path.split(".")[0] + "_0-" + str(len(sentences)) + ".json"
    print(res_filename)
    with open(res_filename, 'w+') as fp:
            json.dump(sf_json, fp)


def main():
    with open("example_data/allCountries.txt", 'r', encoding='UTF8') as cf:
        countries = [c.strip() for c in cf.readlines()]
    with open("example_data/capitals.txt", 'r', encoding='UTF8') as capsf:
        capitals = [cap.strip() for cap in capsf.readlines()]
    mask_sentences("example_data/wiki_data/data_wiki_processed/", countries,
                   capitals, "example_data/masked_sentences/")


if __name__== "__main__":
    main()

