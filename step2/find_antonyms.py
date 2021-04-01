import nltk
from nltk.corpus import wordnet as wn


def prepro_POS_syns(priming_file, res_path):
    """
    filter primes of step 1 for adjectives and create synonyms of those
    :param antonym_file_path: path to the resultfile
    :return: None
    """
    res = []
    with open(priming_file, "r") as f:
        primes = [s.strip() for s in f.readlines()]
    tagged = nltk.pos_tag(primes)
    for prime, tag in tagged:
        if tag[0] == "J":
            syns, ants = generate_synonyms(prime)
            if not ants:
                continue
            res.append(syns[0] + " " + ants[0] + "\n")
            for i, s in enumerate(syns):
                if i < len(ants):
                    pair_str = s + " " + ants[i] + "\n"
                if pair_str not in res:
                    res.append(pair_str)
    with open(res_path + "antonyme_adj.txt", "w+")as ant_res_file:
        ant_res_file.writelines(res)


def generate_synonyms(start_word):
    """
    find list of synonyms and antonyms using wordnet
    :param start_word: list to find synonyms for
    :return: list of synonyms for start word and list of antonyms
    """
    list_synonyms = []
    list_antonyms = []
    for syn in wn.synsets(start_word):
        for lemm in syn.lemmas():
            if not "_" in lemm.name() and not lemm.name() in list_synonyms:
                list_synonyms.append(lemm.name())
                if lemm.antonyms():
                    list_antonyms.append(lemm.antonyms()[0].name())
    return list_synonyms, list_antonyms


def main():
    prepro_POS_syns("example_data_2/primes.txt", "example_data_2/")

if __name__== "__main__":
    main()