import os
from nltk.tokenize.treebank import TreebankWordDetokenizer


twd = TreebankWordDetokenizer()

def tokenizeWiki(data_path):
    """
    detokenize and retrieve sentences from wiki corpus
    saves sentences from articles that meet prerequisites
    :param data_path: path to wiki data
    :return: None
    """
    filenames = os.listdir(data_path)
    print(filenames)
    endOfSent= ["?", "!", "."]
    try:
        os.mkdir(data_path + "/data_wiki_processed/")
    except FileExistsError:
        i = 1
    for fn in filenames:
        if not "englishEtiquetado" in fn:
            continue
        try:
            os.mkdir(data_path + "/data_wiki_processed/" + fn)
        except FileExistsError:
            print(fn, "Already processed")
            continue
        with open(data_path + fn, "r") as data:
            sentence = []
            article_no = 0
            w_in_art = 0

            # iterate through tokenized wiki articles
            while w_in_art < 1000000:
                w_in_art += 1
                try:
                    line = data.readline()
                except UnicodeDecodeError:
                    print("decode error")
                    sentence.append("+")
                    continue
                if line is None:
                    print("Line is none in file ", fn)
                    break
                l = line.strip().split(" ")
                word = l[0]
                # start of a new article
                if "<doc" in word and "title=" in l[2]:
                    article_no += 1
                    print("Article No ", article_no)
                if word:
                    sentence.append(word)

                # append detokenized sentence
                if word in endOfSent:
                    str_sent = twd.detokenize(sentence)

                    # only use sentence with no capitalized word or only one capitalized geographical term
                    named_entity = check_uppercase(sentence)
                    if not check_wiki(sentence) or not named_entity:
                        sentence = []
                        continue
                    print("Good Sent: ", str_sent)
                    with open(data_path + "data_wiki_processed/"
                              + fn + "/article" + str(article_no)
                              + named_entity + ".txt", 'a+') as fp:
                        fp.write(str_sent + "\n")
                    w_in_art = 0
                    sentence = []


def check_uppercase(sent):
    """
    checks if the sentences contain no or only one cased word
    :param sent: A detokenized sentence from a wikipedia article
    :return: True if the sentence contains not more than one cased word
    """
    upper_words = [(i, w) for i, w in enumerate(sent) if w and w[0].isupper()]
    if len(upper_words) < 3 and 0 in dict(upper_words).keys():
        if len(upper_words) == 2:
            print("Two upper")
            return "oneNE"
        print("One upper")
        return "noNE"
    print("Too many named entities")
    return False


def check_wiki(sent):
    """
    checks if the sentences meets the requirements
    :param sent: A detokenized sentence from a wikipedia article
    :return: True if the sentences contains max 5 words and no special characters
    """
    to_check = ["Appendix", "CITE", "ENDOFARTICLE", ">", "_",
                "<", ";", "</doc>", "<doc", "*", "=", "+",
                "com ", "www", "http", "#", "(", ")", "\"", "/", "springer", ":", "]", "["]
    if len(sent) < 5 or len(sent) >25:
        print("wrong length of sentence ", len(sent))
        return False
    for c in to_check:
        for word in sent:
            if c.lower() in word.lower() or not word:
                print("Wrong char sequence: ", word)
                return False
    return True


def main():
    tokenizeWiki("example_data/wiki_data/")

if __name__== "__main__":
    main()

