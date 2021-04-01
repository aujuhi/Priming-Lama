import os
import re
import math
import json
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def load_priming(priming_path):
    """
    preprocess priming words and save to file
    :param priming_path: path to fox news primes
    :return: None
    """
    stop = stopwords.words("english")
    result = []
    re_file = open(priming_path + "priming_fox_comments.txt", "w+")
    with open(priming_path + "fox-news-comments.jsonl", 'r') as f:
        lines = f.readlines()
        for line in lines:
            dic = json.load(line)
            comment = dic["text"]
            line = re.sub('[^A-Za-z\\s]+', '', comment)
            for word in word_tokenize(line):
                word = word.lower()
                if len(word) > 2 \
                        and word not in stop \
                        and word not in result \
                        and "www" not in word \
                        and "http" not in word:
                    re_file.write(word + "\n")
                    result.append(word)
    re_file.close()


def preprocessing(doc, stem):
    """
    preprocess fox comments and articles
    :param doc: comment or article text
    :param stem: determiner if words should be stemmed
    :return: cleaned text
    """
    stopWords = set(stopwords.words("english"))
    doc = re.sub(r'[^\w\s]', '', doc.strip().lower())
    doc_str = ""
    for word in doc.split(" "):
        if word and word not in stopWords:
            if stem:
                doc_str += PorterStemmer().stem(word) + " "
            else:
                doc_str += word + " "
    return doc_str


def get_fox_comments_and_sort_words(wiki_path, priming_path):
    """
    tokenize fox comments and sort by tfidf
    :param wiki_path: path to wiki articles
    :param priming_path: path to fox comments
    :return: None
    """
    docs = []
    res = []
    foldernames = os.listdir(wiki_path)[:10]
    for folder_name in foldernames:
        folder_path = wiki_path + folder_name + "/"
        article_names = os.listdir(folder_path)[:10]
        for article_name in article_names:
            with open(folder_path + article_name, 'r', encoding="utf8") as af:
                try:
                    lines = af.read()
                except UnicodeDecodeError:
                    continue
            topr = preprocessing(lines, True)
            if topr:
                docs.append(topr)
    with open(priming_path + "fox-news-comments.jsonl", 'r') as f:
        fox_doc = ""
        lines = f.readlines()
        for line in lines:
            dic = json.load(line)
            comment = dic["text"]
            comment_pre = preprocessing(comment, False)
            if comment_pre:
                fox_doc += comment_pre
        docs.append(fox_doc)
        scores = {word: tfidf(word, fox_doc, docs) for word in fox_doc.split(" ") if word}
        sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        for word, score in sorted_words:
            if word.isalnum():
                res.append(word)
    with open(priming_path + "primes.txt", 'w+') as rf:
        for word in res:
            rf.write(word + '\n')


def tfidf(word, doc, docs):
    """
    calculates tf idf for a word in a fox-new comments in comparison to document frequency of wiki articles
    :param word: word from fox news comment
    :param doc: fox news comment
    :param docs: wikipedia articles
    :return:
    """
    word = PorterStemmer().stem(word)
    return tf(word, doc) * idf(word, docs)


def tf(word, doc):
    return math.log10(1 + sum([1 for w in doc.split(" ") if w == word]))  # /len(doc)


def idf(word, docs):
    no_doc_with_word = sum([1 for doc in docs if word in doc])
    return math.log10(1 + len(docs) / 1 + no_doc_with_word) + 1


def main():
    get_fox_comments_and_sort_words("example_data/wiki_data/data_wiki_processed/", "example_data/")


if __name__== "__main__":
    main()
