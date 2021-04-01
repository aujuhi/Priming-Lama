import csv
import random
import pandas as pd
import spacy

ethnicities = ["French", "Vietnamese", "Irish", "Japanese", "Greeks",
                     "Italians", "English", "Somalis", "Turks",
                     "Germans", "Jews", "Arabs", "Finns", "Belarusians",
                     "Hungarians", "Assyrians", "Romanians", "Scots",
                     "Portuguese", "Tibetans"]

countries = ["India", "Korea", "Japan", "Argentina", "Chile", "Belgium", "Georgia", "France", "Norway",
                     "Iceland", "Peru", "Colombia", "Iran", "Panama", "Afghanistan", "Czech", "Bosnia", "EU", "Morocco",
                     "Philippines", "Bhutan"]

spacy_en = spacy.load('en_core_web_sm')

def create_val_prems(all_adj_prems_file, val_path, c_or_e="country"):
    all_prems, neg_prem_file, pos_prem_file= get_prem_filenames(all_adj_prems_file, val_path, c_or_e)

    prem_dic_neg, prem_dic_pos = get_prem_dicts(neg_prem_file, pos_prem_file, val_path)
    lines = []
    for key in prem_dic_pos.keys(): #countries:
        le = min(20, len(prem_dic_pos[key]) - 1)
        le_pos = min(20, len(prem_dic_pos[key]) - 1)
        le_neg = min(20, len(prem_dic_neg[key]) - 1)
        # create positive pairs
        for corr_prem in prem_dic_pos[key][:le]:
            rint = random.randint(0, le_neg)
            res_str = key + "|" + corr_prem + "|" + prem_dic_neg[key][rint] + "|1"
            lines.append(res_str)
        # create negative pairs
        for corr_prem in prem_dic_neg[key][:le]:
            rint = random.randint(0, le_pos)
            res_str = key + "|" + corr_prem + "|" + prem_dic_pos[key][rint] + "|2"
            lines.append(res_str)
        # create random pairs
        i = 0
        k = 0
        while i < 3 and k < 500:
            k += 1
            rint = random.randint(0, len(all_prems) - 1)
            prem1 = all_prems[rint]
            rint = random.randint(0, len(all_prems) - 1)
            prem2 = all_prems[rint]
            if prem1 == prem2 or prem1 in prem_dic_pos[key][:10] or prem1 in prem_dic_neg[key][:10]\
                    or prem2 in prem_dic_pos[key][:10] or prem2 in prem_dic_neg[key][:10]:
                continue
            if prem1 in prem_dic_pos[key] and prem2 in prem_dic_neg[key]:
                continue
            if prem1 in prem_dic_neg[key] and prem2 in prem_dic_pos[key]:
                continue
            res_str = key + "|" + prem1 + "|" + prem2 + "|0"
            lines.append(res_str)
            i += 1
    with open(val_path + "val_step1_" + c_or_e + "_pairs.txt", 'w+') as f_c:
        for line in lines:
            f_c.write(line + "\n")


def get_prem_dicts(neg_prem_file, pos_prem_file, val_path):
    # get top k positive and negative results
    prem_dic_pos = {}
    with open(val_path + pos_prem_file) as premf_pos:
        reader = csv.reader(premf_pos)
        for line in reader:
            if line[0] in countries or line[0] in ethnicities:
                # country or ethnic as key
                prem_dic_pos[line[0].strip()] = []
                for prem in line[1:]:

                    # only use adjective prems
                    tagged = spacy_en(prem)
                    for token in tagged:
                        if token.pos_ == "ADJ":
                            prem_dic_pos[line[0].strip()].append(prem)
    prem_dic_neg = {}
    with open(val_path + neg_prem_file) as premf_neg:
        reader = csv.reader(premf_neg)
        for line in reader:
            prem_dic_neg[line[0].strip()] = line[1:]
    return prem_dic_neg, prem_dic_pos


def get_prem_filenames(all_prems_file, val_path, c_or_e):
    # get all premises or adjectives used for lama probing
    with open(val_path + all_prems_file, 'r') as af:
        all_prems = [prem.strip() for prem in af.readlines()]
    # get subset of countries or ethnicities to evaluate
    if c_or_e == "ethnicity":
        pos_prem_file = "ordered_prems_pos_ethnicity.csv"
        neg_prem_file = "ordered_prems_neg_ethnicity.csv"
    if c_or_e == "country":
        pos_prem_file = "ordered_prems_pos_country.csv"
        neg_prem_file = "ordered_prems_neg_country.csv"
    return all_prems, neg_prem_file, pos_prem_file


def filter_prems_adj(val_path, prem_file):
    with open(val_path + prem_file, 'r') as pf:
        premisses = [p.strip() for p in pf.readlines()]
    for prem in premisses:
        if prem == "NoPrem" or prem == "hispanic":
            continue
        tagged = spacy_en(prem)
        for token in tagged:
            if token.pos_ == "ADJ":
                with open(val_path + "all_adj_prems.txt", "a+") as af:
                    af.write(prem + "\n")


def main():
    filter_prems_adj("setup/", "example_primes_country_step1.txt")
    create_val_prems("all_adj_prems.txt", "setup/")


if __name__== "__main__":
    main()
