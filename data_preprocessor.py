import json
from tqdm import tqdm
import copy
from utils import *
import csv

LEN_ABSTRACT = 10000
LESS_FREQUENT_TH = 10
VOCABULARY_SIZE = 5000


def clean_text(text, english_dict, return_tokens=True):
    text = lowercase(text)
    text = remove_underline(text)
    text = remove_non_ascii(text)
    text = remove_digits(text)
    text = remove_punctuation(text)
    text = remove_extra_space(text)

    # get tokens
    tokens = text.split()
    tokens = remove_stopwords(tokens)
    tokens = remove_non_english(tokens, english_dict)

    if return_tokens:
        return tokens
    else:
        return ' '.join(w for w in tokens)


def generate_knowledge_bow():
    scholar_bow = {}
    scholar_names = []
    scholar_abstracts = []
    scholar_interests = []

    with open(SCHOLAR_PUBS_INDEX, 'r') as fp:
        scholar_profiles = json.load(fp)

    with open(VOCABULARY, 'r') as fp:
        vocabs = fp.readlines()
    vocabs = [a.strip() for a in vocabs]

    for scholar in scholar_profiles.keys():
        d = [0 for _ in range(len(vocabs))]

        abstracts = scholar_profiles[scholar]['abstracts']
        interests = scholar_profiles[scholar]['interests']

        for w in abstracts:
            d[w] += 1

        max_val = 1.0 * max(d)
        d = [round(d[i] / max_val, 3) for i in range(len(d))]

        scholar_abstracts.append(d)
        scholar_interests.append(interests)
        scholar_names.append(scholar)

    print len(scholar_interests)
    print len(scholar_abstracts)

    scholar_bow['inputs'] = scholar_abstracts
    scholar_bow['labels'] = scholar_interests
    scholar_bow['names'] = scholar_names

    print "Save dataset..."
    with open(KNOWLEDGE_BOW, 'w') as fp:
        json.dump(scholar_bow, fp)


def generate_nsf_bow():
    with open(NSF_ABSTRACT_BY_ID, 'r') as fp:
        nsf_abstract = json.load(fp)

    with open(VOCABULARY, 'r') as fp:
        vocabs = fp.readlines()
    vocabs = [a.strip() for a in vocabs]
    vocab_dic = dict.fromkeys(vocabs, 0)
    print len(vocabs)

    token_word_index = {u: i for i, u in enumerate(vocabs)}
    nsf_bow = {}
    cnt = 0
    for nsf in nsf_abstract.keys():
        d = [0 for _ in range(len(vocabs))]
        abstract = nsf_abstract[nsf]
        tokens = abstract.replace('-', ' ').replace('_', ' ').split()

        for t in tokens:
            if t in vocab_dic:
                d[token_word_index[t]] += 1

        try:
            max_val = 1.0 * max(d)
            d = [round(d[i] / max_val, 3) for i in range(len(d))]
            nsf_bow[nsf] = d
        except Exception:
            continue
        cnt += 1

        # collect number of NSF award
        if cnt == 20000:
            break

    print len(nsf_bow)

    # save file
    with open(NSF_BOW, 'w') as fp:
        json.dump(nsf_bow, fp)


if __name__ == "__main__":
    # generate knowledge bag-of-words
    generate_knowledge_bow()

    # generate nsf bag-of-words
    generate_nsf_bow()