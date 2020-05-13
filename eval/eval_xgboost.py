import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
from constants import *
import json
from tqdm import tqdm
from numpy.random import seed
from tensorflow import set_random_seed
import random

def convert_to_vector(bow, num_vocab):
    v = [0] * num_vocab

    for k in bow.keys():
        v[k] = bow[k]

    return np.array(v).reshape(1, num_vocab)


def load_input_data():
    with open(NSF_BOW, 'r') as fp:
        nsf_abstract = json.load(fp)

    nsf_abstract = {int(k): v for k, v in nsf_abstract.items()}

    with open(KNOWLEDGE_BOW, 'r') as fp:
        scholar_profiles = json.load(fp)

    with open(SCHOLAR_GRANTS, 'r') as fp:
        scholar_awards = json.load(fp)

    print len(scholar_awards)
    print "processing scholar_awards ... "
    for scholar in scholar_awards.keys():
        name = scholar.split('@')[0]
        award_ids = scholar_awards[scholar]

        del scholar_awards[scholar]
        scholar_awards[name] = award_ids

    print len(scholar_awards)

    with open(VOCABULARY, 'r') as fp:
        vocabs = fp.readlines()
    vocabs = [a.strip() for a in vocabs]

    scholar_names = scholar_profiles['names']
    scholar_abstracts = scholar_profiles['inputs']

    num_scholar = len(scholar_names)
    num_nsf = len(nsf_abstract)
    print "The number of scholars is %d, and the number of NSF grant is %d" % (num_scholar, num_nsf)

    # Get training nsf id and testing nsf id
    nsf_abstract_ids = nsf_abstract.keys()
    train_ids = dict.fromkeys(nsf_abstract_ids[:int(num_nsf * 0.8)], 0)
    test_ids = dict.fromkeys(nsf_abstract_ids[int(num_nsf * 0.8):], 0)
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []

    with tqdm(total=len(scholar_names), unit="sample", desc="Preparing dataset") as p_bar:
        for i in range(len(scholar_names)):
            p_bar.update(1)
            award_ids = scholar_awards[scholar_names[i]]

            # add negative sample
            negative_pool = list(set(nsf_abstract_ids) - set(award_ids))
            negative_samples = []
            for k in range(len(award_ids)):
                negative_samples.append(random.choice(negative_pool))

            idx = 0
            for _id in award_ids:
                if _id in train_ids:
                    # add postive samples
                    vec = scholar_abstracts[i] + nsf_abstract[_id]
                    X_train.append(vec)
                    Y_train.append(1)

                    # add negative samples
                    vec = scholar_abstracts[i] + nsf_abstract[negative_samples[idx]]
                    X_train.append(vec)
                    Y_train.append(0)
                elif _id in test_ids:
                    # add postive samples
                    vec = scholar_abstracts[i] + nsf_abstract[_id]
                    X_test.append(vec)
                    Y_test.append(1)

                    # add negative samples
                    vec = scholar_abstracts[i] + nsf_abstract[negative_samples[idx]]
                    X_test.append(vec)
                    Y_test.append(0)
                idx += 1

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    print X_train.shape
    print Y_train.shape
    print X_test.shape
    print Y_test.shape

    return X_train, Y_train, X_test, Y_test


if __name__ == "__main__":
    # set random seed
    set_random_seed(2)
    seed(1)

    # load input data
    print "Preparing training and testing dataset"
    X_train, Y_train, X_test, Y_test = load_input_data()

    print "Start to train XGBoost"
    Y_train = Y_train.ravel()
    Y_test = Y_test.ravel()

    print X_train.shape
    print Y_train.shape
    print X_test.shape
    print Y_test.shape

    # Train model
    model = XGBClassifier()
    model.fit(X_train, Y_train)

    # make prediction
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test, y_pred)

    print "Evaluate with test dataset"
    print('Accuracy: %.2f' % (accuracy * 100))
    print(classification_report(Y_test, y_pred))
