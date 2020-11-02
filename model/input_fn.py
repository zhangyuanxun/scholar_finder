from constants import *
import json
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from numpy import array
from numpy import argmax
from keras.utils import to_categorical
import random

np.set_printoptions(threshold=np.inf)

BUFFER_SIZE = 1000


def convert_to_vector(bow, num_vocab):
    v = [0] * num_vocab

    for k in bow.keys():
        v[k] = bow[k]

    return np.array(v).reshape(1, num_vocab)


def load_knowledge_bow_data(batch_size=64, test_size=0.2):
    with open(KNOWLEDGE_BOW, 'r') as fp:
        scholar_profiles = json.load(fp)

    with open(VOCABULARY, 'r') as fp:
        vocabs = fp.readlines()
    vocabs = [a.strip() for a in vocabs]

    inputs = scholar_profiles['inputs']
    labels = scholar_profiles['labels']
    names = scholar_profiles['names']

    # create scholar name index mapping
    names_index = [i for i in range(len(names))]

    # create vocabulary index mapping
    token_word_index = {u: i for i, u in enumerate(vocabs)}

    inputs = np.array(inputs)
    print inputs.shape
    labels = np.array(names_index).reshape((len(names_index), -1))

    inputs, labels, names = shuffle(inputs, labels, names, random_state=1)

    # Create training and test sets using an 80-20 split
    inputs_train, inputs_test, labels_train, labels_test, names_train, names_test = train_test_split(inputs, labels,
                                                                                                     names,
                                                                                                     test_size=test_size,
                                                                                                     random_state=0)

    def parse_fn(inputs, lables, names):
        inputs = tf.cast(inputs, tf.float32)
        lables = tf.cast(lables, tf.int32)

        return inputs, lables, names

    train_dataset = (tf.data.Dataset.from_tensor_slices((inputs_train, labels_train, names_train))
                     .map(map_func=parse_fn)
                     .batch(batch_size)
                     .filter(lambda inputs, labels, names: tf.equal(tf.shape(labels)[0], batch_size))
                     .prefetch(1)
                     )
    train_num_sample = len(inputs_train)

    test_dataset = (tf.data.Dataset.from_tensor_slices((inputs_test, labels_test, names_test))
                    .map(map_func=parse_fn)
                    .batch(batch_size)
                    .prefetch(1)
                    )
    test_num_sample = len(inputs_test)

    train_inputs = {'dataset': train_dataset,
                    'num_sample': train_num_sample}

    test_inputs = {'dataset': test_dataset,
                   'num_sample': test_num_sample,
                  }

    inputs_data = {'train': train_inputs,
                   'test': test_inputs,
                   'token_word_index': token_word_index,
                   'vocabs': vocabs,
                   'vocab_size': len(vocabs)}

    return inputs_data


def load_task_bow_data(batch_size=64, test_size=0.2):
    with open(NSF_BOW, 'r') as fp:
        nsf_bow = json.load(fp)

    inputs = []
    labels = []

    for k in nsf_bow.keys():
        inputs.append(nsf_bow[k])
        labels.append(int(k))

    with open(VOCABULARY, 'r') as fp:
        vocabs = fp.readlines()
    vocabs = [a.strip() for a in vocabs]

    # create vocabulary index mapping
    token_word_index = {u: i for i, u in enumerate(vocabs)}

    inputs, labels = shuffle(inputs, labels, random_state=1)

    # Create training and test sets using an 80-20 split
    inputs_train, inputs_test, labels_train, labels_test = train_test_split(inputs, labels,
                                                                            test_size=test_size,
                                                                            random_state=0)

    train_dataset = (tf.data.Dataset.from_tensor_slices((inputs_train, labels_train, labels_train))
                     .map(map_func=parse_fn)
                     .batch(batch_size)
                     .filter(lambda inputs, labels, names: tf.equal(tf.shape(labels)[0], batch_size))
                     .prefetch(1)
                     )
    train_num_sample = len(inputs_train)

    test_dataset = (tf.data.Dataset.from_tensor_slices((inputs_test, labels_test, labels_test))
                    .map(map_func=parse_fn)
                    .batch(batch_size)
                    .prefetch(1)
                    )
    test_num_sample = len(inputs_test)

    train_inputs = {'dataset': train_dataset,
                    'num_sample': train_num_sample}

    test_inputs = {'dataset': test_dataset,
                   'num_sample': test_num_sample,
                   }

    inputs_data = {'train': train_inputs,
                   'test': test_inputs,
                   'token_word_index': token_word_index,
                   'vocabs': vocabs,
                   'vocab_size': len(vocabs)}

    return inputs_data


def load_evaluation_train_test_data():
    with open(NSF_BOW, 'r') as fp:
        nsf_abstract = json.load(fp)

    nsf_abstract = {int(k): v for k, v in nsf_abstract.items()}

    with open(KNOWLEDGE_BOW, 'r') as fp:
        scholar_profiles = json.load(fp)

    with open(SCHOLAR_GRANTS, 'r') as fp:
        scholar_awards = json.load(fp)

    print(len(scholar_awards))
    print("processing scholar_awards ... ")
    for scholar in list(scholar_awards.keys()):
        name = scholar.split('@')[0]
        award_ids = scholar_awards[scholar]

        del scholar_awards[scholar]
        scholar_awards[name] = award_ids

    print(len(scholar_awards))

    with open(VOCABULARY, 'r') as fp:
        vocabs = fp.readlines()
    vocabs = [a.strip() for a in vocabs]

    scholar_names = scholar_profiles['names']
    scholar_abstracts = scholar_profiles['inputs']

    num_scholar = len(scholar_names)
    num_nsf = len(nsf_abstract)
    print("The number of scholars is %d, and the number of NSF grant is %d" % (num_scholar, num_nsf))

    # Get training nsf id and testing nsf id
    nsf_abstract_ids = list(nsf_abstract.keys())
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
                    # add positive samples
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
    print(X_train.shape)
    print(Y_train.shape)
    print(X_test.shape)
    print(Y_test.shape)

    return X_train, Y_train, X_test, Y_test, vocabs


def load_evaluation_train_test_data_tf():
    with open(NSF_BOW, 'r') as fp:
        nsf_abstract = json.load(fp)

    nsf_abstract = {int(k): v for k, v in nsf_abstract.items()}

    with open(KNOWLEDGE_BOW, 'r') as fp:
        scholar_profiles = json.load(fp)

    with open(SCHOLAR_GRANTS, 'r') as fp:
        scholar_awards = json.load(fp)

    print(len(scholar_awards))
    print("processing scholar_awards ... ")
    for scholar in list(scholar_awards.keys()):
        name = scholar.split('@')[0]
        award_ids = scholar_awards[scholar]

        del scholar_awards[scholar]
        scholar_awards[name] = award_ids

    print(len(scholar_awards))

    with open(VOCABULARY, 'r') as fp:
        vocabs = fp.readlines()
    vocabs = [a.strip() for a in vocabs]

    scholar_names = scholar_profiles['names']
    scholar_abstracts = scholar_profiles['inputs']

    num_scholar = len(scholar_names)
    num_nsf = len(nsf_abstract)
    print("The number of scholars is %d, and the number of NSF grant is %d" % (num_scholar, num_nsf))

    # Get training nsf id and testing nsf id
    nsf_abstract_ids = list(nsf_abstract.keys())
    train_ids = dict.fromkeys(nsf_abstract_ids[:int(num_nsf * 0.8)], 0)
    test_ids = dict.fromkeys(nsf_abstract_ids[int(num_nsf * 0.8):], 0)
    X_train_scholar = []
    X_train_task = []
    Y_train = []
    X_test_scholar = []
    X_test_task = []
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
                    X_train_scholar.append(scholar_abstracts[i])
                    X_train_task.append(nsf_abstract[_id])
                    Y_train.append(1)

                    # add negative samples
                    X_train_scholar.append(scholar_abstracts[i])
                    X_train_task.append(nsf_abstract[negative_samples[idx]])
                    Y_train.append(0)

                elif _id in test_ids:
                    # add positive samples
                    vec = scholar_abstracts[i] + nsf_abstract[_id]
                    X_test_scholar.append(scholar_abstracts[i])
                    X_test_task.append(nsf_abstract[_id])
                    Y_test.append(1)

                    # add negative samples
                    X_test_scholar.append(scholar_abstracts[i])
                    X_test_task.append(nsf_abstract[negative_samples[idx]])
                    Y_test.append(0)
                idx += 1

    X_train_scholar = np.array(X_train_scholar)
    X_train_task = np.array(X_train_task)
    X_test_scholar = np.array(X_test_scholar)
    X_test_task = np.array(X_test_task)
    Y_train = np.array(Y_train).astype('int32').reshape((-1, 1))
    Y_test = np.array(Y_test).astype('int32').reshape((-1, 1))
    print(X_train_scholar.shape)
    print(X_train_task.shape)
    print(X_test_scholar.shape)
    print(X_test_task.shape)
    print(Y_train.shape)
    print(Y_test.shape)
    batch_size = 128

    def parse_fn(scholars, tasks, labels):
        scholars = tf.cast(scholars, tf.float32)
        tasks = tf.cast(tasks, tf.float32)
        labels = tf.cast(labels, tf.float32)

        return scholars, tasks, labels

    train_dataset = (tf.data.Dataset.from_tensor_slices((X_train_scholar, X_train_task, Y_train))
                     .map(map_func=parse_fn)
                     .batch(batch_size)
                     .prefetch(1)
                     )

    test_dataset = (tf.data.Dataset.from_tensor_slices((X_test_scholar, X_test_task, Y_test))
                    .map(map_func=parse_fn)
                    .batch(batch_size)
                    .prefetch(1)
                    )

    return train_dataset, test_dataset, vocabs


if __name__ == "__main__":
    # set TensorFlow as eager mode
    tf.enable_eager_execution()
    load_evaluation_train_test_data_tf()
