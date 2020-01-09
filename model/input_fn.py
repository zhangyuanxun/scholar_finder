from constants import *
import json
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from numpy import array
from numpy import argmax
from keras.utils import to_categorical

np.set_printoptions(threshold=np.inf)

BUFFER_SIZE = 1000


def convert_to_vector(bow, num_vocab):
    v = [0] * num_vocab

    for k in bow.keys():
        v[k] = bow[k]

    return np.array(v).reshape(1, num_vocab)


def parse_fn(inputs, lables, names):
    inputs = tf.cast(inputs, tf.float32)
    lables = tf.cast(lables, tf.int32)

    return inputs, lables, names


def load_knowledge_bow_data(batch_size=64, test_size=0.2):
    with open(DATASET_BOW, 'r') as fp:
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


if __name__ == "__main__":
    load_task_bow_data()
