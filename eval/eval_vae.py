import json
from tqdm import tqdm
from tensorflow import set_random_seed
import random
from numpy.random import seed
import sys
import os
from keras.models import Sequential, Model
from keras.layers import Reshape, Dropout, Input, concatenate, Dot, Dense
from sklearn.metrics import classification_report, accuracy_score

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model.model_fn import *
from constants import *


def convert_to_vector(bow, num_vocab):
    v = [0] * num_vocab

    for k in bow.keys():
        v[k] = bow[k]

    return np.array(v).reshape(1, num_vocab)


def prepare_train_test_data(scholar_model_path, scholar_latent_dim, nsf_model_path, nsf_latent_dim):
    with open(VOCABULARY, 'r') as fp:
        vocabs = fp.readlines()
    vocabs = [a.strip() for a in vocabs]

    with open(NSF_BOW, 'r') as fp:
        nsf_abstract = json.load(fp)

    nsf_abstract = {int(k): v for k, v in nsf_abstract.items()}

    with open(KNOWLEDGE_BOW, 'r') as fp:
        scholar_profiles = json.load(fp)

    scholar_names = scholar_profiles['names']
    scholar_abstracts = scholar_profiles['inputs']

    # perform a fake forward pass to compile model
    knowledge_model = Encoder(latent_dim=scholar_latent_dim)
    task_model = Encoder(latent_dim=nsf_latent_dim)

    X_fake = tf.zeros((len(vocabs)))
    X_fake = tf.expand_dims(X_fake, 0)
    knowledge_model(X_fake)
    task_model(X_fake)

    print "Load weights..."
    knowledge_model.load_weights(scholar_model_path + "knowledge_encoder.h5")
    task_model.load_weights(nsf_model_path + "task_encoder.h5")

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

            X = np.array(scholar_abstracts[i])
            X = tf.cast(X, tf.float32)
            X = tf.expand_dims(X, 0)
            z_mean, z_log_var = knowledge_model(X)
            scholar_embedding = tf.add(z_mean, tf.sqrt(tf.exp(z_log_var))).numpy().tolist()[0]

            # add negative sample
            negative_pool = list(set(nsf_abstract_ids) - set(award_ids))
            negative_samples = []
            for k in range(len(award_ids)):
                negative_samples.append(random.choice(negative_pool))

            idx = 0
            for _id in award_ids:
                if _id in train_ids:
                    X = np.array(nsf_abstract[_id])
                    X = tf.cast(X, tf.float32)
                    X = tf.expand_dims(X, 0)
                    z_mean, z_log_var = task_model(X)
                    task_embedding = tf.add(z_mean, tf.sqrt(tf.exp(z_log_var))).numpy().tolist()[0]

                    # add positive samples
                    vec = scholar_embedding + task_embedding
                    X_train.append(vec)
                    Y_train.append(1)

                    # add negative samples
                    X = np.array(nsf_abstract[negative_samples[idx]])
                    X = tf.cast(X, tf.float32)
                    X = tf.expand_dims(X, 0)
                    z_mean, z_log_var = task_model(X)
                    negative_task_embedding = tf.add(z_mean, tf.sqrt(tf.exp(z_log_var))).numpy().tolist()[0]

                    vec = scholar_embedding + negative_task_embedding
                    X_train.append(vec)
                    Y_train.append(0)
                elif _id in test_ids:
                    X = np.array(nsf_abstract[_id])
                    X = tf.cast(X, tf.float32)
                    X = tf.expand_dims(X, 0)
                    z_mean, z_log_var = task_model(X)
                    task_embedding = tf.add(z_mean, tf.sqrt(tf.exp(z_log_var))).numpy().tolist()[0]

                    # add positive samples
                    vec = scholar_embedding + task_embedding
                    X_test.append(vec)
                    Y_test.append(1)

                    # add negative samples
                    X = np.array(nsf_abstract[negative_samples[idx]])
                    X = tf.cast(X, tf.float32)
                    X = tf.expand_dims(X, 0)
                    z_mean, z_log_var = task_model(X)
                    negative_task_embedding = tf.add(z_mean, tf.sqrt(tf.exp(z_log_var))).numpy().tolist()[0]

                    vec = scholar_embedding + negative_task_embedding

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


def model_concatenate(X_train, Y_train, X_test, Y_test, scholar_latent_dim, nsf_latent_dim):
    set_random_seed(2)
    seed(1)

    latent_dim = 50

    print X_train.shape
    print Y_train.shape
    print X_test.shape
    print Y_test.shape

    X_train_schoar = X_train[:, 0: latent_dim]
    X_train_nsf = X_train[:, latent_dim:]
    X_test_scholar = X_test[:, 0: latent_dim]
    X_test_nsf = X_test[:, latent_dim:]

    # create model
    pub_embedding = Input(shape=[latent_dim])
    x = Dense(100, input_dim=latent_dim, kernel_initializer='random_normal', activation='relu')(pub_embedding)
    x = Dropout(0.1)(x)
    x = Dense(100, kernel_initializer='random_normal', activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(100, kernel_initializer='random_normal', activation='relu')(x)
    x = Dropout(0.1)(x)

    nsf_embedding = Input(shape=[latent_dim])
    y = Dense(100, input_dim=latent_dim, kernel_initializer='random_normal', activation='relu')(nsf_embedding)
    y = Dropout(0.1)(y)
    y = Dense(100, kernel_initializer='random_normal', activation='relu')(y)
    y = Dropout(0.1)(y)
    y = Dense(100, kernel_initializer='random_normal', activation='relu')(y)
    y = Dropout(0.1)(y)

    merged = concatenate([x, y])

    # interpretation
    dense3 = Dense(10, activation='relu')(merged)
    outputs = Dense(1, activation='sigmoid')(dense3)

    model = Model(inputs=[pub_embedding, nsf_embedding], outputs=outputs)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print model.summary()

    history = model.fit([X_train_schoar, X_train_nsf], Y_train, epochs=400, batch_size=100)

    ### evaluate with training example ###
    y_pred = model.predict([X_train_schoar, X_train_nsf], batch_size=100, verbose=1)
    y_pred_class = []

    for i in range(y_pred.shape[0]):
        if y_pred[i][0] > 0.5:
            y_pred_class.append(1.0)
        else:
            y_pred_class.append(0.0)

    # Train dataset evaluation
    print "Training dataset evaluation"
    print 'Accuracy: %.2f' % accuracy_score(Y_train, y_pred_class)
    print(classification_report(Y_train, y_pred_class))
    print

    ### evaluate with test example ###
    y_pred = model.predict([X_test_scholar, X_test_nsf], batch_size=100, verbose=0)
    y_pred_class = []

    for i in range(y_pred.shape[0]):
        if y_pred[i][0] > 0.5:
            y_pred_class.append(1.0)
        else:
            y_pred_class.append(0.0)

    # test dataset evaluation
    print "Testing dataset evaluation"
    print 'Accuracy: %.2f' % accuracy_score(Y_test, y_pred_class)
    print(classification_report(Y_test, y_pred_class))
    print
    print(history.history['loss'])


def model_dot():
    set_random_seed(2)
    seed(1)

    latent_dim = 50
    # load input data
    X_train = np.load("X_train.npy")
    X_test = np.load("X_test.npy")
    Y_train = np.load("Y_train.npy")
    Y_test = np.load("Y_test.npy")

    print X_train.shape
    print Y_train.shape
    print X_test.shape
    print Y_test.shape

    X_train_pub = X_train[:, 0: latent_dim]
    X_train_nsf = X_train[:, latent_dim:]
    X_test_pub = X_test[:, 0: latent_dim]
    X_test_nsf = X_test[:, latent_dim:]

    # create model
    pub_embedding = Input(shape=[latent_dim])
    x = Dense(100, input_dim=latent_dim, kernel_initializer='random_normal', activation='relu')(pub_embedding)
    x = Dropout(0.1)(x)
    x = Dense(100, kernel_initializer='random_normal', activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(100, kernel_initializer='random_normal', activation='relu')(x)
    x = Dropout(0.1)(x)

    nsf_embedding = Input(shape=[latent_dim])
    y = Dense(100, input_dim=latent_dim, kernel_initializer='random_normal', activation='relu')(nsf_embedding)
    y = Dropout(0.1)(y)
    y = Dense(100, kernel_initializer='random_normal', activation='relu')(y)
    y = Dropout(0.1)(y)
    y = Dense(100, kernel_initializer='random_normal', activation='relu')(y)
    y = Dropout(0.1)(y)

    dot_product = Dot(name="Dot-Product", axes=1, normalize=True)([x, y])

    model = Model(inputs=[pub_embedding, nsf_embedding], outputs=dot_product)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print model.summary()

    history = model.fit([X_train_pub, X_train_nsf], Y_train, epochs=200, batch_size=100)

    ### evaluate with training example ###
    y_pred = model.predict([X_train_pub, X_train_nsf], batch_size=100, verbose=1)
    y_pred_class = []

    for i in range(y_pred.shape[0]):
        if y_pred[i][0] > 0.5:
            y_pred_class.append(1.0)
        else:
            y_pred_class.append(0.0)

    # Train dataset evaluation
    print "Training dataset evaluation"
    print 'Accuracy: %.2f' % accuracy_score(Y_train, y_pred_class)
    print(classification_report(Y_train, y_pred_class))

    ### evaluate with test example ###
    y_pred = model.predict([X_test_pub, X_test_nsf], batch_size=100, verbose=0)
    y_pred_class = []

    for i in range(y_pred.shape[0]):
        if y_pred[i][0] > 0.5:
            y_pred_class.append(1.0)
        else:
            y_pred_class.append(0.0)

    # test dataset evaluation
    print "Testing dataset evaluation"
    print 'Accuracy: %.2f' % accuracy_score(Y_test, y_pred_class)
    print(classification_report(Y_test, y_pred_class))
    print

    print(history.history['loss'])


if __name__ == "__main__":
    import argparse

    # parse arguments
    parser = argparse.ArgumentParser(description="Description of visualizing scholar's knowledge embedding")
    parser.add_argument("--scholar_model", type=str,
                        help='Set the name of model folder to load scholars pre-trained model')
    parser.add_argument("--scholar_latent_dim", type=int,
                        help='Set the number of latent dimension of scholars pre-trained model')
    parser.add_argument("--nsf_model", type=str,
                        help='Set the name of model folder to load nsf grant pre-trained model')
    parser.add_argument("--nsf_latent_dim", type=int,
                        help='Set the number of latent dimension of nsf grant pre-trained model')

    args = parser.parse_args()
    scholar_model, nsf_model, scholar_latent_dim, nsf_latent_dim = args.scholar_model, \
                                                                   args.nsf_model, \
                                                                   args.scholar_latent_dim, \
                                                                   args.nsf_latent_dim

    print "Input parameters: " + scholar_model, nsf_model, scholar_latent_dim, nsf_latent_dim

    scholar_model_path = OUTPUT_FOLDER + scholar_model + '/'
    if scholar_model is None or not os.path.exists(scholar_model_path):
        parser.error('scholar model folder is not correct.')

    nsf_model_path = OUTPUT_FOLDER + nsf_model + '/'
    if nsf_model is None or not os.path.exists(nsf_model_path):
        parser.error('nsf model folder is not correct.')

    if nsf_model_path == scholar_model_path:
        parser.error('scholar or nsf model folder is not correct.')

    if scholar_latent_dim is None:
        parser.error('scholar latent dimension is not correct.')

    if nsf_latent_dim is None:
        parser.error('nsf latent dimension is not correct.')

    X_train, Y_train, X_test, Y_test = prepare_train_test_data(scholar_model_path, scholar_latent_dim,
                                                               nsf_model_path, nsf_latent_dim)

