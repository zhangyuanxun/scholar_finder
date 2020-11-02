from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Reshape, Dropout, Input, concatenate, Dot, Dense
from sklearn.metrics import classification_report, accuracy_score
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model.input_fn import *
from model.model_fn import *
from constants import *


def convert_to_vector(bow, num_vocab):
    v = [0] * num_vocab

    for k in bow.keys():
        v[k] = bow[k]

    return np.array(v).reshape(1, num_vocab)

#
# def model_dot():
#     set_random_seed(2)
#     seed(1)
#
#     latent_dim = 50
#     # load input data
#     X_train = np.load("X_train.npy")
#     X_test = np.load("X_test.npy")
#     Y_train = np.load("Y_train.npy")
#     Y_test = np.load("Y_test.npy")
#
#     print X_train.shape
#     print Y_train.shape
#     print X_test.shape
#     print Y_test.shape
#
#     X_train_pub = X_train[:, 0: latent_dim]
#     X_train_nsf = X_train[:, latent_dim:]
#     X_test_pub = X_test[:, 0: latent_dim]
#     X_test_nsf = X_test[:, latent_dim:]
#
#     # create model
#     pub_embedding = Input(shape=[latent_dim])
#     x = Dense(100, input_dim=latent_dim, kernel_initializer='random_normal', activation='relu')(pub_embedding)
#     x = Dropout(0.1)(x)
#     x = Dense(100, kernel_initializer='random_normal', activation='relu')(x)
#     x = Dropout(0.1)(x)
#     x = Dense(100, kernel_initializer='random_normal', activation='relu')(x)
#     x = Dropout(0.1)(x)
#
#     nsf_embedding = Input(shape=[latent_dim])
#     y = Dense(100, input_dim=latent_dim, kernel_initializer='random_normal', activation='relu')(nsf_embedding)
#     y = Dropout(0.1)(y)
#     y = Dense(100, kernel_initializer='random_normal', activation='relu')(y)
#     y = Dropout(0.1)(y)
#     y = Dense(100, kernel_initializer='random_normal', activation='relu')(y)
#     y = Dropout(0.1)(y)
#
#     dot_product = Dot(name="Dot-Product", axes=1, normalize=True)([x, y])
#
#     model = Model(inputs=[pub_embedding, nsf_embedding], outputs=dot_product)
#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#     print model.summary()
#
#     history = model.fit([X_train_pub, X_train_nsf], Y_train, epochs=200, batch_size=100)
#
#     ### evaluate with training example ###
#     y_pred = model.predict([X_train_pub, X_train_nsf], batch_size=100, verbose=1)
#     y_pred_class = []
#
#     for i in range(y_pred.shape[0]):
#         if y_pred[i][0] > 0.5:
#             y_pred_class.append(1.0)
#         else:
#             y_pred_class.append(0.0)
#
#     # Train dataset evaluation
#     print "Training dataset evaluation"
#     print 'Accuracy: %.2f' % accuracy_score(Y_train, y_pred_class)
#     print(classification_report(Y_train, y_pred_class))
#
#     ### evaluate with test example ###
#     y_pred = model.predict([X_test_pub, X_test_nsf], batch_size=100, verbose=0)
#     y_pred_class = []
#
#     for i in range(y_pred.shape[0]):
#         if y_pred[i][0] > 0.5:
#             y_pred_class.append(1.0)
#         else:
#             y_pred_class.append(0.0)
#
#     # test dataset evaluation
#     print "Testing dataset evaluation"
#     print 'Accuracy: %.2f' % accuracy_score(Y_test, y_pred_class)
#     print(classification_report(Y_test, y_pred_class))
#     print
#
#     print(history.history['loss'])


# def get_embedding(X_train, X_test, input_size, scholar_latent_dim, task_latent_dim):
#     # perform a fake forward pass to compile model, for the TF1.X limitation
#     knowledge_model = Encoder(latent_dim=scholar_latent_dim)
#     task_model = Encoder(latent_dim=task_latent_dim)
#
#     X_fake = tf.zeros(input_size)
#     X_fake = tf.expand_dims(X_fake, 0)
#     knowledge_model(X_fake)
#     task_model(X_fake)
#
#     print("Load weights...")
#     print(scholar_model_path)
#     print(task_model_path)
#     knowledge_model.load_weights(scholar_model_path + "knowledge_encoder.h5")
#     task_model.load_weights(task_model_path + "knowledge_encoder.h5")
#
#     X_train_scholar = X_train[:, 0: input_size]
#     X_train_task = X_train[:, input_size:]
#     X_test_scholar = X_test[:, 0: input_size]
#     X_test_task = X_test[:, input_size:]
#     print X_train_scholar.shape
#     print X_train_task.shape
#
#     # Get training embedding and testing embedding
#     X_train_embedding = []
#     X_test_embedding = []
#     sess = backend.get_session()
#
#     X_train_scholar = tf.cast(X_train_scholar, tf.float32)
#     X_train_scholar = tf.expand_dims(X_train_scholar, 0)
#
#     X_train_task = tf.cast(X_train_task, tf.float32)
#     X_train_task = tf.expand_dims(X_train_task, 0)
#
#     # get scholar embedding
#     z_mean, z_log_var = knowledge_model(X_train_scholar)
#     scholar_embedding = tf.add(z_mean, tf.sqrt(tf.exp(z_log_var)))
#     scholar_embedding = sess.run(scholar_embedding)
#     print scholar_embedding.shape
#
#     # get task embedding
#     z_mean, z_log_var = task_model(X_train_task)
#     task_embedding = tf.add(z_mean, tf.sqrt(tf.exp(z_log_var)))
#     task_embedding = sess.run(task_embedding).tolist()[0]
#
#     X_train_embedding.append(scholar_embedding + task_embedding)
#
#     for i in range(X_test.shape[0]):
#         scholar_vec = X_test[i][:input_size]
#         scholar_vec = tf.cast(scholar_vec, tf.float32)
#         scholar_vec = tf.expand_dims(scholar_vec, 0)
#
#         task_vec = X_test[i][input_size:]
#         task_vec = tf.cast(task_vec, tf.float32)
#         task_vec = tf.expand_dims(task_vec, 0)
#
#         # get scholar embedding
#         z_mean, z_log_var = knowledge_model(scholar_vec)
#         scholar_embedding = tf.add(z_mean, tf.sqrt(tf.exp(z_log_var)))
#         scholar_embedding = sess.run(scholar_embedding).tolist()[0]
#
#         # get task embedding
#         z_mean, z_log_var = task_model(task_vec)
#         task_embedding = tf.add(z_mean, tf.sqrt(tf.exp(z_log_var)))
#         task_embedding = sess.run(task_embedding).tolist()[0]
#
#         X_test_embedding.append(scholar_embedding + task_embedding)
#
#     return X_train_embedding, X_test_embedding


class MConcatenate(tf.keras.Model):
    def __init__(self, scholar_model_path, task_model_path, scholar_latent_dim, task_latent_dim,
                 input_size):
        super(MConcatenate, self).__init__()
        self.knowledge_embedding_model = Encoder(latent_dim=scholar_latent_dim)
        self.task_embedding_model = Encoder(latent_dim=task_latent_dim)

        # perform a fake forward pass to compile model, for the TF1.X limitation
        X_fake = tf.zeros(input_size)
        X_fake = tf.expand_dims(X_fake, 0)
        self.knowledge_embedding_model(X_fake)
        self.task_embedding_model(X_fake)

        print("Load weights pre-trained models...")
        self.knowledge_embedding_model.load_weights(scholar_model_path + "knowledge_encoder.h5")
        self.task_embedding_model.load_weights(task_model_path + "knowledge_encoder.h5")
        self.knowledge_embedding_model.trainable = False
        self.task_embedding_model.trainable = False

        # define scholar network
        self.scholar_fc1 = tf.keras.layers.Dense(100, activation='relu')
        self.scholar_dropout1 = tf.keras.layers.Dropout(0.1)
        self.scholar_fc2 = tf.keras.layers.Dense(100, activation='relu')
        self.scholar_dropout2 = tf.keras.layers.Dropout(0.1)
        self.scholar_fc3 = tf.keras.layers.Dense(100, activation='relu')
        self.scholar_dropout3 = tf.keras.layers.Dropout(0.1)

        # define task network
        self.task_fc1 = tf.keras.layers.Dense(100, activation='relu')
        self.task_dropout1 = tf.keras.layers.Dropout(0.1)
        self.task_fc2 = tf.keras.layers.Dense(100, activation='relu')
        self.task_dropout2 = tf.keras.layers.Dropout(0.1)
        self.task_fc3 = tf.keras.layers.Dense(100, activation='relu')
        self.task_dropout3 = tf.keras.layers.Dropout(0.1)

        # define concatenating dense
        self.concatenate_fc1 = tf.keras.layers.Dense(10, activation='relu')
        self.concatenate_fc2 = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, x_scholar, x_task, training=False):
        # get embedding for scholar
        z_mean, z_log_var = self.knowledge_embedding_model(x_scholar)
        x_scholar_embedding = tf.add(z_mean, tf.sqrt(tf.exp(z_log_var)))

        # get embedding for task
        z_mean, z_log_var = self.task_embedding_model(x_task)
        x_task_embedding = tf.add(z_mean, tf.sqrt(tf.exp(z_log_var)))

        # scholar network forward pass
        x_scholar = self.scholar_fc1(x_scholar_embedding)
        if training:
            x_scholar = self.scholar_dropout1(x_scholar)

        x_scholar = self.scholar_fc2(x_scholar)
        if training:
            x_scholar = self.scholar_dropout2(x_scholar)

        x_scholar = self.scholar_fc3(x_scholar)
        if training:
            x_scholar = self.scholar_dropout3(x_scholar)

        # task network forward pass
        x_task = self.task_fc1(x_task_embedding)
        if training:
            x_task = self.task_dropout1(x_task)

        x_task = self.task_fc2(x_task)
        if training:
            x_task = self.task_dropout2(x_task)

        x_task = self.task_fc3(x_task)
        if training:
            x_task = self.task_dropout3(x_task)

        # concatenating layer
        x_concatenating = tf.concat([x_scholar, x_task], axis=-1)
        x_concatenating = self.concatenate_fc1(x_concatenating)
        x_concatenating = self.concatenate_fc2(x_concatenating)
        return x_concatenating


def train_step(X_scholar, X_task, labels, model, loss_fn, optimizer):
    with tf.GradientTape(persistent=True) as tape:
        predictions = model(X_scholar, X_task, training=True)
        loss = loss_fn(labels, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss


def test_step(X_scholar, X_task, labels, model):
    predictions = model(X_scholar, X_task, training=False)
    return predictions


def train_and_eval_concatenate(train_dataset, test_dataset,
                               scholar_latent_dim, task_latent_dim,
                               input_size, scholar_model_path, task_model_path):
    num_epoch = 2
    model = MConcatenate(scholar_model_path, task_model_path, scholar_latent_dim, task_latent_dim, input_size)
    loss_fn = tf.keras.losses.binary_crossentropy
    optimizer = tf.train.AdamOptimizer()

    # training  stage
    for epoch in range(num_epoch):
        total_loss = 0
        sys.stdout.flush()
        start = time.time()
        for (batch, (X_scholar, X_task, Y)) in enumerate(train_dataset):
            loss = train_step(X_scholar, X_task, Y, model, loss_fn, optimizer)
            loss = loss.numpy()
            step_loss = np.sum(loss) / loss.shape[0]
            total_loss += step_loss

        print ('Epoch-{}: loss {:.4f}.'.format(epoch + 1, total_loss))
        print ('Time taken for the epoch {} sec\n'.format(time.time() - start))

    # evaluation stage
    y_true_labels = []
    y_pred_probs = []
    y_pred_labels = []
    for (batch, (X_scholar, X_task, Y)) in enumerate(test_dataset):
        y_preds = test_step(X_scholar, X_task, Y, model)
        y_preds = y_preds.numpy().flatten().tolist()
        y_pred_probs.extend(y_preds)

        y_trues = Y.numpy().flatten().tolist()
        y_true_labels.extend(y_trues)

    for i in range(len(y_pred_probs)):
        if y_pred_probs[i] > 0.5:
            y_pred_labels.append(1.0)
        else:
            y_pred_labels.append(0.0)

    # test dataset evaluation
    print "Testing dataset evaluation"
    print 'Accuracy: %.2f' % accuracy_score(y_true_labels, y_pred_labels)
    print(classification_report(y_true_labels, y_pred_labels))
    print


if __name__ == "__main__":
    import argparse
    # set TensorFlow as eager mode
    tf.enable_eager_execution()

    # parse arguments
    parser = argparse.ArgumentParser(description="Description of visualizing scholar's knowledge embedding")
    parser.add_argument("--scholar_model", type=str,
                        help='Set the name of model folder to load scholars pre-trained model')
    parser.add_argument("--scholar_latent_dim", type=int,
                        help='Set the number of latent dimension of scholars pre-trained model')
    parser.add_argument("--task_model", type=str,
                        help='Set the name of model folder to load task pre-trained model')
    parser.add_argument("--task_latent_dim", type=int,
                        help='Set the number of latent dimension of task grant pre-trained model')

    args = parser.parse_args()
    scholar_model, task_model, scholar_latent_dim, task_latent_dim = args.scholar_model, \
                                                                     args.task_model, \
                                                                     args.scholar_latent_dim, \
                                                                     args.task_latent_dim

    print "Input parameters: " + scholar_model, task_model, scholar_latent_dim, task_latent_dim

    scholar_model_path = OUTPUT_FOLDER + scholar_model + '/'
    if scholar_model is None or not os.path.exists(scholar_model_path):
        parser.error('scholar model folder is not correct.')

    task_model_path = OUTPUT_FOLDER + task_model + '/'
    if task_model is None or not os.path.exists(task_model_path):
        parser.error('task model folder is not correct.')

    if task_model_path == scholar_model_path:
        parser.error('scholar or task model folder is not correct.')

    if scholar_latent_dim is None:
        parser.error('scholar latent dimension is not correct.')

    if task_latent_dim is None:
        parser.error('task latent dimension is not correct.')

    # load input data
    print "Preparing training and testing dataset"
    train_dataset, test_dataset, vocabs = load_evaluation_train_test_data_tf()

    # train the model and evaluate the model
    train_and_eval_concatenate(train_dataset, test_dataset, scholar_latent_dim, task_latent_dim,
                               len(vocabs), scholar_model_path, task_model_path)


