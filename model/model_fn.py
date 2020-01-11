import numpy as np
import tensorflow as tf
import time
import sys
import matplotlib
import platform
import os
if platform.system() == "Darwin":
    matplotlib.use('TkAgg')
else:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def xavier_init(fan_in, fan_out, constant=1):
    """ Xavier initialization of network weights"""
    low = -constant*np.sqrt(6.0/(fan_in + fan_out))
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval=low, maxval=high,
                             dtype=tf.float32)


class Encoder(tf.keras.Model):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = tf.keras.layers.Dense(units=500, activation=tf.nn.relu)
        self.fc2 = tf.keras.layers.Dense(units=500, activation=tf.nn.relu)
        self.z_mean = tf.keras.layers.Dense(units=latent_dim, activation=None)
        self.z_log_var = tf.keras.layers.Dense(units=latent_dim, activation=None)

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        return z_mean, z_log_var


class Decoder(tf.keras.Model):
    def __init__(self, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = tf.keras.layers.Dense(units=500, activation=tf.nn.relu)
        self.fc2 = tf.keras.layers.Dense(units=500, activation=tf.nn.relu)
        self.x_hat = tf.keras.layers.Dense(units=output_dim, activation=tf.nn.sigmoid)

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x_hat = self.x_hat(x)
        return x_hat


class VAE(object):
    """Variation Autoencoder (VAE) implementation using TensorFlow.

    Reference "Auto-Encoding Variational Bayes" by Kingma and Welling for more details.
    """
    def __init__(self, input_dim, latent_dim, batch_size, num_epoch, num_sample, model_folder, load_model):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.num_epoch = num_epoch
        self.num_sample = num_sample
        self.model_folder = model_folder
        self.load_model = load_model
        self._build_model()

        if not os.path.exists(self.model_folder):
            os.mkdir(self.model_folder)

    def _build_model(self):
        self.encoder = Encoder(self.latent_dim)
        self.decoder = Decoder(self.input_dim)
        self.optimizer = tf.train.AdamOptimizer()

    def _compute_loss(self, X, X_hat, z_log_var, z_mean):
        # (1) reconstruction loss
        reconstruction_loss = -tf.reduce_sum(X * tf.log(tf.maximum(X_hat, 1e-10))
                                             + (1 - X) * tf.log(tf.maximum(1 - X_hat, 1e-10)), 1)
        # (2) KL loss
        kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), 1)

        # compute loss
        loss = tf.reduce_mean(reconstruction_loss + kl_loss)

        return loss

    def train_step(self, X, Y):
        with tf.GradientTape(persistent=True) as tape:
            # encoder
            z_mean, z_log_var = self.encoder(X)

            # sample noise
            epsilon = tf.random_normal(shape=tf.shape(z_mean), mean=0, stddev=1, dtype=tf.float32)

            # z = mu + sigma * epsilon to approximate from Gaussian distribution z ~ q(z | x)
            z = tf.add(z_mean, tf.multiply(tf.sqrt(tf.exp(z_log_var)), epsilon))

            # decoder
            X_hat = self.decoder(z)

            # compute loss
            loss = self._compute_loss(X, X_hat, z_log_var, z_mean)

        if self.load_model:
            print "Load weights..."
            self.encoder.load_weights(self.model_folder + "knowledge_encoder.h5")
            self.decoder.load_weights(self.model_folder + "knowledge_decoder.h5")
            self.load_model = False

        trainable_variables = self.encoder.trainable_variables + self.decoder.trainable_variables

        # apply gradient for label loss
        gradients_label = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(gradients_label, trainable_variables))

        return loss

    def train(self, dataset):
        print "Model training starts..."
        loss_plot = []
        num_steps = self.num_sample // self.batch_size

        for epoch in range(self.num_epoch):
            sys.stdout.flush()
            start = time.time()
            total_loss = 0
            for (batch, (X, Y, Z)) in enumerate(dataset):
                loss = self.train_step(X, Y)
                total_loss += loss

            epoch_loss = total_loss / num_steps
            if epoch != 0:
                loss_plot.append(epoch_loss)

            print ('Epoch-{}: loss {:.4f}.'.format(epoch + 1, epoch_loss.numpy()))
            print ('Time taken for the epoch {} sec\n'.format(time.time() - start))

        # plot loss
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
        plt.plot(loss_plot)
        plt.xlabel('Epochs', size=12)
        plt.ylabel('Loss', size=12)
        plt.title('Loss Plot')
        fig.tight_layout()
        plt.show(block=False)
        file_name = self.model_folder + 'loss.pdf'
        with PdfPages(file_name) as pdf:
            pdf.savefig(fig)
        plt.close()
        print "Model training ends."

    def save(self):
        self.encoder.save_weights(self.model_folder + "knowledge_encoder.h5")
        self.decoder.save_weights(self.model_folder + "knowledge_decoder.h5")
        print 'Saved trained model at %s ' % self.model_folder

    def reconstruct(self, dataset, vocabs, num_test=5):
        for (batch, (X, Y, Z)) in enumerate(dataset):

            # fake test
            for i in range(1):
                rid = np.random.randint(0, Y.shape[0])

                # encoder
                X_test = tf.expand_dims(X[rid], 0)
                z_mean, z_log_var = self.encoder(X_test)

                # sample noise
                epsilon = tf.random_normal(shape=tf.shape(z_mean), mean=0, stddev=1, dtype=tf.float32)

                # z = mu + sigma * epsilon to approximate from Gaussian distribution z ~ q(z | x)
                z = tf.add(z_mean, tf.multiply(tf.sqrt(tf.exp(z_log_var)), epsilon))

                # decoder
                X_hat = self.decoder(z)

            print "Load weights..."
            self.encoder.load_weights(self.model_folder + "knowledge_encoder.h5")
            self.decoder.load_weights(self.model_folder + "knowledge_decoder.h5")

            for i in range(num_test):
                rid = np.random.randint(0, Y.shape[0])

                # encoder
                X_test = tf.expand_dims(X[rid], 0)
                z_mean, z_log_var = self.encoder(X_test)

                # sample noise
                epsilon = tf.random_normal(shape=tf.shape(z_mean), mean=0, stddev=1, dtype=tf.float32)

                # z = mu + sigma * epsilon to approximate from Gaussian distribution z ~ q(z | x)
                z = tf.add(z_mean, tf.multiply(tf.sqrt(tf.exp(z_log_var)), epsilon))

                # decoder
                X_hat = self.decoder(z)

                fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))

                # first plot
                test_tops = np.argsort(X_test[0])[::-1][:20]

                test_words = []
                test_words_prob = []
                for ii in range(len(test_tops)):
                    test_words.append(vocabs[int(test_tops[ii])])
                    test_words_prob.append(X_test[0, int(test_tops[ii])])

                y_pos = np.arange(20)
                ax1.barh(y_pos, test_words_prob, align='center', linewidth=0.01,
                         color='green', ecolor='black')

                ax1.set_yticks(y_pos)
                ax1.set_yticklabels(test_words)
                ax1.invert_yaxis()  # labels read top-to-bottom
                ax1.set_xlabel('Normalized BOW')
                ax1.set_title("Input: " + Z[rid].numpy())

                # plot 2
                test_tops = np.argsort(X_hat[0])[::-1][:20]
                test_words = []
                test_words_prob = []
                for ii in range(len(test_tops)):
                    test_words.append(vocabs[int(test_tops[ii])])
                    test_words_prob.append(X_hat[0, int(test_tops[ii])])

                y_pos = np.arange(20)
                ax2.barh(y_pos, test_words_prob, align='center', linewidth=0.01,
                         color='green', ecolor='black')

                ax2.set_yticks(y_pos)
                ax2.set_yticklabels(test_words)
                ax2.invert_yaxis()  # labels read top-to-bottom
                ax2.set_xlabel('Normalized BOW')
                ax2.set_title("Reconstruction: " + Z[rid].numpy())

                plt.tight_layout()

                pdf_name = self.model_folder + "demo_reconstruct_bow_" + Z[rid].numpy() + ".pdf"
                with PdfPages(pdf_name) as pdf:
                    pdf.savefig(fig)
                print "Finish %d test" % (i + 1)
            break
