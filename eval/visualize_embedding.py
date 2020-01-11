import matplotlib
import os
import sys
import platform
if platform.system() == "Darwin":
    matplotlib.use('TkAgg')
else:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from model.model_fn import *
from model.input_fn import load_knowledge_bow_data
from constants import *


def visualize_embedding(model_path, num_scholar=15000):
    # set tensorflow as eager mode
    tf.enable_eager_execution()

    latent_dim = 2
    inputs_data = load_knowledge_bow_data(batch_size=num_scholar, test_size=0.0)
    dataset = inputs_data['train']['dataset']

    encoder_model = Encoder(latent_dim=latent_dim)

    for (batch, (X, Y, Z)) in enumerate(dataset):

        # fake test for compile model
        for i in range(1):
            rid = np.random.randint(0, Y.shape[0])

            # encoder
            X_test = tf.expand_dims(X[rid], 0)
            z_mean, z_log_var = encoder_model(X_test)

        print "Load weights..."
        encoder_model.load_weights(model_path + "knowledge_encoder.h5")

        # get real embedding
        z_mean, z_log_var = encoder_model(X)
        z = tf.add(z_mean, tf.sqrt(tf.exp(z_log_var))).numpy()

        # plot embedding
        fig, ax = plt.subplots(figsize=(10, 10))
        import matplotlib as mpl
        mpl.rcParams['axes.linewidth'] = 10  # set the value globally

        ax.scatter(z[:, 0], z[:, 1], c='black', s=2, edgecolors="black")
        ax.grid(False)
        fig.tight_layout()
        plt.show()

        pdf_name = model_path + "scholar_embedding.pdf"
        with PdfPages(pdf_name) as pdf:
            pdf.savefig(fig)


if __name__ == "__main__":
    import argparse

    # parse arguments
    parser = argparse.ArgumentParser(description="Description of visualizing scholar's knowledge embedding")
    parser.add_argument("--model_folder", type=str, help='Set the name of model folder to load pre-trained model')
    parser.add_argument("--num_scholar", help='Set number of scholar for visualization (default = %(default)s)',
                        type=int, default=10000, choices=range(1, 15001), metavar='(1, ..., 15000)')

    args = parser.parse_args()
    model_folder, num_scholar = args.model_folder, args.num_scholar
    print "Input parameters: " + model_folder, num_scholar

    model_path = OUTPUT_FOLDER + model_folder + '/'
    if model_folder is None or not os.path.exists(model_path):
        parser.error('model folder is not correct.')

    visualize_embedding(model_path, num_scholar)