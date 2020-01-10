from model.model_fn import *
from model.input_fn import load_knowledge_bow_data
from constants import *
import matplotlib
import platform

if platform.system() == "Darwin":
    matplotlib.use('TkAgg')
else:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import json
from tsne import tsne


def train_knowledge_embedding(batch_size=128, num_epoch=100, latent_dim=50,
                              test_size=0.0, load_model=False, model_folder='model_embedding'):
    # set TensorFlow as eager mode
    tf.enable_eager_execution()

    inputs_data = load_knowledge_bow_data(batch_size=batch_size, test_size=test_size)

    input_dim = inputs_data['vocab_size']
    num_sample = inputs_data['train']['num_sample']

    print "Total train sample is %d" % num_sample
    model_path = OUTPUT_FOLDER + model_folder + '/'
    vae_model = VAE(input_dim=input_dim, latent_dim=latent_dim, batch_size=batch_size,
                    num_epoch=num_epoch, num_sample=num_sample, model_folder=model_path, load_model=load_model)

    vae_model.train(inputs_data['train']['dataset'])

    # save model
    print "Saving model."
    vae_model.save()
    print "Finish training."


if __name__ == "__main__":
    import argparse

    # parse arguments
    parser = argparse.ArgumentParser(description="Description of training scholar finder Model")
    parser.add_argument("--batch_size", type=int, help='Set the batch size (default = %(default)s)', default=128)
    parser.add_argument("--num_epoch", type=int, help='Set the number of epoch (default = %(default)s)', default=100)
    parser.add_argument("--latent_dim", type=int, help='Set the number of latent dimension (default = %(default)s)',
                        default=50)
    parser.add_argument("--test_size", type=float, help='Set the ratio of data for testing (default = %(default)s)',
                        default=0.0)
    parser.add_argument("--load_model", type=bool, help='Whether train a model from beginning or from existing model'
                                                        ' (default = %(default)s)', default=False)
    parser.add_argument("--model_folder", type=str, help='Set the number of model folder to save model file '
                                                         '(default = %(default)s)', default='model_embedding')
    args = parser.parse_args()

    batch_size, num_epoch, latent_dim, test_size, load_model, model_folder = args.batch_size, \
                                                                             args.num_epoch, \
                                                                             args.latent_dim, \
                                                                             args.test_size, \
                                                                             args.load_model, \
                                                                             args.model_folder

    train_knowledge_embedding(batch_size, num_epoch, latent_dim, test_size, load_model, model_folder)