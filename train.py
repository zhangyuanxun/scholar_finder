from model.model_fn import *
from model.input_fn import load_knowledge_bow_data, load_task_bow_data
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


def train_knowledge_embedding(type, model_folder, batch_size=128, num_epoch=100, latent_dim=50,
                              test_size=0.0, load_model=False):

    # set TensorFlow as eager mode
    tf.enable_eager_execution()

    if type == 'scholar':
        print "Train scholar knowledge embedding"
        inputs_data = load_knowledge_bow_data(batch_size=batch_size, test_size=test_size)
    else:
        print "Train nsf knowledge embedding"
        inputs_data = load_task_bow_data(batch_size=batch_size, test_size=test_size)

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
    parser.add_argument("--type", type=str, help='Set type of knowledge embedding: scholar (scholar) or '
                                                 'nsf grant abstract(nsf)', choices=['scholar', 'nsf'])
    parser.add_argument("--batch_size", type=int, help='Set the batch size (default = %(default)s)', default=128)
    parser.add_argument("--num_epoch", type=int, help='Set the number of epoch (default = %(default)s)', default=100)
    parser.add_argument("--latent_dim", type=int, help='Set the number of latent dimension (default = %(default)s)',
                        default=50)
    parser.add_argument("--test_size", type=float, help='Set the ratio of data for testing (default = %(default)s)',
                        default=0.0)
    parser.add_argument("--load_model", type=str, help='Whether train a model from beginning or from existing model'
                                                        '(default = %(default)s)', choices=['yes', 'no'], default='no')
    parser.add_argument("--model_folder", type=str, help='Set the name of model folder to save model file')

    args = parser.parse_args()
    type = args.type

    type, batch_size, num_epoch, latent_dim, test_size, load_model, model_folder = args.type, \
                                                                                   args.batch_size, \
                                                                                   args.num_epoch, \
                                                                                   args.latent_dim, \
                                                                                   args.test_size, \
                                                                                   args.load_model, \
                                                                                   args.model_folder
    print "Input parameters: " + type, batch_size, num_epoch, latent_dim, test_size, load_model, model_folder
    if type is None:
        parser.error('type is not correct.')
        exit()

    if model_folder is None:
        parser.error('model folder is not correct.')
        exit()

    if load_model is "yes":
        load_model = True
    else:
        load_model = False

    train_knowledge_embedding(type, model_folder, batch_size, num_epoch, latent_dim, test_size, load_model)