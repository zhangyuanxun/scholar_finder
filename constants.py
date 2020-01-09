import os
from os.path import dirname

# Raw dataset path
RAW_NSF_GRANT_INFO = os.path.join(dirname(os.path.realpath(__file__)), 'dataset/raw/nsf_info.csv')
RAW_NSF_GRANT_INVESTIGATORS = os.path.join(dirname(os.path.realpath(__file__)), 'dataset/raw/nsf_investigators.csv')

# Collected dataset path
DATASET_PROCESSED_FOLDER = os.path.join(dirname(os.path.realpath(__file__)), 'dataset/processed/')
SCHOLAR_PROFILES = os.path.join(dirname(os.path.realpath(__file__)), 'dataset/processed/scholar_profiles.json')
SCHOLAR_PROFILES_INDEX = os.path.join(dirname(os.path.realpath(__file__)), 'dataset/processed/scholar_profiles_index.json')
SCHOLAR_AWARDS = os.path.join(dirname(os.path.realpath(__file__)), 'dataset/processed/scholar_awards.json')
TOP_SCHOLARS = os.path.join(dirname(os.path.realpath(__file__)), 'dataset/processed/top_scholars.txt')
NSF_ABSTRACT_BY_ID = os.path.join(dirname(os.path.realpath(__file__)), 'dataset/processed/nsf_abstract_by_id.json')

# Processed data
ELSEVIER_VOCAB = os.path.join(dirname(os.path.realpath(__file__)), 'dataset/processed/elsevier_vocab.txt')
VOCABULARY = os.path.join(dirname(os.path.realpath(__file__)), 'dataset/processed/vocabulary.txt')
GLOVE_50D = os.path.join(dirname(os.path.realpath(__file__)), 'dataset/processed/glove.6B.50d.txt')

# Dataset for training and testing
DATASET_BOW = os.path.join(dirname(os.path.realpath(__file__)), 'dataset/processed/dataset_bow.json')
DATASET_SEQ = os.path.join(dirname(os.path.realpath(__file__)), 'dataset/processed/dataset_seq.json')
DATASET_SEQ_SEP = os.path.join(dirname(os.path.realpath(__file__)), 'dataset/processed/dataset_seq_sep.json')

NSF_BOW = os.path.join(dirname(os.path.realpath(__file__)), 'dataset/processed/nsf_bow.json')

# Output folder
OUTPUT_FOLDER = os.path.join(dirname(os.path.realpath(__file__)), 'output/')


# Number definition
NUM_ABSTRACTS = 20
NUM_ABSTRACTS_LEAST = 5
