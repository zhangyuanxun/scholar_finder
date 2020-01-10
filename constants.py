import os
from os.path import dirname

# Raw dataset path
RAW_NSF_GRANT_INFO = os.path.join(dirname(os.path.realpath(__file__)), 'dataset/raw/nsf_info.csv')
RAW_NSF_GRANT_INVESTIGATORS = os.path.join(dirname(os.path.realpath(__file__)), 'dataset/raw/nsf_investigators.csv')

# Collected dataset path
DATASET_PROCESSED_FOLDER = os.path.join(dirname(os.path.realpath(__file__)), 'dataset/processed/')
SCHOLAR_PUBS = os.path.join(dirname(os.path.realpath(__file__)), 'dataset/processed/scholar_pubs.json')
SCHOLAR_PUBS_INDEX = os.path.join(dirname(os.path.realpath(__file__)), 'dataset/processed/scholar_pubs_index.json')
SCHOLAR_GRANTS = os.path.join(dirname(os.path.realpath(__file__)), 'dataset/processed/scholar_grants.json')
TOP_SCHOLARS = os.path.join(dirname(os.path.realpath(__file__)), 'dataset/processed/top_scholars.txt')
NSF_ABSTRACT_BY_ID = os.path.join(dirname(os.path.realpath(__file__)), 'dataset/processed/nsf_abstract_by_id.json')

# Processed data
ELSEVIER_VOCAB = os.path.join(dirname(os.path.realpath(__file__)), 'dataset/processed/elsevier_vocab.txt')
VOCABULARY = os.path.join(dirname(os.path.realpath(__file__)), 'dataset/processed/vocabulary.txt')
GLOVE_50D = os.path.join(dirname(os.path.realpath(__file__)), 'dataset/processed/glove.6B.50d.txt')

# Dataset for training and testing
KNOWLEDGE_BOW = os.path.join(dirname(os.path.realpath(__file__)), 'dataset/processed/knowledge_bow.json')
NSF_BOW = os.path.join(dirname(os.path.realpath(__file__)), 'dataset/processed/nsf_bow.json')

# Output folder
OUTPUT_FOLDER = os.path.join(dirname(os.path.realpath(__file__)), 'output/')


# Number definition
NUM_ABSTRACTS = 20
NUM_ABSTRACTS_LEAST = 5