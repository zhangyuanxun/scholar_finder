import scholarly
from bs4 import BeautifulSoup
from constants import *
from datetime import datetime
import re
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np

BAD_SYMBOLS = '!"#$%&()*+.,-/:;=?@[\]^_`{|}~'


def print_timestamp():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ": "


def api_get_google_profile(name, institute):
    institute = '@' + institute
    search_query = scholarly.search_author(name)
    author = None
    scholar_profile = {}
    try:
        while True:
            author = next(search_query).fill()
            if institute == author.email:
                break
    except Exception:
        pass

    if author is not None and institute == author.email:
        interests = [i.lower().replace('-', ' ').replace('_', ' ') for i in author.interests]
        scholar_profile['interests'] = interests
        scholar_profile['institute'] = author.email
        abstracts = ""
        cnt = 0
        for pub in author.publications:
            try:
                pub = pub.fill()
                abstract = pub.bib['abstract'].text.lstrip().rstrip()
                abstracts += abstract
                cnt += 1
                if cnt == NUM_ABSTRACTS:
                    break
            except Exception:
                continue

        scholar_profile['abstracts'] = abstracts
        if cnt < NUM_ABSTRACTS_LEAST:
            return None
        else:
            return scholar_profile
    else:
        return None


def lowercase(text):
    """
    Lowercase the text

    Parameters
    ----------
    text : str

    Returns
    -------
    text: str
    """
    return text.lower()


def remove_extra_space(text):
    """
    Remove extra space between two words

    Parameters
    ----------
    text : str

    Returns
    -------
    text: str
    """
    return re.sub(' +', ' ', text)


def remove_url(text):
    """
    Remove urls in the string
    Parameters
    ----------
    text : str

    Returns
    -------
    text: str
    """
    return re.sub(r'http\S+', ' ', text)


def remove_non_ascii(text):
    """
    Remove non ascii characters in the text

    Parameters
    ----------
    text : str

    Returns
    -------
    text: str
    """
    return re.sub(r'[^\x00-\x7F]', ' ', text)


def remove_underline(text):
    return text.replace('_', ' ')


def remove_digits(text):
    """
    Remove all digits in the text

    Parameters
    ----------
    text : str

    Returns
    -------
    text: str
    """
    return re.sub(r'[\d]', '', text)


def remove_bad_symbols(text):
    for c in BAD_SYMBOLS:
        text = text.replace(c, ' ')
    return text


def remove_stopwords(tokens):
    """
    Remove stop words in the text
    Parameters
    ----------
    words : list

    Returns
    -------
    words: list
    """
    stop_words = set(stopwords.words('english'))

    tokens = [w for w in tokens if w not in stop_words]

    return tokens


def remove_punctuation(text):
    """
    Remove all punctuation in the text

    Parameters
    ----------
    text : str

    Returns
    -------
    text: str
    """
    return re.sub(r'[^\w]', ' ', text)


def lemmatizer(tokens):
    wl = WordNetLemmatizer()
    tokens = [wl.lemmatize(w) for w in tokens]
    return tokens


def stemmer(tokens):
    ps = PorterStemmer()
    tokens = [ps.stem(w) for w in tokens]
    return tokens


def remove_non_english(tokens, english_dict):
    tokens = [w for w in tokens if w in english_dict]
    return tokens


def get_work_embedding():
    word_vector = {}
    with open(GLOVE_50D, 'r') as f:
        for line in f:
            word, vector = line.split(" ", 1)
            vector = np.fromstring(vector, 'f', sep=' ')
            word_vector[word] = vector

    word_vector['<pad>'] = word_vector['pad']
    word_vector['<start>'] = word_vector['start']
    word_vector['<end>'] = word_vector['start']

    for k in word_vector.keys():
        if len(k) <= 2:
            del word_vector[k]

    # with open(VOCABULARY, 'r') as fp:
    #     vocabs = fp.readlines()
    # vocabs = [a.strip() for a in vocabs]
    #
    # for v in vocabs:
    #     if v not in word_vector:
    #         print v

    print "Total number of word is %d" % len(word_vector)
    return word_vector


if __name__ == "__main__":
    profile = api_get_google_profile('trupti joshi', 'missouri.edu')
    print profile
    print len(profile['abstracts'])
