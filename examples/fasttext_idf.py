from __future__ import absolute_import, division, unicode_literals

import sys
import io
import os
import numpy as np
import logging
from math import log2

from load_file_from_www import download_file_from_www


# Set PATHs
PATH_TO_SENTEVAL = os.path.join(os.path.dirname(__file__), '..')
PATH_TO_DATA = os.path.join(os.path.dirname(__file__), '..', 'data')
PATH_TO_VEC = os.path.join(os.path.dirname(__file__), 'fasttext', 'ft_native_300_ru_wiki_lenta_nltk_word_tokenize.vec')

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval


# Create dictionary
def create_dictionary(threshold=0):
    words = {}
    doc_number = 0
    input = open('idf.csv', 'r', encoding="utf-8")
    for s in input:
        doc_number += 1
        for word in s:
            words[word] = words.get(word, 0) + 1

    if threshold > 0:
        newwords = {}
        for word in words:
            if words[word] >= threshold:
                newwords[word] = words[word]
        words = newwords
    words['<s>'] = 1e9 + 4
    words['</s>'] = 1e9 + 3
    words['<p>'] = 1e9 + 2

    sorted_words = sorted(words.items(), key=lambda x: -x[1])  # inverse sort
    id2word = []
    word2id = {}
    for i, (w, _) in enumerate(sorted_words):
        id2word.append(w)
        word2id[w] = i

    return id2word, word2id, doc_number, words


# Get word vectors from vocabulary (glove, word2vec, fasttext ..)
def get_wordvec(path_to_vec, word2id):
    word_vec = {}

    with io.open(path_to_vec, 'r', encoding='utf-8') as f:
        # if word2vec or fasttext file : skip first line "next(f)"
        next(f)
        for line in f:
            word, vec = line.split(' ', 1)
            if word in word2id:
                word_vec[word] = np.fromstring(vec, sep=' ')

    logging.info('Found {0} words with word vectors, out of \
        {1} words'.format(len(word_vec), len(word2id)))
    return word_vec


# SentEval prepare and batcher
def prepare(params, samples):
    _, params.word2id, doc_count, word_count = create_dictionary()
    params.word_vec = get_wordvec(PATH_TO_VEC, params.word2id)
    params.wvec_dim = 300
    params.doc_count = doc_count
    params.word_count = word_count
    return


def batcher(params, batch):
    batch = [sent if sent != [] else ['.'] for sent in batch]
    embeddings = []

    for sent in batch:
        sentvec = []
        for word in sent:
            if word in params.word_vec:
                sentvec.append(params.word_vec[word] * log2(params.doc_count / params.word_count[word]))
        if not sentvec:
            vec = np.zeros(params.wvec_dim)
            sentvec.append(vec)
        sentvec = np.mean(sentvec, 0)
        embeddings.append(sentvec)

    embeddings = np.vstack(embeddings)
    return embeddings


# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)


def check():
    # Set params for SentEval
    params_senteval = {
        'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10,
        'classifier': {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128, 'tenacity': 3, 'epoch_size': 2}
    }
    if not os.path.isfile(os.path.join(os.path.dirname(__file__), 'fasttext',
                                       'ft_native_300_ru_wiki_lenta_nltk_word_tokenize.vec')):
        download_file_from_www(
            'http://files.deeppavlov.ai/embeddings/ft_native_300_ru_wiki_lenta_nltk_word_tokenize/'
            'ft_native_300_ru_wiki_lenta_nltk_word_tokenize.vec',
            os.path.join(os.path.dirname(__file__), 'fasttext', 'ft_native_300_ru_wiki_lenta_nltk_word_tokenize.vec')
        )
    if not os.path.isfile(os.path.join(os.path.dirname(__file__), 'fasttext',
                                       'ft_native_300_ru_wiki_lenta_nltk_word_tokenize.bin')):
        download_file_from_www(
            'http://files.deeppavlov.ai/embeddings/ft_native_300_ru_wiki_lenta_nltk_word_tokenize/'
            'ft_native_300_ru_wiki_lenta_nltk_word_tokenize.bin',
            os.path.join(os.path.dirname(__file__), 'fasttext', 'ft_native_300_ru_wiki_lenta_nltk_word_tokenize.bin')
        )
    se = senteval.engine.SE(params_senteval, batcher, prepare)
    transfer_tasks = ['SST2', 'SST3', 'MRPC', 'ReadabilityCl', 'TagCl', 'PoemsCl', 'ProzaCl', 'TREC', 'STS', 'SICK']
    results = se.eval(transfer_tasks)
    return results


if __name__ == "__main__":
    check()
