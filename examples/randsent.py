# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import absolute_import, division, unicode_literals

import argparse
import random
import os, sys
import io
import logging
from load_file_from_www import download_file_from_www

import torch
import torch.nn as nn

import numpy as np

import utils
from models import BOREP, ESN, RandLSTM

# Set PATHs
PATH_TO_SENTEVAL = os.path.join(os.path.dirname(__file__), '..')
PATH_TO_DATA = os.path.join(os.path.dirname(__file__), '..', 'data')
PATH_TO_VEC = os.path.join(os.path.dirname(__file__), 'fasttext', 'ft_native_300_ru_wiki_lenta_nltk_word_tokenize.vec')

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval


class params_embedding:
    # Model parameters
    model = 'lstm'  # Type of model to use (either borep, esn, or lstm, default borep
    task_type = 'downstream'  # Type of task to try (either downstream or probing, default downstream)
    n_folds = 10  # "Number of folds for cross-validation in SentEval (default 10)
    se_batch_size = 16  # Batch size for embedding sentences in SentEval (default 16)
    gpu = 1  # Whether to use GPU (default 0)
    senteval_path = "./SentEval"  # Path to SentEval (default ./SentEval).
    word_emb_file = "./glove.840B.300d.txt"  # Path to word embeddings file (default ./glove.840B.300d.txt)
    word_emb_dim = 300  # Dimension of word embeddings (default 300)

    # Network parameters
    input_dim = 300  # iutput feature dimensionality (default 300)
    output_dim = 2048  # "Output feature dimensionality (default 4096)
    max_seq_len = 96  # Sequence length (default 96)
    bidirectional = 1  # Whether to be bidirectional (default 1).
    init = 'none'  # Type of initialization to use (either none, orthogonal, sparse, normal, uniform, kaiming, "
    # "or xavier, default none).
    activation = None  # Activation function to apply to features  (default none)
    pooling = 'mean'  # Type of pooling (either min, max, mean, hier, or sum, default max)

    # Embedding parameters
    zero = 1  # whether to initialize word embeddings to zero (default 1)
    pos_enc = 0  # Whether to do positional encoding (default 0)
    pos_enc_concat = 0  # Whether to concat positional encoding to regular embedding (default 0)
    random_word_embeddings = 0  # Whether to load pretrained embeddings (default 0)

    # Projection parameters
    projection = 'same'  # Type of projection (either none or same, default same)

    # ESN parameters
    spectral_radius = 1  # Spectral radius for ESN (default 1.)
    leaky = 0  # Fraction of previous state to leak for ESN (default 0)
    concat_inp = 0  # Whether to concatenate input to hidden state for ESN (default 0)
    stdv = 1  # Width of uniform interval to sample weights for ESN (default 1)
    sparsity = 0  # Sparsity of recurrent weights for ESN (default 0)
    # LSTM parameters
    num_layers = 1 #Number of layers for random LSTM (default 1).", default=1)

# Create dictionary
def create_dictionary(sentences, threshold=0):
    words = {}
    for s in sentences:
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

    return id2word, word2id


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
    _, params.word2id = create_dictionary(samples)
    params.word_vec = get_wordvec(PATH_TO_VEC, params.word2id)
    params.wvec_dim = 300
    return


def batcher(params, batch):
    batch = [sent if sent != [] else ['.'] for sent in batch]
    embeddings = []

    for sent in batch:
        sentvec = []
        for word in sent:
            if word in params.word_vec:
                sentvec.append(params.word_vec[word])
        if not sentvec:
            vec = np.zeros(params.wvec_dim)
            sentvec.append(vec)
        sentvec = np.mean(sentvec, 0)
        embeddings.append(sentvec)

    embeddings = np.vstack(embeddings)
    return embeddings

def check():
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Set params for network
    params = params_embedding()

    if params.gpu:
        torch.cuda.manual_seed(seed)
    if params.model == 'lstm':
        network = RandLSTM(params)
    elif params.model == 'esn':
        network = ESN(params)
    else:
        network = BOREP(params)

    # Set params for SentEval
    params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10,
                       'classifier': {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128, 'tenacity': 3, 'epoch_size': 2}}

    se = senteval.engine.SE(params_senteval, batcher, prepare)
    transfer_tasks = ['SST2', 'SST3', 'MRPC', 'ReadabilityCl', 'TagCl', 'PoemsCl', 'ProzaCl', 'TREC', 'STS', 'SICK']
    results = se.eval(transfer_tasks)
    return results

