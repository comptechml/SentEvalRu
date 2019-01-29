from __future__ import absolute_import, division, unicode_literals

import sys
import io
import numpy as np
import logging


# Set PATHs
PATH_TO_SENTEVAL = '../'
PATH_TO_DATA = '../data'

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval


# SentEval prepare and batcher
def prepare(params, samples):
    """ Сделать что-нибудь с обучающей выборкой, если нужно. Если нет - не писать ничего.
    Также эта функция может добавить что-нибудь в параметры, которые использует SentEval """
    # Some code
    return


def batcher(params, batch):
    """ Функция, которая по куску данных (batch) строит его векторное представление (embedding) """
    embeddings = []
    # Some code
    return embeddings


# Set params for SentEval
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                 'tenacity': 3, 'epoch_size': 2}

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    se = senteval.engine.SE(params_senteval, batcher, prepare)
    transfer_tasks = 'SST2'
    results = se.eval(transfer_tasks)
