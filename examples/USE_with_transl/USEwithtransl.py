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
    import tensorflow as tf
    import tensorflow_hub as hub
    import re
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    from yandex.Translater import Translater
    tr=Translater()
    tr.set_key('trnsl.1.1.20190129T181632Z.6ad260c3f03e55a5.ae512973f3fa9c42fec01e4218fb4efd03a61a1b') # Api key found on https://translate.yandex.com/developers/keys
    tr.set_from_lang('ru')
    tr.set_to_lang('en')

    module_url = "https://tfhub.dev/google/universal-sentence-encoder/2"
    embed = hub.Module(module_url)

    l=len(batch)
    for i in range(l):
    tr.set_text(batch[i])
    sentence=tr.translate()
    batch[i]=sentence

    messages=batch    
    with tf.Session() as session:
          session.run([tf.global_variables_initializer(), tf.tables_initializer()])
          message_embeddings = session.run(embed(messages))
    embeddings=format(message_embeddings)
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