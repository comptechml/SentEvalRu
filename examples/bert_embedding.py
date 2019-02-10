from __future__ import absolute_import, division, unicode_literals

import codecs
import json
import sys
import os
import logging
import zipfile

import numpy as np
import tensorflow as tf

from keras_bert import load_trained_model_from_checkpoint

from load_file_from_www import download_file_from_www


# Set PATHs
PATH_TO_SENTEVAL = os.path.join(os.path.dirname(__file__), '..')
PATH_TO_DATA = os.path.join(os.path.dirname(__file__), '..', 'data')
PATH_TO_BERT = os.path.join(os.path.dirname(__file__), 'bert_data', 'multi_cased_L-12_H-768_A-12')

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval


# SentEval prepare and batcher
def prepare(params, samples):
    # Some code
    return


def batcher(params, batch):
    batch = [sent if sent != [] else ['.'] for sent in batch]

    token_input = []
    seg_input = []
    tokens_in_batch = []
    for sent in batch:
        tokens = list(filter(lambda it: it in params['bert']['dict'], sent))
        if len(tokens) > (params['bert']['max_seq_len'] - 2):
            tokens = tokens[:(params['bert']['max_seq_len'] - 2)]
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        token_input.append([params['bert']['dict'][token] for token in tokens] +
                           [0] * (params['bert']['max_seq_len'] - len(tokens)))
        seg_input.append([0] * len(tokens) + [0] * (params['bert']['max_seq_len'] - len(tokens)))
        tokens_in_batch.append(tokens)
    token_input = np.asarray(token_input)
    seg_input = np.asarray(seg_input)

    embeddings = np.zeros((len(batch), params['bert']['embedding_size']), dtype=np.float32)
    predicts = params['bert']['model'].predict([token_input, seg_input])
    for sent_idx in range(len(batch)):
        embeddings[sent_idx] = np.max(predicts[sent_idx][1:(len(tokens_in_batch[sent_idx]) - 1)], axis=0)

    return embeddings


# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)


def check():
    tf.logging.set_verbosity(tf.logging.ERROR)
    # Set params for SentEval
    params_senteval = {
        'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10, 'batch_size': 256,
        'classifier': {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 256, 'tenacity': 3, 'epoch_size': 2},
    }
    if os.path.isdir(PATH_TO_BERT):
        do_download = (not os.path.isfile(os.path.join(PATH_TO_BERT, 'vocab.txt'))) or \
                      (not os.path.isfile(os.path.join(PATH_TO_BERT, 'bert_config.json'))) or \
                      (not os.path.isfile(os.path.join(PATH_TO_BERT, 'bert_model.ckpt.index'))) or \
                      (not os.path.isfile(os.path.join(PATH_TO_BERT, 'bert_model.ckpt.meta'))) or \
                      (not os.path.isfile(os.path.join(PATH_TO_BERT, 'bert_model.ckpt.data-00000-of-00001')))
    else:
        do_download = True
    if do_download:
        if os.path.isfile(os.path.join(PATH_TO_BERT, 'vocab.txt')):
            os.remove(os.path.join(PATH_TO_BERT, 'vocab.txt'))
        if os.path.isfile(os.path.join(PATH_TO_BERT, 'bert_config.json')):
            os.remove(os.path.join(PATH_TO_BERT, 'bert_config.json'))
        if os.path.isfile(os.path.join(PATH_TO_BERT, 'bert_model.ckpt.index')):
            os.remove(os.path.join(PATH_TO_BERT, 'bert_model.ckpt.index'))
        if os.path.isfile(os.path.join(PATH_TO_BERT, 'bert_model.ckpt.meta')):
            os.remove(os.path.join(PATH_TO_BERT, 'bert_model.ckpt.meta'))
        if os.path.isfile(os.path.join(PATH_TO_BERT, 'bert_model.ckpt.data-00000-of-00001')):
            os.remove(os.path.join(PATH_TO_BERT, 'bert_model.ckpt.data-00000-of-00001'))
        if os.path.isdir(PATH_TO_BERT):
            os.removedirs(PATH_TO_BERT)
        download_file_from_www(
            'https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip',
            PATH_TO_BERT + '.zip'
        )
        with zipfile.ZipFile(PATH_TO_BERT + '.zip') as skipthoughts_zip:
            skipthoughts_zip.extractall(os.path.join(os.path.dirname(__file__), 'bert_data'))
        os.remove(PATH_TO_BERT + '.zip')
    token_dict = {}
    with codecs.open(os.path.join(PATH_TO_BERT, 'vocab.txt'), 'r', 'utf8') as vocab_reader:
        for line in vocab_reader:
            token = line.strip()
            token_dict[token] = len(token_dict)
    with codecs.open(os.path.join(PATH_TO_BERT, 'bert_config.json'), 'r', 'utf8') as config_reader:
        config_data = json.load(config_reader)
    params_senteval['bert'] = {
        'model': load_trained_model_from_checkpoint(
            config_file=os.path.join(PATH_TO_BERT, 'bert_config.json'),
            checkpoint_file=os.path.join(PATH_TO_BERT, 'bert_model.ckpt'),
            training=False, seq_len=None
        ),
        'dict': token_dict,
        'max_seq_len': config_data['max_position_embeddings'],
        'embedding_size': config_data['pooler_fc_size']
    }
    del config_data
    se = senteval.engine.SE(params_senteval, batcher, prepare)
    transfer_tasks = ['SST2', 'SST3', 'MRPC', 'ReadabilityCl', 'TagCl', 'PoemsCl', 'TREC', 'STS', 'SICK']
    results = se.eval(transfer_tasks)
    return results


if __name__ == "__main__":
    check()
