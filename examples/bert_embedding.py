from __future__ import absolute_import, division, unicode_literals

import sys
import os
import logging
import zipfile

import numpy as np
import tensorflow as tf

from load_file_from_www import download_file_from_www
import bert_emb.extract_features
import bert_emb.modeling
import bert_emb.tokenizationN


# Set PATHs
PATH_TO_SENTEVAL = os.path.join(os.path.dirname(__file__), '..')
PATH_TO_DATA = os.path.join(os.path.dirname(__file__), '..', 'data')
PATH_TO_BERT = os.path.join(os.path.dirname(__file__), 'bert_emb', 'data', 'multi_cased_L-12_H-768_A-12')

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval


# SentEval prepare and batcher
def prepare(params, samples):
    # Some code
    return


def batcher(params, batch):
    batch = [sent if sent != [] else ['.'] for sent in batch]
    embeddings = []

    features = bert_emb.extract_features.convert_examples_to_features(
        examples=batch, seq_length=params['bert']['max_seq_length'], tokenizer=params['bert']['tokenizer'])
    unique_id_to_feature = {}
    for feature in features:
        unique_id_to_feature[feature.unique_id] = feature
    input_fn = bert_emb.extract_features.input_fn_builder(
        features=features, seq_length=params['bert']['max_seq_length'])
    for result in params['bert']['estimator'].predict(input_fn, yield_single_examples=True):
        unique_id = int(result["unique_id"])
        feature = unique_id_to_feature[unique_id]
        list_of_word_embeddings = []
        for (i, token) in enumerate(feature.tokens):
            new_word_embedding = []
            for (j, layer_index) in enumerate(params['bert']['layer_indexes']):
                layer_output = result["layer_output_%d" % j]
                new_word_embedding += [round(float(x), 6) for x in layer_output[i:(i + 1)].flat]
            list_of_word_embeddings.append(new_word_embedding)
            del new_word_embedding
        embeddings.append(np.max(list_of_word_embeddings, axis=0))
        del list_of_word_embeddings
    
    return embeddings


def initialize_bert(batch_size):
    layer_indexes = [-1, -2, -3, -4]
    bert_config = bert_emb.modeling.BertConfig.from_json_file(os.path.join(PATH_TO_BERT, 'bert_config.json'))
    tokenizer = bert_emb.tokenizationN.FullTokenizer(
        vocab_file=os.path.join(PATH_TO_BERT, 'vocab.txt'),
        do_lower_case=True
    )
    run_config = tf.contrib.tpu.RunConfig(
        master=None,
        tpu_config=tf.contrib.tpu.TPUConfig(num_shards=1, per_host_input_for_training=False)
    )
    model_fn = bert_emb.extract_features.model_fn_builder(
        bert_config=bert_config,
        init_checkpoint=os.path.join(PATH_TO_BERT, 'bert_model.ckpt'),
        layer_indexes=layer_indexes,
        use_tpu=False,
        use_one_hot_embeddings=False
    )
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=False,
        model_fn=model_fn,
        config=run_config,
        predict_batch_size=batch_size)
    return {'tokenizer': tokenizer, 'estimator': estimator, 'max_seq_length': 128, 'layer_indexes': layer_indexes}


# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)


def check():
    # Set params for SentEval
    params_senteval = {
        'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10, 'batch_size': 256,
        'classifier': {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 256, 'tenacity': 3, 'epoch_size': 2},
    }
    if os.path.isdir(PATH_TO_BERT):
        do_download = (not os.path.isfile(os.path.join(PATH_TO_BERT, 'vocab.txt'))) or \
                      (not os.path.isfile(os.path.join(PATH_TO_BERT, 'bert_config.json'))) or \
                      (not os.path.isfile(os.path.join(PATH_TO_BERT, 'bert_model.ckpt'))) or \
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
        if os.path.isfile(os.path.join(PATH_TO_BERT, 'bert_model.ckpt')):
            os.remove(os.path.join(PATH_TO_BERT, 'bert_model.ckpt'))
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
            skipthoughts_zip.extractall(os.path.join(os.path.dirname(__file__), 'bert_emb', 'data'))
        os.remove(PATH_TO_BERT + '.zip')
    params_senteval['bert'] = initialize_bert(batch_size=16)
    se = senteval.engine.SE(params_senteval, batcher, prepare)
    transfer_tasks = ['SST2', 'SST3', 'MRPC', 'ReadabilityCl', 'TagCl', 'PoemsCl', 'TREC', 'STS', 'SICK']
    results = se.eval(transfer_tasks)
    return results


if __name__ == "__main__":
    check()
