from __future__ import absolute_import, division, unicode_literals

import sys
import os
import logging
import json
import zipfile

from load_file_from_www import download_file_from_www


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

    input_file_name = os.path.join(os.path.dirname(__file__), 'bert_emb', 'data', 'file_in.txt')
    output_file_name = os.path.join(os.path.dirname(__file__), 'bert_emb', 'data', 'File_out.jsonl')
    try:
        file = open(input_file_name, "w", encoding="utf-8")
        for sent in batch:
            file.write('{0}\n'.format(' '.join(sent)))
        file.close()

        f = open(output_file_name, 'w')
        f.close()

        os.system(
            'python ./bert_emb/extract_features.py --input_file={0} --output_file={1} --vocab_file={2} '
            '--bert_config_file={3} --init_checkpoint={4} --layers=-1 --max_seq_length=128 --batch_size={5}'.format(
                input_file_name, output_file_name,
                os.path.join(PATH_TO_BERT, 'vocab.txt'),
                os.path.join(PATH_TO_BERT, 'bert_config.json'),
                os.path.join(PATH_TO_BERT, 'bert_model.ckpt'),
                params['batch_size']
            )
        )

        data = []
        f = open(output_file_name)
        for line in f:
            data.append(json.loads(line))

        for j in data:
            vec = []
            for k in j["features"]:
                emg = k["layers"][0]
                for i in emg["values"]:
                    vec.append(i)
            embeddings.append(vec)
    finally:
        if os.path.isfile(input_file_name):
            os.remove(input_file_name)
        if os.path.isfile(output_file_name):
            os.remove(output_file_name)
    
    return embeddings


# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)


def check():
    # Set params for SentEval
    params_senteval = {
        'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10, 'batch_size': 16,
        'classifier': {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128, 'tenacity': 3, 'epoch_size': 2},
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
    se = senteval.engine.SE(params_senteval, batcher, prepare)
    transfer_tasks = ['SST2', 'SST3', 'MRPC', 'ReadabilityCl', 'TagCl', 'PoemsCl', 'TREC', 'STS', 'SICK']
    results = se.eval(transfer_tasks)
    return results


if __name__ == "__main__":
    check()
