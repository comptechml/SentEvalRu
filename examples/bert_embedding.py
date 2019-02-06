from __future__ import absolute_import, division, unicode_literals

import sys
import os
import logging
import json


# Set PATHs
PATH_TO_SENTEVAL = '../'
PATH_TO_DATA = '../data'

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

    input_file_name = 'file_in.txt'
    output_file_name = 'File_out.jsonl'
    try:
        file = open(input_file_name, "w", encoding="utf-8")
        for sent in batch:
            file.write('{0}\n'.format(' '.join(sent)))
        file.close()

        f = open(output_file_name, 'w')
        f.close()

        os.system(
            './bert_emb/extract_features.py --input_file={0} --output_file={1} --vocab_file={2}/vocab.txt '
            '--bert_config_file={2}/bert_config.json --init_checkpoint={2}/bert_model.ckpt --layers=-1 '
            '--max_seq_length=128 --batch_size=8'.format(input_file_name, output_file_name, params['bert_dir'])
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
            print(vec)
            embeddings.append(vec)
    finally:
        if os.path.isfile(input_file_name):
            os.remove(input_file_name)
        if os.path.isfile(output_file_name):
            os.remove(output_file_name)
    
    return embeddings

# Set params for SentEval
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                 'tenacity': 3, 'epoch_size': 2}

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)


def check():
    params_senteval['bert_dir'] = os.path.join('bert_emb', 'data', 'multi_cased_L-12_H-768_A-12')
    se = senteval.engine.SE(params_senteval, batcher, prepare)
    transfer_tasks = ['SST2', 'SST3', 'MRPC', 'ReadabilityCl', 'TagCl', 'PoemsCl', 'TREC', 'STS', 'SICK']
    results = se.eval(transfer_tasks)
    return results


if __name__ == "__main__":
    se = senteval.engine.SE(params_senteval, batcher, prepare)
    transfer_tasks = 'SST2'
    results = se.eval(transfer_tasks)
