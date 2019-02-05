from __future__ import absolute_import, division, unicode_literals

import io
import numpy as np
import logging
import sys
import skip_thought.tools as tools


# Set PATHs
PATH_TO_SENTEVAL = '../'
PATH_TO_DATA = '../data/'
PATH_TO_SKIPTHOUGHT = 'skip-thoughts-files'


# import Senteval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval


def prepare(params, samples):
    return

def batcher(params, batch):
    if any(isinstance(lst, list) for lst in batch):
        batch = [str(" ".join(sent)) for sent in batch]
    embeddings = tools.encode(model, batch,
                                     verbose=False, use_eos=True)
    return embeddings


# Set params for SentEval
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10, 'batch_size': 512}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                 'tenacity': 5, 'epoch_size': 4}
# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)


def check():
    se = senteval.engine.SE(params_senteval, batcher, prepare)
    # transfer_tasks = ['SST2', 'SST3', 'MRPC', 'ReadabilityCl', 'TagCl', 'PoemsCl', 'TREC', 'STS', 'SICK']
    transfer_tasks = 'SST2'
    results =  se.eval(transfer_tasks)
    return results


if __name__ == "__main__":
    model = tools.load_model('skip-thoughts-files/wiki_model.dat.npz', 'skip-thoughts-files/wiki_dict.dat', 'skip-thoughts-files/all.norm-sz100-w10-cb0-it1-min100.w2v')
    #params_senteval['encoder'] = model
    se = senteval.engine.SE(params_senteval, batcher, prepare)
    # transfer_tasks = ['SST2', 'SST3', 'MRPC', 'ReadabilityCl', 'TagCl', 'PoemsCl', 'TREC', 'STS', 'SICK']
    transfer_tasks = 'SST2'
    results = se.eval(transfer_tasks)
