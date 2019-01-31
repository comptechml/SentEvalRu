from __future__ import absolute_import, division, unicode_literals

import io
import numpy as np
import logging
import sys
sys.setdefaultencoding('utf8')
import skip_thought.tools as tools


# Set PATHs
PATH_TO_SENTEVAL = '../'
PATH_TO_DATA = '../data/senteval_data/'
PATH_TO_SKIPTHOUGHT = 'skip-thoughts-files'


# import skipthought and Senteval
sys.path.insert(0, PATH_TO_SKIPTHOUGHT)
import skipthoughts
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval


def prepare(params, samples):
    if isinstance(samples, list):
        return params, samples
    else:
        raise TypeError('Samples should be list of str e.g. ["Хорошая погода.", "Интересный фильм."]')
    return

def batcher(params, batch):
    batch = [str(' '.join(sent), errors="ignore") if sent != [] else '.' for sent in batch]
    embeddings = skipthoughts.encode(params['encoder'], batch,
                                     verbose=False, use_eos=True)
    return embeddings


# Set params for SentEval
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10, 'batch_size': 512}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                 'tenacity': 5, 'epoch_size': 4}
# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    model = tools.load_model('skip-thoughts-files/wiki_model.dat.npz', 'skip-thoughts-files/wiki_dict.dat', 'skip-thoughts-files/all.norm-sz100-w10-cb0-it1-min100.w2v')
    params_senteval['encoder'] = skipthoughts.load_model()
    se = senteval.engine.SE(params_senteval, batcher, prepare)
    transfer_tasks = 'SST2'
    results = se.eval(transfer_tasks)