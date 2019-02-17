from __future__ import absolute_import, division, unicode_literals

import logging
import os
import sys
import zipfile

import skip_thought.tools as tools
from load_file_from_googledrive import download_file_from_google_drive


# Set PATHs
PATH_TO_SENTEVAL = os.path.join(os.path.dirname(__file__), '..')
PATH_TO_DATA = os.path.join(os.path.dirname(__file__), '..', 'data')
PATH_TO_SKIPTHOUGHT = os.path.join(os.path.dirname(__file__), 'skip_thought', 'skip-thoughts-files')


# import Senteval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval


def prepare(params, samples):
    return

def batcher(params, batch):
    if any(isinstance(lst, list) for lst in batch):
        batch = [str(" ".join(sent)) for sent in batch]
    embeddings = tools.encode(params['encoder'], batch, verbose=False, use_eos=True)
    return embeddings


# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)


def check():
    # Set params for SentEval
    params_senteval = {
        'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10, 'batch_size': 128,
        'classifier': {'nhid': 0, 'optim': 'adam', 'batch_size': 128, 'tenacity': 5, 'epoch_size': 4}
    }
    if os.path.isdir(PATH_TO_SKIPTHOUGHT):
        do_download = \
            (not os.path.isfile(os.path.join(PATH_TO_SKIPTHOUGHT, 'wiki_model.dat.npz'))) or \
            (not os.path.isfile(os.path.join(PATH_TO_SKIPTHOUGHT, 'wiki_model.dat.npz.pkl'))) or \
            (not os.path.isfile(os.path.join(PATH_TO_SKIPTHOUGHT, 'wiki_dict.dat'))) or \
            (not os.path.isfile(os.path.join(PATH_TO_SKIPTHOUGHT, 'all.norm-sz100-w10-cb0-it1-min100.w2v')))
    else:
        do_download = True
    if do_download:
        if os.path.isfile(os.path.join(PATH_TO_SKIPTHOUGHT, 'wiki_model.dat.npz')):
            os.remove(os.path.join(PATH_TO_SKIPTHOUGHT, 'wiki_model.dat.npz'))
        if os.path.isfile(os.path.join(PATH_TO_SKIPTHOUGHT, 'wiki_model.dat.npz.pkl')):
            os.remove(os.path.join(PATH_TO_SKIPTHOUGHT, 'wiki_model.dat.npz.pkl'))
        if os.path.isfile(os.path.join(PATH_TO_SKIPTHOUGHT, 'wiki_dict.dat')):
            os.remove(os.path.join(PATH_TO_SKIPTHOUGHT, 'wiki_dict.dat'))
        if os.path.isfile(os.path.join(PATH_TO_SKIPTHOUGHT, 'all.norm-sz100-w10-cb0-it1-min100.w2v')):
            os.remove(os.path.join(PATH_TO_SKIPTHOUGHT, 'all.norm-sz100-w10-cb0-it1-min100.w2v'))
        if os.path.isdir(PATH_TO_SKIPTHOUGHT):
            os.removedirs(PATH_TO_SKIPTHOUGHT)
        download_file_from_google_drive('1pIW0bwoo2gmTyFKnqLYxeXYF6w3ubg6S', PATH_TO_SKIPTHOUGHT + '.zip')
        with zipfile.ZipFile(os.path.join(PATH_TO_SKIPTHOUGHT + '.zip')) as skipthoughts_zip:
            skipthoughts_zip.extractall(os.path.join(os.path.dirname(__file__), 'skip_thought'))
        os.remove(os.path.join(PATH_TO_SKIPTHOUGHT + '.zip'))
    params_senteval['encoder'] = tools.load_model(
        os.path.join(PATH_TO_SKIPTHOUGHT, 'wiki_model.dat.npz'),
        os.path.join(PATH_TO_SKIPTHOUGHT, 'wiki_dict.dat'),
        os.path.join(PATH_TO_SKIPTHOUGHT, 'all.norm-sz100-w10-cb0-it1-min100.w2v')
    )
    se = senteval.engine.SE(params_senteval, batcher, prepare)
    transfer_tasks = ['SST2', 'SST3', 'MRPC', 'ReadabilityCl', 'TagCl', 'PoemsCl', 'ProzaCl', 'TREC', 'STS', 'SICK']
    results = se.eval(transfer_tasks)
    return results


if __name__ == "__main__":
    check()
