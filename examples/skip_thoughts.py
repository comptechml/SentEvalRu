from __future__ import absolute_import, division, unicode_literals

import logging
import os
import sys
import zipfile

import skip_thought.tools as tools
from load_file_from_googledrive import download_file_from_google_drive


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
    embeddings = tools.encode(params['encoder'], batch, verbose=False, use_eos=True)
    return embeddings


# Set params for SentEval
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10, 'batch_size': 512}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                 'tenacity': 5, 'epoch_size': 4}
# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)


def check():
    if os.path.isdir(os.path.join('skip_thought', 'skip-thoughts-files')):
        do_download = \
            (not os.path.isfile(os.path.join('skip_thought', 'skip-thoughts-files', 'wiki_model.dat.npz'))) or \
            (not os.path.isfile(os.path.join('skip_thought', 'skip-thoughts-files', 'wiki_model.dat.npz.pkl'))) or \
            (not os.path.isfile(os.path.join('skip_thought', 'skip-thoughts-files', 'wiki_dict.dat'))) or \
            (not os.path.isfile(os.path.join('skip_thought', 'skip-thoughts-files',
                                             'all.norm-sz100-w10-cb0-it1-min100.w2v')))
    else:
        do_download = True
    if do_download:
        if os.path.isfile(os.path.join('skip_thought', 'skip-thoughts-files', 'wiki_model.dat.npz')):
            os.remove(os.path.join('skip_thought', 'skip-thoughts-files', 'wiki_model.dat.npz'))
        if os.path.isfile(os.path.join('skip_thought', 'skip-thoughts-files', 'wiki_model.dat.npz.pkl')):
            os.remove(os.path.join('skip_thought', 'skip-thoughts-files', 'wiki_model.dat.npz.pkl'))
        if os.path.isfile(os.path.join('skip_thought', 'skip-thoughts-files', 'wiki_dict.dat')):
            os.remove(os.path.join('skip_thought', 'skip-thoughts-files', 'wiki_dict.dat'))
        if os.path.isfile(os.path.join('skip_thought', 'skip-thoughts-files', 'all.norm-sz100-w10-cb0-it1-min100.w2v')):
            os.remove(os.path.join('skip_thought', 'skip-thoughts-files', 'all.norm-sz100-w10-cb0-it1-min100.w2v'))
        if os.path.isdir(os.path.join('skip_thought', 'skip-thoughts-files')):
            os.removedirs(os.path.join('skip_thought', 'skip-thoughts-files'))
        download_file_from_google_drive('1pIW0bwoo2gmTyFKnqLYxeXYF6w3ubg6S',
                                        os.path.join('skip_thought', 'skip-thoughts-files.zip'))
        with zipfile.ZipFile(os.path.join('skip_thought', 'skip-thoughts-files.zip')) as skipthoughts_zip:
            skipthoughts_zip.extractall('skip_thought')
        os.remove(os.path.join('skip_thought', 'skip-thoughts-files.zip'))
    params_senteval['encoder'] = tools.load_model(
        os.path.join('skip_thought', 'skip-thoughts-files', 'wiki_model.dat.npz'),
        os.path.join('skip_thought', 'skip-thoughts-files', 'wiki_dict.dat'),
        os.path.join('skip_thought', 'skip-thoughts-files', 'all.norm-sz100-w10-cb0-it1-min100.w2v')
    )
    se = senteval.engine.SE(params_senteval, batcher, prepare)
    transfer_tasks = ['SST2', 'SST3', 'MRPC', 'ReadabilityCl', 'TagCl', 'PoemsCl', 'TREC', 'STS', 'SICK']
    results = se.eval(transfer_tasks)
    return results


if __name__ == "__main__":
    check()
