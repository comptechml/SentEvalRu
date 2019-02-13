# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

'''

Generic sentence evaluation scripts wrapper

'''
from __future__ import absolute_import, division, unicode_literals
import logging
import time

from senteval import utils
from senteval.sst import SSTEval
from senteval.trec import TRECEval
from senteval.sts import STSBenchmarkEval
from senteval.sick import SICKRelatednessEval
from senteval.mrpc import MRPCEval

class SE(object):
    def __init__(self, params, batcher, prepare=None):
        # parameters
        params = utils.dotdict(params)
        params.usepytorch = True if 'usepytorch' not in params else params.usepytorch
        params.seed = 1111 if 'seed' not in params else params.seed

        params.batch_size = 128 if 'batch_size' not in params else params.batch_size
        params.nhid = 0 if 'nhid' not in params else params.nhid
        params.kfold = 5 if 'kfold' not in params else params.kfold

        if 'classifier' not in params or not params['classifier']:
            params.classifier = {'nhid': 0}

        assert 'nhid' in params.classifier, 'Set number of hidden units in classifier config!!'

        self.params = params

        # batcher and prepare
        self.batcher = batcher
        self.prepare = prepare if prepare else lambda x, y: None

        self.list_tasks = ['SST2', 'SST3', 'MRPC', 'ReadabilityCl', 'TagCl', 'PoemsCl', 'ProzaCl', 'TREC', 'STS', 'SICK']

    def eval(self, name):
        # evaluate on evaluation [name], either takes string or list of strings
        if (isinstance(name, list)):
            self.results = {x: self.eval(x) for x in name}
            return self.results

        tpath = self.params.task_path
        assert name in self.list_tasks, str(name) + ' not in ' + str(self.list_tasks)

        if name == 'SST2':
            self.evaluation = SSTEval(tpath + '/SST/binary', 'SST binary', nclasses=2, seed=self.params.seed)
        elif name == 'SST3':
            self.evaluation = SSTEval(tpath + '/SST/dialog-2016', 'SST3', nclasses=3, seed=self.params.seed)
        elif name == 'ReadabilityCl':
            self.evaluation = SSTEval(tpath + '/Readability classifier', 'readability classifier', nclasses=10, seed=self.params.seed)
        elif name == 'TagCl':
            self.evaluation = SSTEval(tpath + '/Tags classifier', 'tag classifier', nclasses=6961, seed=self.params.seed)
        elif name == 'PoemsCl':
            self.evaluation = SSTEval(tpath + '/Poems classifier', 'poems classifier', nclasses=33, seed=self.params.seed)
        elif name == 'ProzaCl':
            self.evaluation = SSTEval(tpath + '/Proza classifier', 'proza classifier', nclasses=35, seed=self.params.seed)
        elif name == 'TREC':
            self.evaluation = TRECEval(tpath + '/TREC', seed=self.params.seed)
        elif name == 'STS':
            self.evaluation = STSBenchmarkEval(tpath + '/STS', seed=self.params.seed)
        elif name == 'SICK':
            self.evaluation = SICKRelatednessEval(tpath + '/SICK', seed=self.params.seed)
        elif name == 'MRPC':
            self.evaluation = MRPCEval(tpath + '/MRPC', seed=self.params.seed)

        self.params.current_task = name
        self.evaluation.do_prepare(self.params, self.prepare)

        start = time.time()
        self.results = self.evaluation.run(self.params, self.batcher)
        end = time.time()
        self.results["time"] = end - start
        logging.debug('\nTime for task : {0} sec\n'.format(self.results["time"]))

        return self.results
