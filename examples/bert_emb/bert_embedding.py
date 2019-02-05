from __future__ import absolute_import, division, unicode_literals

import sys
import io
import os
import numpy as np
import logging
import json


# Set PATHs
PATH_TO_SENTEVAL = '../../'
PATH_TO_DATA = '../../data'

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval


# SentEval prepare and batcher
def prepare(params, samples):
    # Some code
    return


def batcher(params, batch):
#    batch = [sent if sent != [] else ['.'] for sent in batch]
    embeddings = []
   
    file = open("file_in.txt", "w", encoding="utf-8")
    for sent in batch:
        file.write(str(sent)+'\n')
    file.close() 
    
    f=open("File_out.jsonl", 'w')
    f.close()
    
    os.system('extract_features.py --input_file=file_in.txt --output_file=File_out.jsonl --vocab_file=../multi/vocab.txt --bert_config_file=../multi/bert_config.json --init_checkpoint=../multi/bert_model.ckpt --layers=-1 --max_seq_length=128 --batch_size=8')
    
    data = []
    f=open("File_out.jsonl")
    for line in f:
        data.append(json.loads(line))
            
    for j in data:
        vec=[]
        for k in j["features"]:
            emg=k["layers"][0]
            for i in emg["values"]:
                vec.append(i)
        print(vec)
        embeddings.append(vec)
    #    embeddings = np.vstack(embeddings)
    
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
