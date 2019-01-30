import sys
import numpy as np
import pandas as pd
import warnings
import csv
import fileinput

for i, line in enumerate(fileinput.input('test.txt', inplace=1)):
    sys.stdout.write(line.replace('"', ''))

data = pd.read_csv("test.txt",delimiter='\t', quotechar='"', header=None, names=["partition", "class", "text"])
data.head()

ls=list(data["text"])
m = len(ls)

from yandex.Translater import Translater

newls=[]

for i in range(m):
    tr = Translater()
    tr.set_key('trnsl.1.1.20190129T181632Z.6ad260c3f03e55a5.ae512973f3fa9c42fec01e4218fb4efd03a61a1b') # Api key found on https://translate.yandex.com/developers/keys
    tr.set_from_lang('en')
    tr.set_to_lang('ru')
    st=ls[i]
    str(st)
    tr.set_text(st)
    newls.append(tr.translate())
    #print(tr.translate())

fls = pd.DataFrame(newls)
data[["text"]]=fls

data.to_csv("test_new.txt",index=False, header=False, sep='\t')
