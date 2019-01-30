import sys
import numpy as np
import pandas as pd
import warnings
import csv
import fileinput

for i, line in enumerate(fileinput.input('test.txt', inplace=1)):
    sys.stdout.write(line.replace('"', ''))

data = pd.read_csv("test.txt",delimiter='\t', quotechar='"', header=None, names=["partition", "class", "text"])
#data.head()

ls=list(data["text"])

from googletrans import Translator

newls=[]

key_list = ls
for key in key_list:
    translator = Translator()
    newls.append(translator.translate(key, dest='ru').text)
    #print(translator.translate(key, dest='ru').text)

fls = pd.DataFrame(newls)
data[["text"]]=fls

data.to_csv("test_new.txt",index=False, header=False, sep='\t')
