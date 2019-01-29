import sys
import numpy as np
import pandas as pd
import warnings
import csv
import fileinput

for i, line in enumerate(fileinput.input('msr_paraphrase_train.txt', inplace=1)):
    sys.stdout.write(line.replace('"', '')) 

data = pd.read_csv("msr_paraphrase_test.txt",delimiter='\t', header=None)
data.head()

from googletrans import Translator

ds=list(data)
m = len(ds)
n=data.size/m
n=int(n)
print(n,m)
arr = []
for j in range(m):
    arr.append([0] * n)

for i in range(m):
    translator = Translator()
    for j in range(n):
        arr[i][j]=(translator.translate(data[i][j], dest='ru').text)
       #print(translator.translate(data[i][j], dest='ru').text)

fls = pd.DataFrame(arr)
data=fls
data1=data.transpose()

data1.to_csv("smallRU.txt",index=False, header=False, sep='\t')