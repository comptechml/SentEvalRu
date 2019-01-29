import pandas as pd

path_in = "Data\\rubtsova\\"
path_out = "Data\\rubtsova\\res\\"

df = pd.read_csv(path_in + "positive.csv", sep=';', comment='#', usecols=[3, 4])
df.to_csv(path_out + "positives.csv", sep='\t', header=False)

df = pd.read_csv(path_in + "negative.csv", sep=';', comment='#', usecols=[3, 4])
df.to_csv(path_out + 'negatives.csv', sep='\t', header=False)
