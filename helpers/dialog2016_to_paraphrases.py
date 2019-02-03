import pandas as pd

outfile = open('paraphrases_label_0.csv', 'a') 

df1 = pd.read_csv("dialog2016_bank.csv", sep='\t', comment='#', usecols=[2])

df2 = pd.read_csv("dialog2016_tkk.csv", sep='\t', comment='#', usecols=[2])
frames = [df1, df2]

columns = ['sent1', 'sent2', 'label']
df_out = pd.DataFrame()

c = 0
for index, row in df1.iterrows():
    if c == 0:
        c += 1
        s1 = row['text']
    elif c == 1:
        c += 1
        s2 = row['text']
    elif c == 2:
        c = 0
        df = pd.DataFrame([[s1, s2, 0]], columns=columns)
        df_out.append(df)
        df.to_csv(outfile, sep='\t', index=False, encoding="cp1251", columns=columns, header=False)
