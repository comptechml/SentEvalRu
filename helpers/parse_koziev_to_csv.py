import itertools
import os

path = "Data\koziev\\"
outfile = open('Data\koziev\\res\paraphrases.csv', 'w')


def parsefile():
    texts = []
    new_texts = []
    for filename in os.listdir(path):
        if filename.endswith(".txt"):
            filename = path + filename
            infile = open(filename, 'r')
            for line in infile:
                if line not in ['\n', '\r\n']:
                    texts.append(line.replace('\n', ''))
                else:
                    if len(texts) > 2:
                        new_texts = list(itertools.combinations(texts, 2))
                        for n in new_texts:
                            outfile.write("%s\t%s\t1\n" % (n[0].replace('\n', '\t'), n[1]))
                    elif len(texts) == 2:
                        if texts[0] and texts[1]:
                            outfile.write("%s\t%s\t1\n" % (texts[0], texts[1]))
                    texts[:] = []
                    new_texts[:] = []
            infile.close()


parsefile()
outfile.close()
