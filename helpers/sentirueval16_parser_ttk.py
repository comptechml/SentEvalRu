# -*- coding: utf-8 -*-
from lxml import etree, html
import pandas as pd
import requests
import re


parser = etree.XMLParser(encoding='utf-8')
tree = etree.parse('tkk_test_etalon.xml')
etree.tostring(tree, encoding='utf-8')
root = tree.getroot()
texts = list(root.xpath("//column[@name='text']/text()"))
prep_texts = []
for text in texts:
    sentence = ''
    parsed_text = text.split()
    for word in parsed_text:
        del_http = re.search('http:.*', word)
        del_arob = re.search('@.*', word)
        del_https = re.search('https:.*', word)
        del_hash = re.search('#(.*)', word)
        if del_http or del_arob or del_https:
            continue
        elif del_hash:
            sentence += del_hash.group(1)+" "
        else:
            sentence += word+" "
    prep_texts.append(sentence[:-1])

meaning1 = list(root.xpath("//column[@name='beeline']/text()"))
meaning2 = list(root.xpath("//column[@name='mts']/text()"))
meaning3 = list(root.xpath("//column[@name='megafon']/text()"))
meaning4 = list(root.xpath("//column[@name='tele2']/text()"))
meaning5 = list(root.xpath("//column[@name='rostelecom']/text()"))
meaning6 = list(root.xpath("//column[@name='komstar']/text()"))
meaning7 = list(root.xpath("//column[@name='skylink']/text()"))
dct = {}
for sent_count, sentence in enumerate(prep_texts):
    if meaning1[sent_count] != 'NULL':
        dct[sentence] = meaning1[sent_count]
    elif meaning2[sent_count] != 'NULL':
        dct[sentence] = meaning2[sent_count]
    elif meaning3[sent_count] != 'NULL':
        dct[sentence] = meaning3[sent_count]
    elif meaning4[sent_count] != 'NULL':
        dct[sentence] = meaning4[sent_count]
    elif meaning5[sent_count] != 'NULL':
        dct[sentence] = meaning5[sent_count]
    elif meaning6[sent_count] != 'NULL':
        dct[sentence] = meaning6[sent_count]
    elif meaning7[sent_count] != 'NULL':
        dct[sentence] = meaning7[sent_count]
sentences = []
labels = []
for sentence in dct:
    sentences.append(sentence)
    labels.append(dct[sentence])
data_dict = {'text' : sentences, 'target' : labels}
df = pd.DataFrame(data_dict)
df.to_csv('dialog2016_etalon_tkk.csv', sep = '\t', encoding='utf-8')
