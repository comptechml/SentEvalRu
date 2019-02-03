import pandas as pd
import os
import re

def read_labels(path, label_id, label_target, dataset):
    # по двум меткам (id текста и желаемой эталонной меткой) возвращает словарь, где ключ - id текста, значение - метка
    # input: path : str, label_id : str, label_target: str, dataset: str
    # output: dict
    data = pd.read_csv(path, sep='\t')
    id_target = {}
    for num, line in enumerate(data[label_target]):
        if dataset == 'stihi':
            if line != 'без рубрики':
                id_target[data[label_id][num]] = line
        elif dataset == 'nplus' or dataset == 'proza':
            if pd.notna(line):
                id_target[data[label_id][num]] = line
        elif dataset == 'kp':
            string = re.search('(.*)>(.*)', line)
            if string:
                new_line = string.group(2)
            id_target[data[label_id][num]] = new_line
        elif dataset == 'fontanka':
            id_target['fontanka_'+str(data[label_id][num])] = line
        else:
            id_target[str(data[label_id][num])] = line
    return id_target

def create_csv(dct, path_to_dir, name_of_csv):
    # Создает csv файл из текстов, которые находятся в папке ath_to_dir, и эталонных меток из словаря dict
    # input: dct: dict, path_to_dir : str
    list_of_files = os.listdir(path_to_dir)
    texts = []
    targets = []
    i = 0
    for id in dct:
        i += 1
        path_to_text = id+'.txt'
        if path_to_text in list_of_files:
            if path_to_dir[-1] != '/':
                full_path = path_to_dir+'/'+path_to_text
            else:
                full_path = path_to_dir+path_to_text
            file = open(full_path, encoding='utf-8').readlines()
            texts.append(''.join(str(x) for x in file))
            targets.append(dct[id])
        if i % 1000 == 0:
            print(i, '/', len(dct))
    dict_for_dataframe = {'text': texts, 'target': targets}
    data = pd.DataFrame(data = dict_for_dataframe)
    data.to_csv(name_of_csv, sep='\t', encoding='utf-8')

#Example of using these functions

#stihi_id_target = read_labels('stihi_ru//newmetadata.csv', 'textid', 'textrubric', 'stihi')
#create_csv(stihi_id_target, 'stihi_ru/texts/', 'poems_genre.csv')