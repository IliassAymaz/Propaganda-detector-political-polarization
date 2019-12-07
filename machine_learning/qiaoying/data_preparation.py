# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 13:59:55 2019

@author: 56366
"""
import os.path
import glob
import pandas as pd

train_folder = "./datasets/train-articles"
file_list = glob.glob(os.path.join(train_folder, "*.txt"))
articles_content, articles_id = ([], [])
for filename in file_list:
    with open(filename, "r", encoding="utf-8") as f:
        articles_content.append(f.readlines())
        articles_id.append(os.path.basename(filename).split(".")[0])
articles = dict(zip(articles_id, articles_content))

sentences, spans =  ([], [])
for article_id in articles_id:
    article_content = articles[article_id]
    sentences.extend(article_content)

    train_label_filename = article_id + ".task1-SI.labels"
    try:
        data = pd.read_csv("./datasets/train-labels-task1-span-identification/" + train_label_filename, header = None, sep = "\t")
        data = data.sort_values(data.columns[1], ascending = True)
        
        content_index = 0
        for one_sentence in article_content:
            add = 0
            for _, row in data.iterrows():
                if row[1] > content_index + len(one_sentence):
                    break
                elif row[1] >= content_index and row[2] < content_index + len(one_sentence):
                    spans.append(one_sentence[row[1]-content_index:row[2]-content_index])
                    add = 1
                    break
                else:
                    continue
            
            content_index = content_index + len(one_sentence)
            if add == 0:
                spans.append(None)
    except:
        spans.extend([""] * len(article_content))
        continue
ss_table = pd.DataFrame(list(zip(sentences, spans)), columns =["Sentence", "Span"])
ss_table = ss_table[ss_table["Sentence"] != "\n"]

ss_table.to_csv("train_table.csv", index = False, encoding = "utf-8")
