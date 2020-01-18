from transformers import BertTokenizer, BertForTokenClassification
from embedding_bert import get_bert_embedding

import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
import transformers

tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
bert_model = model_class.from_pretrained(pretrained_weights)

df = pd.read_csv('annotated.csv')
df = df.drop(columns=['sequence_size','Unnamed: 0','ner_country','ner_organization','ner_person','qm_size','em_size','sentiment','polarity','subjectivity','readability','loaded_language'])

df_train = df[:12000].reset_index(drop=True)
df_val = df[12000:13000].reset_index(drop=True)
df_test = df[13000:16666].reset_index(drop=True)

def toBertModel(df):
    c = 0
    bert_model = []
    for index,row in df.iterrows():
        sentence = row['sentence']
        span = row['span']
        bert_model.append(get_bert_embedding(model, tokenizer, sentence, span))
        c+=1
        print(c)
    return bert_model

