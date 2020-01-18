import pandas as pd
from BoW.BoW import get_word_map, document_to_vector2

data = pd.read_csv('annotated.csv')
with open('Bow/words.txt') as f:
    corpus = [line.rstrip() for line in f]

data['bow_sequence'] = ''
print(len(corpus))
word_map = get_word_map(corpus)
max_len = 0
for index, row in data.iterrows():
    v = document_to_vector2(row[1], word_map)
    data.at[index, 'bow_sequence'] = v
    if len(v) > max_len:
        max_len = len(v)

data.to_csv('test.csv')
print(max_len)
