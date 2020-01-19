import pandas as pd
from embedding.BoW.BoW import get_word_map, document_to_vector2

class BowTransformer():
    def __init__(self):
        with open('embedding/Bow/words.txt') as f:
            self.corpus = [line.rstrip() for line in f]
        self.word_map = get_word_map(self.corpus)

    def transform(self,sequence):
        return document_to_vector2(sequence, self.word_map)
'''
data = pd.read_csv('annotated.csv')

t = BowTransformer()
data['bow_sequence'] = ''
max_len = 0
for index, row in data.iterrows():
    v = t.transform(row[1])
    data.at[index, 'bow_sequence'] = v
    if len(v) > max_len:
        max_len = len(v)

data.to_csv('test.csv')
print(max_len)
'''
