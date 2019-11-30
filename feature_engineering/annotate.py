from stanford_sentimentals import StanfordAnnotator
from lexical_features import LexicalAnnotator
import glob
import os.path
import pandas as pd

train_folder = "../data/datasets/train-articles"

def annotate(text):
    s_length, exms, qms = LexicalAnnotator.annotate(text)
    sa = StanfordAnnotator('http://localhost', 9000)

file_list = glob.glob(os.path.join(train_folder, "*.txt"))
articles_content, articles_id = ([], [])
for filename in file_list:
    with open(filename, "r", encoding="utf-8") as f:
        articles_content.append(f.read())
        articles_id.append(os.path.basename(filename).split(".")[0][7:])
articles = dict(zip(articles_id, articles_content))
data = pd.read_csv("../data/datasets/train-task1-SI.labels", header=None, sep='\t')
data['sentence_length'] = ''
data['qm_size'] = ''
data['em_size'] = ''
print(data)
for index, row in data.iterrows():
    article = articles.get(str(row[0]))
    propaganda = article[row[1]:row[2]]
    lex = LexicalAnnotator.annotate(propaganda)[]
    data.set_value(index, 'sentence_length', LexicalAnnotator.annotate(propaganda)[0])
print(data)

