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
data['sequence_size'] = ''
data['qm_size'] = ''
data['em_size'] = ''
data['ner_person'] = ''
data['ner_organization'] = ''
data['ner_country'] = ''
for index, row in data.iterrows():
    article = articles.get(str(row[0]))
    propaganda = article[row[1]:row[2]]

    #LEXICAL ANNOTATION
    lexical = LexicalAnnotator.annotate(propaganda)
    data.at[index, 'sequence_size'] = "%.2f" % lexical[0]
    data.at[index, 'em_size'] =  lexical[1]
    data.at[index, 'qm_size'] = lexical[2]
    continue
    #NER ANNOTATION
    sa = StanfordAnnotator('http://localhost', 9000)
    ner = sa.annotate_ner(propaganda)
    data.at[index, 'ner_person'] = ner[0]

data.to_csv('annotated.csv',index=False)
