from annotators.stanford_annotator import StanfordAnnotator
from annotators.lexical_annotator import LexicalAnnotator
from annotators.polarity_annotator import PolarityAnnotator
from util.progress_bar import ProgressBar
import glob
import os.path
import pandas as pd
import time

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
data['sentiment'] = ''

start_time = time.time()
sa = StanfordAnnotator('http://localhost', 9000)
total = data.shape[0]
for index, row in data.iterrows():
    article = articles.get(str(row[0]))
    propaganda = article[row[1]:row[2]]

    #LEXICAL ANNOTATION
    lexical = LexicalAnnotator.annotate(propaganda)
    data.at[index, 'sequence_size'] = "%.2f" % lexical[0]
    data.at[index, 'em_size'] =  lexical[1]
    data.at[index, 'qm_size'] = lexical[2]

    #NER ANNOTATION
    ner = sa.annotate_ner(propaganda)
    data.at[index, 'ner_person'] = ', '.join(ner[0])
    data.at[index, 'ner_organization'] = ', '.join(ner[1])
    data.at[index, 'ner_country'] = ', '.join(ner[2])

    #SENTIMENT ANNOTATION
    sentiment = sa.annotate_sentiment(propaganda)
    data.at[index, 'sentiment'] = "%.2f" % sentiment

    #POLARITY ANNOTATION
    PolarityAnnotator.annotate(propaganda)

    ProgressBar.printProgressBar(index, total)

print("\n DONE! : " +str(time.time()-start_time))
data.to_csv('annotated.csv',index=False)
