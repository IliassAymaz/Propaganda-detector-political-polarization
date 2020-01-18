from annotators.stanford import StanfordAnnotator
from annotators.lexical import LexicalAnnotator
from annotators.textblob import PolarityAnnotator, SubjectivityAnnotator
from annotators.ibm.ibm import IBMAnnotator
from annotators.readability import ReadabilityAnnotator
from annotators.loaded_language import LoadedLanguageAnnotator
import glob
import os
import pandas as pd
import time

data = pd.read_csv('train_context.csv')

#INITIALIZE TABLE
data['sequence_size'] = ''
data['qm_size'] = ''
data['em_size'] = ''
data['ner_person'] = ''
data['ner_organization'] = ''
data['ner_country'] = ''
data['sentiment'] = ''
data['polarity'] = ''
data['subjectivity'] = ''
data['readability'] = ''

start_time = time.time()
sa = StanfordAnnotator('http://localhost', 9000)
ibm = IBMAnnotator(0,0)
current_dir = os.path.dirname(__file__)
lla = LoadedLanguageAnnotator(os.path.join(current_dir, "data/loaded_language.txt"))
total = data.shape[0]
for index, row in data.iterrows():
    propaganda = row['span']

    if len(propaganda) == 0:
        continue

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

    #TEXTBLOB ANNOTATION
    polarity = PolarityAnnotator.annotate(propaganda)
    subjectivity = SubjectivityAnnotator.annotate(propaganda)
    data.at[index, 'polarity'] =  "%.2f" % polarity
    data.at[index, 'subjectivity'] = "%.2f" % subjectivity

    #READABILITY ANNOTATION
    readability = ReadabilityAnnotator.annotate(propaganda)
    data.at[index, 'readability'] = "%.2f" % readability

    #LOADED LANGUAGE ANNOTATION
    data.at[index, 'loaded_language'] = "%.2f" % lla.annotate(propaganda)

    '''
    #IBM ANNOTATION
    emotions = ibm.annotateEmotions(propaganda)
    data.at[index, 'sadness'] = emotions['sadness']
    data.at[index, 'joy'] = emotions['joy']
    data.at[index, 'fear'] = emotions['fear']
    data.at[index, 'disgust'] = emotions['disgust']
    data.at[index, 'anger'] = emotions['anger']
    '''

    #PROGRESS
    print("\r--%.2f%%--" % (100*(index/float(total))), end="\r")

print("\n DONE! : " +str(time.time()-start_time))
data.to_csv('annotated.csv',index=False)
