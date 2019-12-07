from stanfordcorenlp import StanfordCoreNLP
import json
import xml.etree.ElementTree as ET

class StanfordAnnotator:
    def __init__(self, hostadress, pn):
        self.hostadress = hostadress
        self.pn = pn
        self.nlp = StanfordCoreNLP(hostadress, port=pn)

    #Sentiment values are ranging from 0-4 (Very Negative, Negative, Neutral, Positive, Very Positive)
    def annotate_sentiment(self, text):
        props={'annotators': 'sentiment','pipelineLanguage':'en','outputFormat':'xml'}
        result = self.nlp.annotate(text, properties=props)
        root = ET.fromstring(result)
        total_sentiment = 0
        counter = 0
        for sentence in root.iter('sentence'):
            counter += 1
            total_sentiment += float(sentence.attrib.get('sentimentValue'))
        if counter == 0:
            # Because 2 corresponds to neutral setimental value
            return 2
        return total_sentiment/counter

    #We are only annotating PERSON, ORGANIZATION and COUNTRY here
    def annotate_ner(self, text):
        props={'annotators': 'ner','pipelineLanguage':'en','outputFormat':'json'}
        result = self.nlp.annotate(text, properties=props)
        ner = json.loads(result)
        persons = set()
        organizations = set()
        countries = set()
        for sentence in ner['sentences']:
            for entity in sentence['entitymentions']:
                tag = entity['ner']
                if tag == 'PERSON':
                    persons.add(entity['text'])
                elif tag == 'COUNTRY':
                    countries.add(entity['text'])
                elif tag == 'ORGANIZATION':
                    organizations.add(entity['text'])
        return persons, organizations, countries
