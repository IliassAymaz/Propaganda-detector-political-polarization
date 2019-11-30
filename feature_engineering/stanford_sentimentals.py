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
        return total_sentiment/counter

    #We are only annotating PERSON, ORGANIZATION and COUNTRY here
    def annotate_ner(self, text):
        props={'annotators': 'ner','pipelineLanguage':'en','outputFormat':'json'}
        result = self.nlp.annotate(text, properties=props)
        ner = json.loads(result)
        persons = []
        organizations = []
        countries = []
        for sentence in ner['sentences']:
            for entity in sentence['entitymentions']:
                tag = entity['ner']
                if tag == 'PERSON':
                    persons.append(entity['text'])
                elif tag == 'COUNTRY':
                    countries.append(entity['text'])
                elif tag == 'ORGANIZATION':
                    organizations.append(entity['text'])
        return persons, organizations, countries