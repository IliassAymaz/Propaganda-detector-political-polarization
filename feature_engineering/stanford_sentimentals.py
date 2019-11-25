from stanfordcorenlp import StanfordCoreNLP
import json
import xml.etree.ElementTree as ET

def annotateSentiment(text):
    props={'annotators': 'sentiment','pipelineLanguage':'en','outputFormat':'xml'}
    result = nlp.annotate(text, properties=props)
    root = ET.fromstring(result)
    for sentence in root.iter('sentence'):
        print(sentence.attrib)

def annotateNamedEntities(text):
    props={'annotators': 'ner','pipelineLanguage':'en','outputFormat':'json'}
    result = nlp.annotate(text, properties=props)
    print(result)

nlp = StanfordCoreNLP('http://localhost', port=9000)
text = "Trump has bought Microsoft recently in Africa."
annotateNamedEntities(text)
