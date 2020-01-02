# read annotated.csv and extract NER columns' content in lists
import pandas as pd
import re


data = pd.read_csv('../feature_engineering/annotated.csv')


def clean(l):
    """
    Removes nan and personal pronouns from l
    and returns a list of NER uniques.
    """
    non_nan = [re.sub(r'(?:^|, |s|S|)(?:h|H)(?:e|i).?(?:, |$)', '', str(item)) for item in l]
    non_nan = [item for item in non_nan
               if item != 'nan' and item != '']
    return list(set(non_nan))


persons = clean(list(data['ner_person']))
org = clean(list(data['ner_organization']))
countries = clean(list(data['ner_country']))


def token_to_bow(token, uniques):
    """
    Returns the BoW representation of a NER token.
    """
    vector = [-1]*len(uniques)
    for i in range(len(uniques)):
        if token != '' and token != 'nan' and uniques[i] in token:
            vector[i] = 1
    return vector


bow_persons = []
bow_orgs = []
bow_countries = []
for token in data['ner_person']:
    bow_persons.append(token_to_bow(str(token), persons))
for token in data['ner_organization']:
    bow_orgs.append(token_to_bow(str(token), org))
for token in data['ner_country']:
    bow_countries.append(token_to_bow(str(token), countries))

data['bow_ner_persons'] = bow_persons
data['bow_ner_organization'] = bow_orgs
data['bow_ner_country'] = bow_countries

data.to_csv('annotated_NERtoBoW.csv', sep=',')

