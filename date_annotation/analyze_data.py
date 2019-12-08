import pandas
import time
import random
import csv
from google_search import Search
import os
from os import listdir
from os.path import isfile, join
from difflib import SequenceMatcher


# Paths for the data to be analyzed
BASE_PATH = os.path.dirname(os.path.realpath(__file__))
ARTICLES_PATH = BASE_PATH + "/data/datasets/train-articles/"
ARTICLE_LABELS_PATH = ""

#Google search result title length is 70 characters
MAX_TITLE_LENGTH = 70

def similarity(title, queryTitle):
    title = title[0:MAX_TITLE_LENGTH]
    return SequenceMatcher(None, title, queryTitle).ratio()

def getBestMatch(article):
    similarities = []
    for x in [x for x in Search.query_google(article[1]) if not x[1] == '']:
        similarities.append((x[0], x[1], similarity(article[1], x[0])))
    if len(similarities) > 0:
        return max(similarities,key=lambda item:item[2])
    return ('', 'NULL', 0)
    

articleFiles = [f for f in listdir(ARTICLES_PATH) if isfile(join(ARTICLES_PATH, f))]
articles = []
for article in articleFiles:
    title = ""     
    with open(ARTICLES_PATH + article, 'r') as file:
        for line in file:
            if line in ['\r\n', '\n']:
                break
            else:
                title += line
    articles.append((article.split("article")[1].split(".")[0],title))

results = []
for i in range(365,len(articles)):
    article = articles[i]
    print("Analyzing " + str(i) + "/" + str(len(articles)) + " TITLE = " +article[1])
    bestMatch = getBestMatch(article)
    print(bestMatch)
    if bestMatch[2] > 0.2:
        results.append((article[0], bestMatch[1]))
    else: 
        results.append((article[0], 'NULL'))

with open(BASE_PATH + '/dates.csv','w') as out:
    csv_out=csv.writer(out)
    csv_out.writerow(['id','date'])
    for row in results:
        csv_out.writerow(row)
