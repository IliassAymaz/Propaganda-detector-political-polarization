import time
# A tokenizer tool that will transform our
# documents to a set of list of words
from nltk.tokenize import RegexpTokenizer

# English stop words
from stop_words import get_stop_words

# Stemmer
from nltk.stem.porter import PorterStemmer

# Word ids appender
from gensim import corpora, models
import gensim
import os
import re
import pandas as pd
import numpy as np


def preprocess(articles):
    """The preprocessing stage.
     Articles are tokenized, stripped of stop words
     and stemmed.
     """
    stemmed_articles = []
    for article in articles:
        # Tokenization
        tokenizer = RegexpTokenizer(r'(?<![\w\'\’])\w+?(?=\b|n\'t)')

        result = tokenizer.tokenize(article.lower())

        # Get the English words
        stop_words = get_stop_words('en')

        # Filter them out from the document
        filterd_doc = [item for item in result if not item in stop_words]

        # Stemming
        p_stemmer = PorterStemmer()
        stemmed_article = [p_stemmer.stem(word) for word in filterd_doc]

        stemmed_articles.append(stemmed_article)

        # Document-term matrix
    dictionary = corpora.Dictionary(stemmed_articles)

    # into bag of words
    bg_words = [dictionary.doc2bow(word) for word in stemmed_articles]
    return bg_words, dictionary


def lda(bg_words, dictionary, num_topics, num_words, rootdir):
    """
    Generates the LDA model.
    Takes the dictionary and the bag of words
    from the preprocessing stage.
    """
    start = time.time()
    lda_model = gensim.models.ldamulticore.LdaMulticore(bg_words, num_topics=num_topics, id2word=dictionary, passes=20)
    print('LDA Analysis result for %d topics and %d words :' % (num_topics, num_words))
    # for item in lda_model.print_topics():  # num_topics=num_topics, num_words=num_words):
    for item in lda_model.show_topics(num_topics=num_topics, num_words=num_words, formatted=False):
        print(item)
    end = time.time()
    for subdir, dirs, files in os.walk(rootdir):
        n = len(files)
    print('Executed in ' + str(round(end-start, 2)) + ' seconds on ', len([iq for iq in os.scandir(rootdir)]), 'files')
    return lda_model, bg_words


def identify_docs_by_topic(lda_model, bg_words, article_names):
    # each document is defined by its bag of words
    # we use this to get the classification result on some document
    if input('Show documents by topic? [y/n]') == 'y':
        for i in range(len(bg_words)):
            for j in range(len(lda_model[bg_words[i]])):
                # output that to
                print('Document article%s.txt has a probability of %f%% to be in the topic %d' %
                      (article_names[i], lda_model[bg_words[i]][j][1]*100,
                       lda_model[bg_words[i]][j][0]))


def store_results(df, lda_model, bg_words, article_names, num_topics, num_words):
    """
    Make the output of LDA ready to be appended
    to annotated.csv
    """

    # stores
    data = []
    for i in range(len(bg_words)):
        # get the keywords (topics) from the LDA result
        keywords = lda_model.show_topics(num_topics=num_topics, num_words=num_words, formatted=False)[
            max(lda_model[bg_words[i]], key=lambda x: x[1])[0]
        ][
            1
        ]
        keywords = np.array(keywords)
        # To allow to store a list in the dataframe
        df = df.astype('object')
        for item in range(len(df)):
            if df.at[item, '0'] == int(article_names[i]):
                df.at[item, 'keywords'] = list(keywords[:, 0])
    # print(df)

    # output the resulted LDA keywords to annotated.csv
    df.to_csv('annotated_postLDA.csv', sep=',')

    return data


def main():

    rootdir = '../data/datasets/train-articles/'

    num_topics = 10
    num_keywords = 5
    # Get articles from the dataset
    articles = []
    # list to store article names
    article_names = []
    for subdir, dirs, files in os.walk(rootdir):
        for file in sorted(files):
            # match the number in the article name and store it in the list
            article_names.append(re.findall(r'\d+', file)[0])
            with open(os.path.join(subdir, file), 'r') as article:
                articles.append(article.read())

    bag_words, dict_ = preprocess(articles)
    lda_model, bag_words = lda(bag_words, dict_, num_topics, num_keywords, rootdir)

    # Get documents by topic
    identify_docs_by_topic(lda_model, bag_words, article_names)

    # read data
    annotated_data = pd.read_csv('../feature_engineering/annotated.csv')
    annotated_data['keywords'] = [nan for nan in range(len(annotated_data))]

    # prepare data to be annotated
    store_results(annotated_data, lda_model, bag_words, article_names, num_topics, num_keywords)


if __name__ == '__main__':
    main()
