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


def main():


    def preprocess(articles):
        """The preprocessing stage.
         Articles are tokenized, stripped of stop words
         and stemmed.
         """
        stemmed_articles = []
        for article in articles:
            # Tokenization
            # tokenizer = RegexpTokenizer(r'[^\s\:\#\-\,\;\!\"\_\?\!\.\”\’\“\—]+')
            tokenizer = RegexpTokenizer(r'(?<![\w\'\’])\w+?(?=\b|n\'t)')
            # print(tokenizer)

            result = tokenizer.tokenize(article.lower())

            # print(result)
            # Get the English words
            stop_words = get_stop_words('en')

            # Filter them out from the document
            filterd_doc = [item for item in result if not item in stop_words]

            # print('After removing stop words: \n', filterd_doc)

            # Stemming
            p_stemmer = PorterStemmer()
            stemmed_article = [p_stemmer.stem(word) for word in filterd_doc]

            # print('After stemming: \n', stemmed_article)

            stemmed_articles.append(stemmed_article)

            # Document-term matrix
        dictionary = corpora.Dictionary(stemmed_articles)
        # print(dictionary)

        # into bag of words
        bg_words = [dictionary.doc2bow(word) for word in stemmed_articles]

        # print('Into a bag of words', bg_words)
        return bg_words, dictionary

    def lda(bg_words, dictionary, num_topics, num_words):
        """
        Generates the LDA model.
        Takes the dictionary and the bag of words
        from the preprocessing stage.
        """
        start = time.time()
        lda_model = gensim.models.ldamodel.LdaModel(bg_words, num_topics=num_topics, id2word=dictionary, passes=20)
        print('LDA Analysis result for %d topics and %d words :' % (num_topics, num_words))
        for item in lda_model.print_topics(num_topics=num_topics, num_words=num_words):
            print(item)
        end = time.time()
        for subdir, dirs, files in os.walk(rootdir):
            n = len(files)
        print('Executed in ' + str(round(end-start, 2)) + ' seconds on ', len([iq for iq in os.scandir(rootdir)]), 'files')
        # print(lda_model.print_topics(num_topics=num_topics, num_words=num_words))

    rootdir = '../data/datasets/train-articles/'

    # Get articles from the dataset
    articles = []
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            with open(os.path.join(subdir, file), 'r') as article:
                articles.append(article.read())

    bag_words, dict_ = preprocess(articles)
    lda(bag_words, dict_, 14, 7)


if __name__ == '__main__':
    main()
