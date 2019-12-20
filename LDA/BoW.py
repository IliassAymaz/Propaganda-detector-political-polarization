# Transform the training corpus to a bag of words
# Remove the stop words first? --no, for acccuracy


import re
import os

rootdir = '../data/datasets/train-articles/'

# tokenize
def build_bow(corpus):
    """
    takes the corpus (all documents as str) and 
    builds the vector represetation.
    """
    # lowercase
    corpus = corpus.lower()

    # tokenize into a set of words
    match = re.findall(r'\w+', corpus)

    # get the unique words
    unique_words = list(set(match))

    return unique_words

def document_to_vector(document, uniques):
    """
    Converts a document to a bow vector 
    representation.
    1/0 for word exists/doesn't exist
    """
    print(uniques)
    # tokenize
    words = re.findall(r'\w+', document.lower())

    # vector = {} 
    vector = [0]*len(uniques)
    # list of the words is accessible via vector.keys()
    # list of 0/1 is accessible via vector.values() 

    # seen = []
    for i in range(len(uniques)):
        for j in range(len(words)):
            if uniques[i] == words[j]:
                vector[i] = 1
                continue

    return vector

def get_articles(root_dir):
    for subdir, dirs, files in os.walk(root_dir):


docs = [
    'i love machine learning',
    'machine learning is great',
    'we can programm on any machine'
]

corpus = ' '.join(docs)

uniques = build_bow(corpus)
vector = document_to_vector('i have a great machine.', uniques)
print(vector)