# Transform the training corpus to a bag of words
# Output: vector of 19141 unique words to 
# evalue future tokens/sentences with.

# Use document_to_vector(token, uniques) to get the BoW
# representation of a 'token' that lies 
# on a 'uniques' list.


import re
import os
import spacy


rootdir = '../data/datasets/train-articles/'
nlp = spacy.load('en_core_web_sm')


def get_articles(root_dir):
    """
    Takes a root directory and returns a list of documents text.
    :param root_dir:
    :return: corpus
    """
    corpus = []
    for subdir, dirs, files in os.walk(root_dir):
        for f in files:
            corpus.append(open(subdir+f, 'r').read())
    return corpus


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


# lemmatize
def lemmatize_uniques(joint_uniques):
    """
    lemmatizes the uniques for a more 
    accurate bow representation.
    """
    lemmas = []
    doc = nlp(joint_uniques)
    for token in doc:
        lemmas.append(token.lemma_)
    print(lemmas)
    return lemmas


def lemmatize_document(document):
    """
    Gets a documents and returns a 
    document with lematized tokens.
    """
    lemmas = []
    doc = nlp(document)
    for token in doc:
        lemmas.append(token.lemma_)
    return ' '.join(lemmas)


def document_to_vector(lemmatized_document, uniques):
    """
    Converts a lemmatized document to a bow vector 
    representation.
    1/0 for word exists/doesn't exist
    """
    print(uniques)
    # tokenize
    words = re.findall(r'\w+', lemmatized_document.lower())

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


def save_to_file(filename, _list):
    with open(filename, 'w+') as f_:
        for item in _list:
            f_.write(str(item)+'\n')


def main():
    # Transform our corpus of documents into a bag of words per document
    # The output is a uniques vector: lemmatized_uniques

    docs = get_articles(rootdir)

    corpus = ' '.join(docs)

    uniques = build_bow(corpus)
    lemmatized_uniques = lemmatize_uniques(' '.join(uniques))

    # save uniques to a file
    save_to_file('lemmatized_uniques.txt', lemmatized_uniques)

    lemmatized_documents = []
    for doc in docs:
        lemmatized_documents.append(lemmatize_document(doc))

    document_vectors = []
    for i in range(len(lemmatized_documents)):
        document_vectors.append(document_to_vector(lemmatized_documents[i], lemmatized_uniques))


if __name__ == '__main__':
    main()

