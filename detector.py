from machine_learning.predict_lstm import Predictor

from nltk import tokenize
from nltk.tokenize import TreebankWordTokenizer as twt

def get_sentences(doc):
    return tokenize.sent_tokenize(p)

def chop(seq):
    spans = list(twt().span_tokenize(seq))
    words = []
    for span in spans:
        words.append(seq[span[0]:span[1]])
    bites = []
    c = 0
    for w in words:
        temp = ['', 0, 0]
        for j in range(c,len(words)):
            if j == c:
                temp[0] = words[j]
                temp[1] = spans[j][0]
                temp[2] = spans[j][1]
                bites.append(temp[:])
            else:
                temp[0] += ' ' + words[j]
                temp[2] = spans[j][1]
                bites.append(temp[:])
        c += 1
    return bites

def areNeighbours(seq1, seq2):
    if seq1[2]+1 == seq2[1]:
        return True
    else:
        return False

def areOverlapping(seq1, seq2):
    if seq1[2] < seq2[1]:
        return True
    else:
        return False

def isPart(filtered, item):
    for i in filtered:
        if item == i:
            continue
        if i[1] <= item[1] and i[2] >= item[2]:
            return True
    return False

def removeChunks(filtered):
    for i in filtered:
        if isPart(filtered, i):
            filtered.remove(i)
    return filtered

predictor = Predictor('machine_learning/lstm_128_bow_without_features.h5')
bites = chop('White holds the school board and leadership responsible for spreading what she says is a false sense of what the Bible in the Schools program is and does.')
predictions = []
for b in bites:
    b.append(predictor.get_prediction(b[0])[0][1])
filtered = [x for x in bites if x[3] > 0.8]
filtered = removeChunks(filtered)
print(filtered)
