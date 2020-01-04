from nltk.tokenize import word_tokenize

class LoadedLanguageAnnotator:
    def __init__(self, file):
        self.l = []
        with open(file) as f:
            for line in f:
                self.l.append(line.lower())

    def annotate(self, text):
        words = word_tokenize(text.lower())
        score = 0
        for w in words:
            if w in self.l:
                score += 1
        return score/len(words)

