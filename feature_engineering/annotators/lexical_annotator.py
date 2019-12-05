from nltk.tokenize import sent_tokenize

class LexicalAnnotator:
    pass

    def annotate(text):
        sentence_length = len(text)
        exclamation_marks = text.count('!')
        question_marks = text.count('?')
        return sentence_length, exclamation_marks, question_marks



