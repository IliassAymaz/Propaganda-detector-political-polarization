from nltk.tokenize import sent_tokenize

class LexicalAnnotator:
    pass

    def annotate(text):
        sentences = sent_tokenize(text)
        s_size = len(sentences)
        if s_size == 0:
            return 0, 0, 0
        sentence_length = 0
        exclamation_marks = 0
        question_marks = 0
        for sentence in sentences:
            sentence_length += len(sentence)
            exclamation_marks += sentence.count('!')
            question_marks += sentence.count('?')
        return sentence_length/s_size, exclamation_marks/s_size, question_marks/s_size



