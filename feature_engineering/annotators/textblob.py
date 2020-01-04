from textblob import TextBlob

class PolarityAnnotator:
    pass

    def annotate(text):
        data = TextBlob(text)
        return data.sentiment.polarity

class SubjectivityAnnotator:
    pass

    def annotate(text):
        data = TextBlob(text)
        return data.sentiment.subjectivity


