from textblob import TextBlob

class PolarityAnnotator:
    pass

    def annotate(text):
        data = TextBlob(text)
        return data.sentiment[0]
