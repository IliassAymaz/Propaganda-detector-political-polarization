from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from embedding.transform import BowTransformer
import numpy as np

class Predictor():
    def __init__(self, model_file):
         self.model = load_model(model_file)
         self.t = BowTransformer()

    def get_prediction(self, sequence):
        max_words = 128
        vocab_size = 18933
        bseq = self.t.transform(sequence)
        embedding = [to_categorical((pad_sequences((bseq,), max_words)).reshape(max_words,),vocab_size + 1)]
        return self.model.predict(np.array(embedding))
