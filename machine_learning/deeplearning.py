import pandas as pd
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Embedding, Dense, LSTM
df = pd.read_csv('test.csv')
df = df.drop(columns=['ner_country','ner_organization','ner_person','id','span'])
properties = list(df.columns.values)
properties.remove('propaganda')
properties.remove('bow_sequence')
Z = df[properties].values.tolist()
X = []
c = 0
for i in df['bow_sequence']:
    if c%100 == 0:
        print(c)
    c+=1
    X.append(eval(i))
X = np.asarray(X)
labels = df['propaganda'].to_list()
num_classes = 2 
max_words = 128
y = to_categorical(labels, num_classes=num_classes)

def generator(X_data, y_data, batch_size):
    
    samples_per_epoch = len(X_data)
    number_of_batches = samples_per_epoch/batch_size
    counter=0
            
    while 1:
        X_batch = np.array([to_categorical((pad_sequences((sent,), max_words)).reshape(max_words,),vocab_size + 1) for sent in X_data[batch_size*counter:batch_size*(counter+1)]])
        y_batch = y_data[batch_size*counter:batch_size*(counter+1)]
        counter += 1
        yield X_batch,y_batch
        
        #restart counter to yeild data in the next epoch as well
        if counter >= number_of_batches:
            counter = 0

vocab_size = 18933
X_train, X_test, y_train, y_test, Z_train, Z_test = train_test_split(X, y, Z, test_size=0.3, random_state=0)


model = Sequential()
model.add(Dense(512, input_shape=(max_words, vocab_size + 1)))
model.add(LSTM(128))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy',
          optimizer='adam',
                    metrics=['accuracy'])

model.fit_generator(generator(X_train,y_train,100),steps_per_epoch = len(X_train)/100, epochs=5)

test_loss, test_acc = model.evaluate_generator(generator(X_test,y_test,100), steps = len(X_test)/100)
print('Test accuracy:', test_acc)
model.save('lstm_128_bow.h5')
