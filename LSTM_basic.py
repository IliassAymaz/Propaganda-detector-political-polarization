import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

#data preparation..
initial_data = [[i for i in range(100)]]
initial_data = np.array(initial_data, dtype=float)
target_data= [[i for i in range(1,101)]]
target_data= np.array(target_data, dtype=float)

#splitting data  into test and train..
initial_data = initial_data.reshape((1, 1, 100)) 
target_data= target_data.reshape((1, 1, 100)) 
x_t=[i for i in range(100,200)]
x_t=np.array(x_t).reshape((1,1,100));
y_t=[i for i in range(101,201)]
y_t=np.array(y_t).reshape(1,1,100)

#RNN modle with LSTM..
mod = Sequential()  
mod.add(LSTM(100, input_shape=(1, 100),return_sequences=True))
mod.add(Dense(100))
mod.compile(loss='mean_absolute_error', optimizer='adam',metrics=['accuracy'])
mod.fit(initial_data, target_data, nb_epoch=10000, batch_size=1, verbose=2,validation_data=(x_t, y_t))



predict = mod.predict(initial_data)