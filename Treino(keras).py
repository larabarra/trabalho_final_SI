#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 08:03:04 2021

@author: lara
"""
#hidden layer - tanh

from tensorflow import keras
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def normaliza(array):
    a = np.min(array)
    b = np.max(array)
    
    normalizado = np.zeros(array.shape)

    for i in range (0,len(array)):
        normalizado [i] = (array[i] - a)/(b-a)

    return normalizado
 

dataset = pd.read_csv("dados.csv")
np.random.shuffle(dataset.values)
'''
scaler = MinMaxScaler()
scaler.fit(dataset)
'''
x = (dataset.iloc[:,0]) 
y = dataset.iloc[:,1] 
x = x.values.reshape(-1,1)
y = y.values.reshape(-1,1)

x_normalizado = normaliza(x)
y_normalizado = normaliza(y)

x_train, x_test, y_train, y_test = train_test_split(x_normalizado,  y_normalizado , test_size = 0.20) #separa 20% para o treino


model = keras.models.Sequential()
model.add(keras.layers.Dense(60,input_shape=(1,), activation = 'relu'))
model.add(keras.layers.Dense(60, activation = 'tanh'))
model.add(keras.layers.Dense(60, activation = 'tanh'))
model.add(keras.layers.Dense(60, activation = 'tanh'))
model.add(keras.layers.Dense(60, activation = 'tanh'))
model.add(keras.layers.Dense(60,activation='softmax'))

model.compile(optimizer='adam', loss = keras.losses.MeanAbsolutePercentageError(), metrics = ['accuracy'])

model.fit(x_train, y_train, batch_size = 40, epochs = 1)
print("--------------------------------------------")
print("Avaliação: ", model.evaluate(x_test,y_test))
print("Avaliação treino: ", model.evaluate(x_train,y_train))

teste = model.predict(x_test)
#traino = model.predict(x_train)'''