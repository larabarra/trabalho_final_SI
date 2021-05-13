#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle

def normaliza(array):
    
    a = np.min(array)
    b = np.max(array)
    
    normalizado = np.zeros(array.shape)
    
    for i in range (0,len(array)):
        normalizado [i] = (array[i] - a)/(b-a) 
    
    
    return normalizado

def desnormaliza(array, valores):
    
    a = np.min(array)
    b = np.max(array)
    
    valores.reshape(-1,1)
    valores_f = valores.flatten()
    original = np.zeros(valores.shape)
   
    for i in range(0,len(valores)):
            original[i] = (valores_f[i]*(b-a)) + a  
    
    return original
        
def erro_medio(reais, estimados):
    erro = np.zeros(reais.shape)
    for i in range(0,len(estimados)):
        if reais[i] == 0:
            erro[i] = abs((reais[i] - estimados[i]))
        else:
            erro[i] = abs((reais[i] - estimados[i]))
    
    return np.sum(erro)/len(estimados)

def separa(x,y):
    
    x_test = np.zeros((20,1))
    y_test = np.zeros((20,))
    x_train = np.zeros((41,1))
    y_train = np.zeros((41,))
    
    count_test = 0
    count_train = 0
    
    for i in range (0, len(x)):
        if (i > 6 and i <= 13) or (i >= 32 and i<= 38) or (i>=46 and i<=51):
            x_test[count_test] = x[i]
            y_test[count_test] = y[i]
            count_test+=1
            continue
        else:
            x_train[count_train] = x[i]
            y_train[count_train] = y[i]
            count_train+=1
            continue
    
    return x_train, x_test, y_train, y_test


dataset = pd.read_csv("dados.csv")
dataset.dropna()

x = dataset.iloc[:,0]
y = dataset.iloc[:,1]
x_normalizado = normaliza(x)
y_normalizado = normaliza(y)

x_train, x_test, y_train, y_test =  separa(x_normalizado,y_normalizado)


mlp = MLPRegressor(hidden_layer_sizes=(97,46,68,16),activation='relu', solver='lbfgs', max_iter=150,random_state = 22,verbose = True)
#mlp = MLPRegressor(hidden_layer_sizes=(97,46,68,16),activation='relu', solver='lbfgs', max_iter=1000,random_state = 22,verbose = True)
#mlp = MLPRegressor(hidden_layer_sizes=(90,90,90,90),activation='relu', solver='lbfgs', max_iter=1000,random_state = 22,verbose = True)
#mlp = MLPRegressor(hidden_layer_sizes=(22,5,17,12,38),activation='relu', batch_size = 2 ,learning_rate_init = 0.0030, shuffle = True, random_state = 4 , solver='adam', max_iter= 1000, verbose = False)
mlp.fit(x_train, y_train)

predict_train = mlp.predict(x_train)
predict_test = mlp.predict(x_test)
predict_data = mlp.predict(x_normalizado.reshape(-1,1))
#pickle.dump(mlp, open('treino.sav', 'wb'))


x_test_original = desnormaliza(x, x_test)
x_train_original = desnormaliza(x, x_train)
y_test_original = desnormaliza(y, y_test)
y_train_original = desnormaliza(y, y_train)


predict_train_original = desnormaliza(y, predict_train)
predict_test_original = desnormaliza(y, predict_test)
predict_data_original = desnormaliza(y, predict_data)

erro = erro_medio(y_test,predict_test_original)


plt.plot(x,predict_data_original,color='red')
plt.plot(x,y,color='blue')
#plt.plot(x,y,color='blue')
#plt.plotr(x_test_original,y_test_original,color='green')
#plt.plot(x_test_original,predict_test_original,color='green')
#plt.plot(x_test_original,y_test_original,color='red')
#print("Erro total:", np.sum(erro))

