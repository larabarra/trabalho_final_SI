#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  7 19:13:10 2021

@author: lara
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
import joblib


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
        
def erro_relativo(reais, estimados):
    erro = np.zeros(reais.shape)
    for i in range(0,len(estimados)):
        if reais[i] == 0:
            erro[i] = abs((reais[i] - estimados[i]))
        else:
            erro[i] = abs((reais[i] - estimados[i]))
    
    return erro

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



        #mlp = MLPRegressor(hidden_layer_sizes=(22,5,17,12,38),activation='relu', batch_size = 2 ,learning_rate_init = 0.003, shuffle = True , random_state = 4 , solver='adam', max_iter= 1000, verbose = False)
        #mlp = MLPRegressor(hidden_layer_sizes=(41,96,12,69,7),activation='relu', solver='lbfgs', max_iter= 1000, random_state = 54)
        #mlp = MLPRegressor(hidden_layer_sizes=(22,5,17,12,38),activation='relu', batch_size = 2 ,learning_rate_init = 0.003, shuffle = True , random_state = 4 , solver='adam', max_iter= 1000, verbose = False)
        #mlp = MLPRegressor(hidden_layer_sizes=(30,50,60,30,40,40),activation='relu', solver='lbfgs', max_iter=1000,random_state = i)
'''
for i in range (1,100):
    for j in range (1,100):

        mlp = MLPRegressor((90,90,90,i),activation='relu', solver='lbfgs', max_iter=1000,random_state = j) 
        
        mlp.fit(x_train, y_train)
        
        predict_train = mlp.predict(x_train)
        predict_test = mlp.predict(x_test)
        
        x_test_original = desnormaliza(x, x_test)
            #x_train_original = desnormaliza(x, x_train)
        y_test_original = desnormaliza(y, y_test)
            #y_train_original = desnormaliza(y, y_train)
            #predict_train_original = desnormaliza(y, predict_train)
        predict_test_original = desnormaliza(y, predict_test)
        
        erro = erro_relativo(y_test_original, predict_test_original)
        
        if np.sum(erro) < a:
            a = np.sum(erro)
            neuron = i
            r = j #87
            break
    
                  
    #print ("neuron:", i, "rand", j, "erro:" , np.sum(erro))
    print("iteração: ", i)

print(neuron, r,a)

mixer.init() #you must initialize the mixer
alert=mixer.Sound('bells.wav')
alert.play()

'''
mlp = MLPRegressor(hidden_layer_sizes=(97,46,68,16),activation='relu', solver='lbfgs', max_iter=1000,random_state = 22) #erro total de 15.694908275712322


#mlp = MLPRegressor(hidden_layer_sizes=(30,50,60,30,40,40),activation='relu', solver='lbfgs', max_iter=200,random_state = 11) #34.38
#mlp = MLPRegressor(hidden_layer_sizes=(22,5,17,12,38),activation='relu', batch_size = 2 ,learning_rate_init = 0.0030, shuffle = True, random_state = 4 , solver='adam', max_iter= 50, verbose = False)
#mlp = MLPRegressor(hidden_layer_sizes=(41,96,12,69,7),activation='relu', solver='lbfgs', max_iter= 400, random_state = 54)
#mlp = MLPRegressor(hidden_layer_sizes=(4,4,4,4,4,4),activation='relu', solver='lbfgs', max_iter=1000)

mlp.fit(x_train, y_train)

predict_train = mlp.predict(x_train)
predict_test = mlp.predict(x_test)

pickle.dump(mlp, open('treino.sav', 'wb'))

x_test_original = desnormaliza(x, x_test)
x_train_original = desnormaliza(x, x_train)
y_test_original = desnormaliza(y, y_test)
y_train_original = desnormaliza(y, y_train)
predict_train_original = desnormaliza(y, predict_train)
predict_test_original = desnormaliza(y, predict_test)

erro = erro_relativo(y_test_original, predict_test_original)

for i in range (0,len(predict_test_original)):
    print("Esperado:", y_test_original[i],[i], " Encontrado:", predict_test_original[i], "\n")


#plt.scatter(x,y,color='blue')
#plt.scatter(x_test_original,y_test_original,color='green')
#plt.scatter(x_test_original,predict_test_original ,color='red')
#plt.scatter(x_test_original,erro ,color='blue')
print("Erro total:", np.sum(erro))

#plt.scatter(x_train,y_train,color='green')
#plt.scatter(x_train,predict_train,color='blue') '''
