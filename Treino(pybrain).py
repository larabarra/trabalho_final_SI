#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
# _________________________________________________________________
from pybrain.tools.shortcuts import buildNetwork 
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer 
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import FullConnection
from sklearn.model_selection import train_test_split

import matplotlib.pylab as plt
import os
from pybrain.structure import LinearLayer, TanhLayer, SigmoidLayer, GaussianLayer

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
    original = []
    
    for i in range(0,len(valores)):
        original.append((valores[i]*(b-a)) + a) 

    return original

def erro_relativo(reais, estimados):
    erro = []
    for i in range(0,len(estimados)):
        if reais[i] == 0:
            erro.append(abs((reais[i] - estimados[i])))
        else:
            erro.append(abs((reais[i] - estimados[i])))
    
    return erro

def separa(x,y):
    
    x_test = np.zeros((20,1))
    y_test = np.zeros((20,1))
    x_train = np.zeros((41,1))
    y_train = np.zeros((41,1))
    
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
 
os.system("clear")

dataset = pd.read_csv("dados.csv")

x = (dataset.iloc[:,0]) 
y = dataset.iloc[:,1] 
x = x.values.reshape(-1,1)
y = y.values.reshape(-1,1)

x_normalizado = normaliza(x)
y_normalizado = normaliza(y)

x_train, x_test, y_train, y_test = separa(x_normalizado,  y_normalizado) #separa 20% para o treino

rede = FeedForwardNetwork()

camadaEntrada = LinearLayer(1)    
 

camadaSaida = LinearLayer(1) 

a = 1000
b = 0

for i in range(1,50):
    camadaOculta1 = TanhLayer(i) 
#    camadaOculta2 = TanhLayer(20) 
#   camadaOculta3 = TanhLayer(40)
#   camadaOculta4 = TanhLayer(40)
#   camadaOculta5 = TanhLayer(40)
#    camadaOculta6 = TanhLayer(40)

    rede.addInputModule(camadaEntrada)
    rede.addModule(camadaOculta1)
#rede.addModule(camadaOculta2)
#rede.addModule(camadaOculta3)
#rede.addModule(camadaOculta4)
#rede.addModule(camadaOculta5)
#rede.addModule(camadaOculta6)
    rede.addOutputModule(camadaSaida)


    entradaoculta1 = FullConnection(camadaEntrada, camadaOculta1)
#oculta1oculta2 = FullConnection(camadaOculta1, camadaOculta2)
#oculta2oculta3 = FullConnection(camadaOculta2, camadaOculta3)
#oculta3oculta4 = FullConnection(camadaOculta3, camadaOculta4)
#oculta4oculta5 = FullConnection(camadaOculta4, camadaOculta5)
#oculta5oculta6 = FullConnection(camadaOculta5, camadaOculta6)
    oculta1saida = FullConnection(camadaOculta1, camadaSaida)

    rede.addConnection(entradaoculta1)
#rede.addConnection(oculta1oculta2)
#rede.addConnection(oculta2oculta3)
#rede.addConnection(oculta3oculta4)
#rede.addConnection(oculta4oculta5)
#rede.addConnection(oculta5oculta6)
    rede.addConnection(oculta1saida)

    rede.sortModules()


    ds = SupervisedDataSet(1,1)

    for i,j in zip(x_train, y_train):
        ds.addSample(i,j)
    
    trainer = BackpropTrainer(rede,ds,learningrate=0.01)

    treino = [] #erro
    for i in range(0,1000):
        print(i)
        treino.append(trainer.train())

    predict_test = []
    for i in x_test:
        predict_test.append(rede.activate(i))


#x_test_original = desnormaliza(x, x_test)
#x_train_original = desnormaliza(x, x_train)
    y_test_original = desnormaliza(y, y_test)
#y_train_original = desnormaliza(y, y_train)
#predict_train_original = desnormaliza(y, predict_train)
    predict_test_original = desnormaliza(y, predict_test)

    erro = sum(erro_relativo(y_test_original, predict_test_original))
    
    if np.sum(erro) <=a:
        a = np.sum(erro)
        b = i
            
print("valor do erro:", a, "Valor do i:", b)
'''
plt.scatter(x_test_original,y_test_original,color='green')
plt.scatter(x_test_original,predict_test_original,color='red')

for i in range (0,len(predict_test_original)):
        print("Esperado:", y_test_original[i], " Encontrado:", predict_test_original[i], "\n")'''
