#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 07:45:11 2021

@author: lara
"""

import pandas as pd
from sklearn import model_selection
import joblib
import numpy as np

def normaliza(a,b,entrada):
    
    normalizado = (entrada - a)/(b-a)   
        
    return normalizado

def desnormaliza(a,b,saida):
    
    original = saida[0]*(b-a) + a 
    
    return original

dataset_normalizado = pd.read_csv("dados.csv")
dataset_normalizado.dropna()



x = dataset_normalizado.iloc[:,0]
y = dataset_normalizado.iloc[:,1]

x_min = np.min(x)
x_max = np.max(x)
y_min = np.min(y)
y_max = np.max(y)


mpl = joblib.load('treino.sav')

entrada = int(input("Insira o valor da entrada:\n"))

entrada_normalizada = normaliza(x_min,x_max,entrada)

entrada_normalizada = entrada_normalizada.reshape(1,-1)

saida = mlp.predict(entrada_normalizada)

saida = desnormaliza(y_min, y_max, saida)
print(saida)

