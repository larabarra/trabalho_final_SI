#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 20:51:27 2021

@author: lara
"""
from pybrain.tools.shortcuts import buildNetwork # para criar a rede
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer #algoritmo para treinar
import pickle #para salvar o treino
import csv
import matplotlib.pylab as plt
import numpy as np
import os
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, TanhLayer, SigmoidLayer, GaussianLayer
from pybrain.structure import FullConnection
from random import random
from pybrain.structure import FeedForwardNetwork
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from sklearn.preprocessing import StandardScaler
from pybrain.structure import LinearLayer, SigmoidLayer, BiasUnit
from sklearn.model_selection import train_test_split
import pandas as pd 

dataset = pd.read_csv("dados.csv")

x = dataset.iloc[:,0]
y = dataset.iloc[:,1]
x = x.values.reshape(-1,1)
y = y.values.reshape(-1,1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20) #separa 20% para o treino

rede = buildNetwork(1,5,1)
base = SupervisedDataSet(1,1)

for i, j in zip(x_train, y_train):
    ds.addSample(i[0]/1000,j[0]/1000)

treinamento = BackpropTrainer(rede,dataset = base, learningrate = 0.01, momentum = 0.06)

for e in range (1,1000):
    erro = treinamento.train()
    print("Erro: " , erro)

for t in x_test:
    print(rede.activate(t/1000)*1000)