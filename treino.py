#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
# _________________________________________________________________
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

# _________________________________________________________________
#                        FUNCOES
"""
Abre file em modo de leitura e le a coluna passada como parametro
"""
def le_arquivo(arq,coluna):
    with open(arq,'r') as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            yield row[coluna]

def y(x):
    return (x + 900) / 1000

# _________________________________________________________________
#                        MAIN
# quanto tempo vai treinar ?
n_treinos = 1000 # depois de 5000 não tem efeito
os.system("clear")
entrada = []
saida = []

for i in le_arquivo('dados.csv','Entrada'):
    entrada.append(i)
list(map(int,entrada))

for i in le_arquivo('dados.csv','Saída'):
    saida.append(i)
list(map(int,saida))

#                   Cria a rede neural
network = FeedForwardNetwork()
os.system("rm treino")

#                   DEFINIÇÕES DA REDE
camadaEntrada = LinearLayer(1)      #1 neuronio de entrada
camadaIntermediaria1 = TanhLayer(90) #4 neuronios na 1a camada intermediaria
camadaIntermediaria2 = TanhLayer(90) #4 neuronios na 2a camada intermediaria
camadaIntermediaria3 = TanhLayer(90)
camadaIntermediaria4 = TanhLayer(90)
camadaIntermediaria5 = TanhLayer(90)
camadaIntermediaria6 = TanhLayer(90)

camadaSaida = LinearLayer(1)        #1 neuronio de saida

#                    CONFIGURA A REDE
network.addInputModule(camadaEntrada)
network.addModule(camadaIntermediaria1)
network.addModule(camadaIntermediaria2)
network.addModule(camadaIntermediaria3)
network.addModule(camadaIntermediaria4)
network.addModule(camadaIntermediaria5)
network.addModule(camadaIntermediaria6)
network.addOutputModule(camadaSaida)

#                   CONECTA AS CAMADAS
entrada_meio1 = FullConnection(camadaEntrada,camadaIntermediaria1)
meio1_meio2 = FullConnection(camadaIntermediaria1,camadaIntermediaria2)
meio2_meio3 = FullConnection(camadaIntermediaria2,camadaIntermediaria3)
meio3_meio4 = FullConnection(camadaIntermediaria3,camadaIntermediaria4)
meio4_meio5 = FullConnection(camadaIntermediaria4,camadaIntermediaria5)
meio5_meio6 = FullConnection(camadaIntermediaria5,camadaIntermediaria6)
meio4_saida = FullConnection(camadaIntermediaria6,camadaSaida)

#                ADICIONA AS CAMADAS Á REDE
network.addConnection(entrada_meio1)
network.addConnection(meio1_meio2)
network.addConnection(meio2_meio3)
network.addConnection(meio3_meio4)
network.addConnection(meio4_meio5)
network.addConnection(meio5_meio6)
network.addConnection(meio4_saida)

network.sortModules() #Torna a rede utilizavel, faz as ligações

ds = SupervisedDataSet(1,1) #1 entrada, 1 saida

# Abrir arquivo para registro do conhecimento
file_obj = open('treino','w')

treino = [] # lista para armazenar o erro da aprendizagem
iteracao = []

#               NORMALIZANDO OS VALORES
entrada_normalizada = []
saida_normalizada = []

for i in range(60):
    """
    if  ( i  >= 1 and i <= 15):
        x_temp = float((i + 0.5)) / 1000
        entrada_normalizada.append(x_temp)
        saida_normalizada.append((x_temp**2))
    """
    if ( i >=31 and i <= 45):
        for j in range(0,100,2):
            x_temp = (i + float(j/100))
            entrada_normalizada.append(x_temp/1000)
            saida_normalizada.append(y(x_temp))

    """
    if ( i >= 46 and i <= 60):
        x_temp = float((i + 0.5)) / 1000
        entrada_normalizada.append(x_temp)
        saida_normalizada.append((x_temp*(-53)+3338))
    """
    entrada_normalizada.append(float(entrada[i])/1000)
    saida_normalizada.append(float(saida[i])/1000)

for i in range(len(entrada_normalizada)-1):
    ds.addSample(entrada_normalizada[i],saida_normalizada[i])

#                       APRENDER -> learningrate=0.01
trainer = BackpropTrainer(network,ds,learningrate=0.01)
print("Epocas:")
for i in range(0,n_treinos):
    #print(trainer.trainUntilConvergence())
    iteracao.append(i)
    if ( i == n_treinos - 1):
        print("Erro:")
        print(trainer.train()) # erro
    else:
        treino.append(trainer.train())
    print(i)

#             REGISTRA O CONHECIMENTO NO ARQUIVO
pickle.dump(network,file_obj)
file_obj.close()

"""
#       PARA PLOTAR O GRÁFICO ( muitos dados está estourando a funcao)
plt.plot(iteracao,treino)
plt.show()
"""
