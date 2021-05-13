#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 15:31:08 2021

@author: lara
"""

def desnormaliza(array, valores):
    
    a = np.min(array)
    b = np.max(array)
    
    valores.reshape(-1,1)
    valores_f = valores.flatten()
    original = np.zeros(valores.shape)
   
    j = 0
    for i in range(0,len(valores)):
        if (valores_f[i] % int(valores_f[i])) >= 0.5 and valores_f[i] != 0:
               original[i] = (valores_f[i]*(b-a)) + a + 1
        else:
             original[i] = (valores_f[i]*(b-a)) + a
        
    #u = statistics.mean(list(array))
    #o = statistics.pstdev(list(array))    
    
    #for i in range(0,len(valores)):
        #original[i] = (valores[i]*(o)) + u
    
    return original