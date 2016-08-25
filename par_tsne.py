# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 10:27:56 2016

@author: paoloinglese
"""


import numpy as np
from rbm import *
from scipy.special import expit
from tsne_backprop import tsne_backprop

def train_par_tsne(train_X, train_labels, layers, training='CD1'):
    
       origX=train_X
       no_layers=len(layers)
       network=[]
       for i in range(no_layers):
           
           print "Training layer %d (size %d -> %d)..." % (i,train_X.shape[1],layers[i])
           
           if i != no_layers-1:
               
               if training == 'CD1':
                   curr_rbm,_ =train_rbm(train_X, layers[i], 0.1, 30))
               elif training == 'PCD':
                   curr_rbm,_=train_rbm_pcd(train_X, layers[i])
               elif training == 'None':
                   v=train_X.shape[1]
                   curr_rbm={'W' : np.random.randn(v,layers[i])*0.1,
                             'bias_upW' : np.zeros(layers[i]),
                             'bias_downW' : np.zeros(v)}
               else:
                   raise Exception("Unknown training procedure")
                   
               train_X = expit(np.dot(train_X,curr_rbm['W'])+curr_rbm['bias_upW'])
           
           else:
                
                if training != 'None':
                    curr_rbm=train_linear_rbm(train_X, layers[i], 0.01, 30)
                else:
                    v=train_X.shape[1]
                    curr_rbm={'W' : np.random.randn(v,layers[i])*0.1,
                              'bias_upW' : np.zeros(layers[i]),
                              'bias_downW' : np.zeros(v)}
                              
           network.append(curr_rbm)
           
       network, err = tsne_backprop(network, origX, train_labels, 30, 30, 1)
       return network,err

