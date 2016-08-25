# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 09:14:10 2016

@author: paolo
"""

import timeit
import numpy as np
from scipy.special import expit


def compute_recon_err(machine,X):
    
    if type(machine) is list:
        
        err=np.zeros(len(machine))
        vis=X
        for i in range(len(machine)):
            if i < len(machine):
                hid=expit(np.dot(vis,machine[i]['W'])+machine[i]['bias_upW'])
            else:
                hid=vis*machine[i]['W']+machine[i]['bias_upW']
        
        rec=expit(np.dot(hid,machine[i]['W'].T)+machine[i]['bias_downW'])
        err[i]=np.sum((vis-rec)**2)/len(X)
        vis=hid
        
    else:
        
        hid=expit(np.dot(X,machine['W'])+machine['bias_upW'])
        rec=expit(np.dot(hid,machine['W'].T)+machine['bias_downW'])
        err=np.sum((X-rec)**2)/len(X)
    
    return err
        
        
def train_rbm(X, h=30, eta=0.1, max_iter=30, weight_cost=0.0002):
    
    initial_momentum=0.5
    final_momentum=0.9
    
    (n,v) = X.shape    
    batch_size=100
    machine = {'W' : np.random.randn(v,h)*0.1,
               'bias_upW' : np.zeros(h),
               'bias_downW' : np.zeros(v)
               }
    delta_W=np.zeros((v,h))
    delta_bias_upW=np.zeros(h)
    delta_bias_downW=np.zeros(v)
    
    # Main loop
    train_err=np.zeros(max_iter)
    for i in range(max_iter):
        
        start_time=timeit.default_timer()        
        
        ind=np.random.permutation(n)
        
        if i < 5:
            momentum=initial_momentum
        else:
            momentum=final_momentum
        
        for batch in np.arange(0,n,batch_size):
            if batch+batch_size <= n:
                
                vis1=(X[ind[batch:np.min([batch+batch_size,n])]]).astype(float)
                
                hid1=expit(np.dot(vis1,machine['W'])+machine['bias_upW'])
                
                hid_states=np.asarray(hid1 > np.random.rand(hid1.shape[0],hid1.shape[1]), dtype=float)
                
                vis2=expit(np.dot(hid_states,machine['W'].T)+machine['bias_downW'])
                
                hid2=expit(np.dot(vis2,machine['W'])+machine['bias_upW'])

                posprods=np.dot(vis1.T,hid1)
                negprods=np.dot(vis2.T,hid2)
                delta_W=momentum*delta_W+eta*((posprods-negprods)/batch_size-weight_cost*machine['W'])
                delta_bias_upW=momentum*delta_bias_upW+eta/batch_size*(np.sum(hid1,axis=0)-np.sum(hid2,axis=0))
                delta_bias_downW=momentum*delta_bias_downW+eta/batch_size*(np.sum(vis1,axis=0)-np.sum(vis2,axis=0))
                
                machine['W']+=delta_W
                machine['bias_upW']+=delta_bias_upW
                machine['bias_downW']+=delta_bias_downW
        
        err_tmp=compute_recon_err(machine,X)
        print "Iteration %d (train rec. error = %f)" % (i,err_tmp)
        
        print timeit.default_timer() - start_time
        
        train_err[i]=err_tmp
    
    print " "
    
    err=compute_recon_err(machine,X)
    
    return machine, err
    
    
def train_linear_rbm(X, h=20, eta=0.001, max_iter=50, weight_cost=0.0002):

    initial_momentum=0.5
    final_momentum=0.9
    
    (n,v) = X.shape    
    batch_size=100
    machine = {'W' : np.random.randn(v,h)*0.1,
               'bias_upW' : np.zeros(h),
               'bias_downW' : np.zeros(v)
               }
    delta_W=np.zeros((v,h))
    delta_bias_upW=np.zeros(h)
    delta_bias_downW=np.zeros(v)
    
    # Main loop
    for i in range(max_iter):

        err=0        
        ind=np.random.permutation(n) 
        if i < 5:
            momentum=initial_momentum
        else:
            momentum=final_momentum
            
        for batch in np.arange(0,n,batch_size):
            if batch+batch_size <= n:
                
                vis1=(X[ind[batch:np.min([batch+batch_size,n])]]).astype(float)
                
                hid1=np.dot(vis1,machine['W'])+machine['bias_upW']
                
                hid_states=hid1+np.random.randn(hid1.shape[0],hid1.shape[1])
                
                vis2=expit(np.dot(hid_states,machine['W'].T)+machine['bias_downW'])
                
                hid2=np.dot(vis2,machine['W'])+machine['bias_upW']

                posprods=np.dot(vis1.T,hid1)
                negprods=np.dot(vis2.T,hid2)
                delta_W=momentum*delta_W+eta*((posprods-negprods)/batch_size-weight_cost*machine['W'])
                delta_bias_upW=momentum*delta_bias_upW+(eta/batch_size)*(np.sum(hid1,axis=0)-np.sum(hid2,axis=0))
                delta_bias_downW=momentum*delta_bias_downW+(eta/batch_size)*(np.sum(vis1,axis=0)-np.sum(vis2,axis=0))
                
                machine['W']+=delta_W
                machine['bias_upW']+=delta_bias_upW
                machine['bias_downW']+=delta_bias_downW
                
                err+=np.sum(np.sum((vis1-vis2)**2,axis=0))
        
        print "Iteration %d (train rec. error ~ %f)" % (i,err/n)
    
    return machine         
    

#from scipy.io import loadmat
#
#train_mnist = loadmat('/home/paolo/source-test/parametric_tsne/mnist_test.mat')
#train_X = train_mnist['test_X']
#train_labels = train_mnist['test_labels']
#layers=[250,50,2]
#train_rbm(train_X,layers[0],0.01,3)