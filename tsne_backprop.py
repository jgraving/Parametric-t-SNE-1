# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 21:20:14 2016

@author: paolo
"""

import scipy.spatial.distance
#from scipy.optimize import minimize
from minimize import *
import numpy as np
import numba as nb
import timeit


def Hbeta(D, beta):
    
    P = np.exp(-D * beta)
    sumP = np.sum(P,axis=0)
    H = np.log(sumP) + beta * np.sum(D*P,axis=0) / sumP
    P = P / sumP
    return H,P
    

@nb.jit(nb.float32(nb.float32))
def pdist_python(x):
    delta = sum(x.T**2)
    return delta[:,None] + delta[None,:] - 2*np.dot(x,x.T)


def compute_gaussian_perplexity(x, u=15.0, tol=1e-4):
    
    n = len(x)
    P = np.zeros((n,n))
    beta = np.ones(n)
    logU = np.log(u)

    print "Computing pairwise distances..."
#    start_time = timeit.default_timer()
#    DD = scipy.spatial.distance.pdist(x, 'sqeuclidean')
#    DD = scipy.spatial.distance.squareform(DD)
#    print timeit.default_timer() - start_time

    start_time = timeit.default_timer()
    DD = pdist_python(x)
    print timeit.default_timer() - start_time
    
    print "Computing P-values..."
    for i in range(n):
        
        min_beta = -np.inf
        max_beta = np.inf
        
        if i % 500 == 0:
            print "Computed P-values {} of {} datapoints...".format(i,n)
        
        Di = DD[i, np.arange(len(DD))!=i]
        H, thisP = Hbeta(Di, beta[i])

        Hdiff = H - logU
        tries = 0
        
        while np.abs(Hdiff) > tol and tries < 50:
            if Hdiff > 0:
                min_beta = beta[i]
                if np.isinf(max_beta):
                    beta[i] *= 2.0
                else:
                    beta[i] = (beta[i] + max_beta) / 2.0
            else:
                max_beta = beta[i]
                if np.isinf(min_beta):
                    beta[i] /= 2.0
                else:
                    beta[i] = (beta[i] + min_beta) / 2.0
            
            H, thisP = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1
        
        P[i, np.arange(len(P))!=i] = thisP
        
    print "Mean value of sigma: %f" % np.mean(np.sqrt(np.reciprocal(beta)))
    print "Minimum value of sigma: %f" % np.min(np.sqrt(np.reciprocal(beta)))
    print "Maximum value of sigma: %f" % np.max(np.sqrt(np.reciprocal(beta)))
    
    return P,beta   
    
    
#def compute_Pvalues(X, perplexity):
#    
#    P,_=compute_gaussian_perplexity(X, perplexity, 1e-5)
#    P[np.isnan(P)] = 0
#    P=(P + P.T) / 2 # symmetrize the probs
#    P=P / np.sum(P)
#    P[P<np.finfo(float).eps]=np.finfo(float).eps
#    return P
#    
#    
#def compute_Qvalues(x, v, n):
#    
#    sum_x = np.sum(x ** 2.0, axis=1)
#    num = 1 + (sum_x[:,np.newaxis] + (sum_x.T - 2.0*np.dot(x,x.T)))/v
#    Q=num ** -((v+1)/2.0)
#    np.fill_diagonal(Q,0) #np.arange(0,len(Q),n+1)
#    Q /= np.sum(Q)
#    Q[Q<np.finfo(float).eps]=np.finfo(float).eps
#    num = 1/num
#    np.fill_diagonal(num,0)
#    
#    return Q, num
#    
#
#def tsne_cost(x, X, P, network, v):
#    
#    no_layers=len(network)
#    
#    ii=0
#    for i in range(no_layers):
#        network[i]["W"] = x[ii:ii+network[i]["W"].size].reshape(network[i]["W"].shape)
#        ii += network[i]["W"].size
#        network[i]["bias_upW"] = x[ii:ii+network[i]["bias_upW"].size].reshape(network[i]["bias_upW"].shape)
#        ii += network[i]["bias_upW"].size
#    
#    activations = []
#    activations.append(np.append(X,np.ones((len(X),1)),axis=1))
#    for i in range(no_layers-1):
#        act_tmp=expit(np.dot(activations[i],np.vstack((network[i]["W"],network[i]["bias_upW"]))))
#        activations.append(np.append(act_tmp, np.ones((len(act_tmp),1)),axis=1))
#    activations.append(np.dot(activations[-1],np.vstack((network[-1]["W"],network[-1]["bias_upW"])))) 
#    
#    Q,num=compute_Qvalues(activations[-1],v,len(X))
#    C=np.sum(P * np.log((P+np.finfo(float).eps)/(Q+np.finfo(float).eps)))
#    
#    return C
#
#
#def tsne_grad(x, X, P, network, v):    
#    
#    n=len(X)
#    no_layers=len(network)
#    
#    ii=0
#    for i in range(no_layers):
#        network[i]["W"] = x[ii:ii+network[i]["W"].size].reshape(network[i]["W"].shape,order='F')
#        ii += network[i]["W"].size
#        network[i]["bias_upW"] = x[ii:ii+network[i]["bias_upW"].size].reshape(network[i]["bias_upW"].shape)
#        ii += network[i]["bias_upW"].size
#    
#    activations = []
#    activations.append(np.append(X,np.ones((len(X),1)),axis=1))
#    for i in range(no_layers-1):
#        act_tmp=expit(np.dot(activations[i],np.vstack((network[i]["W"],network[i]["bias_upW"]))))
#        activations.append(np.append(act_tmp, np.ones((len(act_tmp),1)),axis=1))
#    activations.append(np.dot(activations[-1],np.vstack((network[-1]["W"],network[-1]["bias_upW"]))))  
#    
#    Ix=np.zeros(activations[-1].shape)
#    Q,num=compute_Qvalues(activations[-1],v,len(X))
#    
#    stiffness=4*((v+1)/(2*v))*(P-Q)*num
#    for i in range(n):
#        Ix[i]=np.sum((activations[-1][i]-activations[-1])*stiffness[:,i,np.newaxis],axis=0)
#    
#    dW = no_layers*[0]
#    db = no_layers*[0]
#    for i in range(no_layers-1,-1,-1):
#        delta=np.dot(activations[i].T,Ix)
#        dW[i] = delta[:-1]
#        db[i] = delta[-1]
#        
#        if i > 0:
#            Ix = np.dot(Ix,np.vstack((network[i]['W'],network[i]['bias_upW'])).T) * \
#                activations[i] * (1-activations[i])
#            Ix = Ix[:,:-1]
#    
#    dC=[]
#    ii=0
#    for i in range(no_layers):
#        dC = np.append(dC,dW[i].reshape(-1,order='F'))
#        ii=ii+dW[i].size
#        dC = np.append(dC,db[i].reshape(-1,order='F'))
#        ii=ii+db[i].size
#        
#    return dC
    

def tsne_backprop(network, train_X, train_labels, max_iter, perplexity=30, v=1):
    
    n = len(train_X)
    batch_size = np.min([1000, n])
    ind = np.random.permutation(n)
    err = np.zeros(max_iter)
    
    print "Precomputing P-values..."
    curX=[]
    P=[]
    for batch in np.arange(0,n,batch_size):
        if batch+batch_size <= n:
            curX.append((train_X[ind[batch:np.min([batch+batch_size,n])]]).astype(float))
            Ptmp,_ = compute_gaussian_perplexity(curX[-1],perplexity,1e-5)
            Ptmp[np.isnan(Ptmp)] = 0
            Ptmp = (Ptmp+Ptmp.T) / 2.0
            Ptmp = Ptmp / np.sum(Ptmp)
            Ptmp[Ptmp<np.finfo(float).eps]= np.finfo(float).eps
            P.append(Ptmp)
    
    for it in range(max_iter):
        print "Iteration {}".format(it)
        b=0
        for batch in np.arange(0,n,batch_size):
            if batch+batch_size <= n:
                
                x0 = [];
                for i in range(len(network)):
                    x0 = np.append(x0, np.append(network[i]['W'].reshape(-1,order='F'), \
                        network[i]['bias_upW'].reshape(-1,order='F')))
                        
                # perform conjugate gradient - Poliak-Rebiere + 3 linesearches
#                opts = {'maxiter' : 3,
#                        'disp' : True}
#                args = (curX[b], P[b], network, v)
#                xmin=minimize(tsne_cost, x0, jac=tsne_grad, args=args,\
#                                method='CG', options=opts)
                xmin,_,_=minimize(x0, tsne_grad, 3, curX[b], P[b], network, v)
                #x=minimize(x, 'tsne_grad', 3, curX[b], P[b], network, v)
                #x = fmin_cg(tsne_cost, x0, fprime=tsne_grad, maxit=3, disp=True,args=args)
                b+=1
                
                # store new solution
                ii=0
                for i in range(len(network)):
                    network[i]['W'] = xmin[ii:ii+network[i]['W'].size].reshape(network[i]['W'].shape,order='F')
                    ii=ii+network[i]['W'].size
                    network[i]['bias_upW'] = xmin[ii:ii+network[i]['bias_upW'].size].reshape(network[i]['bias_upW'].shape,order='F')
                    ii=ii+network[i]['bias_upW'].size
                
        #activations=run_data_through_network(network,curX[1])
        #Q,_=compute_Qvalues(activations[-1],v,len(X))
        
        
    return network,err

                