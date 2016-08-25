# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 19:35:54 2016

@author: paolo
"""

import numpy as np
from tsne_grad import *


def minimize(X, f, length, *args):
    
    RHO=0.01
    SIG=0.5
    INT=0.1
    EXT=3.0
    MAX=20
    RATIO=100
    red=1
    
    if length > 0:
        S='Linesearch'
    
    i=0
    ls_failed=False
    fX=[]
    f1, df1 = f(X, *args)
    s=-df1
    d1=-np.dot(s.T,s)
    z1=red/(1-d1)
    
    while i < np.abs(length):
        i+=1
        
        X0=X; f0=f1; df0=df1
        X+=z1*s
        f2,df2=f(X, *args)
        d2=np.dot(df2.T,s)
        f3=f1; d3=d1; z3=-z1
        if length > 0:
            M = MAX
        else:
            M = np.min([MAX,-length-i])
        success=False; limit=-1
        
        while True:
            while ((f2 > f1+z1*RHO*d1) | (d2 > -SIG*d1)) & (M > 0):
                limit=z1
                if f2 > f1:
                    z2 = z3-(0.5*d3*z3*z3)/(d3*z3+f2-f3)
                else:
                    A = 6*(f2-f3)/z3 + 3*(d2+d3)
                    B = 3*(f3-f2) - z3*(d3+2.0*d2)
                    z2 = (np.sqrt(B*B-A*d2*z3*z3)-B)/A
                if (np.isnan(z2)) | (np.isinf(z2)):
                    z2 = z3/2
                z2 = np.max([np.min([z2,INT*z3]),(1-INT)*z3])
                z1 += z2
                X += z2*s
                f2,df2=f(X, *args)
                M-=1; i+=np.sum(np.array(length<0,dtype=float))
                d2=np.dot(df2.T,s)
                z3-=z2
            if (f2>f1+z1*RHO*d1) | (d2>-SIG*d1):
                break
            elif d2>SIG*d1:
                success=True; break
            elif M==0:
                break
            A = 6*(f2-f3)/z3 + 3*(d2+d3)
            B = 3*(f3-f2) - z3*(d3+2.0*d2)
            z2 = -d2*z3*z3 / (B+np.sqrt(B*B-A*d2*z3*z3))
            if not(np.isreal(z2)) | (np.isnan(z2)) | (np.isinf(z2)) | (z2<0):
                if limit < -0.5:
                    z2=z1*(EXT-1)
                else:
                    z2=(limit-z1)/2.0
            elif (limit > -0.5) & (z2+z1>limit):
                z2 = (limit-z1)/2.0
            elif (limit < -0.5) & (z2+z1>z1*EXT):
                z2 = z1*(EXT-1.0)
            elif z2 < -z3*INT:
                z2 = -z3*INT
            elif (limit > -0.5) & (z2 < (limit-z1)*(1.0-INT)):
                z2 = (limit-z1)*(1.0-INT)
            f3=f2; d3=d2; z3=-z2
            z1+=z2; X+=np.dot(z2,s)
            f2,df2=f(X, *args)
            M-=1; d2=np.dot(df2.T,s)
        
        if success:
            f1=f2
            fX=np.append(fX,f1).T
            print "%s %6i; Value %4.6e" % (S,i,f1)
            s = (np.dot(df2.T,df2)-np.dot(df1.T,df1))/(np.dot(df1.T,df1))*s - df2
            tmp = df1
            df1=df2
            df2=tmp
            d2=np.dot(df1.T,s)
            if d2>0:
                s=-df1
                d2=-np.dot(s.T,s)
            z1=z1*np.min([RATIO, d1/(d2-np.finfo(np.double).tiny)])
            d1=d2
            ls_failed=False
        else:
            X=X0
            f1=f0
            df1=df0
            if ls_failed | (i>abs(length)):
                break
            tmp=df1
            df1=df2
            df2=tmp
            s=-df1
            d1=-np.dot(s.T,s)
            z1=1/(1-d1)
            ls_failed=True
            
    return X,fX,i