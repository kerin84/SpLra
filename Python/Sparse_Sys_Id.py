#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 12:47:14 2022

@author: kerin
"""

import numpy as np
import scipy.linalg as la 
from scipy import signal
import control as ctl
#from control.matlab import lsim
import matplotlib.pyplot as plt
from lsspsolver import lsspsolver
from statsmodels.tools.eval_measures import rmse
import pandas as pd
#from Lra_Modeling import lra
from blkhankel import blkhank


def Ar_Model(X,L,solver,tol=1e-2,delta=1e-2):
    
    n,m=X.shape
    Hx=blkhank(X,L,m-L+1)
    
    HLx = Hx[:n*(L-1),:]
    XL = Hx[n*(L-1):,:]
    
    if solver==1:
        A= XL.dot(la.pinv(HLx))
       
    if solver==2:
        A = lsspsolver(HLx.T,XL.T,tol,delta)
        A=A.T 
        
    Xp=A.dot(HLx)
    error=XL-Xp
    rmse_train=rmse(X[:,L-1:],Xp,axis=1)
    
    return A,HLx,XL,Xp,error,rmse_train

def Ar_2_ss(A,n,m,h):
    
    n1,m1=A.shape
    Ass=np.block([[np.zeros((m1-n1,n1)),np.eye(m1-n1)],[A]])
    B=np.zeros((m1-n1+n,m))
    C=np.block([np.zeros((n1,m1-n1)),np.eye(n1)])
    D=np.eye(n)
    
    sys=signal.StateSpace(Ass,B,C,D,dt=h)
    
    return sys

def x0_ss_estimate(sys,L,Ytrain,Utrain):
    
    O=ctl.obsv(sys.A,sys.C)
    n,m=O.shape
    
    if L==2:
        t,y=ctl.forced_response(sys,U=Utrain[:,:L]) 
        xini=la.pinv(O[:m,:]).dot(Ytrain[:,:L-1].T.reshape(m,1)-y[:,0].reshape(m,1))
    else:    
        t,y=ctl.forced_response(sys,U=Utrain[:,:L-1])
        xini=la.pinv(O[:m,:]).dot(Ytrain[:,:L-1].T.reshape(m,1)-y.reshape(m,1))
    
    return xini

def Ar_Model_sim(A,U,x0,N):
    
    n,m=A.shape
    y=x0[-n:,:].copy()
    
    for i in range(N):
        x1= A.dot(x0)+U[:,i:i+1]
        y=np.append(y,x1,axis=1)
        x0=np.block([[x0[n:,:]],[x1]])
        
    return  y

def dyn_sys_sim(A,Hu,fun,y0,N):
    
    n = A.shape[0]
    y =y0[-n:,:].copy()
   
    for i in range(N):
        u0 = Hu[:,i:i+1]
        x1= A.dot(fun(y0,u0))
        y=np.append(y,x1,axis=1)
        y0=np.block([[y0[n:,:]],[x1]])
        
    return y

def Ar_Model_ForeCast(A,y0,sigma,N,M=10000):
    
    n,m=A.shape
    y=np.zeros((M,N+1))
    
    for i in range(M):
        U=sigma*np.random.randn(n,N)
        y[i,:]=Ar_Model_sim(A,U,y0,N)
        
    max_y=np.max(y,axis=0)
    min_y=np.min(y,axis=0)  
    mean_y=np.mean(y,axis=0)
    return max_y,min_y,mean_y

    
def Ar_IO_Model(X,U,L,delay=True,tol=1e-2,delta=1e-2):
    
    n,m=X.shape
    Hx=blkhank(X,L,m-L+1)
    HLx = Hx[:n*(L-1),:]
    XL=Hx[n*(L-1):,:]
    
    n1,m1=U.shape
    Hu=blkhank(U,L,m-L+1)
    if delay==True:
        Hu=Hu[:n1*(L-1),:]
    HL=np.block([[HLx],[Hu]])
    
    A  = lsspsolver(HL.T,XL.T,tol,delta)
    A=A.T 
       
    A1,A2=A[:,:n*(L-1)],A[:,n*(L-1):]
    
    Xp=A.dot(HL)
    error=X[:,L-1:]-Xp
    rmse_train=rmse(X[:,L-1:],Xp,axis=1)
    
    return A1,A2,HL,XL,Xp,error,rmse_train

def Ar_IO_sim(A1,A2,y0,Hu,N):
    
    n,m=A1.shape
    y=y0[-n:,:]
    for i in range(N):
        y1=A1.dot(y0)+A2.dot(Hu[:,i:i+1])
        y=np.append(y,y1,axis=1)
        y0=np.block([[y0[n:,:]],[y1]])
    
    return y 
    
def Ar_IO_ss_Model(A,B,n,m,h):
    
    n1,m1=A.shape
    n2,m2=B.shape
    L=int(m1/n1)
    Lu=int(m2/n2)
    
    if L==Lu:
        D=np.zeros((n,m))
       
    else :
        D=B[-n:,-m:]  
        
    
    P0=A[:,:n]
    Q0=B[:,:m]+P0.dot(D)
    A1=P0
    B1=Q0
    
    for i in range(1,L):
        P0=A[:,i*n:(i+1)*n]
        Q0=B[:,i*m:(i+1)*m]+P0.dot(D)
        
        A1=np.block([[A1],[P0]])
        B1=np.block([[B1],[Q0]])
        
    A0=np.block([[np.zeros((n,m1-n))],[np.eye(m1-n)]]) 
    A=np.block([A0,A1])
    C=np.block([np.zeros((n,m1-n)),np.eye(n)])
    
    sys=ctl.ss(A,B1,C,D,dt=h)
        
    return sys

def sparsity(X):
    spar=(np.count_nonzero(X)/X.size)*100
    return spar

def show_heatmap(data):
    plt.matshow(data.corr())
    plt.xticks(range(data.shape[1]), data.columns, fontsize=11, rotation=90)
    plt.gca().xaxis.tick_bottom()
    plt.yticks(range(data.shape[1]), data.columns, fontsize=11)

    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title("Feature Correlation Heatmap", fontsize=11)
    plt.show()

def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)    
    


     
    


    
    
    
    
    
    
    
    
    
    
    
    
