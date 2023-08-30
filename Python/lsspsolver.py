#!/usr/bin/env python3
"""
Created on Wed Mar 31 02:57:52 2021
LSSPSOLVER  Sparse linear least squares solver
   Code by Fredy Vides
   For Paper, "Computing Sparse Semilinear Models for Approximately Eventually Periodic Signals"
   by F. Vides
@author: Fredy Vides
"""
import numpy as np
from numpy.linalg import svd,lstsq,norm


def lsspsolver(A,Y,tol=1e-3,delta=1e-3,L=100):
    
    N=Y.shape[1]
    M=A.shape[1]
    X=np.zeros((M,N))
    
    u,s,v=svd(A,full_matrices=0)
    rk=sum(s>tol)
    u=u[:,:rk]
    s=s[:rk]
    s=1/s
    s=np.diag(s)
    v=v[:rk,:]
    A=np.dot(u.T,A)
    Y=np.dot(u.T,Y)
    X0=np.dot(v.T,np.dot(s,Y))
    
   
    for k in range(N):
        w=np.zeros((M,))
        K=1
        Error=1+tol
        c=X0[:,k]
        x0=c
        ac=abs(c)
        f=np.argsort(-ac)
        N0=int(max(sum(ac[f]>delta),1))
        while (K<=L) & (Error>tol):
            ff=f[:N0]
            X[:,k]=w
            c = lstsq(A[:,ff],Y[:,k],rcond=None)[0]
            X[ff,k]=c
            Error=norm(x0-X[:,k],np.inf)
            x0=X[:,k]
            ac=abs(x0)
            f=np.argsort(-ac)
            N0=int(max(sum(ac[f]>delta),1))
            K=K+1
    return X