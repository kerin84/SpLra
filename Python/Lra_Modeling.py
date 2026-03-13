#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 12:38:37 2022

@author: kerin
"""

import numpy as np
import scipy.linalg as la
import control as cnt
import numpy.matlib
from lsspsolver import lsspsolver

def lra(
    D: np.ndarray,
    r: int,
    tol: float = 1e-3,
    delta: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    D = np.asarray(D)
    if D.ndim != 2:
        raise ValueError("D must be a 2D array.")

    n, m = D.shape
    if not (1 <= r < n):
        raise ValueError("r must satisfy 1 <= r < number of rows in D.")

    u, s, v = la.svd(D, full_matrices=0)
    R = u[:, r:].T
    P = u[:, :r]
    v = v.T
    Dh = u[:, :r].dot(np.diag(s[:r]).dot(v[:, :r].T))

    nres = n - r
    trailing = R[-nres:, -nres:]
    if np.linalg.matrix_rank(trailing) < nres:
        raise ValueError("Unable to invert residual basis block; choose a different r.")

    x = -la.inv(trailing).dot(R[:, :-nres])
    x_sp = lsspsolver(Dh[:-nres, :].T, Dh[-nres:, :].T, tol=tol, delta=delta).T
    R_sp = np.block([-x_sp, np.eye(R.shape[0])])

    return R, R_sp, P, Dh, x, x_sp

def Ar_Model_sim_lra(A,delta_H,delta_y,x0,N):
    
    n,m=A.shape
    y=x0[-n:,:].copy()
    
    for i in range(N):
        x1= A.dot(x0+delta_H[:,i:i+1])-delta_y[:,i:i+1]
        y=np.append(y,x1,axis=1)
        x0=np.block([[x0[n:,:]],[x1]])
       
    return  y

def R2ss(R,m,L,h):
    
    p,t=R.shape
    L1=L+1
    q=int(t/L1)
    n=L*p
    
    R=np.reshape(R,(p,q,L1),order='F')
    Q=R[:,0:m,:]
    P=-R[:,m:q,:]
    Pl_inv=la.pinv(P[:,:,-1])
    Ql=Q[:,:,-1]
    D=Pl_inv.dot(Ql)
    
    P0=-Pl_inv.dot(P[:,:,0])
    Q0=Pl_inv.dot(Q[:,:,0]-P[:,:,0].dot(Pl_inv.dot(Ql)))
    A1=P0
    B1=Q0
    
    for i in range(1,L):
        P0=-Pl_inv.dot(P[:,:,i])
        Q0=Pl_inv.dot(Q[:,:,i]-P[:,:,i].dot(Pl_inv.dot(Ql)))
        
        A1=np.block([[A1],[P0]])
        B1=np.block([[B1],[Q0]])
    
    A=np.block([[np.zeros((p,(L-1)*p))],[np.eye((L-1)*p)]])
    A=np.block([A,A1])

    C=np.block([np.zeros((p,n-p)),np.eye(p)])
    
    sys=cnt.ss(A,B1,C,D,dt=h)
    
    return sys

def pq_2_tf(P,Q,L,m,dt):
    
    p = Q.shape[0]
    L1=L+1
    num = Q.tolist() 
    den = np.zeros((p,m*L1))
    
    for i in range(p):
        den[i,:]=np.matlib.repmat(P[i,:],m,1)
        
    sys=cnt.tf(num,den, dt)
    return sys
    
def R2tf(R,L,m,dt):
    
    p,t=R.shape
    L1=L+1
    q=int(t/L1)
    
    R_sp=np.reshape(R,(p,q,L1),order='F')
    Q,P = np.flip(-R_sp[:,0:m,:],axis=-1) , np.flip(R_sp[:,m:q,:],axis=-1)

    num = Q.tolist() 
    den = np.zeros((p,m,L1))
    
    for i in range(p):
        den[i,:,:]=np.matlib.repmat(P[i,i,:],m,1)
        
    sys=cnt.tf(num,den, dt)
    return sys

def AutR_2_ss(x,h):
    
    n1,m1=x.shape
    A=np.block([[np.zeros((m1-n1,n1)),np.eye(m1-n1)],[x]])
    C=np.block([np.zeros((n1,m1-n1)),np.eye(n1)])
    
    return A,C

def Aut_sys_sim(A,C,x0,N):
   
    x=x0.copy()
    y=C.dot(x)
    
    for i in range(N):
        x1= A.dot(x0)
        y1=C.dot(x1)
        x=np.append(x,x1,axis=1)
        y=np.append(y,y1,axis=1)
        x0=x1
    
    return  y,x

    
    
