# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 22:26:10 2023

@author: kerin
"""

from blkhankel import blkhank
import numpy as np
import scipy.linalg as la
from Lra_Modeling import lra,R2tf,R2ss,AutR_2_ss,Aut_sys_sim
import control as cnt
from control.matlab import impulse
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib as mpl
from statsmodels.tools.eval_measures import rmse
from sklearn.metrics import r2_score

import warnings
from Sparse_Sys_Id import x0_ss_estimate
warnings.filterwarnings('ignore')

import scienceplots
plt.style.use('science')
mpl.rcParams['lines.linewidth'] = 0.6


def Sparse_lra_sysid(w,L,m, x0 = 1, tol=1e-3 , delta=1e-3):
    
    n1,m1 = w.shape
    p = n1-m
    
    U,Y = w[:m,:],w[m:,:]
    Hr=blkhank(w,L+1,m1-L)
    R,R_sp,P,Dh,x,x_sp=lra(Hr,Hr.shape[0]-p,tol,delta)
    
    sys_tf=R2tf(R_sp,L,m,dt=1)
    sys_ss = R2ss(R_sp,m,L,1)
    
    if x0==0:
        x0=np.zeros((L*p,1))
    else:
        x0=x0_ss_estimate(sys_ss,L+1,Y,U) 
        
    Uh = np.block([U[:,:L],Dh[-n1:-p,:]])
    t,Yr=cnt.forced_response(sys_ss,U=Uh,X0=x0)
    wh = np.block([[Uh],[Yr]])
    
    M=la.norm(w-wh,'fro')
    fit = 100*(1-(M/la.norm(w-np.mean(w,axis=1).reshape(w.shape[0],1),'fro')))
    
    return M,fit,wh,R_sp,sys_tf,sys_ss,x0

def Sparse_lra_Aut_sysid(w,L,tol=1e-3,delta=1e-3):
    
    n1,m1 = w.shape
    Hr=blkhank(w,L+1,m1-L)
    R,R_sp,P,Dh,x,x_sp=lra(Hr,Hr.shape[0]-n1,tol,delta)
    
    A,C=AutR_2_ss(x_sp,1)
    wh,Xh = Aut_sys_sim(A,C,Hr[:-n1,0:1],m1-L-1)
    wh = np.block([w[:,:L],wh])
    
    M = la.norm(w-wh,'fro')
    fit = 100*(1-(M/la.norm(w-np.mean(w,axis=1).reshape(w.shape[0],1),'fro')))
    
    return M,fit,A,C,wh,Hr,Dh,x_sp

def Ar_Model_lra(X,L,tol,delta):
    
    n,m=X.shape
    H=blkhank(X,L+1,m-L)
    
    R,R_sp,P,Dh,x,x_sp = lra(H,H.shape[0]-n,tol,delta)
    HLx,XL=Dh[:n*L,:],Dh[n*L:,:]
    
    Xp=x_sp.dot(HLx)                         
    error=R_sp.dot(H)
    
    return x,x_sp,R_sp,H,Dh,Xp,error
    
def tol_delta_tuning(w,L,m):
    
    x=np.linspace(0.,0.3,300)
    M=np.zeros(len(x))
    
    for i in np.arange(len(x)):
        
        M[i] = Sparse_lra_sysid(w,L,m,x[i],x[i])[0]
    
    plt.plot(x,M)         


def lag_est(w,Lmax,m,x0=1,tol=1e-3,delta=1e-3):
    
    L=np.arange(2,Lmax)
    M=np.zeros(len(L))
    
    for i in L:
        
        M[i-2] = Sparse_lra_sysid(w,i,m,x0,tol,delta)[0]
        
    with plt.style.context(['science']):
        
      plt.figure(figsize=(9,3))
      plt.plot(L,M,'.-',label='$misfit$') 
      
      plt.title('Estimacion del Lag')
      plt.xlabel('$L$') 
      plt.ylabel('$misfit$')
      plt.legend()
      
      plt.show()    
    
    
def lag_est_aut(w,Lmax,tol,delta):
    
    L=np.arange(2,Lmax)
    M=np.zeros(len(L))
    
    for i in L:
        
        M[i-2] = Sparse_lra_Aut_sysid(w,i,tol,delta)[0]
        
    plt.figure(figsize=(6,2.5))
    plt.plot(L,M,'.-')  
    
def lag_est_ar(yid,yval,Lmax,tol,delta):
    
    L=np.arange(2,Lmax)
    e_id=np.zeros(len(L))
    e_val=np.zeros(len(L))
    
    for i in L:
        
        x,x_sp,R_sp,H,Dh,Xp,error = Ar_Model_lra(yid,i,tol,delta)
        e_id[i-2] = la.norm(error)/np.sqrt(yid.shape[1]-i)
        Hval = blkhank(yval,i+1,yval.shape[1]-i)
        e_val[i-2] = la.norm(R_sp.dot(Hval))/ np.sqrt(yval.shape[1]-i)                                        
    
    with plt.style.context(['science']):
        
      plt.figure(figsize=(9,3))
      plt.plot(L,e_id,'.-',label='$\epsilon_{id}$') 
      plt.plot(L,e_val,'.-',color='r',label='$\epsilon_{val}$')   
      
      plt.title('Estimacion del Lag')
      plt.xlabel('$L$') 
      plt.ylabel('rmse($\epsilon$)')
      plt.legend()
      
      plt.show()
    
def Ar_lra_forecast(sys_ss,yid,yval,e_id,x0,sigma2,M,N,j):
    
    n,m = e_id.shape
    t,Y,X=cnt.forced_response(sys_ss,U=e_id,X0=x0,return_x=True)
    xfcast=X[:,-1]

    if n==1:
        Y=Y.reshape(1,len(Y))
        
    Yj = Y[j:j+1,:]    
        
    Yfcast=np.zeros((yid.shape[0],N,M))
    
    for i in range(M):
        e_val = sigma2*np.random.randn(n,N)
        tfcast,Yfcast0=cnt.forced_response(sys_ss,U=e_val,X0=xfcast)
        Yfcast[:,:,i] = Yfcast0
    
    y_max=np.max(Yfcast[j:j+1,:,:],axis=2)
    y_min=np.min(Yfcast[j:j+1,:,:],axis=2)
    y_mean=np.mean(Yfcast[j:j+1,:,:],axis=2)
    
    Ytot=np.append(Yj,y_mean)
    t1=np.arange(t[-1]+1,t[-1]+N+1)
    t_total=np.append(t,t1)

    with plt.style.context(['science']):
        
       plt.figure(figsize=(9,3.5))
       plt.plot(t_total[-2*N:],Ytot[-2*N:],label='$y_{sim}$')
       plt.plot(t1,yval[j,:],'--',color='r',label='$y_{val}$')
       plt.fill_between(t1,y_min[0,:], y_max[0,:],facecolor='k', alpha=0.17)
       
       plt.title('Simulación + Predicción')
       plt.xlabel('$t$') 
       plt.ylabel('Amplitud')
       plt.legend()
       
       plt.show()
    
    return t,Yfcast    
    
def sparsity_ss(sys_ss):
    
    X = np.block([[sys_ss.A,sys_ss.B],[sys_ss.C,sys_ss.D]])
    spar=(np.count_nonzero(X)/X.size)*100
    return spar