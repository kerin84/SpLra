#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 00:56:48 2022

@author: kerin
"""
import numpy as np

def blkhank(w,i,j):
    
    if w.ndim == 3 :
        
        q,N,T = w.shape
        
        H=np.zeros((i*q,j*N))
        
        for ii in range(1,i+1):
        
            for jj in range(1,j+1):
            
                H[(ii-1)*q :ii*q,(jj - 1)*N:(jj*N)] = w[:,:,ii+jj-2]
    
    else:
        
        q,T=w.shape
        if T<q:
           w=w.T
           q,T=w.shape
       
        H=np.zeros((i*q,j))
    
        for ii in range(1,i+1):
        
            H[((ii-1)*q):(ii*q),:]=w[:,(ii-1):(ii+j-1)]
        
    return  H        
    
    