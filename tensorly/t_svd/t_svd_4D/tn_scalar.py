#!/usr/bin/env python3
#_*_coding:utf-8_*_
'the tensor of size m*n*k*l transform to tensor-scalar'
__author__='hai.li'
import numpy as np
def tensor_scalar(t,m,n,k,l):
    ts=np.empty_like(t)
    for i in range(0,m):
        for j in range(0,n):
            for p in range(0,l):
                for q in range(0,k):
                    ts[i*l*n*k+j*l*k+p*k+q]=t[m*n*k*p+m*n*q+n*i+j]
    return ts
def scalar_tensor(t,m,n,k,l):
    ts=np.empty_like(t)
    for i in range(0,m):
        for j in range(0,n):
            for p in range(0,l):
                for q in range(0,k):
                    ts[m*n*k*p+m*n*q+n*i+j]=t[i*l*n*k+j*l*k+p*k+q]
    return ts 
def compress(s,r):
    b=[]
    length=np.size(s)
    for i in range(length):
        b.append(i)
    lis=[(key,val) for key,val in zip(s,b)]
    lis2=sorted(lis,key=lambda x:x[0],reverse=False)
    for i in range(r):
        s[lis2[i][1]]=0
    return s
   
