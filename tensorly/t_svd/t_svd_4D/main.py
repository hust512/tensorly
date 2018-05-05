#!/usr/bin/env python
#_*_coding:utf-8_*_
'fourth tensor SVD'
__author__='hai.li'
import numpy as np
import sys
sys.path.append("/home/haili/tensor")
import tn_scalar #ac
def main():
    #create tensor of t

    m,n,k,l=eval(input('enter the m n k l:'))
    print(m,n,k,l)
    t=np.random.random_sample((m*n*k*l))*100
    print(t)
    print(type(t))

    # tensor to tensor-scalar
    ts=tn_scalar.tensor_scalar(t,m,n,k,l)
    print(ts)

    # a set of tensor-scalar do  fft 
    for i in range(m*n):
        t_array=ts[i*k*l:k*l*(i+1)].reshape(l,k)
        if i==0:
            tf=np.fft.fft2(t_array)
        else:
            tf=np.vstack((tf,np.fft.fft2(t_array)))
    print(tf,type(tf))
    tf=tf.flatten()
    print(tf,type(tf))

    # tensor-scalar to tensor
    tf1=tn_scalar.scalar_tensor(tf,m,n,k,l)
    print('tf1',tf1)

    # a set of M do SVD
    for i in range(k*l):
        u,s,v=np.linalg.svd(tf1[m*n*i:m*n*(i+1)].reshape((m,n)))
        if i==0:
            U,S,V=u,s,v
        else:
            U,V,S=np.vstack((U,u)),np.vstack((V,v)),np.vstack((S,s))
    print('U…………………………\n',U) 
    print('S…………………………\n',S) 
    print('V…………………………\n',V)
    
    #r=np.size(S)//3
    #tn_scalar.compress(S,r)
    #print('compress S\n',S)

    #U*S*V
    if m>n:
        t=n
        S1=S.flatten()
        V1=V.flatten()
        U1=U[:,0:t].flatten()
        for i in range(k*l):
            d=np.dot(U1[m*t*i:m*t*(i+1)].reshape(m,t),np.diag(S1[t*i:t*(i+1)]))
            re=np.dot(d,V1[n*n*i:n*n*(i+1)].reshape(n,n))
            if i==0:
                result=re
            else:
                result=np.vstack((result,re))
    else:
        t=m
        S1=S.flatten()
        U1=U.flatten()
        V1=V.flatten()
        for i in range(k*l):
            d=np.dot(U1[m*m*i:m*m*(i+1)].reshape(m,m),np.diag(S1[t*i:t*(i+1)]))
            re=np.dot(d,V1[n*n*i:(n*n*i+t*n)].reshape(t,n))
            if i==0:
                result=re
            else:
                result=np.vstack((result,re))
    result1=[int(result.flatten()[i]-tf1[i]) for i in range(n*m*k*l)]
    print('result\n',result)
    print('result1\n',result1)
if __name__=='__main__':
    main()
