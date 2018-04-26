from numpy import *
import numpy as np
from numpy import linalg as la
from scipy.fftpack import fft,ifft,dct,idct

def dct_trans(y):
    n = max(y.shape)
    X = y
    a = zeros((1, 1, n))
    I = eye(n)
    C=dct(I, axis = 0, norm = 'ortho')
    Z = diag(ones(n-1), 1)
    W = diag(C[:,0])
#   b = np.linalg.inv(W)*C*(Z+I)*X
    b = np.dot(np.dot(np.dot(np.linalg.inv(W),C),(Z+I)),X)
    for i in range(n):
        a[0, 0, i] = b[i]
    return a
    
    
def idct_trans(y):
    n = max(y.shape)
    X = y
    a = zeros((1, 1, n))
    I = eye(n)
    C=dct(I, axis = 0, norm = 'ortho')
    Z = diag(ones(n-1), 1)
    W = diag(C[:,0])
#    b = np.linalg.inv(I+Z)*(C.T*(W*X))
    b = np.dot(np.linalg.inv(I+Z), np.dot(C.T,np.dot(W,X)))
    for i in range(n):
        a[0, 0, i] = b[i]
    return a



def t_svd(M, flag = 'fft'):
	[n1 ,n2 ,n3] = M.shape
	D = zeros((n1 ,n2 ,n3), dtype = complex)
	D = trans(M, flag)
#	print(D)
# 	print(flag)
#   	D = fft(M, axis = -1)
#   	print(D.shape)    
	Uf = zeros((n1,n1,n3), dtype = complex)
	Thetaf = zeros((n1,n2,n3), dtype = complex)
	Vf = zeros((n2,n2,n3), dtype = complex)	

	for i in range(n3):
		temp_U ,temp_Theta, temp_V = la.svd(D[: ,: ,i], full_matrices=True);
		Uf[: ,: ,i] = temp_U;
		Thetaf[:n2, :n2, i] = np.diag(temp_Theta)
		Vf[:, :, i] = temp_V;
	U = zeros((n1,n1,n3))
	Theta = zeros((n1,n2,n3))
	V = zeros((n2,n2,n3))
#	U = ifft(Uf, axis = 2).real
#	Theta = ifft(Thetaf, axis = -1).real
#	V = ifft(Vf, axis = -1).real
	U = invtrans(Uf, flag).real
	Theta = invtrans(Thetaf, flag).real
	V = invtrans(Vf, flag).real
	return U, Theta, V
 
 
def my_dct(A):
    [n1, n2, n3] = A.shape
    a = zeros((n1, n2, n3))
    for i in range(n1):
        for j in range(n2):
            g = dct_trans(A[i, j, :])
            a[i, j, :] = g
    return a
def my_idct(A):
    [n1, n2, n3] = A.shape
    a = zeros((n1, n2, n3))
    for i in range(n1):
        for j in range(n2):
            g = idct_trans(A[i, j, :])
            a[i, j, :] = g
    return a
 #########################################################
 
def trans(A, flag = 'fft'):
    [n1, n2, n3] = A.shape
    a = np.zeros((n1, n2, n3))
    if flag == 'fft':
        a = fft(A, axis = -1)
    elif flag  == 'dct':
        a = my_dct(A)
#        print(a.dtype, flag)
    return a
    
    
def invtrans(A, flag = 'fft'):
    [n1, n2, n3] = A.shape
    a = np.zeros((n1, n2, n3))
    if flag  == 'fft':
        a = ifft(A, axis = -1)
    elif flag == 'dct':
        a = my_idct(A)
#        print(a.dtype, flag)        
    return a

###############################################################
'''
def t_prod(A, B):
	[a1 ,a2 ,a3] = A.shape
	[b1 ,b2 ,b3] = B.shape
#    A = trans(A)
#    B = trans(B)
	A = fft(A, axis = -1)
	B = fft(B, axis = -1)
	C = np.zeros((a1,b2,b3), dtype = complex)
	for i in arange(b3):
		C[:,:,i]=np.dot(A[:,:,i],B[:,:,i])
	C = ifft(C, axis = -1)
	return Cm = my_dct(M)
M_inv = my_idct(m)

 '''
def t_prod(A, B, flag = 'fft'):
    [a1, a2, a3] = A.shape
    [b1, b2, b3] = B.shape
    A = trans(A, flag)
    B = trans(B, flag)
    C = np.zeros((a1,b2,b3), dtype = complex)
    for i in arange(b3):
        C[:, :,i] = np.dot(A[:, :, i], B[:, :, i])
    C = invtrans(C, flag)
    return C
    
#############################################################
#####################################################
M = np.random.random(10*5*2).reshape((10, 5, 2))
a1,b1,c1  = t_svd(M, flag = 'fft')
M_svd = t_prod(t_prod(a1,b1, flag = 'fft'), c1, flag = 'fft')
err = M-M_svd
print(np.linalg.norm(err))
print(np.allclose(M, M_svd))

    


    

    
#Y=np.array(arange(5))
#y=dct_trans(Y)
#Y_inv=idct_trans(y)
#print(np.allclose(Y,Y_inv))

#m = trans(M, flag = 'dct')
#M_inv = invtrans(m, flag = 'dct')
#print(np.allclose(M, M_inv))