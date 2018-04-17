from numpy import *
import numpy as np
from numpy import linalg as la
from scipy.fftpack import fft,ifft

def t_svd( M):
	[n1 ,n2 ,n3] = M.shape
	D = zeros((n1 ,n2 ,n3), dtype = complex)
   	D = fft(M, axis = -1)
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
	U = ifft(Uf, axis = 2).real
	Theta = ifft(Thetaf, axis = -1).real
	V = ifft(Vf, axis = -1).real
	return U, Theta, V

###############################################################
def t_prod(A, B):
	[a1 ,a2 ,a3] = A.shape
	[b1 ,b2 ,b3] = B.shape
	A = fft(A, axis = -1)
	B = fft(B, axis = -1)
	C = np.zeros((a1,b2,b3), dtype = complex)
	for i in arange(b3):
		C[:,:,i]=np.dot(A[:,:,i],B[:,:,i])
	C = ifft(C, axis = -1)
	return C

#############################################################