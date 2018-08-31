# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 11:05:08 2018

@author: lsj
"""
import numpy as np
from L_trans import trans, invtrans
from L_svd import L_svd
from L_prod import L_prod
#from proxF import proxF
#from FindX import FindX
def L_svd_compress(M, k2, flag = 'fft'):
   [n1, n2, n3] = M.shape
   n0 = min(n1, n2)
#   if k2>n0 or k2<1:
#       raise ValueError('k2 is illegal')
   U, S, V = L_svd(M, flag = 'fft')
#   U = trans(U, flag)
#   S = np.zeros((S.shape), dtype = complex)
   S = trans(S, flag)
#   V = trans(V, flag)
   sTrueV = np.zeros((n0, n3), dtype = complex)
   for i in range(n3):
       s = S[:, :, i]
       s = np.diag(s)
       sTrueV[:, i] = s
   k2 = FindX(sTrueV, k2)
   sTrueV = proxF(sTrueV, k2)
   for i in range(n0):
       for j in range(n3):
           S[i, i, j] = sTrueV[i, j]
   S = invtrans(S, flag).real
   Mm = L_prod(L_prod(U, S, flag), V, flag).real
#   print(Mm.shape)
   return Mm


def FindX(M, k2):
    M = M.reshape(1, M.size)
#    M = numpy.fromnumeric.sort(M)
    M = np.sort(M)
    return M[0, k2]

def proxF(x, k2):
#    s = 1- min(k2/abs(x), 1)
    x[(k2/abs(x))>1] = 0
#    s = 1-x
#    x = x * s
#    x = x + k2
    return x
