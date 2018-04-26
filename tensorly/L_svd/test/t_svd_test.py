# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 16:45:55 2018

@author: lsj
"""

import numpy as np
import sys
sys.path.append("..")
from L_svd import L_svd
from L_prod import L_prod
def t_svd_test():
    M = np.random.random(10*5*2).reshape((10, 5, 2))
    a1,b1,c1  = L_svd(M, flag = 'fft')
    M_svd = L_prod(L_prod(a1,b1, flag = 'fft'), c1, flag = 'fft')
    err = M-M_svd
    print(np.linalg.norm(err))
    print(np.allclose(M, M_svd))
if __name__=='__main__':
    t_svd_test()