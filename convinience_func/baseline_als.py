# -*- coding: utf-8 -*-

from scipy import sparse
import numpy as np

def baseline_als(y, lam, p, niter=10):
"""
baseline correction of positive peaks
argument : y     -- data array
           p     -- for asymmetry generally from 0.001 to 0.1
           lam   -- for smoothness generally from 2 to 9 in log scale
           niter -- number of iteration
return   : z     -- smoothed baseline array

Paul H.C. Eilers and Hans F.M. Boelens (2005)
Baseline Correction with Asymmetric Least Squares

"""
    L = len(y)
    D = sparse.csc_matrix(np.diff(np.eye(L), 2))
    w = np.ones(L)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = sparse.linalg.spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z