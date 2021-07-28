import numpy as np
import random
import scipy as sp
               
from cython import cdivision, boundscheck, wraparound
from cython.parallel import prange
from libc.math cimport pow as cpow
    
@wraparound(False)  
@boundscheck(False) 
def frscore(labels, **kwargs):
    """
    dissimilarity score as in 10.1021/acs.jctc.7b00779
    """   
    cdef int i, j, nlabels, M
    cdef int [:] near, L
    cdef double [:,:] cD
    cdef double DS = 0.
    cdef double ef = 0.
    cdef double ES = 0.

    D  = kwargs.get('D')
    NN = kwargs.get('NN')
    N_loc = int(kwargs.get('N_loc'))
    nlabels = len(labels)
    M = N_loc

    if nlabels >= N_loc:
        M = N_loc
    else:
        M = nlabels

    L = np.asarray(labels, dtype=np.intc)
    #neighbours = np.argsort(D, axis=1)[0,:M+1]
    near = np.asarray(NN, dtype=np.intc)[0,:M+1]
    cD = D
    
    for j in range(M):
        ef = cpow(2., M-j)
        ES += ef
        for i in range(nlabels):
            DS += ef*cD[near[j], L[i]]
    DS = DS/(nlabels*M*ES)
    return DS
               
def sumdist(labels, **kwargs):
    """
    this is just the inverse of the sum of distances
    """
    D = kwargs.get('D')
    DT = np.sum(D[labels])
    return 1.0/DT

def gauk(labels, **kwargs):
    D = kwargs.get('D')
    sigma = float(kwargs.get('sigma'))
    return np.exp( (-D**2)/(2.*sigma**2) )
