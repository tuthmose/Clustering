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
    cdef int i, j, nlabels, N_loc, nthread
    cdef double [:,:] cD
    cdef double DS #, ES
    cdef double [:] ef

    D  = kwargs.get('D')
    X =  kwargs.get('X')
    N_loc = int(kwargs.get('N_loc'))
    try:
        nthread = int(kwargs.get('nthread'))
    except:
        nthread = 4
    
    nlabels = len(labels)
    cD = np.sort(D[labels, :][:, labels], axis=0)
    DS = 0.
    #ES = 0.

    expfact = np.array([2.**(N_loc-i-1) for i in range(N_loc)])
    ef = expfact
    #ES = np.sum(expfact)
    
    for j in prange(N_loc, num_threads=nthread, nogil=True):
        for i in range(nlabels):
            DS += ef[j]*cD[j+1,i]
    
    return DS/(nlabels*N_loc)
               
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
