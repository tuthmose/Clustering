#cython: language_level=3

import numpy as np
import random
import sys

from math import exp,sqrt
from scipy.spatial.distance import cdist,pdist,squareform

from cython import cdivision, boundscheck, wraparound
from cython.parallel import prange
    
from libc.math cimport fmin as cmin
from libc.math cimport fmax as cmax

from partition_cluster import *

class PAM(KMedoids):
    """
    see Kaufman, L. and Rousseeuw, P.J. (1987), Clustering by means of Medoids,
    in Statistical Data Analysis Based on the L1–Norm and 
    Related Methods, edited by Y. Dodge, North-Holland, 405–416.
    From pseudocode found in https://doi.org/10.1007/978-3-030-32047-8_16
    """
    def do_clustering(self, X=None, D=None, W=None, doswap=True):
        """
        initialize and run main loop
        X(npoints,nfeatures) is the feature matrix
        D(npoints,npoints) is the distance/dissimilarity (for PAM)
        W(npoints) are the weights
        """
        self.X = X
        self.D = D
        self.W = W
        self.init2()
        if self.N <= self.K:
            raise ValueError("N<K does not make sense")
        medoids, non_medoids = self.BUILD()
        if self.debug:
            self.clusters = self.assign(medoids)
            self.inertia  = 0.
            for m in medoids:
                points = np.where(self.clusters==m)[0]
                self.inertia += self.calc_cost(points, m)
            print("inertia and medoids b4 swap",self.inertia, medoids)
        if doswap:
            medoids = self.SWAP(medoids, non_medoids)
        self.medoids = medoids
        self.clusters = self.assign(medoids)
        self.inertia  = 0.
        for m in medoids:
            points = np.where(self.clusters==m)[0]
            self.inertia += self.calc_cost(points, m)
        return self.inertia, self.medoids
    

    @wraparound(False)  
    @boundscheck(False) 
    def BUILD(self):
        """
        BUILD phase
        S is the set of selected objects (current medoids)
        U is the set of other points
        U \intersection S  = 0
        """
        cdef double [:,:] cD = self.D
        cdef float dmax = np.max(self.D)
        cdef double [:] cG
        cdef int I, i, j, k, imax, M
        cdef float Dj 
        cdef double [:] cW = np.ones(self.N)
        if np.any(self.W):
            cW = self.W
        S = list()
        #object with global minimum distance and distances from it
        s0 = np.argmin(np.sum(self.D,axis=1))
        points = list(range(self.N))
        S.append(s0)
        while len(S) < self.K:
            U = list(set(points).difference(S))
            M = self.N - len(S)
            gain = np.zeros(M)
            cG = gain
            for I in range(0,M,1):
                i = U[I]
                for j in U:
                    if i != j:
                        Dj = dmax
                        for k in S:
                            Dj = cmin(Dj, cD[k,j])
                        cG[I] += cW[i]*cmax(Dj - cD[i,j], 0.)
            # take new centroid
            imax = U[np.argmax(gain)]
            S.append(imax)
        return S, U
        
    @wraparound(False)  
    @boundscheck(False) 
    def SWAP(self, S, U):
        """
        SWAP phase
        take all pairs (i,h) \in U \times S
        compute cost of swapping i and h
        """
        # constants and index variables
        cdef int i, j, h, si, uj, uh
        cdef int iimin, hhmin, imin, hmin
        cdef int nswap=0 
        cdef int L=self.N - self.K
        cdef double Kijh, dd, dmax
        dmax = np.max(self.D)
        # arrays and memoryviews
        DEj = np.empty((L, 2))
        Tih = np.empty((L, self.K))
        cdef double [:,:] cD = self.D
        cdef double [:,:] cDEj 
        cdef double [:,:] cTih 
        cdef double [:] cW = np.ones(self.N)
        if np.any(self.W):
            cW = self.W
        # begin swap phase
        while True:
            DEj = np.sort(self.D[U, :][:, S])[:,:2]
            Tih = np.zeros((L, self.K))
            cTih = Tih
            cDEj = DEj
            # try to swap i and h
            for i in range(0, self.K, 1):
                si = S[i]
                for h in range(0, L, 1):
                    uh = U[h]
                    for j in range(0, L, 1):
                        uj = U[j]
                        if uj != uh:
                            if cD[uj, si] > cDEj[j, 0]:
                                Kijh = cmin(cD[uj, uh] - cDEj[j, 0], 0.)
                            elif cD[uj, si] == cDEj[j, 0]:
                                Kijh = cmin(cD[uj, uh], cDEj[j, 1]) - cDEj[j, 0]
                            cTih[h, i] += (cW[i]/cW[h])*Kijh
            Tmin = np.amin(Tih)
            if Tmin < 0:
                pmin  = np.where(Tih==Tmin)
                hhmin = pmin[0][0]
                iimin = pmin[1][0]
                imin  = S[iimin]
                hmin  = U[hhmin]
                S[iimin] = hmin
                U[hhmin] = imin
                nswap += 1
            else:
                break
        self.nswap = nswap
        return S
