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
    in Statistical Data Analysis Based on the {\displaystyle L{1}}L{1}–Norm and 
    Related Methods, edited by Y. Dodge, North-Holland, 405–416.
    From psudocode found in https://doi.org/10.1007/978-3-030-32047-8_16
    """
    def do_clustering(self, X=None, D=None, W=None, doswap=True, medbuild=False):
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
        self.D = np.asarray(self.D,dtype='float')
        TD, medoids, non_medoids = self.BUILD()
        if medbuild and doswap:
            self.clusters = self.assign(medoids)
            self.inertia  = 0.
            for m in medoids:
                points = np.where(self.clusters==m)[0]
                self.inertia += self.calc_cost(points, m)
            print("inertia and medoids b4 swap",self.inertia,medoids)
        if doswap:
            medoids = self.SWAP(TD, medoids, non_medoids)
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
        dmax = np.max(self.D)
        #
        S = list()
        #object with global minimum distance and distances from it
        s0 = np.argmin(np.sum(self.D,axis=1))
        TD = np.min(np.sum(self.D,axis=1))
        S.append(s0)
        U = np.arange(0,self.N,dtype='int32')
        U[s0] = -1
        cdef int [:] cU = U
        cdef int i, j, k, imax
        cdef float Dj = dmax
        cdef float gi
        gain = np.zeros(self.N)
        cdef double [:] cG = gain
        #
        while len(S) < self.K:
            for i in range(self.N):
                if cU[i] != -1:
                    g_i = 0.
                    for j in range(self.N):
                        if i != j and cU[j] != -1:
                            Dj = dmax
                            for k in S:
                                Dj = cmin(Dj,cD[k,j])
                            g_i += cmax(Dj-cD[i,j],0.)
                cG[i] = g_i            
            # take new centroid
            imax = np.argmax(gain)
            S.append(imax)
            cU[imax] = -1
            TD = TD - cG[imax]
        nU = [s for s in U if s !=-1 ]
        return TD, S, nU
        
    @wraparound(False)  
    @boundscheck(False) 
    def SWAP(self,TD, S, U):
        """
        SWAP phase
        take all pairs (i,h) \in U \times S
        compute cost of swapping i and h
        """
        self.nswap = 0
        Dj = np.empty((self.N-self.K,2))
        Tih = np.empty((self.N-self.K,self.K))
        #
        cdef int ii,jj,j,hh,h
        cdef int iimin,hhmin,imin,hmin
        cdef int nstep=0, nswap=0
        cdef double [:,:] cT  = Tih
        cdef double [:,:] cD = self.D
        cdef double [:,:] cDj = Dj
        cdef double dji, djh, Kijh
        while True:
            #Tih = np.empty((self.N-self.K,self.K))
            #for jj in range(self.N-self.K):
            #    for j in range(self.K):
            #        cT[jj,j] = 0
            #Dj, Ej
            for jj,j in enumerate(U):
                dd = self.D[j,:][S]
                Dj[jj,0] = np.sort(dd)[0]
                Dj[jj,1] = np.sort(dd)[1]
            # try to swap i and h
            for ii,i in enumerate(S):
                for hh,h in enumerate(U):
                   for jj,j in enumerate(U):
                       if j != h:
                           dji = cD[j,i]
                           djh = cD[j,h]
                           if cDj[j,0] > dji:
                               Kijh = cmin(djh - cDj[jj,0],0.)
                           elif cDj[j,0] == dji:
                               Kijh = cmin(djh,cDj[jj,1]) - cDj[jj,0]
                   cT[ii,hh] += Kijh
            Tmin = np.amin(Tih)
            if Tmin < 0:
                pmin  = np.where(Tih==Tmin)
                iimin = pmin[0][0]
                hhmin = pmin[1][0]
                imin  = S[iimin]
                hmin  = U[hhmin]
                S[iimin] = hmin
                U[hhmin] = imin
                nswap += 1
            elif not np.any(Tih) <= 0:
                break
            if nstep > self.niter:
                break
            nstep += 1
        self.nstep = nstep
        self.nswap = nswap
        return S
