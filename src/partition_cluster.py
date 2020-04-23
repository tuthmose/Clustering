import numpy as np
import random
import sys

from math import exp,sqrt
from scipy.spatial.distance import cdist,pdist,squareform

# Partition based clustering methods.
#   from kmeans.py:
#   - kMeans
#   - kMedians
#   - kMedoids
# from pam:
# - PAM (cython serial version)

from kmeans import *
from pam    import *
    
class gromos_clustering:
    """
    See Torda, A. E. & van Gunsteren, W. F. 
    Journal of computational chemistry 15, 1331–1340 (1994).
    A sort of average linkage
    """
    def __init__(self, **kwargs):
        """
        metric is the type of distance used
        X(npoints,nfeatures) is the feature matrix
        D(npoints,npoints) is the distance/dissimilarity (for PAM)
        C is the cutoff
        scaledist is used to std distances 
        """
        prop_defaults = {
            "metric"    : "euclidean",
            "C"         : 1.0,
            "scaledist" : False
        }
        for (prop, default) in prop_defaults.items():
            setattr(self, prop, kwargs.get(prop, default))         
        # check some input
        assert isinstance( self.C, float )
        
    def init2(self, X, D):
        if self.X is not None:
            self.N = self.X.shape[0]
        else:
            self.N = self.D.shape[0]
        if self.metric=="precomputed" and self.D is None:
            raise ValueError("missing precomputed distance matrix")
        elif self.X is not None:
            self.D = pdist(self.X,metric=self.metric)
        if self.scaledist:
            self.D = (self.D - np.mean(self.D))/np.std(self.D)
        self.D = squareform(self.D)
            
    def do_clustering(self, X=None, D=None):
        self.X = X
        self.D = D
        self.init2(X, D)
        clusters = -1*np.ones(self.N,dtype='int')
        #
        a=0
        while True:
            #point non assigned with highest number of neighbours
            new_med = np.argsort(np.count_nonzero(self.D[clusters==-1] <= self.C,axis=0))[-1]
            nn = 0
            for point in range(self.N):
                if self.D[point,new_med] <= self.C and clusters[point]==-1:
                    clusters[point] = new_med
                    nn += 1
            #print(clusters)
            if np.count_nonzero(clusters!=-1) == self.N or nn <= 1:
                break
        #
        self.clusters = clusters
        self.medoids  = set(list(clusters[clusters!=-1]))
        self.singletons = np.where(clusters==-1)[0]
        self.inertia = .0
        for m in self.medoids:
            self.inertia = self.inertia + np.sum(self.D[clusters==m,:][:,m])
        return self.inertia, len(self.medoids), len(self.singletons)
        
