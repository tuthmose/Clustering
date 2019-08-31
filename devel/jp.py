import numpy as np
import scipy as sp
import scipy.spatial.distance as distance
from sklearn.neighbors import NearestNeighbors

class jarvis_patrick(object):
    """
    Jarvis Patrick clustering
    basic algorithm
    """

    def __init__(self, **kwargs):
        """
        metric is the type of distance used
        K is the number of nearest neighbors evaluated
        Kmin is minimum number of shared NN needed for two 
        points to be in the same cluster
        the other keywords are arguments to sklearn.nn
        """
        prop_defaults = {
            "metric"    : "euclidean",
            "K"         : None,
            "Kmin"      : None,
            "debug"     : False
        }
        for (prop, default) in prop_defaults.items():
            setattr(self, prop, kwargs.get(prop, default))         
        # check some input
        assert isinstance( self.K, int )
        assert isinstance( self.Kmin, int )

    def init2(self, X, D):
        # check X and/or D
        self.X = X
        self.D = D
        if self.metric=="precomputed" and self.D is None:
            raise ValueError("missing precomputed distance matrix")
        elif self.X is None:
            raise ValueError("Provide either a feature matrix or a distance matrix")
        else:
            self.D = distance.squareform(distance.pdist(X,metric=self.metric))
        self.N = self.D.shape[0]        
        
    def adiancency_matrix(self):
        A = np.eye(self.N,dtype='int')
        for i in range(self.N-1):
            for j in range(i+1,self.N):
                # mutual N. N.
                if i in self.NB[j] and j in self.NB[i]:
                    #kmin neighbors in common
                    common_NN = set(self.NB[i]).union(self.NB[j])
                    if len(common_NN) >= self.Kmin:
                        A[i,j] = 1
                        A[j,i] = 1
        return A
    
    def build_clusters(self):
        if self.debug:
            print(self.A)
        # filt noise:
        noise  = np.where(np.count_nonzero(self.A,axis=0)==1)[0]
        nnoise = len(noise)
        # build clusters
        assigned = list()
        cluster = -np.ones(self.N,dtype='int')
        ncluster = 0        
        for e in range(self.N):
            if e in assigned or e in noise:
                if self.debug:
                    print(e,assigned,noise)
                continue
            members = np.where(self.A[e]>0)[0]
            cluster[members] = ncluster
            assigned = assigned + list(members)
            ncluster += 1
        return ncluster, nnoise, cluster
        
    def do_clustering(self, X=None, D=None):
        self.init2(X, D)
        # nearest neighs
        self.NB = np.zeros((self.N,self.K),dtype='int')
        for i in range(self.N):
            self.NB[i] = np.argsort(self.D[i])[:self.K]
        #adiancency matrix
        self.A = self.adiancency_matrix()
        # build clusters
        self.ncluster, self.nnoise, self.cluster = self.build_clusters()
        #return
        return self.ncluster, self.nnoise, self.cluster
    
#class brown_martin(jarvis_patrick):
#    xxx