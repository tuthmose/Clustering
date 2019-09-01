import math
import numpy as np
import scipy as sp
import scipy.spatial.distance as distance

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
        
    def find_nbrs(self):
        self.NB = np.zeros((self.N,self.K),dtype='int')
        for i in range(self.N):
            self.NB[i] = np.argsort(self.D[i])[:self.K]
            
    def same_cluster(self, I, J, A):
        #kmin neighbors in common
        common_NN = set(self.NB[I]).intersection(self.NB[J])
        if len(common_NN) >= self.Kmin:
            A[I,J] = 1
            A[J,I] = 1        
            
    def adiancency_matrix(self):
        A = np.eye(self.N,dtype='int')
        for i in range(self.N-1):
            for j in range(i+1,self.N):
                # mutual N. N.
                if i in self.NB[j] and j in self.NB[i]:
                    self.same_cluster(i, j, A)
        return A
    
    def build_clusters(self):
        if self.debug:
            print("Ad. matrix\n",self.A)
        # filt noise:
        noise  = np.where(np.count_nonzero(self.A,axis=0)==1)[0]
        nnoise = len(noise)
        # build clusters
        assigned = list()
        clusters = -np.ones(self.N,dtype='int')
        nclusters = 0
        if self.debug:
            print("elem., assigned, noise")
        for e in range(self.N):
            if e in assigned or e in noise:
                if self.debug:
                    print(e,assigned,noise)
                continue
            members = np.where(self.A[e]>0)[0]
            if len(members) > 0:
                if self.debug:
                    print("members",e,members)
                clusters[members] = nclusters
                assigned = assigned + list(members)
            assigned.append(e)
            nclusters += 1
        print(assigned)
        assert len(assigned) + len(noise) == self.N
        return nclusters, nnoise, clusters
        
    def do_clustering(self, X=None, D=None):
        self.init2(X, D)
        # nearest neighs
        self.find_nbrs()
        #adiancency matrix
        self.A = self.adiancency_matrix()
        # build clusters
        self.nclusters, self.nnoise, self.clusters = self.build_clusters()
        #return
        return self.nclusters, self.nnoise, self.clusters
    
    def clean(self):
        """
        delete results of last clustering
        """
        del self.D, self.A, self.NB, self.nclusters, self.nnoise, self.clusters
    
class brown_martin(jarvis_patrick):
    """
    T is distance threshold in which to look for N. N.
    Rmin is the ratio of lenght of the NN lists
    """
    def __init__(self,**kwargs):
        prop_defaults = {
            "metric"    : "euclidean",
            "T"         : None,
            "Rmin"      : None,
            "debug"     : False
        }
        for (prop, default) in prop_defaults.items():
            setattr(self, prop, kwargs.get(prop, default))         
        # check some input
        assert isinstance( self.T, float )
        assert isinstance( self.Rmin, float )
        
    def find_nbrs(self):
        self.NB = list()
        for i in range(self.N):
            nbrs = np.where(self.D[i] <= self.T)[0]
            self.NB.append(nbrs)
        if self.debug:
            print(self.NB)
        
    def same_cluster(self, I, J, A):
        L1 = len(self.NB[I])
        L2 = len(self.NB[J])
        Lmin = math.floor(self.Rmin*L2)
        common_NN = set(self.NB[I]).intersection(self.NB[J])
        if len(common_NN) >= Lmin:
            A[I,J] = 1
            A[J,I] = 1 

class SNN(jarvis_patrick):
    """
    See L. Ertoz, M. Steinbach, and V. Kumar, 
    A New Shared Nearest Neighbor Clustering Algorithm and its Applications
    - K is the minimum number of neighbors
    - minPTS is the minimum number of neighbor point with density > epsilon
    - epsilon is the reachbility treshold
    minPTS is the same of DBSCAN
    """
    
    def __init__(self,**kwargs):
        prop_defaults = {
            "metric"    : "euclidean",
            "K"         : None,            
            "minPTS"    : None,
            "epsilon"   : None,
            "debug"     : False
        }
        for (prop, default) in prop_defaults.items():
            setattr(self, prop, kwargs.get(prop, default))         
        # check some input
        assert isinstance( self.K, int )
        assert isinstance( self.minPTS, int )
        assert self.K > self.minPTS
        
    def build_clusters(self):
        clusters = list()
        nclusters = 0
        #assign core points to clustes
        for cp in self.core_points:
            members = np.where(self.SNN[cp,self.core_points] <= self.epsilon)[0]
            
                

    def do_clustering(self, X=None, D=None):
        self.init2(X, D)
        # nearest neighs
        self.find_nbrs()
        #Shared nearest neighbors graph
        self.SNN = self.snn_matrix()
        # find core points
        self.core_points = self.find_core_points()
        # build clusters
        self.nclusters, self.nnoise, self.clusters = self.build_clusters()
        #return
        return self.nclusters, self.nnoise, self.clusters
    