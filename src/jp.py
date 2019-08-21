import numpy as np
import sys
from math import exp,sqrt
from numpy.random import choice,randint,seed
from scipy.spatial.distance import cdist,pdist,squareform
from sklearn import neighbors


class Jarvis_Patrick:
    """
    Jarvis Patrick clustering
    """
    
    def __init__(self,**kwargs):
        """
        metric is the type of distance used
        K is the minimum number of neighbors
        link_str is (simple|weighted) to calculate the weight of edges
        simple if just the number of common neighbors
        weights is calculated according to 
        L. Ertoz, M. Steinbach, and V. Kumar, 
        Workshop on clustering high dimensional data and its applications
        minPTS is the minimum number of neighbor point with density > epsilon
        epsilon is the minimum XXX
        epsilon is used only by SNN
        """
        prop_defaults = {
            "metric"    : "euclidean",
            "leaf_size" : 40,
            "minPTS"    : None,
            "epsilon"   : None,
            "K"         : None,
            "link_str"  : "simple"
        }
        for (prop, default) in prop_defaults.items():
            setattr(self, prop, kwargs.get(prop, default))         
        # check some input
        assert isinstance( self.K, int )
        assert isinstance( self.minPTS, int )
        assert self.K > self.minPTS
        assert isinstance( self.leaf_size, int )
        assert self.link_str in ("simple","weighted")
        #if self.metric is None:
         #   raise ValueError("metric must be provided")
        
    def calc_link_st(self,neigh1,neigh2):
        """
        calculate strenght of link between nodes and 1 from
        the number of common neighbors
        """
        if self.link_str == "simple":
            count = len(set(neigh1).intersection(neigh2))
        elif self.link_str == "weighted":
            shared = list(set(neigh1).intersection(neigh2))
            count = 0
            for l in shared:
                count += (self.K-neigh1.index(l)-1)*(self.K-neigh1.index(l)-1)
        return count
    
    def train(self,X=None):
            """
            initialize and run main loop
            """
            #build kd tree of nearest neighbours
            assert(isinstance(X,np.ndarray))
            npoints = X.shape[0]
            kdtree = neighbors.KDTree(X,metric=self.metric,leaf_size=self.leaf_size)
            #knn_idx includes the point with at least K neighbors
            knn_d, knn_idx = kdtree.query(X,self.K)
            nngraph = np.zeros(npoints*(npoints-1)//2,dtype='int')
            #nngraph = np.zeros((npoints,npoints),dtype='int')
            # similarity of nodes i and j
            for i in range(npoints-1):
                for j in range(i+1,npoints):
                    nngraph[i+j*(j-1)//2] = self.calc_link_st(knn_idx[i],knn_idx[j])
                    #nngraph[i,j] = self.calc_link_st(knn_idx[i],knn_idx[j])
            # if similarity is enoungh join otherwise node is a singleton (noise)
            clusters = dict()
            nclusters = 0
            for i in range(npoints-1):
                for j in range(i+1,npoints):
                    if nngraph[i+j*(j-1)//2] >= self.minPTS:
                        if i in clusters and j not in clusters:
                            clusters[j] = clusters[i]
                        elif i not in clusters and j in clusters:
                            clusters[i] = clusters[j]
                        else:
                            nclusters += 1
                            clusters[i] = nclusters
                            clusters[j] = nclusters
            self.labels = list()
            for i in range(npoints):
                if i in clusters.keys():
                    self.labels.append(clusters[i])
                else:
                    self.labels.append(-1)
            self.labels = np.asarray(self.labels)
            return nclusters

        
                        
