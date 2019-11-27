import math
import numpy as np
import scipy as sp
import scipy.spatial.distance as distance
import sys
from math import exp,sqrt
from numpy.random import choice,randint,seed
from scipy.spatial.distance import cdist,pdist,squareform

class DBSCAN(object):
    """
    simple DBSCAN implementation
    """
    def __init__(self,**kwargs):
        """
        metric is the type of distance used
        minPTS is the minimum number of points to be considered core point
        epsilon minimum distance to be considered neighbors
        """
        prop_defaults = {
            "metric"  : "euclidean",
            "minPTS"  : None,
            "epsilon" : None
        }
        for (prop, default) in prop_defaults.items():
            setattr(self, prop, kwargs.get(prop, default))         
        # check some input
        assert isinstance( self.minPTS, int )
        assert isinstance( self.epsilon, float )

    def init2(self, X, D):
        # check X and/or D
        if self.metric=="precomputed" and D is None:
            raise ValueError("missing precomputed distance matrix")
        elif X is None:
            raise ValueError("Provide either a feature matrix or a distance matrix")
        else:
            D = squareform(pdist(X,metric=self.metric))
        self.N = D.shape[0]
        return D
            
    def do_clustering(self, X=None, D=None):
        """
        clusters (-1, or cluster ID, 0:N-1), cluster number (start from 0)
        X(npoints,nfeatures) is the feature matrix
        D(npoints,npoints) is the distance/dissimilarity matrix
        """
        D = self.init2(X,D)
        clusters = -2 * np.ones(self.N,dtype='int')
        nclusters = 0
        for point in range(self.N):
            if clusters[point] != -2:
                #already assigned
                continue
            neighbors = self.find_neighbors(point,D)
            if len(neighbors) < self.minPTS:
                # a noise point (can be assigned as leaf later on)
                clusters[point] = -1
            elif len(neighbors) >= self.minPTS:
                #we have Unassigned point with enough density -> new cluster
                nclusters = self.grow_cluster(point,neighbors,nclusters,clusters,D)
        noise  = np.where(clusters==-1)[0]
        return nclusters, len(noise), clusters
            
    def grow_cluster(self,point,neighbors,nclusters,clusters,D):
        # now seach in all neighborhoods for connected points
        clusters[point] = nclusters
        pcounter = 0
        queue = list(neighbors)
        while pcounter < len(queue):
            point = queue[pcounter]
            if clusters[point] == -1:
                # a density reachable point (leaf)
                 clusters[point] = nclusters
            elif clusters[point] == -2:
                #another unassigned cluster
                neighbors = self.find_neighbors(point,D)
                if len(neighbors) >= self.minPTS:
                    #another core point; add the eps-neighborhood
                    #to the queue
                    clusters[point] = nclusters
                    queue = queue + list(neighbors)
            pcounter += 1
        # add the point eps-neighborhood to the cluster
        clusters[queue] = nclusters       
        nclusters += 1
        return nclusters
    
    def find_neighbors(self,point,D):
        Dp = np.where(D[:,point] <= self.epsilon)
        return Dp[0]
