import numpy as np
import sys
from math import exp,sqrt
from numpy.random import choice,randint,seed
from scipy.spatial.distance import cdist,pdist,squareform

class density_peaks(object):
    def __init__(self,**kwargs):
        prop_defaults = {
            "metric"  : "euclidean",
            "kernel"  : "gaussian",
            "cutoff"  : 0.0,
            "percent" : 2.0,
        }
        for (prop, default) in prop_defaults.items():
            setattr(self, prop, kwargs.get(prop, default))          
        if self.kernel not in kernels:
            raise NotImplementedError("no kernel %s %s in" \
            % (self.kernel,self.__class__.__name__))
        if self.cutoff == 0.0:
            raise ValueError("cutoff must > 0")
        elif self.cutoff=="auto":
            print("Determining cutoff using a % of neighbors=",percent)
            self.cutoff = self.use_percent()

    def use_percent(self):
        """
        calculate average distance so that
        average number of neighbors within
        is perc of total points
        """
        dd = np.sort(self.D.ravel())
        N = dd.shape[0]
        frac = N*self.percent/100. 
        cutoff = dd[int(frac)]
        return cutoff

    def calc_rho(self):
        """
        calculate local density
        """
        rho = np.zeros(self.N)
        for i in range(self.N):
            mydist = self.D[i,:]
            if self.kernel == "flat":
                dens = float(len(mydist[mydist<self.cutoff]))-1.
            elif self.kernel == "gaussian":
                dens = np.sum(np.exp( -(mydist/self.cutoff)**2 ))-1
            rho[i] = dens
        rank = np.argsort(rho)[::-1]
        return rho,rank

    def calc_delta(self):
        """
        loop on ordered densities; for each point find
        minimum distance from other points with higher density
        the point with the highest density is assigned as 
        neighbor to build clusters
        """
        maxd = np.max(self.D)
        nneigh = np.zeros(self.N,dtype=np.int64)
        delta  = maxd*np.ones(self.N)
        delta[self.order[0]] = -1.
        for i in range(1,self.N):
            I = self.order[i]
            Imin = self.order[:i]
            dd = np.min(self.D[I,Imin])
            delta[I] = dd
            nn = self.order[np.where(self.D[I,Imin]==dd)[0][0]]
            nneigh[I] = nn
        delta[self.order[0]] = np.max(delta)
        #delta[self.order[0]] = np.max(self.D[self.order[0]])
        return delta,nneigh

    def decision_graph(self, X=None, D=None):
        """
        calculate decision graph for data set
        """
        # check X and/or D
        if self.metric=="precomputed" and self.D is None:
            raise ValueError("missing precomputed distance matrix")
        elif self.X is not None:
            self.D = squareform(pdist(self.X,metric=self.metric))
        self.N = self.D.shape[0]
        #calculate decision graph
        self.rho,self.order = self.calc_rho()
        self.delta,self.nneigh = self.calc_delta()
        return self.rho,self.delta

    def get_centroids(self,**kwargs):
        """
        search for points above level of mean 
        values of rho and delta for candidate 
        cluster centroids and for outliers
        """
        rmin = 0.0
        dmin = 0.0
        for key,value in kwargs.items():
            if key=="rmin":
                rmin = value
            if key=="dmin":
                dmin = value
        rmask = self.rho   > rmin
        dmask = self.delta > dmin
        self.centroids = np.where(rmask & dmask)[0]
        self.nclusters = len(self.centroids)
        self.points = np.where((~rmask) | (~dmask))[0]
        return self.centroids, self.points

    def assign_points(self):
        """
        assign a point to the same cluster 
        of its nearest neighbor with highest 
        density
        """
        self.clusters = -np.ones(self.N,dtype=np.int64)
        #initialize cluster labels to centroids
        for i in self.centroids:
            self.clusters[i] = i
        for point in self.order:
            if self.clusters[point] == -1:
                self.clusters[point] = self.clusters[self.nneigh[point]]
        if np.any(self.cluster==-1):
            raise ValueError("Error: Unassigned points are present")
        return self.clusters

    def create_halo(self):
        """
        create halo of points for each cluster that
        may be assigned as noise
        """
        self.robust_clusters = -np.ones(self.N,dtype=np.int64)
        rho_b = dict()
        for myc in self.centroids:
            rho_b[myc] = 0.0
            NN = np.asarray(list(range(self.N)))
            intpoints  = self.clusters == myc
            extpoints = ~intpoints
            intpoints = NN[intpoints]
            extpoints = NN[extpoints]
            for i in intpoints:
                for j in extpoints:
                    if self.D[i,j] <= self.cutoff:
                        rho_b[myc] = max(0.5*(self.rho[i]+self.rho[j]),rho_b[myc])
        for p,P in enumerate(self.clusters):
            if self.rho[p] >= rho_b[P]:
                self.robust_clusters[p] = self.clusters[p]
        return self.robust_clusters
    
   
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
    
