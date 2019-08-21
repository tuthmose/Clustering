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
            "X"  : None,
            "D" : None
        }
        for (prop, default) in prop_defaults.items():
            setattr(self, prop, kwargs.get(prop, default))         
        if self.metric=="precomputed" and self.D is None:
            raise ValueError("missing precomputed distance matrix")
        elif self.X is not None:
            self.D = squareform(pdist(self.X,metric=self.metric))  
        if self.kernel not in kernels:
            raise NotImplementedError("no kernel %s %s in" \
            % (self.kernel,self.__class__.__name__))
        self.N = self.D.shape[0]
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
        neighbor to build cluster
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

    def decision_graph(self):
        """
        calculate decision graph for data set
        """
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
        if np.any(self.clusters==-1):
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
        boot is the initialization method (random|kmeans++|input array)
        niter is the number of iterations for each run
        conv is the convergence criterion
        nrun is the number of restarts (run with lowest SSE is returned)
        voronoi is to use PAM or Voronoi iteration        
        X(npoints,nfeatures) is the feature matrix
        D(npoints,npoints) is the distance/dissimilarity (for PAM)
        K is the number of clusters
        """
        prop_defaults = {
            "metric"  : "euclidean",
            "X"       : None,
            "D"       : None,
            "minPTS"  : None,
            "epsilon" : None
        }
        for (prop, default) in prop_defaults.items():
            setattr(self, prop, kwargs.get(prop, default))         
        # check some input
        assert isinstance( self.minPTS, int )
        assert isinstance( self.epsilon, float )
        if self.X is None and self.D is None:
            raise ValueError("Either features or distances must be provided")
        elif self.X is None:
            assert isinstance(D,np.ndarray)
        else:
            assert isinstance(self.X,np.ndarray)
            self.D = squareform(pdist(self.X,metric=self.metric))  
        self.N = self.D.shape[0]
            
    def do_clustering(self):
        """
        labels (-1, or cluster ID), visited (0/1) and cluster number (start from 0)
        """
        self.labels = np.zeros(self.N,dtype='int')
        C = 0
        for p in range(self.N):
            if self.labels[p] != 0:
                continue
            neighs = self.regionQuery(p)
            if len(neighs) == 0:
                self.labels[p] = -1
                continue
            #we have Unassigned point with enough density -> 
            #new cluster
            C += 1
            self.labels[p] = C
            self.growCluster(C,p,neighs)
            
    def growCluster(self,C,p,neighbors):
        """
        add point to a cluster or create a new one
        every point found among neighbor must
        be visit
        """
        queue = list()
        queue.append(p)
        to_visit = 0
        while to_visit < len(queue): 
            neighs_2 = self.regionQuery(p)
            if len(neighs_2) == 0:
                to_visit += 1
                continue
            for n in neighs_2:
                #point was not dense but density connected
                # or it was still to visit
                if self.labels[n] <= 0:
                    self.labels[n] = C
                    if self.labels[n] == 0:
                        queue.append(n)
            to_visit += 1
    
    def regionQuery(self,point):
        Dp = np.where(self.D[:,point] <= self.epsilon)
        return Dp[0]
    
