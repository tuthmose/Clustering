import numpy as np
import sys
from math import exp,sqrt
from numpy.random import choice,randint,seed
from scipy.spatial.distance import cdist,pdist,squareform


class PartitionClustering:
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
            "metric"    : "euclidean",
            "boot"      : "random",
            "niter"     : 500,
            "conv"      : 1e-5,
            "nrun"      : 10,
            "voronoi"   : False,
            "X"         : None,
            "D"         : None,
            "K"         : None
        }
        for (prop, default) in prop_defaults.items():
            setattr(self, prop, kwargs.get(prop, default))         
        # check some input
        if not isinstance( self.boot, np.ndarray ):
            assert isinstance( self.K, int )
        else:
            self.K = self.boot.shape[0]
        assert isinstance( self.nrun, int )
        assert isinstance( self.niter, int )
 
    def do_clustering(self):
        """
        initialize and run main loop
        """
        self.init2()
        SSE_run = np.empty(self.nrun)
        centers_run  = list()
        clusters_run = list()
        for run in range(self.nrun):
            centers = self.init_clustering()
            sse_prev = .0
            clusters = np.empty(self.N,dtype='int')
            for it in range(self.niter):
                clusters = self.assign(centers,clusters,it)
                sse,centers  = self.newcenters(clusters)
                if abs(sse - sse_prev) <= self.conv and it > 1:
                    break
                else:
                    sse_prev = sse
            SSE_run[run]  = sse
            clusters_run.append(clusters)
            centers_run.append(centers)                    
        minRun = np.argmin(SSE_run)
        self.clusters = clusters_run[minRun]
        self.centers  = centers_run[minRun]
        self.inertia = SSE_run[minRun]
        del clusters_run, centers_run, SSE_run
        return self.inertia
   
    def assign(self):
        return None
    
    def newcenters(self):
        return None
    
    def init_clustering(self):
        """
        select centers to initialize KMeans
        """
        method = type(self).__name__
        if  isinstance( self.boot, np.ndarray ):
            if method is 'PAM':
                return self.boot
            else:    
                return self.X[self.boot]            
        if self.boot=='random':
            centers  = self.boot_random(method)
            return centers
        elif self.boot == 'kmeans++':
            #raise ValueError("kmeans++ nyi")
            try:
                centers = self.kmeanspp(method)
            except:
                centers = self.boot_random()
            return centers
        elif isinstance(self.boot,np.ndarray) and method is not 'PAM':
            assert self.boot.shape == (self.K,self.nfeatures) 
        elif isinstance(self.boot,np.ndarray) and method is 'PAM':
            assert self.boot.shape == (self.K)
            
    def boot_random(self,algo):
        """
        standard initialization with random 
        choice of coordinates or medoids
        """
        seed()
        centers = list()
        centers.append(randint(0,self.N))
        while len(centers) < self.K:
            newc = randint(0,self.N)
            if not newc in centers:
                centers.append(newc)
        if algo is 'PAM':
            return centers
        else:    
            return self.X[centers]
        
    def kmeanspp(self,algo):
        """
        initialization with kmeans++
        """
        #choose 1st center
        seed()
        centers = list()
        centers.append(randint(0,self.N))
        # squared distances; we start working only with points 
        # in the dataset
        # the probability of picking a point x as new center
        # is ~ to the distance from the nearest already picked center
        #https://datasciencelab.wordpress.com/2014/01/15/improved-seeding-for-clustering-with-k-means/
        while len(centers) < self.K:
            if self.X is not None:
                D2 = np.array([min([np.linalg.norm(x-c)**2 for c in centers]) for x in self.X])
            else:
                D2 = np.array([min([d[c]**2 for c in centers]) for d in self.D])
            cumprob = (D2/D2.sum()).cumsum()
            coin = np.random.random()
            try:
                ind = np.where(cumprob >= coin)[0][0]
            except:            
                print(cumprob,coin)
                quit()
            centers.append(ind)
        if algo is not 'PAM':
            centers = self.X[centers]
        return centers
            
class KMeans(PartitionClustering):
    """
    K-Means clustering
    """
    
    def init2(self):
        """
        check some input specific for KMeans
        """
        assert isinstance(self.X,np.ndarray)
        self.N = self.X.shape[0]
        self.nfeatures = self.X.shape[1]
        #we need coordinates for KMeans
       
    def newcenters(self,clusters):
        """
        calculate coordinates of new centers, given
        clusters,  and calculate 
        Sum of Squared Errors
        """
        centers = np.empty((self.K,self.nfeatures))
        for i in range(self.K):
            points = self.X[clusters==i]
            centers[i] = np.mean(points,axis=0)
        # having assigned centers, calculate cost
        sse = .0
        dist = cdist(self.X,centers,metric=self.metric)
        for i in range(self.K):
            sse = sse + np.sum(np.sum(dist[clusters==i,i]**2,axis=0))                        
        return sse,centers    
    
    def assign(self,centers,clusters,it):
        """
        assign points to clusters and calculate 
        Sum of Squared Errors
        """
        clusters = np.empty(self.N,dtype='int')
        #https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html
        # calcola le distanze di tutti i punti da tutti i centroidi
        dist = cdist(self.X,centers,metric=self.metric)
        #per ogni punto cerca il centroide più vicino
        for pj in range(self.N):
            nearest = np.argmin(dist[pj])
            #cl = np.unravel_index(nearest,(1,self.K))
            clusters[pj] = nearest
        return clusters
    
class PAM(PartitionClustering):
    """
    K-Medoids, std implementation
    """
    
    def init2(self):
        """
        do some checks
        """
        #with kmedoids we can use distance matrix
        if self.X is not None:
            self.N = self.X.shape[0]
        else:
            self.N = self.D.shape[0]
        if self.metric=="precomputed" and self.D is None:
            raise ValueError("missing precomputed distance matrix")
        #elif self.metric=="precomputed":
        #    print("WARNING: K-Medoids should be used with l1 norm (cityblock)")
        elif self.X is not None:
            self.D = squareform(pdist(self.X,metric=self.metric))  
           
    def assign(self,centers,clusters,it):
        """ 
        assign points to nearest Medoid
        """
        if it == 0 or self.voronoi is True:
            clusters = np.empty(self.N,dtype='int')
            for pj in range(self.N):
                nearest = np.argmin(self.D[centers,:][:,pj])
                clusters[pj] = centers[nearest]
        return clusters
    
    def newcenters(self,clusters):
        """
        search new centers and calculate cost
        to swap and SSE
        """
        sse = .0
        centers = list(set(clusters))
        for ce in centers:
            sse = sse + np.sum(self.D[clusters==ce,:][:,ce]**2)
        tmpcent = list()
        if self.voronoi is False:
            for ce in centers:
                newmed = randint(0,self.N)
                #clusters[clusters==ce] = newmed
                newcost = np.sum(self.D[clusters==ce,:][:,newmed]**2)
                oldcost = np.sum(self.D[clusters==ce,:][:,ce]**2)
                if newcost < oldcost:
                    sse = sse + newcost - oldcost
                    tmpcent.append(newmed)
                else:
                    tmpcent.append(ce)
        else:
            tmpcent = centers
        return sse,tmpcent
            
class KMedians(KMeans):
    """
    use median along each dimension to find
    centroids
    """   
    
    def newcenters(self,clusters):
        """
        calculate coordinates of new centers, given
        clusters,  and calculate 
        Sum of Squared Errors
        """
        centers = np.empty((self.K,self.nfeatures))
        for i in range(self.K):
            points = self.X[clusters==i]
            centers[i] = np.median(points,axis=0)
        # having assigned centers, calculate cost
        sse = .0
        dist = cdist(self.X,centers,metric=self.metric)
        for i in range(self.K):
            sse = sse + np.sum(np.sum(dist[clusters==i,i]**2,axis=0))                        
        return sse,centers 
    
    
