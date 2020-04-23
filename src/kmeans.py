import numpy as np
import random
import sys

from math import exp,sqrt
from scipy.spatial.distance import cdist,pdist,squareform

# KMeans like methods under the same parent class
# - KMeans
# - KMedoids 
# - KMedians

class PartitionClustering:
    def __init__(self,**kwargs):
        """
        metric is the type of distance used
        boot is the initialization method (random|kmeans++|input array)
        niter is the number of iterations for each run
        conv is the convergence criterion
        nrun is the number of restarts (run with lowest SSE is returned; not for PAM)       
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
            "K"         : None,
            "random_state": None
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
        if self.random_state is not None:
            random.seed(self.random_state)         
 
    def do_clustering(self, X=None, D=None, W=None):
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
        SSE_run = np.empty(self.nrun)
        centers_run  = list()
        clusters_run = list()
        iter_run     = list()
        for run in range(self.nrun):
            centers = self.init_clustering()
            clusters, centers, sse, it = self.main_loop(centers)
            SSE_run[run]  = sse
            clusters_run.append(clusters)
            centers_run.append(centers)
            iter_run.append(it)
        minRun = np.argmin(SSE_run)
        self.clusters = clusters_run[minRun]
        self.centers  = centers_run[minRun]
        self.inertia = SSE_run[minRun]
        self.final_iter = iter_run[minRun]
        del clusters_run, centers_run, SSE_run
        return self.inertia, self.final_iter
   
    def assign(self):
        return None
    
    def newcenters(self):
        return None
    
    def init_clustering(self):
        return None
                          
class KMeans(PartitionClustering):
    """
    K-Means clustering
    """
    def boot_random(self):
        """
        standard initialization with random 
        choice of coordinates or medoids
        """
        centers = random.sample(list(range(self.N)),self.K)
        return self.X[centers]
        
    def kmeanspp(self):
        """
        initialization with kmeans++
        """
        #choose 1st center
        centers = list()
        centers.append(np.random.randint(0,self.N))
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
                print("Error in kmeans++",cumprob,coin)
            centers.append(ind)
        centers = self.X[centers]
        return centers
    
    def init_clustering(self):
        if self.boot == 'random':
            centers = self.boot_random()
        elif self.boot == 'kmeans++':
            centers = self.kmeanspp()
        else:
            raise ValueError('init method not supported')
        return centers
    
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
            centers[i] = np.average(points,axis=0,weights=self.W)
        # having assigned centers, calculate cost
        sse = .0
        dist = cdist(self.X,centers,metric=self.metric)
        for i in range(self.K):
            sse = sse + np.sum(np.sum(dist[clusters==i,i]**2,axis=0))                        
        return sse,centers    
    
    def assign(self,centers):
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
    
    def main_loop(self, centers):
        sse_prev = 0.
        for it in range(self.niter):
            clusters = self.assign(centers)
            sse,centers = self.newcenters(clusters)
            conv = abs(sse - sse_prev)
            if conv <= self.conv and it > 3:
                break
        return clusters, centers, sse, it
                
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
    
class KMedoids(PartitionClustering):
    """
    KMeans like solution of KMedoids
    problem
    This is NOT Partition Around Medoids (PAM)
    but a different solution of the same problem
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
           
    def boot_random(self):
        """
        standard initialization with random 
        choice of coordinates or medoids
        """
        if self.random_state is not None:
            random.seed(self.random_state)
        centers = random.sample(list(range(self.N)),self.K)
        return centers
    
    def kmeanspp(self):
        """
        initialization with kmeans++
        """
        centers = list()
        centers.append(np.random.randint(0,self.N))
        while len(centers) < self.K:
            D = np.array([min([d[c] for c in centers]) for d in self.D])
            cumprob = (D/D.sum()).cumsum()
            coin = np.random.random()
            try:
                ind = np.where(cumprob >= coin)[0][0]
            except:            
                print("Error in kmeans++",cumprob,coin)
            centers.append(ind)
        return centers
    
    def init_clustering(self):
        if self.boot == 'random':
            centers = self.boot_random()
        elif self.boot == 'kmeanspp':
            centers = self.kmeanspp()
        else:
            raise ValueError('init method not supported')
        return centers
    
    def calc_cost(self, points, medoid):
        """
        compute cost with current medoids
        """
        cost = .0
        if self.W == None:
            cost = cost + np.sum(self.D[points,:][:,medoid])
        else:
            cost = cost + np.sum(self.W[points]*self.D[points,:][:,medoid])/np.sum(self.W[points])
        return cost
    
    def assign(self,centers):
        """ 
        assign points to nearest Medoid
        """
        D = self.D[:,centers]
        labels = np.argmin(D,axis=1)
        clusters = np.array([centers[l] for l in labels])
        return clusters
       
    def newcenters(self,clusters):
        """
        search new centers and calculate cost
        for swapping
        """
        oldcenters = list(set(clusters))
        centers = list(set(clusters))
        for i, oc in enumerate(oldcenters):
            non_medoids =  np.where(clusters==oc)[0]
            oldcost = self.calc_cost(non_medoids, oc)
            for nm in non_medoids:
                newcost = self.calc_cost(non_medoids, nm)
                if newcost < oldcost:
                    oldcost = newcost
                    centers[i] = nm
        cost = 0.
        for ce in centers:
            non_medoids =  np.where(clusters==ce)[0]            
            cost = cost + self.calc_cost(non_medoids, ce)
        return cost, centers
    
    def main_loop(self, centers):
        cost_prev = 0.
        for it in range(self.niter):
            clusters = self.assign(centers)
            cost,centers = self.newcenters(clusters)
            conv = abs(cost - cost_prev)
            if conv <= self.conv and it > 3:
                break
        return clusters, centers, cost, it
