import numpy as np
import sys
from math import exp,sqrt
import random
from scipy.spatial.distance import cdist,pdist,squareform

# Partition based clustering methods.
# - KMeans
# - KMedoids (PAM)
# - KMedians

# G Mancini September 2019

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
        conv_run     = list()
        iter_run     = list()
        for run in range(self.nrun):
            centers = self.init_clustering()
            sse_prev = .0
            #clusters = np.empty(self.N,dtype='int')
            for it in range(self.niter):
                clusters = self.assign(centers)
                sse,centers  = self.newcenters(clusters)
                conv = abs(sse - sse_prev)
                if conv <= self.conv and it > 3:
                    break
                else:
                    sse_prev = sse
            SSE_run[run]  = sse
            clusters_run.append(clusters)
            centers_run.append(centers)
            conv_run.append(conv)
            iter_run.append(it)
        minRun = np.argmin(SSE_run)
        self.clusters = clusters_run[minRun]
        self.centers  = centers_run[minRun]
        self.inertia = SSE_run[minRun]
        self.final_conv = conv_run[minRun]
        self.final_iter = iter_run[minRun]
        del clusters_run, centers_run, SSE_run
        return self.inertia, self.final_conv, self.final_iter
   
    def assign(self):
        return None
    
    def newcenters(self):
        return None
    
    def init_clustering(self):
        return None
    
class KMeans2(PartitionClustering):
    """
    KMeans like solution of KMedoids
    problem
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
    
    def init_clustering(self):
        if self.boot == 'random':
            centers = self.boot_random()
        else:
            raise ValueError('init method not supported')
        return centers
    
    def assign(self,centers):
        """ 
        assign points to nearest Medoid
        """
        clusters = np.empty(self.N,dtype='int')
        for pj in range(self.N):
            nearest = np.argmin(self.D[centers,:][:,pj])
            clusters[pj] = centers[nearest]
        return clusters
    
    def calc_cost(self,clusters,medoids):
        """
        compute cost with current medoids
        """
        cost = .0
        if self.W == None:
            for m in medoids:
                cost = cost + np.sum(self.D[clusters==m,:][:,m])
        else:
            for m in medoids:
                cost = cost + np.sum(self.W[clusters==m]*self.D[clusters==m,:][:,m])/np.sum(self.W[clusters==m])
        return cost
    
    def newcenters(self,clusters):
        """
        search new centers and calculate cost
        for swapping
        """
        centers = list(set(clusters))
        oldclusters = clusters
        for i,ce in enumerate(centers):
            non_medoids = np.where(clusters[clusters==ce])[0]
            oldcost = self.calc_cost(clusters, centers)            
            for nm in non_medoids:
                if nm == ce:
                    continue
                #swapping
                centers[i] = nm
                tmpclusters = self.assign(centers)
                if len(set(tmpclusters)) < self.K:
                    centers[i] = ce
                    continue
                newcost = self.calc_cost(tmpclusters, centers)
                if newcost > oldcost:
                    centers[i] = ce
                else:
                    clusters = tmpclusters
                    oldcost  = newcost
        clusters = self.assign(centers)                    
        cost = self.calc_cost(clusters, centers)                    
        return cost,centers   
                       
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
    
    
