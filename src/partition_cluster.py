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
    
class PAM(KMedoids):
    """
    see Kaufman, L. and Rousseeuw, P.J. (1987), Clustering by means of Medoids,
    in Statistical Data Analysis Based on the {\displaystyle L{1}}L{1}–Norm and 
    Related Methods, edited by Y. Dodge, North-Holland, 405–416.
    From psudocode found in https://doi.org/10.1007/978-3-030-32047-8_16
    """
    
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
        TD, medoids, non_medoids = self.BUILD()
        TD, self.medoids, non_medoids = self.SWAP(TD, medoids, non_medoids)
        self.clusters = self.assign(self.medoids)
        self.inertia = TD
        return self.inertia, self.medoids
    
    def BUILD(self):
        """
        BUILD phase
        S is the set of selected objects (current medoids)
        U is the set of other points
        U \intersection S  = 0
        """
        S = list()
        #object with global minimum distance
        s0 = np.argmin(np.sum(self.D,axis=1))
        TD = np.min(np.sum(self.D,axis=1))
        S.append(s0)
        points = np.arange(0,self.N,dtype='int')
        mask = np.zeros(self.N,dtype='int')
        mask[s0] = 1
        U = np.ma.array(points,mask=mask)
        while len(S) < self.K:
            g = list()
            for i in range(U.shape[0]):
                I = U[i]
                if I is np.ma.masked:
                    continue
                u_ij = self.D[I]
                U[i] = np.ma.masked
                Dj   = np.min(self.D[U != np.ma.masked][S],axis=0)
                C_ij = np.max(Dj-u_ij,0)
                g.append(np.sum(C_ij))
                U[i] = I
            g = np.asarray(g)
            si = np.argmax(g)
            TD = TD - np.max(g)
            S.append(si)
            U[si] = np.ma.masked
        return TD, S, U
        
    def SWAP(self,TD, S, U):
        """
        swap phase
        """
        self.nswap = 0
        niter = 0
        while True:
            for i, mk in enumerate(S):
                DT_best = 0.
                #for this swap
                for j, xj in enumerate(U[U!=np.ma.masked]):
                    DT = 0.
                    oldcost = np.sum(self.D[U!=np.ma.masked,:][:,mk])
                    newcost = np.sum(self.D[U!=np.ma.masked,:][:,xj])
                    DT = newcost - oldcost
                    #for l, x0 in enumerate(U[U!=np.ma.masked]):
                    #    DT = DT - self.D[x0,mk] + self.D[x0,xj]
                    if DT < DT_best:
                        #update best
                        DT_best = DT
                        m_best = mk
                        i_best = i
                        j_best = j
                        x_best = xj
            niter += 1
            if DT_best >= 0.:
                break
            #swap if DT* < 0
            S[i_best] = x_best
            U[m_best] = m_best
            U[x_best] = np.ma.masked
            S = sorted(S)
            self.nswap += 1
            if niter > self.niter:
                break
        # SWAP Done
        TD = TD + DT_best
        return TD, S, U

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
        
