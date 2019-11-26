import math
import numpy as np
import scipy as sp
import scipy.spatial.distance as distance
import sys
from math import exp,sqrt
from numpy.random import choice,randint,seed
from scipy.spatial.distance import cdist,pdist,squareform

# Density based cluster methods:
# Density peaks
# DBSCAN
# Jarvis Patrick
# Shared nearest neighbors

# G Mancini, September 2019

# a scikit.learn KDTree should be used but it does not
# take precomputed distances
# btw you can pass the SNN graph to sklearn.cluster.DBSCAN
# with metric=precomputed

class density_peaks(object):
    """
    Rodriguez, A.; Laio, A. 
    Clustering by Fast Search and Find of Density Peaks. 
    Science 2014, 344 (6191), 1492–1496. 
    https://doi.org/10.1126/science.1242072.
    """
    def __init__(self,**kwargs):
        prop_defaults = {
            "metric"  : "euclidean",
            "kernel"  : "gaussian",
            "cutoff"  : 0.0,
            "percent" : 2.0,
        }
        for (prop, default) in prop_defaults.items():
            setattr(self, prop, kwargs.get(prop, default))
        kernels = ["flat","gaussian"]
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

    def calc_rho(self, D):
        """
        calculate local density
        """
        rho = np.zeros(self.N)
        for i in range(self.N):
            mydist = D[i,:]
            if self.kernel == "flat":
                dens = float(len(mydist[mydist<self.cutoff]))-1.
            elif self.kernel == "gaussian":
                dens = np.sum(np.exp( -(mydist/self.cutoff)**2 ))-1
            rho[i] = dens
        rank = np.argsort(rho)[::-1]
        return rho,rank

    def calc_delta(self, D):
        """
        loop on ordered densities; for each point find
        minimum distance from other points with higher density
        the point with the highest density is assigned as 
        neighbor to build clusters
        """
        maxd = np.max(D)
        nneigh = np.zeros(self.N,dtype=np.int64)
        delta  = maxd*np.ones(self.N)
        delta[self.order[0]] = -1.
        for i in range(1,self.N):
            I = self.order[i]
            Imin = self.order[:i]
            dd = np.min(D[I,Imin])
            delta[I] = dd
            nn = self.order[np.where(D[I,Imin]==dd)[0][0]]
            nneigh[I] = nn
        delta[self.order[0]] = np.max(delta)
        #delta[self.order[0]] = np.max(self.D[self.order[0]])
        return delta,nneigh

    def decision_graph(self, X=None, D=None):
        """
        calculate decision graph for data set
        """
        # check X and/or D
        if self.metric=="precomputed" and D is None:
            raise ValueError("missing precomputed distance matrix")
        elif X is not None:
            D = squareform(pdist(X,metric=self.metric))
        self.N = D.shape[0]
        #calculate decision graph
        self.rho,self.order = self.calc_rho(D)
        self.delta,self.nneigh = self.calc_delta(D)
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

    def create_halo(self, D):
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
                    if D[i,j] <= self.cutoff:
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

class jarvis_patrick(object):
    """
    Jarvis Patrick clustering
    basic algorithm.
    See  Jarvis, R. A.; Patrick, E. A. 
    Clustering Using a Similarity Measure Based on 
    Shared Nearest Neighbors IEEE Trans. Comput. 1973, 
    C22, 1025-1034.
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
            "debug"     : False,
            "saveA"     : False
        }
        for (prop, default) in prop_defaults.items():
            setattr(self, prop, kwargs.get(prop, default))         
        # check some input
        assert isinstance( self.K, int )
        assert isinstance( self.Kmin, int )
        assert self.K >= self.Kmin

    def init2(self, X, D):
        """
        preprocess initial data and calculate
        distance matrix if necessary
        """
        if self.metric=="precomputed" and D is not None:
            pass
        elif self.metric=="precomputed" and D is None:
            raise ValueError("missing precomputed distance matrix")
        elif X is None:
            raise ValueError("Provide either a feature matrix or a distance matrix")
        else:
            D = distance.squareform(distance.pdist(X,metric=self.metric))
        self.N = D.shape[0]
        return D
        
    def find_nbrs(self,D,W=None):
        """
        find the kNN of all points
        """        
        NB = np.zeros((self.N,self.K),dtype='int')
        if W is None:
            for i in range(self.N):
                NB[i] = np.argsort(D[i])[:self.K]
        else:
            for i in range(self.N):
                NB[i] = np.argsort((1./W[i])*D[i])[:self.K]            
        return NB
            
    def same_cluster(self, I, J, A, NB):
        """
        kNN graph; create a link between two points
        if the share at least Kmin neighbors 
        **including themselves**
        not weighted
        """
        common_NN = set(NB[I]).intersection(NB[J])
        #print(I,J,common_NN,NB[I],NB[J])
        if len(common_NN) >= self.Kmin:
            A[I,J] = 1
            A[J,I] = 1
            
    def adiancency_matrix(self, NB):
        """
        build kNN graph
        """        
        A = np.eye(self.N,dtype='int')
        for i in range(self.N-1):
            for j in range(i+1,self.N):
                # mutual N. N.
                if i in NB[j] and j in NB[i]:
                    self.same_cluster(i, j, A, NB)
        return A
    
    def grow_cluster(self,point, members, nclusters, clusters, A):
        """
        given a point with some shared neighbors
        create a new cluster and add all linked points
        seach in neighborhoods of all connected points
        """        
        clusters[point] = nclusters
        pcounter = 0
        queue = list(members)
        while pcounter < len(queue):
            point = queue[pcounter]
            #cannot have clusters == -1 with simple
            #neighbors
            if clusters[point] == -2:
                #another unassigned point
                members = np.where(A[point]>0)[0]
                if len(members) >0:
                    #another core point; add the eps-neighborhood
                    #to the queue
                    clusters[point] = nclusters
                    queue = queue + list(members)
            pcounter += 1
        # add the point eps-neighborhood to the cluster
        clusters[queue] = nclusters       
        nclusters += 1
        return nclusters, clusters
    
    def build_clusters(self, A):
        """
        given kNN graph, create clusters
        """        
        if self.debug:
            print("Ad. matrix\n",A)
        # build clusters
        clusters = -2 * np.ones(self.N,dtype='int')
        nclusters = 0
        for point in range(self.N):
            if clusters[point] != -2:
                #already assigned
                continue                
            members = np.where(A[point]>0)[0]
            if len(members) == 1:
                # a noise point; can this happen? 
                clusters[point] = -1
            elif len(members) > 1:
                #Unassigned point with neighbours
                nclusters, clusters = self.grow_cluster(\
                    point, members, nclusters, clusters, A)            
            else:
                raise ValueError
        return nclusters, clusters
        
    def do_clustering(self, X=None, D=None, W=None, do_conn=True):
        """
        X = features
        D = distances (precomputed)
        W = weights
        do_conn == False; class was already instatiated with 
        data and adiancency matrix was calculated; we are only
        changing P and K
        """
        if do_conn:
            D = self.init2(X, D)
            # nearest neighs
            NB = self.find_nbrs(D,W)
            #adiancency matrix
            A = self.adiancency_matrix(NB)
        elif not np.any(self.A):
            raise ValueError("No adiacency matrix available")
        if self.saveA:
            self.A = A
        # build clusters
        nclusters, clusters = self.build_clusters(A)
        nnoise = np.count_nonzero(clusters==-1)
        #return
        return nclusters, nnoise, clusters
    
class brown_martin(jarvis_patrick):
    """
    Jarvis Patrick variant. See 
    Brown, R. D.; Martin. Y. C. 
    Use of Structure-Activity Data To Compare Structure-Based 
    Clustering Methods and Descriptors for Use in Compound 
    Selection J. Chem. Inf. Comput. Sci. 1996, 36, 572-584
    - T is distance threshold in which to look for N. N.
    - Rmin is the ratio of lenght of the NN lists
    """
    def __init__(self,**kwargs):
        prop_defaults = {
            "metric"    : "euclidean",
            "T"         : None,
            "Rmin"      : None,
            "debug"     : False,
            "saveA"     : False
        }
        for (prop, default) in prop_defaults.items():
            setattr(self, prop, kwargs.get(prop, default))         
        # check some input
        assert isinstance( self.T, float )
        assert isinstance( self.Rmin, float )
        
    def find_nbrs(self, D):
        NB = list()
        for i in range(self.N):
            nbrs = np.where(D[i] <= self.T)[0]
            NB.append(nbrs)
        if self.debug:
            print(NB)
        return NB
        
    def same_cluster(self, I, J, A, NB):
        L1 = len(NB[I])
        L2 = len(NB[J])
        Lmin = math.floor(self.Rmin*L2)
        common_NN = set(NB[I]).intersection(NB[J])
        if len(common_NN) >= Lmin:
            A[I,J] = 1
            A[J,I] = 1 

class SNN(jarvis_patrick):
    """
    See L. Ertoz, M. Steinbach, and V. Kumar, 
    (1) Ertoz, L.; Steinbach, M.; Kumar, V. 
    A New Shared Nearest Neighbor Clustering Algorithm and Its Applications. 
    In Workshop on clustering high dimensional data and its applications 
    at 2nd SIAM international conference on data mining; 2002; pp 105–115.
    - K is the number of neighbors to look at for kNN search
    - minPTS is the minimum number of neighbor point with density > epsilon
      is the same of DBSCAN
    - epsilon is the reachbility threshold i.e. the Kmin parameter of standard
      JP
    grow_cluster is almost a verbatim copy from DBSCAN
    """
    
    def __init__(self,**kwargs):
        prop_defaults = {
            "metric"    : "euclidean",
            "K"         : None,            
            "minPTS"    : None,
            "epsilon"   : None,
            "link_str"  : "simple",
            "debug"     : False,
            "saveSNN"   : False
        }
        for (prop, default) in prop_defaults.items():
            setattr(self, prop, kwargs.get(prop, default))         
        # check some input
        assert isinstance( self.K, int ) or isinstance( self.K, np.int64 )
        assert isinstance( self.epsilon, int ) or isinstance( self.epsilon, np.int64 )
        assert isinstance( self.minPTS, int ) or isinstance( self.minPTS, np.int64 )
        assert self.K > self.minPTS
        assert self.K > self.epsilon
        self.Kmin = self.epsilon
        
    def calc_snn_graph(self, NB):
        snn_graph = self.K * np.eye(self.N,dtype='int')
        if self.link_str == "simple":
            for i in range(self.N-1):
                for j in range(i+1,self.N):
                    if i in NB[j] and j in NB[i]:
                #how many shared neighbors
                #this is the standard JP strenght
                        N_common_NN = len(set(NB[i]).intersection(NB[j]))
                        if N_common_NN >= self.epsilon:
                            snn_graph[i,j] = N_common_NN
                            snn_graph[j,i] = N_common_NN
        elif self.link_str == "weighted":
            # consider the order of the shared neighbors
            for i in range(self.N-1):
                for j in range(i+1,self.N):
                    if i in NB[j] and j in NB[i]:
                        shared = list(set(NB[i]).intersection(NB[j]))
                        count = 0
                        for s in shared:
                            ipos, = np.where(NB[i]==s)
                            jpos, = np.where(NB[j]==s)
                            count += (self.K + 1 - ipos)*(self.K + 1 - jpos)
                        if count >= self.epsilon:
                            snn_graph[i,j] = count
                            snn_graph[j,i] = count
        return snn_graph
    
    def calc_snn_density(self, point, SNN_graph):
        neighbors = np.where(SNN_graph[point] != 0)[0]
        rho = len(neighbors)
        return rho, neighbors

    def do_clustering(self, X=None, D=None, W=None):
        """
        clusters (-1, or cluster ID, 0:N-1), cluster number (start from 0)
        X(npoints,nfeatures) is the feature matrix
        D(npoints,npoints) is the distance/dissimilarity matrix
        W(npoints) are the weights
        """
        D = self.init2(X,D)
        ###
        # Do the jarvis patrick steps
        NB = self.find_nbrs(D,W)
        SNN_graph = self.calc_snn_graph(NB)
        ###
        # Do the DBSCAN steps
        clusters = -2 * np.ones(self.N,dtype='int')
        nclusters = 0
        for point in range(self.N):
            if clusters[point] != -2:
                #already assigned
                continue
            rho, neighbors = self.calc_snn_density(point, SNN_graph)
            if rho < self.minPTS:
                # a noise point (can be assigned as leaf later on)
                clusters[point] = -1
            elif rho >= self.minPTS:
                #we have Unassigned point with enough density -> new cluster
                nclusters = self.grow_cluster(\
                    point,neighbors,nclusters,clusters, SNN_graph)
        noise  = np.where(clusters==-1)[0]
        return nclusters, len(noise), clusters
            
    def grow_cluster(self,point, neighbors, nclusters, clusters, D):
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
                #another unassigned point
                rho, neighbors = self.calc_snn_density(point, D)
                if rho >= self.minPTS:
                    #another core point; add the eps-neighborhood
                    #to the queue
                    clusters[point] = nclusters
                    queue = queue + list(neighbors)
            pcounter += 1
        # add the point eps-neighborhood to the cluster
        clusters[queue] = nclusters       
        nclusters += 1
        return nclusters
