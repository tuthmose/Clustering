import math
import numpy as np
import scipy as sp
import scipy.spatial.distance as distance

# a scikit.learn KDTree should be used but it does not
# take precomputed distances

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
        self.A = np.zeros((self.N,self.N),dtype='bool')
        
    def find_nbrs(self):
        NB = np.zeros((self.N,self.K),dtype='int')
        for i in range(self.N):
            NB[i] = np.argsort(self.D[i])[:self.K]
        return NB
            
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
    
    def grow_cluster(self,i,clusters,nclusters):
        members = np.where(self.A[i]>0)[0]
        if nclusters==0 or np.all(clusters[members]==-1):
            #first cluster
            #or neither this element nor its connections are in a existing cluster
            clusters[members] = nclusters
            label = nclusters
            nclusters += 1
        else:
            C = set(clusters[members])
            try:
                C.remove(-1)
            except:
                pass
            C = list(C)
            label = C[0]
            clusters[members] = label
            for c in C[1:]:
                clusters[clusters==c] = label
                if c != label:
                    nclusters = nclusters-1
        # add any other element connected to members
        for m in members:
            new_m = np.where(self.A[m]>0)[0]
            if len(new_m)>0:
                clusters[new_m] = label
        return nclusters
    
    def build_clusters(self):
        if self.debug:
            print("Ad. matrix\n",self.A)
        # filt noise:
        noise  = np.where(np.count_nonzero(self.A,axis=0)==1)[0]
        # build clusters
        clusters = -np.ones(self.N,dtype='int')
        nclusters = 0
        for i in range(self.N):
            if i in noise:
                continue
            members = np.where(self.A[i]>0)[0]
            nclusters = self.grow_cluster(i,clusters,nclusters)
        #labels = list(set(clusters))
        #nclusters = len(labels)
        return nclusters, len(noise), clusters#, labels
        
    def do_clustering(self, X=None, D=None, do_conn=True):
        if do_conn:
            self.init2(X, D)
            # nearest neighs
            self.NB = self.find_nbrs()
            #adiancency matrix
            self.A = self.adiancency_matrix()
        elif not np.any(self.A):
            raise ValueError("No adiacency matrix available")
        # build clusters
        #self.nclusters, self.nnoise, self.clusters, self.labels = self.build_clusters()
        self.nclusters, self.nnoise, self.clusters = self.build_clusters()
        #return
        return self.nclusters, self.nnoise, self.clusters
    
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
      is the same of DBSCAN
    - epsilon is the reachbility threshold i.e. the Kmin parameter of standard
      JP
    """
    
    def __init__(self,**kwargs):
        prop_defaults = {
            "metric"    : "euclidean",
            "K"         : None,            
            "minPTS"    : None,
            "epsilon"   : None,
            "link_str"  : "simple",
            "debug"     : False
        }
        for (prop, default) in prop_defaults.items():
            setattr(self, prop, kwargs.get(prop, default))         
        # check some input
        assert isinstance( self.K, int )
        assert isinstance( self.minPTS, int )
        assert self.K > self.minPTS 
        
    def calc_snn_graph(self, NB):
        #note that self.NB is 2D but SNN_graph is 1D
        #also, in JP the first NN of a point is the point itself
        snn_graph = np.zeros(self.N*(self.N+1)//2,dtype='int')
        if self.link_str == "simple":
            for i in range(self.N-1):
                for j in range(i,self.N):           
                #how many shared neighbors
                #this is the standard JP strenght
                    count = len(set(NB[i]).intersection(NB[j]))
                    snn_graph[i+j*(j-1)//2] = count
        elif self.link_str == "weighted":
            # consider the order of the shared neighbors
            for i in range(self.N-1):
                for j in range(i,self.N):
                    shared = list(set(NB[i]).intersection(NB[j]))
                    count = 0
                    for s in shared:
                        ipos, = np.where(self.NB[i]==s)
                        jpos, = np.where(self.NB[j]==s)
                        count += (self.K + 1 - ipos)*(self.K + 1 - jpos)
                    snn_graph[i+j*(j-1)//2] = count
        return snn_graph
    
    def calc_snn_density(self, SNN_graph):
        SNN_density = np.zeros(self.N-1)
        for i in range(self.N):
            rho = 0.
            for j in range(i,self.N):
                if SNN_graph[ !=]
                    rho += lambda i,j: 1 if self.D[i,j] >= self.epsilon else 0.
            SNN_density[i] = rho
        return SNN_density
    

    def do_clustering(self, X=None, D=None):
        self.init2(X, D)
        # nearest neighs
        NB = self.find_nbrs()
        #Shared nearest neighbors graph
        SNN_graph = self.calc_snn_graph(NB)
        # SNN density
        SNN_density = self.calc_snn_density(SNN_graph)
        #build clusters
        self.nclusters, self.nnoise, self.clusters = self.build_clusters(XXX)
        #return
        return self.nclusters, self.nnoise, self.clusters
