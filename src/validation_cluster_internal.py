import numpy as np
import numpy.ma as MA
from math import log,sqrt
from scipy.spatial.distance import cdist,pdist,squareform
from scipy.special import binom

# Internal cluster validation criteria:

# Base class
# - Davies Bouldin index
# - Dunn index (single, average, complete)
# - Calinski - Harabasz score
# - f(K) function

# Range K class
# - Gap Statistic nyi

# G Mancini November 2019

# note that in many cases, the same operation
# (e.g. WSS for a cluster) is done for different
# criteria. This could be optimized.
          
def find_centroid(D,beta=1.,mask=None):
    """
    find centroid on input distance matrix using similarity scores:
    s_ij = e^-beta rmsd_ij / sigma(rmsd)
    c = argmax_i sum_j sij
    i.e. the frame with maximum similarity from all the others
    """
    similarity = np.exp(-beta*D/np.std(D))
    if mask is None:
        centroid = (similarity.sum(axis=1)).argmax()
    else:
        maskt = np.expand_dims(mask,axis=0).T
        sim_m = MA.array(similarity,mask=~(mask*maskt))
        centroid = (sim_m.sum(axis=1)).argmax()
    return centroid  

def core2centers(**kwargs):
    """
    find a centroid point for each cluster, given a list of labels
    (1 to ncluster); -1==noise
    if given coordinates, return the point closest to the center of
    mass
    if given distances, return the point with the highest similarity
    """
    ## !! TIES NYI
    D = False
    X = False
    beta = 1.0
    metric = "euclidean"
    clusters = False
    for key, value in kwargs.items():
        if key is "clusters":
            clusters = value
        if key is "X":
            X = value
        if key is "D":
            D = value
        if key is "metric":
            metric = value
        if key is "beta":
            beta = value
    if not "clusters":
        raise ValueError("Missing cluster list")
    if np.any(X) and np.any(D):
        print("Given X and D; using features")
    labels = set(clusters)
    centroids = list()
    if np.any(X):
        nfeatures = X.shape[1]
        for L in labels:
            if L==-1:
                centroids.append(-1)
                continue
            points = X[clusters==L,:]
            COM = np.average(points,axis=0)
            COM = np.expand_dims(COM,axis=0)
            dcom = cdist(points,COM,metric=metric)
            C = np.where(dcom==np.min(dcom))[0][0]
            mask = np.sum(np.equal(X,points[C]),axis=1)
            C1 = np.where([mask==nfeatures])[1][0]
            centroids.append(C1)
    else:
        for L in labels:
            if L==-1:
                centroids.append(-1)
                continue
            C = find_centroid(D,beta,clusters==L)
            centroids.append(C)
    return np.asarray(centroids)

def assign_centroid_label(clusters,labels):
    """
    given a list of clusters and labels for the
    centroids, return the list with clusters labels
    the order of labels must be the same of those 
    in the clusters list
    """
    new_labels = list()
    for i in clusters:
        new_labels.append(labels[i])
    return np.asarray(new_labels)

class cluster_eval(object):
    def __init__(self,**kwargs):
        """
        create evaluation object using
        either coordinates (X) or precomputed
        distance matrix (D)
        """
        prop_defaults = {
            "metric"  : "euclidean",
            "clusters": None,
            "X"       : None,
            "D"       : None,
        }
        for (prop, default) in prop_defaults.items():
            setattr(self, prop, kwargs.get(prop, default))                
        #precomputed distance?
        if self.metric is "precomputed" and self.D is None:
            raise ValueError("metric=precomputed but missing distance matrix")
        elif self.metric is not "precomputed" and self.X is not None:
            self.D = squareform(pdist(self.X,metric=self.metric)) 
        return None

    def com(self,**kwargs):
        """
        find center of mass of given
        coordinate set or index of point
        in dataset nearest to the COM
        """
        if self.X is None:
            raise ValueError("Must have coordinates to find center of mass")
        weights = None
        nearest = False
        for key, value in kwargs.items():
            if key=="weights":
                weights = value
            if key=="nearest":
                nearest = value
            if key=="coord":
                coord = value
        if weights==None:
            R = coord 
        else:
            if weights.ndim == 1:
                weights.shape = (coord.shape[0],1)
            R = coord*weights
        COM = R.sum(0)/R.shape[0]
        if nearest is False:
            result = COM
        else:
            COM.shape = (1,COM.shape[0])
            dist = cdist(R,COM,metric=self.metric)
            mindist = np.min(dist)
            result = np.where(dist[:,0]==mindist)[0]
        return result

    def filt_noise(self):
        """
        filt noise accordinf to kwd
        """
        X = None
        D = None
        if self.noise == "ignore":
            """
            discard coordinates of noise points
            and scale labels subctracting number of
            noise points beyond them
            points do not need to be ordered
            """
            if self.X is not None:
                X = self.X[self.clusters!=-1]
            oldc = list(set(self.clusters))
            cent = dict()
            clust = list()
            if self.D is not None:
                D = self.D[self.clusters!=-1,:][:,self.clusters!=-1]
            for i in oldc:
                if i==-1: continue
                c = self.clusters[:i+1]
                j = len(c[c==-1])
                cent[i] = i-j
            for i in self.clusters:
                if i==-1: continue
                clust.append(cent[i])
            cent = list(cent.values())
            clust = np.asarray(clust)
        elif self.noise == "uniq":
            """
            form new cluster with all noise points
            by creating a new label corresponding
            to the centroid of noise points
            """
            X = self.X
            D = self.D
            clust = np.copy(self.clusters)
            if self.X is not None:
                points = X[self.clusters==-1]
                cnoise = self.com(coord=points,weight=None,nearest=True)
                mask = np.sum(np.equal(X,points[cnoise]),axis=1)
                C1 = np.where([mask==X.shape[1]])[1][0]
                cnoise = C1
                print(C1)
            else:
                mask = self.clusters==-1
                cnoise = find_centroid(self.D,beta=1.,mask=mask)
                print(cnoise)
            clust[self.clusters==-1] = cnoise
            cent = list(set(clust))
        elif self.noise == "singles":
            """
            assign a new label for each noise point
            """
            X = self.X
            D = self.D
            clust = list()
            for i,j in enumerate(self.clusters):
                if j == -1:
                    clust.append(np.int64(i))
                else:
                    clust.append(j)
            cent = list(set(clust))
            clust = np.asarray(clust)
        return X,clust,cent,D

    def __call__(self,**kwargs):
        """
        Get or calculate distance matrix
        according to metric.
        Manage noise with the following options:
        - discarded if noise="ignore"
        - treated as a unique cluster if noise="uniq" (default)
        - trated as size=1 cluster if noise="singles"
        When using "ignore" a normalization factor can be added
        as (ndata - ncluster)/ndata
        """
        #keywords
        self.noise = "uniq"
        noise_filt = ["uniq","ignore","singles"]
        self.methods = ["DBI","Dunn","CH","f_K"]        
        method = None
        norm_noise = False
        #Dunn index kwds
        inter = "allav"
        intra = "allav"
        intra_avail = ("center","allav","allmax")
        inter_avail = ("center","allav","allmin")
        # use true coordinate centers in CH instead
        # of nearest points (centroids)
        usec = False
        Skm1 = 1.
        Nd   = None
        for key, value in kwargs.items():
            if key == "noise":
                self.noise = value
            if key == "method":
                method = value
            if key == "inter":
                inter = value
            if key == "intra":
                intra = value
            if key == "use_centroid":
                usec = value
            if key == "norm_noise":
                norm_noise = True
            if key == "Skm1":
                Skm1 = value
            if key == "Nd":
                Nd = value
        if self.X is not None and Nd == None:
            Nd = self.X.shape[1]
        elif Nd == None and method == 'f_K':
            raise ValueError("Need # dimensions for f_K")    
        #check kwds
        if method not in self.methods:
            raise NotImplementedError("no method %s in %s" \
            % (method,self.__class__.__name__))
        if self.noise not in noise_filt:
            raise NotImplementedError("no noise filtering %s in %s" \
            % (self.noise,self.__class__.__name__))
        if intra not in intra_avail:
            raise NotImplementedError("Dunn: no intracluster distance %s in %s" \
            % (intra,self.__class__.__name__))
        if inter not in inter_avail:
            raise NotImplementedError("Dunn: no intercluster distance %s in %s" \
            % (inter,self.__class__.__name__))
        ### filt data set
        labels = set(self.clusters)
        if -1 in labels and norm_noise:
            nnoise = len(self.clusters[self.clusters==-1])       
        if -1 in labels:
            filt_X,filt_clust,self.centroids,filt_D = self.filt_noise()
        else:
            filt_X = self.X
            filt_clust = self.clusters
            self.centroids = list(labels)
            filt_D = self.D
        self.N = len(self.centroids)
        try: assert self.N > 1
        except AssertionError:
            print("Error :at least two custers are needed")
            if method is "CH":
                return (False,False)
            else:
                return False
        self.NData = len(self.clusters)
        #cluster size
        size = list()
        for i in self.centroids:
            size.append(len(filt_clust[filt_clust==i]))
        size = dict(zip(list(self.centroids),size))
        #call required method
        if method == "DBI":
             result = self.davies_bouldin(filt_X,filt_clust,size,filt_D)
        if method == "Dunn":
             result = self.dunn(filt_X,filt_clust,size,filt_D,wid=intra,bwd=inter)
        if method == "CH":
            result = self.CHscore(filt_X,filt_clust,size,filt_D,usec)
        if method == "f_K":
            result = self.f_K(filt_X,filt_clust,filt_D,usec,Skm1,Nd)
        if self.noise=="ignore" and norm_noise:
            noise_const = (self.NData - nnoise)/self.NData
            result = result*noise_const
        return result

    def dunn(self,coord,clust,size,dist,wid=None,bwd=None):
        """
        calculates Dunn index
        Available definitions of INTERcluster distance (inter):
        - center: distance between centroids
        - nearest: distance between nearest points
        - allav: average distance between all pairs
        Available definitions of INTRAcluster distance (intra):
        - center: average distance of all points from centroid
        - allav: average distance between all pairs
        - allmax: maximum distance distance between all pairs
        """
        #setup
        within = list()
        between = list()
        for i in self.centroids[:-1]:
            if size[i] == 1:
                dI = 0.
            elif wid == "allav":
                dI = np.mean(dist[clust==i,:][:,clust==i])
            elif wid == "allmax":
                dI = np.max(dist[clust==i,:][:,clust==i])
            elif wid=="center":
                dI = np.mean(dist[clust==i,i])
            within.append(dI)
            for j in self.centroids[self.centroids.index(i)+1:]:
                if bwd == "allav":
                    dij = np.mean(dist[clust==i,:][:,clust==j])
                elif bwd == "center":
                    dij = dist[i,j]
                elif bwd == "allmin":
                    dij = np.min(dist[clust==i,:][:,clust==j])
                between.append(dij)
        W = np.max(np.asarray(within))
        between = np.asarray(between)
        between = between[np.nonzero(between)]
        B = np.min(between,axis=-1)
        return B/W

    def davies_bouldin(self,coord,clust,size,dist):
        """
        calculates Davies Bouldin index 
        """
        DBI = 0.0
        for i in self.centroids:
            if self.X is None:
                dispersion_i = np.mean(dist[clust==i,i],axis=-1)
            else:
                ci = np.mean(coord[clust==i],axis=0)
                dispersion_i = np.mean(cdist(coord[clust==i],[ci],metric=self.metric))
            Rij = np.zeros(self.N)
            for nj,j in enumerate(self.centroids):
                if i==j:
                    continue
                elif self.X is None:
                    Rij[nj] = (np.mean(dist[clust==j,j])+dispersion_i)/dist[i,j]
                else:
                    cj = np.mean(coord[clust==j],axis=0)
                    dispersion_j = np.mean(cdist(coord[clust==j],[cj],metric=self.metric))
                    dij = np.linalg.norm(ci-cj)
                    Rij[nj] = (dispersion_i+dispersion_j)/dij
            DBI += np.max(Rij)
        return DBI / self.N

    def CHscore(self,coord,clust,size,dist,use_centroid=False):
        """
        Calinski Harabasz score
        1.Caliński, T. & Harabasz, J. A dendrite method for cluster analysis. 
        Communications in Statistics 3, 1–27 (1974).
        use either centroids (if only distances are available)
        or true centers (if coordinates/features are available)
        """
        if self.X is not None and use_centroid is False:
            COM = self.com(coord=coord,weight=None,nearest=False)
            COM = np.expand_dims(COM,axis=0)
            near = False
        else:
            COM_id = find_centroid(dist)
            near = True
        WSS = 0.
        BSS = 0.
        for i in self.centroids:
            if near:
                dw = np.sum(dist[clust==i,i]**2)
                db  = size[i]*(dist[i,COM_id]**2)
            else:
                coord_i = coord[clust==i]
                center_i = self.com(coord=coord_i,weight=None,nearest=False)
                center_i = np.expand_dims(center_i,axis=0)
                dw = np.sum(cdist(coord_i,center_i,metric=self.metric)**2)
                db = size[i]*(cdist(center_i,COM,metric=self.metric)[0][0])**2
            WSS += dw
            BSS += db
        CHs = ( BSS/(self.N -1) ) / ( WSS/(self.NData - self.N) )
        return np.asarray((CHs, WSS))
    
    def f_K(self,coords,clusters,dist,use_centroid,Skm1,Nd):
        """
        
        """
        #almost copied from 
        #https://datasciencelab.wordpress.com/2014/01/21/selection-of-k-in-k-means-clustering-reloaded/
        thisk = np.unique(clusters).shape[0]
        #throw away recursive definition of a_k
        a_k = lambda k, Nd: 1. - 3./(4.*Nd) if k == 2 else a_k(k-1, Nd) + (1-a_k(k-1, Nd))/6.
        ##
        Sk = 0.
        if self.X is not None and use_centroid is False:
            for c in self.centroids:
                coord_c = coords[clusters==c]
                mu_c = self.com(coord=coord_c,weight=None,nearest=False)
                Sk += np.sum(cdist(coord_c,np.expand_dims(mu_c,axis=0),metric=self.metric)**2)
        else:
            for c in self.centroids:
                Sk += np.sum(dist[clusters==c,:][:,c]**2)
        #
        if thisk == 1:
            fs = 1
        elif Skm1 == 0:
            fs = 1
        else:
            fs = Sk/(a_k(thisk,Nd)*Skm1)
        return fs, Sk 
