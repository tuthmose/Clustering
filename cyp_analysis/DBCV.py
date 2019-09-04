# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 16:25:54 2017

@author: g.mancini
"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import cdist,pdist,squareform
from sklearn.neighbors import NearestNeighbors
np.seterr(all='raise')

class DBCV:
    
    def __init__(self,**kwargs):
        """
        create Density Based Cluster Validity
        estimator (see Moulavi et al,
        http://epubs.siam.org/doi/abs/10.1137/1.9781611973440.96)
        and check input data
        """
        X = None
        D = None
        clusters = None
        self.metric = "euclidean"
        self.NF = 0
        for key,value in kwargs.items():
            if key is  "X":
                X = value
            if key is  "D":
                D = value
            if key is  "clusters":
                clusters = value
            if key is  "metric":
                self.metric = value
            if key is "NF":
                self.NF = value
        if X is None and self.NF==0:
            raise ValueError("Need either feature matrix or number\
        of features")
        elif self.NF != 0 and X is not None:
            if self.NF != X.shape[1]:
                raise ValueError("Shape of feature matrix different from nfeatures")
        if D is None and X is None:
            raise ValueError("Need either feature or distance matrix")
        elif X is not None:
            D = squareform(pdist(X,metric=self.metric))
        if clusters is None:
            raise ValueError("Need cluster labels")
        Noise = clusters==-1
        N_noise = len(Noise)
        Points = ~Noise
        self.Ntot = len(clusters)
        self.clusters  = clusters[Points]
        self.centroids = np.sort(np.asarray(list(set(self.clusters))))
        self.Nclust = len(self.centroids)
        self.D = D[Points,:][:,Points]
        if X is not None: 
            self.X = X[Points,:]
        self.Nobj = self.D.shape[0]
        self.Size = [len(self.clusters[self.clusters==i]) for i in self.centroids]
        self.Size = np.asarray(self.Size)

    def calc_score(self,**kwargs):
        """
        DBCV algorithm
        """
        measures = ('acore','kernel')
        self.meas = 'acore'
        for key,value in kwargs.items():
            if key is "meas":
                self.meas = value
        if self.meas not in measures:
            raise ValueError("measure of density is either acore or kernel")
        if self.Nclust==0:
            return -1.5
        #0. add size=1 clusters to noise
        cleanup = self.elim_singles()
        if not cleanup:
            return -1.5
        #1. calculate density
        if self.meas is "acore":
            self.AC, self.MRD = self.apts_coredist()
        else:
            self.AC, self.MRD = self.kdens()
        if self.AC is None:
            return -1.5
        #2. calculate minimum spanning trees
        self.MSTs = self.create_MSTs()
        #3. select internal edges for each MST
        self.EdgeI = self.select_edgesI()
        #4. calculate sparness and density separation
        try:
            SPAR, SEP = self.DSC_DSPC()
        except:
            return -1.5
        #5. calculate validity index of each cluster
        VC  = np.zeros(self.Nclust)
        if self.Nclust == 1:
            if SPAR[0]==0 and SEP[0]==0:
                return 1.5
            VC[0] = (SEP[0]-SPAR[0]) /max(SEP[0],SPAR[0])
        else:
            for i in range(self.Nclust):
                mask = np.ones(self.Nclust,dtype='bool')
                mask[i] = False
                try:
                    Num = np.min(SEP[i,mask]-SPAR[i])
                    Den = max(np.min(SEP[i,mask]),SPAR[i])
                    VC[i] = Num / Den
                except:
                    return -1.5
        result = (1./self.Ntot)*np.sum(self.Size[:]*VC[:])
        return result

    def elim_singles(self):
        todel = np.zeros(self.Nclust,'bool')
        for i in range(self.Nclust):
            sz = self.Size[i]
            if sz == 1:
                todel[i] = True
            else:
                todel[i] = False
        if not np.any(~todel):
            return False
        if np.any(todel):
            cen = self.centroids[~todel]
            tokeep = self.clusters==cen
            if not np.any(tokeep):
                return False
            self.clusters = self.clusters[tokeep]
            self.centroids = cen
            sz = self.Size[todel].sum()
            self.Nobj = self.Nobj - sz
            self.Nclust = self.Nclust-len(todel[todel==1])
            self.D = self.D[tokeep,:][:,tokeep]
        todel = np.zeros(self.Nclust,'bool')
        for i in range(self.Nclust):
            cen = self.centroids[i]
            dd = self.D[self.clusters==cen,:][:,self.clusters==cen]
            if not np.any(dd):
                todel[i] = True
            else:
                todel[i] = False
        if not np.any(~todel):
            return False
        if np.any(todel):
            cen = self.centroids[~todel]
            tokeep = self.clusters==cen
            if not np.any(tokeep):
                return False
            self.clusters = self.clusters[tokeep]
            self.centroids = cen
            sz = self.Size[todel].sum()
            self.Nobj = self.Nobj - sz
            self.Nclust = self.Nclust-len(todel[todel==1])
            self.D = self.D[tokeep,:][:,tokeep]
        return True

    def kdens(self):
        cutoff = np.median(self.D)
        acore = np.zeros(self.Nobj)
        for i in range(self.Nobj):
            d = self.D[i,:]
            dens = np.sum(np.exp( -(d/cutoff)**2 ))-1
            acore[i] = 1./dens
        MRD = np.empty((self.Nobj,self.Nobj))
        if acore is None:
            return None,None
        for i in range(self.Nobj-1):
            MRD[i,i] = acore[i]
            for j in range(i+1,self.Nobj):
                MRD[i,j] = max(acore[i],acore[j],self.D[i,j])
        return acore, MRD

    def apts_coredist(self):
        """
        calculate apts_coredist and return
        mutual reachability distance matrix
        """
        acore = np.empty(self.Nobj)
        for i in range(self.Nobj):
            Cl = self.clusters[i]
            mask = self.clusters == Cl
            nmemb = len(mask[mask>0])
            NN_i = np.sort(self.D[i,mask])
            NN_i = NN_i[NN_i>0.]
            DEN = nmemb-1
            NUM = ((1./NN_i)**self.NF).sum()
            acore[i] = (NUM/DEN)**(-1./self.NF)
        MRD = np.empty((self.Nobj,self.Nobj))
        if acore is None:
            return None,None
        for i in range(self.Nobj-1):
            MRD[i,i] = acore[i]
            for j in range(i+1,self.Nobj):
                MRD[i,j] = max(acore[i],acore[j],self.D[i,j])
        return acore, MRD

    def create_MSTs(self):
        """
        calculate Mutual Reachability Distance Graphs
        for each cluster (in matrix format) and return
        minimum spanning trees
        """
        MSTs = dict()
        for i in range(self.Nclust):
            graph = np.zeros((self.Nobj,self.Nobj))
            mask = self.clusters==self.centroids[i]
            for j in range(self.Nobj-1):
                if not mask[j]: continue
                graph[j,j] = self.AC[j]
                for k in range(j+1,self.Nobj):
                    if not mask[k]: continue
                    graph[j,k] = self.MRD[j,k]
                    graph[k,j] = graph[j,k]
            csrgraph = csr_matrix(graph)
            mst_i = minimum_spanning_tree(csrgraph).toarray()
            MSTs[i] = mst_i + mst_i.transpose()
        return MSTs

    def select_edgesI(self):
        """
        Select internal edges (degree>1) from MSTs
        """
        edgeI = dict()
        for i in range(self.Nclust):
            mst  = self.MSTs[i]
            edgeI[i] = np.count_nonzero(mst,axis=1)>1
        return edgeI

    def DSC_DSPC(self):
        """
        calculate density sparseness and separation
        from MSTs internal edges
        """
        MM = max(np.max(self.D),np.max(self.AC))
        mm = np.min(self.MRD)
        #sparseness = MM*np.ones(Nclust)
        separation = mm*np.ones((self.Nclust,self.Nclust))
        sparseness = np.zeros(self.Nclust)
        if self.Nclust > 1:
            separation = np.zeros((self.Nclust,self.Nclust))
            for i in range(self.Nclust-1):
                separation[i,i] = np.min(self.MRD[self.clusters==self.centroids[i]])
                eI = self.EdgeI[i]
                if len(eI[eI>0])>0:
                    sparseness[i] = np.max(self.MSTs[i][eI])
                    for j in range(i+1,self.Nclust):
                        eJ = self.EdgeI[j]
                        if len(eJ[eJ>0])>0:
                            separation[i,j] = np.min(self.MRD[eI,:][:,eJ])
                        else:
                            separation[i,j] = np.min(self.MRD[eI,:]\
                                [:,self.clusters==self.centroids[i]])
                        separation[j,i] = separation[i,j]
            eI = self.EdgeI[self.Nclust-1]
            if len(eI[eI>0])>0:
                sparseness[self.Nclust-1] = np.max(self.MSTs[self.Nclust-1][eI])
        else:
            eI = self.EdgeI[0]
            if len(eI[eI>0])>0:
                sparseness[0] = np.max(self.MSTs[0][self.EdgeI[0]])
                separation[0] = np.min(self.MRD[self.EdgeI[0]])
        return sparseness, separation
