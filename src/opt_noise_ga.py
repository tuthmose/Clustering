import numpy as np
import scipy as sp
import sys

import hdbscan

import myclusters
import myvalidation

sys.path.append("/home/gmancini/Dropbox/appunti/old_search_algos_28_08_2020/EvolutionaryAlgorithms/src")
import ga_evolution
import ga_population

from sklearn.metrics import silhouette_score

# a collection of functions to optimize SNN parameters
# with GA using internal validation and noise correction as
# VI_{adj} = 

# perhaps it works with other JP related methods
#
# G Mancini Dec 2021

class paramgenerator:
    """
    template used to change  used to change clustering parameters given predefined grids
    """
    def __init__(self ):
        return None

    def newparams(self, param):
        """
        return a new param picked by one domain
        """
        return None
 
    def __call__(self, population, prob):
        """
        mutate a model with uniform probability
        changing one if its parameters
        """
        mutated = list()
        for chrm in population.get_ch_labels():
            coin = np.random.rand()
            if coin <= prob:
                mutated.append(chrm)
                newchrm = self.newparams(population.chromosomes[chrm])
                population.chromosomes[chrm] = newchrm
        return mutated

class SNN_paramgenerator(paramgenerator):
    """
    generate parameters for Shared Nearest Neighbours
    """
    def __init__(self, maxK):
        self.maxK = maxK

    def newparams(self, param):
        """
        return a new param picked by one domain with
        uniform probability
        param[0]: k
        param[1]: minpts
        param[2]: epsilon
        """
        param[0] = np.random.randint(2, high=self.maxK)
        if param[0] > 4:
            param[1] = np.random.randint(2, high=param[0]-1)//2
            param[2] = np.random.randint(2, high=param[0]-1)//2
        else:
            param[1] = 1
            param[2] = 1
        assert param[1] < param[0]
        assert param[2] < param[0]
        return param

class HDBSCAN_paramgenerator(paramgenerator):
    """
    generate parameters for HDBSCAN
    """
    def __init__(self, min_clust_size, max_clust_size, max_min_pts):
        self.min_clust_size = min_clust_size
        self.max_clust_size = max_clust_size
        self.max_min_pts = max_min_pts

    def newparams(self, param):
        """
        return a new param picked by one domain with
        uniform probability
        param[0]: clust_size
        param[1]: minpts
        """
        param[0] = np.random.randint(self.min_clust_size, high=self.max_clust_size)
        param[1] = np.random.randint(2, high=self.max_min_pts)
        return param

class paraminterpolator:
    """
    object used to mix clustering parameters given two sets of parameters
    p0 = k
    p1 = minpts
    p2 = epsilon
    """
    def __init__(self):
        return None

    def __call__(self, alpha, p0, p1):
        """
        do crossover
        """
        child0, child1 = self.newparams(alpha, p0, p1)
        return child0, child1

class SNN_paraminterpolator(paraminterpolator):
    def newparams(self, alpha, p0, p1):
        """
        return two sets of new parameters given the old ones
        """
        new0 = (alpha*p0 + (1.-alpha)*p1).astype(int)
        new1 = ((1. - alpha)*p0 + alpha*p1).astype(int)
        new0[0] = max(new0[0], 2)
        new0[0] = max(new0[1], 2)
        new0[1] = min(new0[1], new0[0]-1)
        new0[2] = min(new0[2], new0[1]-1)
        new1[1] = min(new1[1], new1[0]-1)
        new1[2] = min(new1[2], new1[1]-1)
        assert new0[0] > new0[1]
        assert new0[0] > new0[2]
        assert new1[0] > new0[1]
        assert new1[0] > new1[2]
        return new0, new1

class HDBSCAN_paraminterpolator(paraminterpolator):
    def newparams(self, alpha, p0, p1):
        """
        return two sets of new parameters given the old ones
        """
        new0 = (alpha*p0 + (1.-alpha)*p1).astype(int)
        new1 = ((1. - alpha)*p0 + alpha*p1).astype(int)
        new0[0] = max(new0[0], 2)
        new0[1] = max(new0[1], 2)
        new1[0] = max(new1[0], 2)
        new1[1] = max(new1[1], 2)
        return new0, new1

def gen_model(method, param):
    """
    return a model 
    """
    if method == "SNN":
        k = np.random.randint(2, high=param[0])
        if k <= 4:
            minpts = 1
            eps = 1
        else:
            minpts = np.random.randint(2, high=k-2)
            eps = minpts = np.random.randint(2, high=k-2)
        assert k > minpts
        assert k > minpts
        return np.array((k, minpts, eps))
    elif method == "HDBSCAN":
        clust_size = np.random.randint(2, high=param[0])
        minpts = np.random.randint(2, high=param[1])
        return np.array((clust_size, minpts))
    else:
        raise ValueError("method not supported")

class param_evaluator:
    """
    do clustering with given model
    """
    def __init__(self, **kwargs):
        self.vmethod  = kwargs['eval_method']
        self.interd = kwargs['interd']
        self.intrad = kwargs['intrad']
        self.data    = kwargs['data']
        self.npoints = kwargs['data'].shape[0]
        self.scaleN  = kwargs['scaleN']
        self.apply_penalty = kwargs['penalty']
        self.metric = kwargs['metric']
        if not kwargs['precomputed']:
            self.D = sp.spatial.distance.squareform(sp.spatial.distance.pdist(kwargs['data'], metric=kwargs['metric']))
        else:
            self.D = kwargs['data']

    def run_cluster(self):
        """
        do the clustering to be validated
        """
        return None

    def validate(self, clusters, D):
        """
        apply the selected internal validation criterion
        """
        myeval = myvalidation.cluster_eval(clusters=clusters, D=D, metric="precomputed", noise="ignore")
        if self.vmethod != "silhouette":
            score = myeval(method=self.vmethod, inter=self.interd, intra=self.intrad)
        else:
            fD = D[clusters!=-1, :][:, clusters!=-1]
            fC = clusters[clusters!=-1]
            score = 1 + silhouette_score(fD, fC, metric='precomputed')
        return score

    def __call__(self, model):
        """
        run cluster analysis and calculate score and penalty
        """
        try:
            clusters = self.run_cluster(model, self.D)
            nnoise = len(clusters[clusters==-1])
            score = self.validate(clusters, self.D)
            # apply noise penalty
            if self.scaleN != 0:
                nnoise_s = int(nnoise*self.scaleN)
            if self.vmethod == 'silhouette' or self.vmethod == 'Dunn':
                if self.apply_penalty:
                    penalty = 1. - nnoise/self.npoints
                    score = penalty*score
                return 1./score
            else:
                if self.apply_penalty:
                    penalty = 1. + nnoise/self.npoints
                    score = penalty*score
                return score
        except:
            return 1000.

class SNN_param_evaluator(param_evaluator):
    def run_cluster(self, model, D):
        """
        calculate fitness for SNN method
        """
        my_estimator = myclusters.SNN(K=model[0], minPTS=model[1], epsilon=model[2], metric='precomputed')
        ncluster, nnoise, clusters = my_estimator.do_clustering(X=None, D=D)
        return clusters

class HDBSCAN_param_evaluator(param_evaluator):
    def run_cluster(self, model, D):
        """
        calculate fitness for HDBSCAN method
        """
        my_estimator = hdbscan.HDBSCAN(min_cluster_size=int(model[0]), min_samples=int(model[1]), metric="precomputed")
        labels = my_estimator.fit_predict(D)
        return labels
