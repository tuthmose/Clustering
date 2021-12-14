import numpy as np
import scipy as sp
import sys

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
    object used to change clustering parameters given predefined grids
    p0 = k
    p1 = minpts
    p2 = epsilon
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

class paraminterpolator:
    """
    object used to mix clustering parameters given two sets of parameters
    p0 = k
    p1 = minpts
    p2 = epsilon
    """
    def __init__(self):
        return None

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
    
    def __call__(self, alpha, p0, p1):
        """
        do crossover
        """
        child0, child1 = self.newparams(alpha, p0, p1)
        return child0, child1

def gen_model(maxK):
    """
    return a model k, minpts, eps
    """
    k = np.random.randint(2, high=maxK)
    if k <= 4:
        minpts = 1
        eps = 1
    else:
        minpts = np.random.randint(2, high=k-2)
        eps = minpts = np.random.randint(2, high=k-2)
    assert k > minpts
    assert k > minpts
    return np.array((k, minpts, eps))

def param_evaluator(model, **kwargs):
    """
    do clustering and evaluate model
    """
    # set options
    method  = kwargs['eval_method']
    npoints = kwargs['data'].shape[0]
    scaleN  = kwargs['scaleN']
    penalty = kwargs['penalty']
    # do clustering
    my_estimator = myclusters.SNN(K=model[0], minPTS=model[1], epsilon=model[2], metric='precomputed')
    if not kwargs['precomputed']:
        D = sp.spatial.distance.squareform(sp.spatial.distance.pdist(kwargs['data']))
    else:
        D = kwargs['data']
    ncluster, nnoise, clusters = my_estimator.do_clustering(X=None, D=D)
    if ncluster <= 1 or npoints==nnoise:
        return 1000.
    # do validation
    myeval = myvalidation.cluster_eval(clusters=clusters, D=D, metric="precomputed", noise="ignore")
    if method != "silhouette":
        score = myeval(method=method,inter=kwargs['interd'],intra=kwargs['intrad'])
    else:
        fD = D[clusters!=-1, :][:, clusters!=-1]
        fC = clusters[clusters!=-1]
        score = 1 + silhouette_score(fD, fC, metric='precomputed')
    # apply noise penalty
    if scaleN != 0:
        nnoise_s = int(nnoise*scaleN)
    if method == 'silhouette' or method == 'Dunn':
        if penalty:
            penalty = 1. - nnoise/npoints
            score = penalty*score
        return 1./score
    else:
        if penalty:
            penalty = 1. + nnoise/npoints
            score = penalty*score
        return score
