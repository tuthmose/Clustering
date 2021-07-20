from copy import deepcopy

import numpy as np
import random
import scipy as sp

def expD_similarity(labels, **kwargs):
    """
    s_{ij} = e^{ \frac{-beta d_{ij}} {\sigma}}
    S = \sum_{ij} s_{ij}
    """
    beta = float(kwargs.get('beta'))
    D = kwargs.get('D')
    return np.sum(np.exp(-beta*D[labels,:][:,labels] / np.std(D[labels,:][:,labels])))

def gau_similarity(labels, **kwargs):
    """
    s_{ij} = e^{ \frac{-beta d_{ij}} {\sigma}}
    S = \sum_{ij} s_{ij}
    """
    sigma = float(kwargs.get('sigma'))
    D = kwargs.get('D')
    return np.sum(np.exp(-D[labels,:][:,labels]**2 / 2.*sigma**2))

def gau_kernel(labels, **kwargs):
    """
    s_{ij} = e^{ \frac{-beta d_{ij}} {\sigma}}
    S = \sum_{ij} s_{ij}
    """
    sigma = float(kwargs.get('sigma'))
    D = kwargs.get('D')    
    return np.exp(-D[labels,:][:,labels]**2 / 2.*sigma**2)

def max_var(labels,  **kwargs):
    """
    The similarity is the inverse of the sum of variances along dimensions
    (columns) of the data set. 
    S = 1. / sum_i^ndim sum_j^NTs \frac{x_{ij}^2}{NTs} -\mu_i^2
    """
    X = kwargs.get('x')    
    DS = np.sum([np.var(X[labels], axis=0) for i in range(X.shape[1])])
    return 1./DS

def mindist(labels,  **kwargs):
    """
    simply minimize the inverse sum of the distance matrix with the given metric
    """
    D = kwargs.get('D')    
    DS = np.sum(D[labels,:][:,labels])
    return DS
               
class simpleGRASP:
    
    def __init__(self, **kwargs):
        """
        main fuction for Greedy Randommized Adaptive Procedure
        return the N_select elements from the input data set X that
        implemented as a minimization problem
        - n_ini:    number of seed elements b4 construction (1)
        - n_iter:   number of iterations (100)
        - n_local:  number of passes in local search (100)
        - temp:     temperature for simulated annealing local search (500)
        - n_neigh:   number of neighbours to consider for simulated annealing (0.01)
        - boltzmann:
        - c_rate:   cooling rate for SA in (0,1] (0.995)
        - alpha:    greedyness factor in [0,1] (0.75)
        - verbose:  verbosity level (1)
        - metric:   metric of distance between data points (euclidean)
        - seed:     random seed
        - score:    score function
        - skwds:    keyword for the score function (a string like "k1:v1 k2:v2")
        - kernel:   transform coordinates according to this kernel
        - kkwds:    keyword for the score function (a string like "k1:v1 k2:v2")        
        """
        # get values
        prop_defaults = {
            'n_ini'     : 1,
            'n_iter'    : 100,
            'n_local'   : 100,
            'temp'      : 500.,
            'n_neigh'   : 0.01,
            'boltzmann' : 1.,
            'c_rate'    : 0.99,
            'alpha'     : 0.75,
            'verbose'   : 1,
            'metric'    : 'euclidean',
            'seed'      : None,
            'score'     : None,            
            'skwds'     : None,
            'kernel'    : None,
            'kkwds'     : None            
            }
        for (prop, default) in prop_defaults.items():
            setattr(self, prop, kwargs.get(prop, default))
        # check input
        assert isinstance(self.n_iter, int)
        assert isinstance(self.n_local, int)
        assert isinstance(self.n_ini, int)
        assert isinstance(self.temp, float)
        assert self.alpha >= 0. and self.alpha <= 1.
        assert self.c_rate > 0. and self.c_rate <= 1.
        assert self.score is not None
        if self.skwds is not None:
            mykwargs = self.skwds.split()
            skwds = dict()
            for k in mykwargs:
                kv = k.split(":")
                skwds[kv[0]] = kv[1]
            self.skwds = skwds
        else:
            self.skwds = dict()
        if self.kkwds is not None and self.kernel is None:
            raise ValueError("Kernel keywords set but no transformation kernel given")
        elif self.kernel is not None:
            mykwargs = self.kkwds.split()
            kkwds = dict()
            for k in mykwargs:
                kv = k.split(":")
                kkwds[kv[0]] = kv[1]
            self.kkwds = kkwds
        else:
            self.kkwds = dict()            
        return None
    
    def run(self, X, N_sel=0.05):
        """
        Get input data and run GRASP
        - X:     input data
        - N_sel: fraction of needed points
        """
        
        #check input
        assert isinstance(X, np.ndarray)
        self.X = X
        if self.seed is not None:
            assert isinstance(self.seed,int)
        self.rng = np.random.default_rng(self.seed)
        self.N_sel = int(N_sel * self.X.shape[0])
        self.n_neigh = int(self.n_neigh * self.X.shape[0])
        assert self.n_ini < self.N_sel
        self.all_labels = set(list(range(self.X.shape[0])))
        best_score  = False
        best_labels = None
        
        #distance, transformations and initialization
        self.D = sp.spatial.distance.squareform(sp.spatial.distance.pdist(X,metric=self.metric))
        self.kkwds['X'] = self.X
        self.kkwds['D'] = self.D        
        if self.kernel is not None:
            self.D = self.kernel(np.arange(self.X.shape[0]), **self.kkwds)
        self.skwds['X'] = self.X
        self.skwds['D'] = self.D
        init_labels = self.init_solution()
        
        # main loop
        for i in range(self.n_iter):
            if self.verbose >1:
                print("building",i)
            build_score, build_labels = self.construction(init_labels)
            if self.n_local > 0:
                if self.verbose > 1:
                    print("local optimization",i)
                opt_score, opt_labels = self.local_search(build_score, build_labels)
            elif self.n_local == 0:
                opt_score  = build_score                
                opt_labels = build_labels
            else:
                raise ValueError("n_local is >=0")
            if not best_score or opt_score < best_score:
                best_labels = opt_labels
                best_score  = opt_score
                if self.verbose > 0:
                    print("--- iter, best_score, opt_score, build_score ",i,best_score, opt_score, build_score)
        return best_score, best_labels, self.X[best_labels]    

    def build_RCL(self, solution):
        """
        Build the restricted candidate list from
        a set of candidate feature vectors C
        """
        candidates = list(self.all_labels.difference(solution))
        RCL = list()
        # gain = list()
        #for c in candidates:
        #    tmp = solution + [c]
        #    print(tmp)
        #    gain.append(self.score(tmp, self.X, self.D))
        gain = [self.score(solution + [c], **self.skwds) for c in candidates]
        v_min = np.min(gain)
        v_max = np.max(gain)
        for i, g in enumerate(gain):
            if g >= v_min and g <= v_min + self.alpha*(v_max - v_min):
                RCL.append(i)        
        return RCL

    def construction(self, labels):
        """
        Build a candidate solution selecting a random
        element from the RCL for the needed number of elements
        """
        if len(labels) == self.N_sel:
            M = 0
        else:
            M = len(labels)
        build_labels = [i for i in labels]
        for i in range(self.N_sel - M):
            #if i % 10 == 0: print(i)
            # build RCL
            RCL = self.build_RCL(build_labels)
            # pick a random element from RCL
            selected = np.random.choice(RCL)
            build_labels.append(selected)
        build_score = self.score(build_labels, **self.skwds)
        return build_score, build_labels

    def local_search(self, build_score, build_labels):
        """
        select a random element in the neighbourhood 
        of a solution and accept if there is gain
        or limited loss (SA like)
        """          
        # do SA
        current_labels = deepcopy(build_labels)
        current_score  = build_score
        for l, label in enumerate(current_labels):
            temp_labels = deepcopy(current_labels)
            current_temp   = self.temp
            for it in range(self.n_local):
                neighs = np.argsort(self.D[l])[:self.n_neigh]
                swap = np.random.choice(neighs)
                temp_labels[l] = swap
                temp_score = self.score(temp_labels, **self.skwds)
                if temp_score < current_score:
                    current_labels[l] = swap
                    current_score = temp_score
                else:
                    Ediff   = (temp_score - current_score)/ (self.boltzmann * current_temp)
                    Bweight = np.exp(-Ediff)
                    coin = np.random.rand()
                    if Bweight > coin:
                        current_labels[l] = swap
                        current_score = temp_score
                current_temp = self.c_rate * current_temp
        return current_score, current_labels

    def init_solution(self):
        """
        pick n_ini elements w/o repetition to seed the solution
        """
        #labels = self.rng.integers(0, high=self.X.shape[0], size=self.n_ini, endpoint=False)
        dj = np.sum(self.D, axis=1)
        labels = np.argsort(dj)[:self.n_ini]
        return labels        
        