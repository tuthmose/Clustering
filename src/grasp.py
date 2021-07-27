from copy import deepcopy

import numpy as np
import random
import scipy as sp
               
def sumdist(labels, **kwargs):
    """
    this is just the inverse of the sum of distances
    """
    D = kwargs.get('D')
    DT = np.sum(D[labels])
    return 1.0/DT

def gauk(labels, **kwargs):
    D = kwargs.get('D')
    sigma = float(kwargs.get('sigma'))
    return np.exp( (-D**2)/(2.*sigma**2) )
                
class simpleGRASP:
    
    def __init__(self, **kwargs):
        """
        main fuction for Greedy Randommized Adaptive Procedure
        return the N_select elements from the input data set X that
        implemented as a minimization problem
        - n_ini:    number of seed elements b4 construction (1)
        - n_iter:   number of iterations (100)
        - do_local: do a linear search in the neighbourhood of each point after construction
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
            'do_local'  : False,
            'temp'      : 500.,
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
        assert isinstance(self.do_local, bool)
        assert isinstance(self.temp, float)
        if isinstance(self.n_ini, int):
            self.restart = False
        elif isinstance(self.n_ini, list):
            self.restart = True
        else:
            raise ValueError("n_ini is an int or a label list")
            
        assert self.alpha >= 0. and self.alpha <= 1.
        assert self.c_rate > 0. and self.c_rate <= 1.
        if self.alpha == 0:
            print("--- alpha=0 is a greedy run w/o randomness in the RCL")
        
        # score and transformation kernel
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
            
        #random seed 
        if self.seed is not None:
            assert isinstance(self.seed,int)
        self.rng = np.random.default_rng(self.seed)
        return None
    
    def run(self, X, N_sel=0.05):
        """
        Get input data and run GRASP
        - X:     input data
        - N_sel: fraction of needed points
        """
        print("-- Starting GRASP")
        #check input
        assert isinstance(X, np.ndarray)
        self.X = X
        self.N_sel = int(N_sel * self.X.shape[0])
        if self.restart:
            print("--- restarting", self.N_sel, len(self.n_ini))
            assert len(self.n_ini) == self.N_sel
        else:
            assert self.n_ini < self.N_sel
        self.N_max = 4 * self.N_sel
            
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
            init_labels = self.init_solution()
            build_score, build_labels = self.construction(init_labels)
            if self.verbose >1:
                print("building", i, build_score)            
            if self.do_local:
                opt_score, opt_labels = self.local_search(build_score, build_labels)
                if self.verbose > 1:
                    print("local optimization", i, opt_score)                
            else:
                opt_score  = build_score                
                opt_labels = build_labels
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
        gain = [self.score(solution + [c], **self.skwds) for c in candidates]
        v_min = np.min(gain)
        v_max = np.max(gain)
        for i, g in enumerate(gain):
            if g >= v_min and g <= v_min + self.alpha*(v_max - v_min):
                if self.alpha > 0:
                    RCL.append((i, g))
                else:
                    RCL.append(i)
        if self.alpha > 0 :
            RCL = sorted(RCL, key=lambda cand: cand[1])
            RCL = [c[0] for c in RCL]
        return RCL

    def construction(self, labels):
        """
        Build a candidate solution selecting a random
        element from the RCL for the needed number of elements
        """
        if isinstance(labels, int):
            labels = [labels]
        for i in range(self.N_sel - self.n_ini):
            RCL = self.build_RCL(labels)
            selected = self.rng.choice(RCL, replace=False)
            labels.append(selected)
        score = self.score(labels, **self.skwds)
        return score, labels

    def local_search(self, build_score, build_labels):
        """
        perform a linear search on the N_max nearest neighbours 
        of each node not nearer to another node
        """
        opt_labels = deepcopy(build_labels)
        # assign elements to nearest node
        N = self.X.shape[0]
        nodes = -np.ones(N, dtype='int')
        D = sp.spatial.distance.squareform(sp.spatial.distance.pdist(self.X))
        #for pj in range(N):
        #    nearest = np.argmin(self.D[pj,opt_labels])
        #    nodes[pj] = opt_labels[nearest]
        #print(set(nodes))
        print(opt_labels)
        points = tuple(set(np.arange(N, dtype='int')).difference(opt_labels))
        for pk in opt_labels:
            Pj = tuple(set(opt_labels).difference([pk]))
            oD = np.min(D[Pj, :][:, points])
            print(oD)
            nD = np.where(D[pk] <= oD)
            print(nD, D[pk,pk], D[pk,Pj])
            nodes[nD[0]] = o
            raise ValueError
        print(set(nodes), opt_labels, self.D[:, diff].shape)
        # do the LS
        temp_labels = deepcopy(opt_labels)
        for i in range(self.N_sel):
            # order the elements
            dd = self.D[i, nodes==i]
            NN = np.argsort(dd)[1:][::-1]
            base_gain = self.score(opt_labels, **self.skwds)
            for j in NN:
                if j not in opt_labels:
                    temp_labels[i] = j
                    gain = self.score(temp_labels, **self.skwds)
                    if gain < base_gain:
                        opt_labels[i] = j
        opt_score = self.score(opt_labels, **self.skwds)
        return opt_score, opt_labels

    def init_solution(self):
        """
        pick n_ini elements w/o repetition to seed the solution
        """
        #labels = self.rng.integers(0, high=self.X.shape[0], size=self.n_ini, endpoint=False)
        if self.restart:
            labels = self.n_ini
        else:
            labels = self.rng.choice(self.n_ini, replace=False)
        return labels        
        
