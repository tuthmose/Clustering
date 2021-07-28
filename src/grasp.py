from copy import deepcopy

import numpy as np
import random
import scipy as sp
                              
class simpleGRASP:
    
    def __init__(self, **kwargs):
        """
        main fuction for Greedy Randommized Adaptive Procedure
        return the N_select elements from the input data set X that
        implemented as a minimization problem
        - n_ini:    number of seed elements b4 construction (1)
        - n_iter:   number of iterations (100)
        - do_local: do a linear search in the neighbourhood of each point after construction
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
        if isinstance(self.n_ini, int):
            self.restart = False
        elif isinstance(self.n_ini, list):
            self.restart = True
        else:
            raise ValueError("n_ini is an int or a label list")
            
        assert self.alpha >= 0. and self.alpha <= 1.
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
        #check input
        assert isinstance(X, np.ndarray)
        self.X = X
        self.N_sel = int(N_sel * self.X.shape[0])
        print("-- Starting GRASP with ",self.N_sel," points")
        
        if self.restart:
            print("--- restarting", self.N_sel, len(self.n_ini))
            assert len(self.n_ini) == self.N_sel
        else:
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
        
        self.NN = np.argsort(self.D, axis=1)        
        self.skwds['X']  = self.X
        self.skwds['D']  = self.D
        self.skwds['NN'] = self.NN
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
        print("building",solution, candidates)        
        RCL = list()
        gain = [self.score(solution + [c], **self.skwds) for c in candidates]
        v_min = np.min(gain)
        v_max = np.max(gain)
        if self.alpha > 0:
            print(v_min, v_min+self.alpha*(v_max - v_min),v_max)
            for i, g in enumerate(gain):
                if g >= v_min and g <= v_min + self.alpha*(v_max - v_min):
                    RCL.append(i)
        elif self.alpha == 0:
            print("greedy")
            gain_best = np.argmin(gain)
            RCL = [gain_best]
        return RCL

    def construction(self, labels):
        """
        Build a candidate solution selecting a random
        element from the RCL for the needed number of elements
        """
        if isinstance(labels, int):
            labels = [labels]
        RCL = self.build_RCL(labels)
        if self.verbose > 2:
            print("---- built RCL")
        for i in range(self.N_sel - self.n_ini):
            #RCL = self.build_RCL(labels)
            if self.verbose > 2:
                print("---- building step ",i)
            while len(labels) < self.N_sel:
                selected = self.rng.choice(RCL, replace=False)
                if selected not in labels:
                    labels.append(selected)
                print(selected, RCL, labels)
                raise ValueError                    
        score = self.score(labels, **self.skwds)
        return score, labels

    def local_search(self, build_score, build_labels):
        """
        perform a linear search on the "Voronoi cell" (if not empty)
        of each node not nearer to another node
        """
        opt_labels = deepcopy(build_labels)
        # assign elements to nearest node
        N = self.X.shape[0]
        nodes = -np.ones(N, dtype='int')
        for o in opt_labels:
            nodes[o] = o
        points = tuple(set(np.arange(N, dtype='int')).difference(opt_labels))
        for pk in opt_labels:
            Pj = tuple(set(opt_labels).difference([pk]))
            oD = np.min(self.D[Pj, :][:, points])
            nD = np.where(self.D[pk,points] <= oD)
            nodes[nD[0]] = o
        # do the LS
        for k, pk in enumerate(opt_labels):
            if len(nodes[nodes == pk]) > 1:
                temp_labels = deepcopy(opt_labels)
                # order the elements
                dd = self.D[k, nodes==pk]
                NN = np.argsort(dd)[1:][::-1]
                base_gain = self.score(opt_labels, **self.skwds)
                for j in NN:
                    if j not in opt_labels:
                        temp_labels[k] = j
                        gain = self.score(temp_labels, **self.skwds)
                        if gain < base_gain:
                            opt_labels[k] = j
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
        
