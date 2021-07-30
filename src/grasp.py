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
        - n_ini:    fraction of seed elements for initialization (1)
        - N_sel:    fraction of needed points (default 0.01 of total points)
        - maxBuild: maximum attempts to create a solution in the building phase
                    (default N_sel*2)
        - nBuild    fraction of random elements selected to create candidate list for RCL
                    and for initialization
        - n_iter:   number of iterations (10)
        - do_local: do a linear search in the neighbourhood of each point after construction
        - alpha:    greedyness factor in [0,1] (0.75)
        - verbose:  verbosity level (1)
        - metric:   metric of distance between data points (euclidean)
        - seed:     random seed
        - score:    score function
        - skwds:    keyword for the score function (a string like "k1:v1 k2:v2")
        """
        # get values
        prop_defaults = {
            'N_sel'     : 0.01,
            'n_ini'     : 0.001,
            'maxBuild'  : None,
            'nBuild'    : 0.2,
            'n_iter'    : 10,
            'do_local'  : False,
            'alpha'     : 0.1,
            'verbose'   : 1,
            'metric'    : 'euclidean',
            'seed'      : None,
            'score'     : None,            
            'skwds'     : None
            }
        for (prop, default) in prop_defaults.items():
            setattr(self, prop, kwargs.get(prop, default))
            
        # check input
        assert self.N_sel > 0. and self.N_sel < 1.
        assert self.n_ini > 0. and self.n_ini < 1.
        assert self.nBuild > 0. and self.nBuild < 1.
        assert isinstance(self.n_iter, int)
        assert isinstance(self.do_local, bool)
        assert self.alpha >= 0. and self.alpha <= 1.
        if self.alpha == 0:
            print("--- alpha=0 is a greedy run w/o randomness in the RCL")
        
        # score 
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
            
        #random seed 
        if self.seed is not None:
            assert isinstance(self.seed,int)
        self.rng = np.random.default_rng(self.seed)
        return None
    
    def run(self, X, init_labels=None):
        """
        Get input data and run GRASP
        - X:        input data
        """
        #check input
        assert isinstance(X, np.ndarray)
        self.X = X
        
        self.N_sel  = int(self.N_sel * self.X.shape[0])
        self.n_ini  = max(int(self.n_ini * self.X.shape[0]), 1)
        self.nBuild = int(self.nBuild * self.X.shape[0])
        if self.maxBuild == None:
            self.maxBuild = 2*self.N_sel
        else:
            self.maxBuild = int(maxBuild * self.X.shape[0])
        assert self.maxBuild > self.N_sel
        
        print("-- Starting GRASP with ",self.N_sel," points and ",self.maxBuild," building attempts")
            
        self.all_labels = set(list(range(self.X.shape[0])))
        best_score  = False
        best_labels = None

        if init_labels == None:
            init_labels = self.init_solution()
        
        # main loop
        for i in range(self.n_iter):
            if i>0:
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
        pcandidates = list(self.all_labels.difference(solution))
        candidates = self.rng.choice(pcandidates, size=self.nBuild, replace=False)
        RCL = list()
        gain = [self.score(solution + [c], **self.skwds) for c in candidates]
        v_min = np.min(gain)
        v_max = np.max(gain)
        if self.alpha > 0:
            for i, g in enumerate(gain):
                if g >= v_min and g <= v_min + self.alpha*(v_max - v_min):
                    RCL.append(candidates[i])
        elif self.alpha == 0:
            RCL.append(candidates[np.argmin(gain)])
        return len(RCL), RCL

    def construction(self, labels):
        """
        Build a candidate solution selecting a random
        element from the RCL for the needed number of elements
        """
        attempts = 0
        while len(labels) < self.N_sel and attempts < self.maxBuild:
            nRCL, RCL = self.build_RCL(labels)
            if self.verbose > 2:
                print("---- built RCL of len ", nRCL)
            if nRCL > 1:
                labels.append(self.rng.choice(RCL, replace=False))
            else:
                labels.append(RCL.pop())
            attempts += 1
        if attempts >= self.maxBuild:
            print("ERROR: could build a solution with given paramenters")
            raise ValueError("Decrease N_sel or increase alpha")
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
        labels = [int(self.rng.uniform(high=self.X.shape[0]))]
        #if self.n_ini > 1:
        #    for l in range(1, self.n_ini):
        #        pcandidates = self.all_labels.difference(labels)
        #        candidates = self.rng.choice(pcandidates, size=self.nBuild/2., replace=False)
        #        gain = [self.score(solution + [c], **self.skwds) for c in candidates]
        #        ng = np.argsort(gain)
        ##       roulette selection
        #        coin = self.rng.uniform(high=np.sum(gain))
        #        gsum = 0.
        #        for i in ng:
        #           gsum = gsum + gain[i]
        #           if gsum >= coin:
        #               labels.append(i)
        #               break
        return labels        
        
