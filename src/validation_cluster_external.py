import numpy as np
import numpy.ma as MA
from math import log,sqrt
from scipy.spatial.distance import cdist,pdist,squareform
from scipy.special import binom

# External cluster validation criteria
# Mostly useful for test/development

# G Mancini Sept. 2019
   
class ext_cluster_eval(object):
    """
    implements external cluster evaluation methods
    such as NMI and Jaccard
    right now, uses only prediction and prior labels
    """
    def __init__(self,**kwargs):
        for key, value in kwargs.items():
            if key is 'ground':
                self.ground = value
            if key == 'prediction':
                self.prediction = value
        assert isinstance(self.ground,np.ndarray)
        assert isinstance(self.prediction,np.ndarray)
        labels_ground = set(self.ground)
        labels_pred = set(self.prediction)
        self.labelsG = list(labels_ground)
        self.labelsP = list(labels_pred)
        
    def compute(self):
        return None
        
class NMI(ext_cluster_eval):
    
    def compute_mi(self):
        """
        compute mutual information
        """
        N = float(self.ground.shape[0])
        GR = list()
        MJ = list()
        result = 0.0
        for l in self.labelsG:
            GR.append(self.ground == l)
            MJ.append(len(self.ground[GR[l]]))
        for l in self.labelsP:
            cl_i = self.prediction == l
            n_i = len(self.prediction[cl_i])
            for LL,ll in enumerate(GR):
                n_ij = len(self.ground[np.logical_and(cl_i,ll)])
                try:
                    result = result + n_ij * log (N*n_ij/(n_i*MJ[LL]))
                except:
                    continue
        return result/N
    
    def compute(self):
        """
        calculates Normalized mutual information
        """
        N = float(self.ground.shape[0])
        H_clus = 0.0
        H_part = 0.0
        for l in self.labelsP:
            cl_i = self.prediction[self.prediction == l]
            prob_ci = float(len(cl_i))/N 
            try:
                H_clus = H_clus - prob_ci*log(prob_ci)
            except:
                continue
        for l in self.labelsG:
            gr_j = self.ground[self.ground == l]
            prob_pj = float(len(gr_j))/N 
            try:
                H_part = H_part - prob_pj*log(prob_pj)
            except:
                continue
        MI  = self.compute_mi()
        result = MI/sqrt(H_clus*H_part)
        return result
    
class Jaccard(ext_cluster_eval):
    def compute(self):
        N = float(self.ground.shape[0])
        TP = 0
        FP = 0
        FN = 0
        GR = list()
        #
        for l in self.labelsG:
            GR.append(self.ground == l)
            m_j  = len(self.ground[GR[l]])
            FN = FN + binom(m_j,2)            
        #
        for l in self.labelsP:
            cl_i = self.prediction == l
            n_i = len(self.prediction[cl_i])
            FP = FP + binom(n_i,2)
            for ll in GR:
                n_ij = len(self.ground[np.logical_and(cl_i,ll)])
                TP = TP + binom(n_ij,2)
        FP = FP - TP
        FN = FN - TP
        #print(TP,FP,FN)
        Jac = TP/(TP+FP+FN)
        return Jac
