# various clustering related utilities for MD trajectories 
# G Mancini September 2019

# NB to convert RMSD matrix from gmx, where 376 is number of FRAMES in data set,
# 4 the byte size (4 for a float) and %06f the desired output format
# hexdump -v -e '376/4 "%06f "' -e '"\n"' rmsdm_hex.dat > rmsdm.dat
#              no. of entries/ sizeof(one entry) format of each entry

#import mdtraj as md
import numpy as np
import scipy.stats as st
import scipy.spatial.distance as sd

def size_indep_rho(RMSD,rgyr):
    """
    size indep. distance based on rmsd and radius of gyr
    by Maiorov and Crippen
    Size-Independent Comparison of Protein Three-Dimensional Structures. 
    Proteins: Structure, Function, and Bioinformatics 1995, 22 (3), 273–283.
    """
    D2  = RMSD**2
    R2  = rgyr**2
    R2i = np.expand_dims(R2,axis=0).T
    rho = 2.*RMSD / np.sqrt(R2+R2i-D2)
    return rho

def check(RMSD,rgyr):
    npoints = rgyr.shape[0]
    rho = np.zeros((npoints,npoints))
    for i in range(npoints-1):
        for j in range(i,npoints):
            szind = 2.*RMSD[i,j]/np.sqrt(rgyr[i]**2+rgyr[j]**2-RMSD[i,j]**2)
            rho[i,j] = szind
            rho[j,i] = szind
    return rho  

def USR(coords,mass=None):
    """
    ultrafrash shape recognition
    see Ballester, Proc. roy. soc. A, 2007
    """
    # - ctd = mol. geom. centroid
    ctd = np.average(coords,axis=0,weights=mass)
    D_ctd = np.linalg.norm(coords-ctd,axis=1)
    # - cst = nearest atom to ctd    
    cst = coords[np.argmin(D_ctd)]
    # - fct = farthest atom from ctd
    fct = coords[np.argmax(D_ctd)]
    # - distances from fct
    D_fct = np.linalg.norm(coords-fct,axis=1)
    # - ftf = farthest from fct    
    ftf = coords[np.argmax(D_fct)]
    usr = np.empty(12)
    #distances from cst and ftf
    D_cst = np.linalg.norm(coords-cst,axis=1)
    D_ftf = np.linalg.norm(coords-ftf,axis=1)
    for i,j in enumerate((D_ctd,D_cst,D_fct,D_ftf)):
        usr[i*3]   = np.mean(j)
        usr[i*3+1] = np.std(j)
        usr[i*3+2] = np.cbrt(st.skew(j))
        #usr[i*3+1] = np.var(j)
        #usr[i*3+2] = st.skew(j)
    return usr

def usr_mat(traj, sel = False, return_matrix=False):
    """
    given a mdtraj trajectory object calculates USR for sel atom indexes
    operates on all frames
    optionally return a USR distance matrix
    the use of manhattan d. is part of the procedure
    """
    nframes = traj.xyz.shape[0]
    usr = np.empty((nframes,12))
    if sel:
        for f in range(nframes):
            usr[f] = USR(traj.xyz[f][:,sel])
    else:
        for f in range(nframes):
            usr[f] = USR(traj.xyz[f])
    if return_matrix:
        distmat = np.zeros((nframes,nframes))
        for f in range(nframes-1):
            for g in range(f+1,nframes):
                distmat[f,g] = sd.cityblock(usr[f],usr[g])
        return distmat
    else:
        return usr
    
def one_2_two_ndx(k,n):
    """
    return i,j indexes for
    single index 1D triangular
    matrix
    """
    #k = ((n*(n-1))/2) - ((n-i)*((n-i)-1))/2 + j - i - 1
    i = n - 2 - int(sqrt(-8*k + 4*n*(n-1)-7)/2.0 - 0.5)
    j = k + i + 1 - n*(n-1)/2 + (n-i)*((n-i)-1)/2
    return i, j

def tri_2_square(dim, flat):
    """
    return symmetric square matrix from
    flat 1D array
    """
    tri = np.zeros((dim, dim))
    tri[np.triu_indices(dim, 1)] = flat
    tri[np.tril_indices(dim, 1)] = flat
    return tri

def gini_coefficent(data, reverse=False, diff=False):
    """
    given a distribution of values estimate the Gini coefficent
    as misure of the distribution inequality
    see https://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    reverse=True means to consider the weight as 1./data
    if diff is a float values are transformed as v=diff - v
    """
    if diff != False:
        values = diff - data
    else:
        values = data
    assert np.any(values) > 0
    if np.any(values) < 1e-6:
        values += 1e-6
    if reverse:
        values = 1./values
    num = np.abs(np.subtract.outer(values, values)).mean()
    G = 0.5*(num/np.mean(values))
    return G

def cont_coord_num(X, ion, ox, smooth, r0):
    """
    continuous coordination number as defined in
    doi:10.1021/ct400988e
    s = \sum_{ox water} 1 - \frac{1}{1+exp(-a(r_i - r0)}
    a is a smoothing parameter
    r0 is the first shell cutoff
    X are the coordinates
    ion the ion index
    ox the oxygen atom indexes
    """
    xion = X[ion].reshape(1,-1)
    dist = sd.cdist(X[ox], xion)[:,0]
    sfunc = lambda r: 1.-1./(1+np.exp(-smooth*(r-r0)))
    sarray= np.apply_along_axis(sfunc, 0, dist)
    s = np.sum(sarray)
    return s
