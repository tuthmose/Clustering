# various utilities for MD trajectories 
# to use before clustering

import mdtraj as md
import numpy as np
import scipy.stats as st
import scipy.spatial.distance as sd

def size_indep_rho(RMSD,rgyr):
    """
    size indep. distance based on rmsd and radius of gyr
    by Maiorov and Crippen
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
    # - ctd mol. geom. centroid
    ctd = np.average(coords,axis=0,weights=mass)
    D_ctd = np.linalg.norm(coords-ctd,axis=1)
    # - cst nearest atom to ctd    
    cst = coords[np.argmin(D_ctd)]
    # - fct farthest atom from ctd
    fct = coords[np.argmax(D_ctd)]
    # - distances from fct
    D_fct = np.linalg.norm(coords-fct,axis=1)
    # - ftf farthest from fct    
    ftf = coords[np.argmax(D_fct)]
    usr = np.empty(12)
    #distances from cst and ftf
    D_cst = np.linalg.norm(coords-cst,axis=1)
    D_ftf = np.linalg.norm(coords-ftf,axis=1)
    for i,j in enumerate((D_ctd,D_cst,D_fct,D_ftf)):
        usr[i*3]   = np.mean(j)
        usr[i*3+1] = np.var(j)
        usr[i*3+2] = st.skew(j)
    return usr

def usr_mat(traj,sel = False):
    """
    given a mdtraj trajectory object
    calculates USR for sel atom indexes
    then generate USR distance matrix
    operates on all frames
    """
    nframes = traj.xyz.shape[0]
    usr = np.empty(nframes)
    if sel:
        for f in range(nframes):
            usr[f] = USR(traj.xyz[f][:,sel])
    else:
        for f in range(nframes):
            usr[f] = USR(traj.xyz[f])                  
    distmat = np.zeros(nframes,nframes)
    for f in range(nframes-1):
        for g in range(f+1,nframes):
            distmat[f,g] = sd.cityblock(usr[f],usr[g])
    return distmat
    