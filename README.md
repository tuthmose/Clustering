## Clustering algorithms and external/internal validation criteria

`src/` includes source files for clustering and clustering metrics metods
`devel/` includes work in progress modules
`test/` include jupyter notebooks with various tests 
`validation/` includes notes about clustering validation

Algorithms included were selected because:
- they were not present at the time in scikit-learn or extras (but perhaps they are now) e. g. Density Peaks or SNN
- self teaching

all clustering algorithms are imported by `myclusters.py`
all validation ones are imported by `mymetrics.py`
`mdutils` includes various helper functions for Molecular Dynamics 
trajectories including, e. g. USR distance 

Implementation is pure Python/numpy and *slow*. I did not even
bother with unraveling triangular matrices (not often at least).

The following conventions in source and notebook holds:

1. *X* when given or defined, is ALWAYS the feature or 
    coordinate matrix, [npoints x nfeatures]
2. *D*, the distance matrix is always expected to be a nxn square symmetric matrix 
    with pair distances between the data *ALL* data set 
    elements:
      0    1   2   3
      0 - d00 d01 ...
      1 - d10 d11 ...
      2 - ... 
      3 - ...
3. *W* are the weights of data points ([npoints x 1] or None)
4. *clusters* is always expected to be a list of all elements the data set where the elements are identified with the label of the corresponding CLUSTER (0 to n) or cluster centroids IF AVAILABLE; If centroids are available set(clusters) ALWAYS gives the centroids labels cluster labels are ALWAYS positive integers a label of -1 ALWAYS indentifies noise. 


please cite one or more of the following studies:
- (1) Mancini, G.; Fusè, M.; Lipparini, F.; Nottoli, M.; Scalmani, G.; Barone, V. Molecular Dynamics Simulations Enforcing Nonperiodic Boundary Conditions: New Developments and Application to the Solvent Shifts of Nitroxide Magnetic Parameters. J. Chem. Theory Comput. 2022, acs.jctc.2c00046. https://doi.org/10.1021/acs.jctc.2c00046.
(2) Mancini, G.; Fusè, M.; Lazzari, F.; Chandramouli, B.; Barone, V. Unsupervised Search of Low-Lying Conformers with Spectroscopic Accuracy: A Two-Step Algorithm Rooted into the Island Model Evolutionary Algorithm. J. Chem. Phys. 2020, 153 (12), 124110. https://doi.org/10.1063/5.0018314.
(3)  Mancini, G.; Del Galdo, S.; Chandramouli, B.; Pagliai, M.; Barone, V. Computational Spectroscopy in Solution by Integration of Variational and Perturbative Approaches on Top of Clusterized Molecular Dynamics. J. Chem. Theory Comput. 2020, 16 (9), 5747–5761. https://doi.org/10.1021/acs.jctc.0c00454.

