**Clustering algorithms and external/internal validation criteria**

`src/` includes source files for clustering and clustering metrics metods
`devel/` includes work in progress modules
`test/` include jupyter notebooks with various tests 
`validation/` includes notes about clustering validation

Algorithms included were selected because:
- (i) they were not present at the time in scikit-learn (but perhaps they are) e. g. Density Peaks or pseudoF
- (ii) they were but I needed some feature changed (e. g. SNN or PAM).

all clustering algorithms are imported by `myclusters.py`
all validation ones are imported by `mymetrics.py`
`mdutils` includes various helper functions for Molecular Dynamics 
trajectories includeing the USR distance (pure Python version)

Implementation is pure Python/numpy and *slow*. I did not even
bother with unraveling triangular matrices.

