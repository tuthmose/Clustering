**Clustering algorithms and external/internal validation criteria**

`src/` includes source files for clustering and clustering metrics metods
`devel/` includes work in progress modules
`test/` include jupyter notebooks with various tests 
`validation/` includes notes about clustering validation

Algorithms included were selected because:
- (i)  they were not present at the time in scikit-learn or extras (but perhaps they are now) e. g. Density Peaks or SNN
- (ii) self teaching

all clustering algorithms are imported by `myclusters.py`
all validation ones are imported by `mymetrics.py`
`mdutils` includes various helper functions for Molecular Dynamics 
trajectories including, e. g. USR distance 

Implementation is pure Python/numpy and *slow*. I did not even
bother with unraveling triangular matrices (not often at least).

