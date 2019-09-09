def myrmsd(X1,X2,mask=None):
    #Kabsch algorithm
    #see https://cnx.org/contents/HV-RsdwL@23/Molecular-Distance-Measures
    if mask is None:
        mask = list(range(X1.shape[0]))
    assert X1[mask].shape == X2[mask].shape
    nat = len(mask)
    Com1 = np.mean(X1[mask],axis=0)
    Com2 = np.mean(X2[mask],axis=0)
    C1 = X1[mask]-Com1
    C2 = X2[mask]-Com2
    Cov = np.dot(C1.T, C2)
    V, S, W = np.linalg.svd(Cov)
    d = np.sign(np.linalg.det(Cov))
    D = np.eye(3)
    D[2,2] = d
    R = np.dot(V,np.dot(D,W))
    rotC2 = np.dot(C2, R)
    displ = np.mean(C1-rotC2,axis=0)
    rotC2 = rotC2 -displ
    dim = C2.shape[1]
    rmsd = 0.
    for v, w in zip(C1, rotC2):
        rmsd = sum([(v[i] - w[i])**2.0 for i in range(dim)])
    return np.sqrt(rmsd/nat)
