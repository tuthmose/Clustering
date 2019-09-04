import numpy as np
inmport scipy as sp

print("IMPORT ADDITIONAL MODULES FOR CLUSTERING\n")
from sklearn.neighbors import kneighbors_graph,radius_neighbors_graph
from sklearn.decomposition import PCA
import myclusters2
import mymetrics2
import DBCV
import time
print("METRIC DEFINED GLOBALLY")

start = time.time()
print "Loading the matrix"
# Ignore first columns, cpptraj writes frame number in column 1
draw = np.loadtxt('dmat_raw_comp.dat',dtype=np.float64)   # check matrix
draw = draw[:,1:]/10.0                                    # make it nm from Ang

rho_r = np.loadtxt("rgyr_r.dat")[:,1]
rho_r = myclusters2.size_indep_rho(draw,rgyr_r)
nfeatures = 897

print("Total time for loading in Mins:", ( time.time() - start) /60.0)
print("Computing  Minpts-Epsilon pairs")

spacedef="precomputed"
ndata = draw.shape[1]  # check matrix 
met1=np.zeros( (5,5), dtype=np.float )
perc = [1.0,1.25,1.5,1.75,2.0]
mynoise="ignore"

outf= [ "R18-1.DP" , "R18-125.DP" , "R18-15.DP" , "R18-175.DP", "R18-2.DP" ]
print("Estimating  DP")

for i  in [0,1,2,3,4]:
    estimator = myclusters2.density_peaks(cutoff="auto",percent=perc[i],metric=spacedef, kernel="gaussian",D = draw)  #check matrix
    rho,delta = estimator.decision_graph()
    N = np.vstack((rho,delta))  #save this stack
    #np.savetxt( outf[i], N, fmt="%5.3f")
    
    centroids,points = estimator.get_centroids(rmin=1250,dmin=0.16)
    clusters = estimator.assign_points()

    print("Creating halos")
    robust_clusters = estimator.create_halo()
    hal = 0 ; non_hal = 0
    for c in centroids:
        N = len(robust_clusters[robust_clusters==c])
        H = len(clusters[clusters==c]) - N
        hal += H 
        non_hal += N
        #print "%d elements and %d HALO points found in cluster %d" % (N,H,c)
    Cor1 = ( ndata - hal )/float(ndata)
    print("ndata,non_hal,cor",ndata,non_hal,Cor1)
    Cor2 = 1.0/Cor1

    # Print details and plot decision graphs
    print("EST cutoff:", estimator.cutoff)
    print("Found Nclus",estimator.nclusters)
    #print "Chosen rmin:100, dmin=0.2"
    print("Centroids: ",centroids)

    # Obtain metrics and save it in Met-1 array
    newlabels = estimator.assign_points()
    
    met1[i,0]  =  metrics.silhouette_score( draw , newlabels , metric = spacedef) / Cor1     #check matrix 
    eval1 = mymetrics2.cluster_eval(metric = spacedef, clusters = newlabels, D = draw )       #check matrix

    # collect GM-DBI
    met1[i,1] = eval1(noise=mynoise,method="DBI") / Cor2
    # collect GM-DUNN
    met1[i,2] = eval1(noise=mynoise,method="Dunn",inter="center",intra="allmax")/Cor1
    # collect GM-PSF,WSS
    met1[i,3:]= eval1(noise=mynoise,method="psF",centroid=True)
    met1[i,3] = met1[i,3]/Cor1
    met1[i,4] = met1[i,4]/Cor2
    dens_score = DBCV.DBCV(clusters=newlabels,metric=spacedef,NF=nfeatures,D=draw)
    met1[i,5] = dens_score.calc_score(meas='kernel')    
    
    print "**** End of Run %d ********"  % (i+1)

np.savetxt("R18-MET_rmsd_r.DP",met1,fmt="%6.4f")


for i  in [0,1,2,3,4]:
    estimator = myclusters2.density_peaks(cutoff="auto",percent=perc[i],metric=spacedef, kernel="gaussian",D = rho_r)  #check matrix
    rho,delta = estimator.decision_graph()
    N = np.vstack((rho,delta))  #save this stack
    np.savetxt( outf[i], N, fmt="%5.3f")
    
    centroids,points = estimator.get_centroids(rmin=1250,dmin=0.16)
    clusters = estimator.assign_points()

    print("Creating halos")
    robust_clusters = estimator.create_halo()
    hal = 0 ; non_hal = 0
    for c in centroids:
        N = len(robust_clusters[robust_clusters==c])
        H = len(clusters[clusters==c]) - N
        hal += H 
        non_hal += N
        print("%d elements and %d HALO points found in cluster %d" % (N,H,c))
    Cor1 = ( ndata - hal )/float(ndata)
    print("ndata,non_hal,cor",ndata,non_hal,Cor1)
    Cor2 = 1.0/Cor1

    # Print details and plot decision graphs
    print("EST cutoff:", estimator.cutoff)
    print("Found Nclus",estimator.nclusters)
    #print "Chosen rmin:100, dmin=0.2"
    print("Centroids: ",centroids)

    # Obtain metrics and save it in Met-1 array
    newlabels = estimator.assign_points()
    
    met1[i,0]  =  metrics.silhouette_score( rho_r, newlabels , metric = spacedef) / Cor1     #check matrix 
    eval1 = mymetrics2.cluster_eval(metric = spacedef, clusters = newlabels, D = rho_r )       #check matrix

    # collect GM-DBI
    met1[i,1] = eval1(noise=mynoise,method="DBI") / Cor2
    # collect GM-DUNN
    met1[i,2] = eval1(noise=mynoise,method="Dunn",inter="center",intra="allmax")/Cor1
    # collect GM-PSF,WSS
    met1[i,3:]= eval1(noise=mynoise,method="psF",centroid=True)
    met1[i,3] = met1[i,3]/Cor1
    met1[i,4] = met1[i,4]/Cor2
    dens_score = DBCV.DBCV(clusters=newlabels,metric=spacedef,NF=nfeatures,D=rho_r)
    met1[i,5] = dens_score.calc_score(meas='kernel')    
    
    print "**** End of Run %d ********"  % (i+1)

np.savetxt("R18-MET_rho_r.DP",met1,fmt="%6.4f")
