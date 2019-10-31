%module pyusr

%{
#define SWIG_FILE_WITH_INIT
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include "usr.h"
%}

%include "/usr/local/lib/python3.7/dist-packages/numpy/numpy.i"

%init %{
import_array();
%}

#define DIM 3
#define NCOMP 12
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

%apply (int DIM1, int DIM2, double *IN_ARRAY2) {(int natoms, int dim, double *X)};
%apply (int DIM1, double* IN_ARRAY1) {(int n2, double *W)};
%apply (int DIM1, double* INPLACE_ARRAY1) {(int ncomp, double *USR)};

%apply (int DIM1, int DIM2, int DIM3, double *IN_ARRAY3) {(int nframes, int natoms, int dim, double *TRJ)};
%apply (int DIM1, int DIM2, double* INPLACE_ARRAY2) {(int nf, int nc, double *USRmat)};

%inline %{
        void pyUSR(int natoms, int dim, double *X, int n2, double *W, int ncomp, double *USR, int verbosity)
        {
            calc_usr(X, USR, natoms, W, verbosity);
        }
%}

%inline %{
        void pyUSRmat(int nframes, int natoms, int dim, double *TRJ, int nf, int nc, 
                      double *USRmat,int n2, double *W, int nthr, int verbosity)
        {
            int i, j, k;
            double *usrvec = calloc(nframes*12,sizeof(double));
            omp_set_num_threads(nthr);

            //#pragma omp parallel for if(nframes>=100)
            for(i=0; i < nframes; i++)
                calc_usr(&TRJ[i*natoms*3], &usrvec[i*12], natoms, W, verbosity);
            
            for(i=0; i < nframes; i++) 
                //#pragma omp parallel for if(nframes>=100)
                for(j=1; j < nframes; j++)
                    for(k=0; k < 12; k++)
                        USRmat[i*nframes+j] += fabs( usrvec[i*12+k] - usrvec[j*12+k] );
        }
%}

%clear (int nframes, int natoms, int dim, double *TRJ);
%clear (int nf, int nc, double *USRmat);

%clear (int natoms,int dim, double *X);
%clear (int natoms, double *W);
%clear (int ncomp, double *USR);
