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
//n2 is needed for the numpy mask, same for nf
%apply (int DIM1, double* IN_ARRAY1) {(int n2, double *weights)};
%apply (int DIM1, double* INPLACE_ARRAY1) {(int ncomp, double *USR)};

%apply (int DIM1, int DIM2, int DIM3, double *IN_ARRAY3) {(int nframes, int natoms, int dim, double *TRJ)};
%apply (int DIM1, int DIM2, double* INPLACE_ARRAY2) {(int nf, int ncomp, double *USRvec)};

%inline %{
        //calculate USR for a single frame
        void pyUSR(int natoms, int dim, double *X, int n2, double *weights, int ncomp, double *USR, int verbosity)
        {
            calc_usr(X, USR, natoms, weights, verbosity);
        }
%}

%inline %{
        //calculate USR for a set of frames
        void pyUSRvec(int nframes, int natoms, int dim, double *TRJ, int n2, double *weights,
                      int nf, int ncomp, double *USRvec, int nthr, int verbosity)
        {
            int i, j, k;
            
            omp_set_num_threads(nthr);
            #pragma omp parallel for if(nframes>=100)
            for(i=0; i < nframes; i++)
                calc_usr(&TRJ[i*natoms*3], &USRvec[i*12], natoms, weights, verbosity);
        }
%}

%clear (int nframes, int natoms, int dim, double *TRJ);
%clear (int nf, int nc, double *USRmat);

%clear (int natoms,int dim, double *X);
%clear (int natoms, double *weights);
%clear (int nframes, double *usrvec);
%clear (int ncomp, double *USR);
