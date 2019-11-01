#include <stdio.h>
#include <stdlib.h>
#include "math.h"
#include <omp.h>
//
//#include <cblas.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_cblas.h>
#include <gsl/gsl_blas.h>

void get_momenta (double *dist, double *usr, int natoms, int s, double Ni, double N1i);
void get_dist (double *dist, double *X, double *y, int natoms);
void calc_usr(double *X, double *usr, int natoms, double *weights, int verbosity);
   
