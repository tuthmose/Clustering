#include <usr.h>

void get_momenta(double *dist, double *usr, int natoms, int s, double Ni, double N1i)
{
    /*
     * calculate statistics and assign USR vector components 
     * according to distances in dist
     * s = 0 -> centroid
     * s = 3 -> clostest  to centroid
     * s = 6 -> farthest  from centroid
     * s = 9 -> farthest  from farthest  from centroid
     */
    double mu=0., mu2=0., mu3=0.;
    
    #pragma omp parallel for simd reduction(+:mu,mu2,mu3)
    for(int i=0;i<natoms;i++)
    {
        mu  += dist[i];
        mu2 += dist[i] * dist[i];
        mu3 += pow(dist[i],3);
    }

    usr[s]   = mu*Ni;
    usr[s+1] = mu2*Ni - usr[s]*usr[s];
    //usr[s+1] = (mu2 -pow(mu,2)*Ni)*N1i;
    usr[s+2] = ((mu3*Ni -pow(usr[s],3) -3.*usr[s]*usr[s+1])/powf(usr[s+1],1.5));
    
}


void get_dist(double *dist, double *X, double *y, int natoms)
{
    /*
     * calculate Euclidean distances
     * X = coordinates
     * y = reference point (ctd, cst, fct, ftf)
     */

    double vx, vy, vz;
    
    for(int i=0;i<natoms;i++)
    {
        #pragma omp simd
        {
            vx = pow((X[i*3]   - y[0]),2);
            vy = pow((X[i*3+1] - y[1]),2);
            vz = pow((X[i*3+2] - y[2]),2);
            dist[i] = sqrt(vx+vy+vz);
        }
    }
}


void calc_usr(double *X, double *usr, int natoms, double *weights, int verbosity)
{

    double xav=0., yav=0., zav=0.;
    double ctd[3] = {0.,0.,0.};
    double *dist = calloc(natoms,sizeof(double));
    double W = 0.;
    gsl_vector_view DIST;
    double Ni  = 1./((double) natoms);
    double N1i = 1./((double) natoms - 1.);
    
    #pragma omp simd reduction(+:xav,yav,zav,W)
    for(int i=0; i<natoms; i++)
    {
        xav += weights[i]*X[i*3];
        yav += weights[i]*X[i*3+1];
        zav += weights[i]*X[i*3+2];
        W += weights[i];
    }
    ctd[0] = xav / W;
    ctd[1] = yav / W;
    ctd[2] = zav / W;
    
    // distances to centroid; atom closest to centroid
    get_dist(dist,X,ctd,natoms);
    DIST = gsl_vector_view_array(dist,natoms);
    // momenta to ctd
    get_momenta(dist,usr,natoms,0,Ni,N1i);
    
    gsl_vector_view CTD = gsl_vector_view_array(ctd,3);
    
    if(verbosity >= 1)
    {
        gsl_vector_fprintf(stdout,&CTD.vector,"%f");     
        gsl_vector_fprintf(stdout,&DIST.vector,"%f");
    }
    
    // cst and fct
    int cst = gsl_vector_min_index(&DIST.vector);
    int fct = gsl_vector_max_index(&DIST.vector);
    
    // distances and momenta to cst   
    get_dist(dist,X,&X[cst*3],natoms);  
    get_momenta(dist,usr,natoms,3,Ni,N1i);
    
    // distances and momenta to fct
    get_dist(dist,X,&X[fct*3],natoms);
    get_momenta(dist,usr,natoms,6,Ni,N1i);
    
    //ftf
    int ftf = gsl_vector_max_index(&DIST.vector);
    get_dist(dist,X,&X[ftf*3],natoms);
    get_momenta(dist,usr,natoms,9,Ni,N1i);
    
    free(dist);
    
}

void main(int argc, char *argv[])
{
    // the main function is used to read a single
    // xyz matrix w/o atomic numbers for test purposes

    // cmd line: <input file> <natoms> <verbosity>
    
    int natoms = atoi(argv[2]);
    int verbosity = atoi(argv[3]);
    int usew = 0; //weights nyi
    double usr[12];
    int k;

    // read data
    double *X = malloc(natoms * 3 * sizeof(double *));
    if (X == NULL) 
    {
        printf("failed to allocate memory for X");
        exit(0);
    }

    // scan input file
    FILE *fp;
    fp = fopen(argv[1], "r");
    for (int i=0; i < natoms;i++)
    {
        for (int j=0; j< 3; j++)
        {
            k = fscanf(fp,"%lf", &X[j + i*3]);
            }
    }    
    fclose(fp);

    double *weights = malloc(natoms * sizeof(double));
    if(usew == 0)
    {
        for(int i=0;i<natoms;i++)
            weights[i] = 1.0;
    }
    //end read data
    
    //print input
    if(verbosity >= 1)
    {
        printf("Input\n");
        for(int i = 0; i< natoms; i++)
            printf("%d %f %f %f %f\n",i,weights[i],X[i*3],X[i*3+1],X[i*3+2]);
    }
        
    calc_usr(X, usr, natoms, weights, verbosity);

    if(verbosity >= 1)
    {   
        for(int i = 0; i< 12; i++)
           printf("USR %d %f\n",i,usr[i]);
    }else
    {
        for(int i = 0; i< 12; i++)
           printf("  %10.8f  ",usr[i]);
        printf("\n");
    }    
    free(X);
    free(weights);
}

