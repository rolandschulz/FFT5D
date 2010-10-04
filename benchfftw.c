#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <assert.h>
#include <math.h>
#include <string.h>

#include "fft5d.h"
#include <fftw3.h>

/*compile:
gcc   -Wall  benchfftw.c -l fftw3f -o benchfftw  -std=gnu99 -g -I . -D NOGMX -l fftw3f_threads
*/

void init_random(real* x, int l);

int main(int argc,char** argv)
{
#define OUTOFPLACE 128
    int N=0,K=0,M=0,N_measure=10,nthreads=1;
    int flags = 0;
    int c,a;
	
    const char* helpmsg = "\
Usage: %s [OPTION] N\n\
   or: %s [OPTION] N M K \n\
\n\
Compute parallel FFT on NxNxN or NxMxK size data on 2D prozessor grid\n\
Correctness is checked by comparison to FFTW\n\
\n\
Options:\n\
  -P CPUS   CPUS number of processors used in 1st dimension\n\
            default is to use most squared CPU grid\n\
  -B        compute backward transform (default: forward)\n\
  -R        use real input (forward) or output (backward) data\n\
            default is complex to complex\n\
  -D        activate debuging\n\
  -N RUNS	RUNS number of measurement runs for timing (default: 10)\n\
  -A RANK   RANK order: 0: default, 1: x, 2: y, 3: z\n\
  -T THREADS \n\
  -O        Out-of place\n";		
		
    while ((c = getopt (argc, argv, "BORhDP:N:A:T:")) != -1)

	switch (c) { 
	    case 'B':flags|=FFT5D_BACKWARD;break;
	    case 'R':flags|=FFT5D_REALCOMPLEX;break;
	    case 'O':flags|=OUTOFPLACE;break;
	    case 'A':a=atoi(optarg);break;
	    case 'N':N_measure=atoi(optarg);break;
	    case 'T':nthreads=atoi(optarg);break;
	    default:
		printf("Unknown Option: %c, %d\n\n",c,c);
		printf(helpmsg,argv[0],argv[0]);
		abort();
	}

    if (argc-optind==1) {
	N=M=K=atoi(argv[optind]);
    } else if (argc-optind==3) {
	K=atoi(argv[optind+2]);
	M=atoi(argv[optind+1]);
	N=atoi(argv[optind]);
    } else {
	printf(helpmsg,argv[0],argv[0]);
	abort();
    } 
    
    unsigned fftw_flags = FFTW_MEASURE;//PATIENT;

#ifndef NTHREADS
    printf("Running on %d threads\n", nthreads);
    FFTW(init_threads)();
    FFTW(plan_with_nthreads)(nthreads);
#endif

    srand(time(0));
    FFTW(plan) p2=0;
    int rN=N;
    t_complex *in=0,*out=0;
    in = (t_complex*) FFTW(malloc)(N*M*K*sizeof(t_complex));
    if (flags&OUTOFPLACE)
	out = (t_complex*) FFTW(malloc)(N*M*K*sizeof(t_complex));
    else 
	out = in;

    if (flags&FFT5D_REALCOMPLEX) {
	if (!(flags&FFT5D_BACKWARD)) {
	  printf("real forward %d %d %d\n", K, M, rN);
	    p2 = FFTW(plan_dft_r2c_3d)(K, M, rN, (real*)in, (FFTW(complex)*)out, fftw_flags);
	} else {
	  printf("real backward\n");
	    p2 = FFTW(plan_dft_c2r_3d)(K, M, rN, in, (real*)out, fftw_flags);
	}
    } else {
	
	if (a>0) {
	    fftw_iodim dims[3];
	
	    dims[0].n  = N;
	    dims[1].n  = M;
	    dims[2].n  = K;
	    
	    dims[0].is = 1;     /*N M K*/
	    dims[1].is = N;
	    dims[2].is = N*M;
	    if (a==1) {
		dims[0].os = 1;     /*N M K*/
		dims[1].os = N;
		dims[2].os = N*M;
	    } else if (a==2) {
		dims[0].os = M*K;       //M K N
		dims[1].os = 1;
		dims[2].os = M; 
	    } else if (a==3) {
		dims[0].os = K;       //K N M
		dims[1].os = K*N;
		dims[2].os = 1;
	    } else {
		printf("wrong value for A"); abort();
	    }

	    printf("complex forward with rotation\n");
	    p2 = FFTW(plan_guru_dft)(/*rank*/ 3, dims,
		       /*howmany*/ 0, /*howmany_dims*/0 ,
		       in, out,
		       /*sign*/ -1, /*flags*/ fftw_flags);
	} else {
	     printf("complex\n");
	    p2 = FFTW(plan_dft_3d)(K, M, N, (FFTW(complex)*)in, (FFTW(complex)*)out, (flags&FFT5D_BACKWARD)?1:-1, fftw_flags);
	}
	assert(p2);

    }
    
    init_random((real*)in,N*M*K*sizeof(t_complex)/sizeof(real));
    //bzero(in,N*M*K*sizeof(FFTW(complex)));

    double min_time=1e30,stime=0;
    for (int t=0;t<8;t++) {
      for (int m=0;m<N_measure;m++) {
	stime-=MPI_Wtime();
	FFTW(execute)(p2);
	stime+=MPI_Wtime();
      }
      min_time=fmin(min_time,stime/N_measure);
    }
    printf("%lf\n",min_time*1000);

    FFTW(destroy_plan)(p2);
    FFTW(free)(in);
#ifndef NTHREADS
    FFTW(cleanup_threads)();
#endif
    if (flags&OUTOFPLACE) FFTW(free)(out);
}

void init_random(real* x, int l) {
    int i;
    for (i=0;i<l;i++) {
        x[i]=((real)rand())/RAND_MAX;
    }
}

#include <sys/time.h>
double MPI_Wtime() {
    struct timeval tv;
    gettimeofday(&tv,0);
    return tv.tv_sec+tv.tv_usec*1e-6;
}
