#ifndef FFTLIB_H_
#define FFTLIB_H_

#include <complex.h>

#include <fftw3.h>
#ifdef FFT_MPI_TRANSPOSE
#include <fftw3-mpi.h>
#endif


#ifdef PFFT_SINGLE
#define FFTW(x) fftwf_##x
typedef FFTW(complex) type;
typedef float rtype;
#define EPS __FLT_EPSILON__
#define MPI_RTYPE MPI_FLOAT
#else
#define FFTW(x) fftw_##x
typedef FFTW(complex) type;
typedef double rtype;
#define EPS __DBL_EPSILON__
#define MPI_RTYPE MPI_DOUBLE
#endif

struct pfft_time_t {
	double fft,local,mpi1,mpi2;
};
typedef struct pfft_time_t *pfft_time;

struct pfft_plan_t {
	type *lin;
	type *lin2;
	FFTW(plan) p11,p12,p13;
#ifdef FFT_MPI_TRANSPOSE
	FFTW(plan) mpip1,mpip2;
#else
	MPI_Comm cart1, cart2;
#endif
	
	int N1,M0,K0,K1;
	int N,M,K;
	int P[2];
	int coor[2];
}; 

typedef struct pfft_plan_t *pfft_plan;

void pfft_execute(pfft_plan plan,pfft_time times);
pfft_plan pfft_plan_3d(int N, int M, int K, MPI_Comm comm, int P0, int direction, int realcomplex, type** lin, type** lin2);
void pfft_local_size(pfft_plan plan,int* N1,int* M0,int* K0,int* K1,int** coor);
void pfft_destroy(pfft_plan plan);

#ifndef __USE_ISOC99
inline double fmax(double a, double b);
inline double fmin(double a, double b);
#endif

#endif /*FFTLIB_H_*/
