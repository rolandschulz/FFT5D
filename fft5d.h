#ifndef FFT5D_H_     
#define FFT5D_H_

#include <complex.h>

#include <fftw3.h>
#ifdef FFT5D_MPI_TRANSPOSE
#include <fftw3-mpi.h>
#endif


#ifdef FFT5D_SINGLE
#define FFTW(x) fftwf_##x
typedef FFTW(complex) type;
typedef float rtype;
#define FFT5D_EPS __FLT_EPSILON__
#define MPI_RTYPE MPI_FLOAT
#else
#define FFTW(x) fftw_##x
typedef FFTW(complex) type;
typedef double rtype;
#define FFT5D_EPS __DBL_EPSILON__
#define MPI_RTYPE MPI_DOUBLE
#endif

struct fft5d_time_t {
	double fft,local,mpi1,mpi2;
};
typedef struct fft5d_time_t *fft5d_time;

typedef enum fftflags {
	FFT5D_ORDER_YZ=1,
	FFT5D_BACKWARD=2,
	FFT5D_REALCOMPLEX=4,
	FFT5D_DEBUG=8,
	FFT5D_NOMEASURE=16,
	FFT5D_INPLACE=32,
	FFT5D_NOMALLOC=64
} fft5dflags;

struct fft5d_plan_t {
	type *lin;
	type *lout;
	FFTW(plan) p1d[3];
#ifdef FFT5D_MPI_TRANSPOSE
	FFTW(plan) mpip[2];
#else
	MPI_Comm cart[2];
#endif
	int N[3],M[3],K[3]; //local length in transposed coordinate system
	int C[3],rC[3]; //global length (of the one global axes) 
	//C!=rC for real<->complex. then C=rC/2 but with potential padding
	int P[2]; //size of processor grid
//	int fftorder;
//	int direction;
//	int realcomplex;
	fft5dflags flags;
	//int N0,N1,M0,M1,K0,K1;
	int NG,MG,KG;
	//int P[2];
	int coor[2];
}; 

typedef struct fft5d_plan_t *fft5d_plan;

void fft5d_execute(fft5d_plan plan,fft5d_time times);
fft5d_plan fft5d_plan_3d(int N, int M, int K, MPI_Comm comm, int P0, fft5dflags flags, type** lin, type** lin2);
void fft5d_local_size(fft5d_plan plan,int* N1,int* M0,int* K0,int* K1,int** coor);
void fft5d_destroy(fft5d_plan plan);

void fft5d_compare_data(const type* lin, const type* in, fft5d_plan plan, int bothLocal, int normarlize);
#ifndef __USE_ISOC99
inline double fmax(double a, double b);
inline double fmin(double a, double b);
#endif

#endif /*FFTLIB_H_*/
