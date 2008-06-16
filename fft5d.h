#ifndef FFT5D_H_     
#define FFT5D_H_

#include <complex.h>

#include <fftw3.h>
#ifdef FFT_MPI_TRANSPOSE
#include <fftw3-mpi.h>
#endif


#ifdef FFT5D_SINGLE
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

struct fft5d_time_t {
	double fft,local,mpi1,mpi2;
};
typedef struct fft5d_time_t *fft5d_time;

enum fft5dorder {
	ZY,
	YZ
};

struct fft5d_plan_t {
	type *lin;
	type *lout;
	FFTW(plan) p1d[3];
#ifdef FFT_MPI_TRANSPOSE
	FFTW(plan) mpip[2];
#else
	MPI_Comm cart[2];
#endif
	int N[3],M[3],K[3],C[3],rC[3],P[2];
	int fftorder;
	int direction;
	int realcomplex;
	//int N0,N1,M0,M1,K0,K1;
	int NG,MG,KG;
	//int P[2];
	int coor[2];
}; 

typedef struct fft5d_plan_t *fft5d_plan;

void fft5d_execute(fft5d_plan plan,fft5d_time times,int debug);
fft5d_plan fft5d_plan_3d(int N, int M, int K, MPI_Comm comm, int P0, int direction, int realcomplex, int inplace, int fftorder, type** lin, type** lin2);
void fft5d_local_size(fft5d_plan plan,int* N1,int* M0,int* K0,int* K1,int** coor);
void fft5d_destroy(fft5d_plan plan);

void compare_data(const type* lin, const type* in, fft5d_plan plan, int debug);
#ifndef __USE_ISOC99
inline double fmax(double a, double b);
inline double fmin(double a, double b);
#endif

#endif /*FFTLIB_H_*/
