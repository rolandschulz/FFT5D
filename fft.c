//============================================================================
// Name        : fft.cpp
// Author      : Roland Schulz
// Version     :
// Copyright   : GPL
// Description : Hello World in C, Ansi-style
//============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#include <complex.h>

#include <fftw3.h>
#ifdef FFT_MPI_TRANSPOSE
#include <fftw3-mpi.h>
#endif
#include <mpi.h>
#include <float.h>
#include <math.h>

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




void init_random(rtype* x, int l, int dim, int ld) {
	int i,j;
	for (i=0;i<((dim==1)?1:l);i++) {
		for (j=0;j<l;j++) {
			x[j+i*ld]=((rtype)rand())/RAND_MAX;
		}
	}
}



void swap(type *a, type *b) {
	type t=*a;
	*a=*b;
	*b=t;
}

//largest factor smaller than sqrt
int lfactor(int z) {  
	int i;
	for (i=sqrt(z);;i--)
		if (z%i==0) return i;
}

//largest factor 
int l2factor(int z) {  
	int i;
	for (i=z/2;;i--)
		if (z%i==0) return i;
}

double max(double a, double b) {
	return (a>b)?a:b;
}
//largest prime factor: WARNING: slow recursion
int lpfactor(int z) {
	int f = l2factor(z);
	if (f==1) return z;
	return max(lpfactor(f),lpfactor(z/f));
}

void avg(double* d, int n) { //avg excluding first
	int i;
	d[0]=0;
	for (i=1;i<n;i++) {
		d[0]+=d[i];
	}
	d[0]/=(n-1);
	d[1]=(d[1]-d[0])*(d[1]-d[0]);
	for (i=2;i<n;i++) {
		d[1]+=(d[i]-d[0])*(d[i]-d[0]);
		}
	d[0]*=1000;
	d[1]=1000000 * d[1]/(n-1);
}



inline double min(double a, double b) {
	return (a<b)?a:b;
}

int main(int argc,char** argv)
{
	MPI_Init( &argc, &argv );
	int size,prank;
	MPI_Comm_size(MPI_COMM_WORLD,&size);
	MPI_Comm_rank(MPI_COMM_WORLD,&prank);
	int N = 10, M = 10, K = 10;
	int P0 = lfactor(size);
	switch (argc) { //falltrough!
	case 5:
		P0=atoi(argv[4]);
	case 4:
		K=atoi(argv[3]);
	case 3:
		M=atoi(argv[2]);
	case 2:
		N=atoi(argv[1]);
	}
	if (P0==0 || size%P0!=0) {
		if (prank==0) printf("FATAL: Number of processors %d not evenly dividable by %d\n",size,P0);
		MPI_Finalize();
		return 1;
	}
	int P[] = {P0,size/P0}; //per proc 4*4*4 or 8*4*2


	
	if (prank==0) {
		printf("N: %d, M: %d, K: %d, P: %dx%d\n",N,M,K,P[0],P[1]);
		if (max(max(lpfactor(N),lpfactor(M)),lpfactor(K))>7) {
			printf("WARNING: FFT very slow with prime factors larger 7\n");
			printf("Change FFT size or in case you cannot change it look at\n");
			printf("http://www.fftw.org/fftw3_doc/Generating-your-own-code.html\n");
		}
	}
	
	if (N==0 || M==0 || K==0) {
		if (prank==0) printf("FATAL: Datasize cannot be zero in any dimension\n");
		MPI_Finalize();
		return 1;
	}

#ifdef PFFT_COMPLEX
	int rM = M;
	M = M/2+1;
#endif
	int Nx=ceil((double)N/P[0]);
	int My=ceil((double)M/P[1]);
	int Kx=ceil((double)K/P[0]),Ky=ceil((double)K/P[1]);
	
	int i,l,coor[2],x,y,z;
	
	int wrap[]={0,0};
	MPI_Comm cart;
	MPI_Cart_create(MPI_COMM_WORLD,2,P,wrap,1,&cart); //true: reorder
	MPI_Cart_get(cart,2,P,wrap,coor);
	int rdim1[] = {0,1}, rdim2[] = {1,0};
	MPI_Comm cart1, cart2;
	MPI_Cart_sub(cart, rdim1 , &cart1);
	MPI_Cart_sub(cart, rdim2 , &cart2);
	

	type* in = (type*) FFTW(malloc)(sizeof(type) * N*M*K);

	
	int lsize = max(Nx*My*Ky*P[1],Nx*My*Kx*P[0]);
	type *lin = (type*)FFTW(malloc)(sizeof(type) * lsize); //local in
	type *lin2 = (type*)FFTW(malloc)(sizeof(type) * lsize); //local in

	FFTW(plan) p11,p12,p13,p2;
#ifdef PFFT_COMPLEX
	p2 = FFTW(plan_dft_r2c_3d)(K, M, N, (rtype*)in, (FFTW(complex)*)in, FFTW_ESTIMATE);
	p11 = FFTW(plan_many_dft_r2c)(1, &rM, Nx*Ky,   (rtype*)lin, &rM, 1,   rM, 
			(FFTW(complex)*)lin2, &M, 1,   M, FFTW_MEASURE|FFTW_DESTROY_INPUT);//prod: FFTW_MEASURE
#else
	p2 = FFTW(plan_dft_3d)(K, M, N, (FFTW(complex)*)in, (FFTW(complex)*)in, FFTW_FORWARD, FFTW_ESTIMATE);
	p11 = FFTW(plan_many_dft)(1, &M, Nx*Ky,   (FFTW(complex)*)lin, &M, 1,   M, 
			(FFTW(complex)*)lin2, &M, 1,   M, FFTW_FORWARD, FFTW_MEASURE|FFTW_DESTROY_INPUT);//prod: FFTW_MEASURE
#endif
	p12 = FFTW(plan_many_dft)(1, &K, Nx*My,   (FFTW(complex)*)lin, &K, 1,   K, 
			(FFTW(complex)*)lin2, &K, 1,   K, FFTW_FORWARD, FFTW_MEASURE|FFTW_DESTROY_INPUT);//prod: FFTW_MEASURE
	p13 = FFTW(plan_many_dft)(1, &N, Kx*My,   (FFTW(complex)*)lin, &N, 1,   N, 
			(FFTW(complex)*)lin2, &N, 1,   N, FFTW_FORWARD, FFTW_MEASURE|FFTW_DESTROY_INPUT);//prod: FFTW_MEASURE
			
	
	#ifdef FFT_LOCAL_TRANSPOSE
	FFTW(plan) p12 = FFTW(plan_many_dft)(1, &N, Nx*Ny,   (FFTW(complex)*)lin, &N, Nx*Ny,   1, 
														 (FFTW(complex)*)lin2, &N, 1,   N, FFTW_FORWARD, FFTW_MEASURE|FFTW_DESTROY_INPUT);//prod: FFTW_MEASURE
#endif
	
#ifdef FFT_MPI_TRANSPOSE
	FFTW(plan) mpip1 = FFTW(mpi_plan_many_transpose)(P[1], P[1], Nx*Ky*My*2, 1, 1, (rtype*)lin, (rtype*)lin2, cart1, FFTW_MEASURE);
	FFTW(plan) mpip2 = FFTW(mpi_plan_many_transpose)(P[0], P[0], Nx*My*Kx*2, 1, 1, (rtype*)lin, (rtype*)lin2, cart2, FFTW_MEASURE);
//	FFTW(plan) mpip1 = FFTW(mpi_plan_many_transpose)(P[1], N, Nx*Ny*2, 1, Ny, (rtype*)lin, (rtype*)lin2, cart1, FFTW_MEASURE|FFTW_MPI_TRANSPOSED_OUT);
//	FFTW(plan) mpip2 = FFTW(mpi_plan_many_transpose)(P[0], N, Nx*Ny*2, 1, Nx, (rtype*)lin, (rtype*)lin2, cart2, FFTW_MEASURE|FFTW_MPI_TRANSPOSED_OUT);//|FFTW_MPI_TRANSPOSED_IN);
#endif
	init_random((rtype*)in,N*M*K*sizeof(type)/sizeof(rtype),1,1);
	
#ifdef DEBUG2
	if (prank==0) {
		for(z=0;z<K;z++) {
			for (y=0;y<M;y++) {
				for (x=0;x<N;x++) {				
					printf("%f+%fi ",((rtype*)in)[(x+y*N+z*N*M)*2],((rtype*)in)[(x+y*N+z*N*M)*2+1]);
				}
				printf("\n");
			}
		}
	}
#endif
	
	for(y=0;y<M;y++) {  //y i
		for (x=0;x<min(Nx,N-coor[0]*Nx);x++) { //x j  2nd of min for non dividable
			for (z=0;z<min(Ky,K-coor[1]*Ky);z++) { //z k
				//lin[i]=in[i+coor[0]*N+coor[1]*N*N];  //in[Px][i][Pz]
				lin[y+x*M+z*M*Nx]=in[(coor[0]*Nx+x)+y*N+(coor[1]*Ky+z)*M*N];  //in[Px][i][Pz]
				//lin[i*2+1]=in[coor[0]*2+1+i*N+coor[1]*N*N]; //imaginary part
			}
		}
	}
	
#ifdef DEBUG2
	printf("%d %d: copy in lin\n",coor[0],coor[1]);
	for (z=0;z<Ky;z++) {
		for (y=0;y<M;y++) {
			printf("%d %d: ",coor[0],coor[1]);
			for(x=0;x<Nx;x++) {
				printf("%f ",((rtype*)lin)[(y+x*M+z*Nx*M)*2]);
			}
			printf("\n");
		}
	}
#endif

	//lin: y,x,z
	
#define N_measure 100
	int m;
	double time_fft[N_measure]={0},time_local[N_measure]={0},time_mpi1[N_measure]={0},time_mpi2[N_measure]={0},time;
	
	for (m=0;m<N_measure;m++) {
	time=MPI_Wtime(); 
	FFTW(execute)(p11); //in:lin out:lin2
	time_fft[m]=MPI_Wtime()-time;

#ifdef DEBUG2
	printf("%d %d: FFT in y\n",coor[0],coor[1]);
	for (z=0;z<Ky;z++) {
		for (y=0;y<M;y++) {
			printf("%d %d: ",coor[0],coor[1]);
			for(x=0;x<Nx;x++) {
				printf("%f ",((rtype*)lin2)[(y+x*M+z*Nx*M)*2]);
			}
			printf("\n");
		}
	}
#endif

	time=MPI_Wtime(); 
	//prepare for AllToAll
	//1. (most outer) axes (y) is split into P[1] parts of size Ny for sending 
	for (i=0;i<P[1];i++) { //index cube along long axis
		for (z=0;z<Ky;z++) { //3. z l
			for (x=0;x<Nx;x++) { //2. x k
				for (y=0;y<min(My,M-My*i);y++) { //1. y j
					lin[y+x*My+z*Nx*My+i*Nx*My*Ky]=lin2[(i*My+y)+x*M+z*M*Nx];
				}
			}
		}
	}
	time_local[m]=MPI_Wtime()-time;
	
	

	//send, recv
	time=MPI_Wtime();

#ifdef FFT_MPI_TRANSPOSE
    FFTW(execute)(mpip1);
#else
    MPI_Alltoall(lin,Nx*My*Ky*sizeof(type)/sizeof(rtype),MPI_RTYPE,lin2,Nx*My*Ky*sizeof(type)/sizeof(rtype),MPI_RTYPE,cart1);
#endif
	time_mpi1[m]=MPI_Wtime()-time;


	
	time=MPI_Wtime();
#ifdef FFT_LOCAL_TRANSPOSE
	for (i=0;i<P[1];i++) { //index cube along long axis
		for (l=0;l<Ny;l++) { //3.
			for (j=0;j<Ny;j++) { //1.
				for (k=0;k<Nx;k++) { //2.
					lin[k+j*Nx+l*Nx*Ny+i*Nx*Ny*Ny]=lin2[j+k*Ny+l*Nx*Ny+i*Nx*Ny*Ny];
				}
			}
		}
	}	
#else
	
	//bring back in matrix form (could be avoided by storing blocks as eleftheriou)
	//thus make z ( 1. axes) again contiguos
	//also local transpose 1 and 3 
	//then z,x,y
	for (i=0;i<P[1];i++) { //index cube along long axis
		for (y=0;y<My;y++) { //1.j
			for (x=0;x<Nx;x++) { //2.k
				for (z=0;z<min(Ky,K-Ky*i);z++) { //3.l
					lin[(i*Ky+z)+x*K+y*K*Nx]=lin2[y+x*My+z*Nx*My+i*Nx*My*Ky];
				}
			}
		}
	}	
#endif
	time_local[m]+=MPI_Wtime()-time;

#ifdef DEBUG2
	printf("%d %d: transposed y-z\n",coor[0],coor[1]);
	for (z=0;z<K;z++) {
		for (y=0;y<My;y++) {
			printf("%d %d: ",coor[0],coor[1]);
			for(x=0;x<Nx;x++) {
				printf("%f ",((rtype*)lin)[(z+x*K+y*Nx*K)*2]);
			}
			printf("\n");
		}
	}
#endif
	
	time=MPI_Wtime();
#ifdef FFT_LOCAL_TRANSPOSE
	FFTW(execute)(p12);
#else
	FFTW(execute)(p12);
#endif
	time_fft[m]+=MPI_Wtime()-time;
	
#ifdef DEBUG2
	printf("%d %d: FFT in z\n",coor[0],coor[1]);
	for (z=0;z<K;z++) {
			for (y=0;y<My;y++) {
				printf("%d %d: ",coor[0],coor[1]);
				for(x=0;x<Nx;x++) {
					printf("%f ",((rtype*)lin2)[(z+x*K+y*Nx*K)*2]);
				}
				printf("\n");
			}
	}
#endif
	//prepare alltoall. split 1 axes (z) into P[0] parts with size Nx 
	time=MPI_Wtime();
	for (i=0;i<P[0];i++) { //index cube along long axis
		for (y=0;y<My;y++) { //3.y l 
			for (x=0;x<Nx;x++) { //2.x k
				for (z=0;z<min(Kx,K-Kx*i);z++) { //1.z j
					lin[z+x*Kx+y*Nx*Kx+i*Nx*Kx*My]=lin2[(i*Kx+z)+x*K+y*K*Nx];
				}
			}
		}
	}
	
	time_local[m]+=MPI_Wtime()-time;

	time=MPI_Wtime();
#ifdef FFT_MPI_TRANSPOSE
    FFTW(execute)(mpip2);
#else
	MPI_Alltoall(lin,Nx*Kx*My*sizeof(type)/sizeof(rtype),MPI_RTYPE,lin2,Nx*Kx*My*sizeof(type)/sizeof(rtype),MPI_RTYPE,cart2);
#endif
	time_mpi2[m]=MPI_Wtime()-time;

	time=MPI_Wtime();
	//make x contiguous  and also transpose 1 and 2
	//now x,z,y

	for (i=0;i<P[0];i++) { //index cube along long axis
		for (y=0;y<My;y++) { //3. y l
			for (z=0;z<Kx;z++) { //1. z j
				for (x=0;x<min(Nx,N-Nx*i);x++) { //2. x k
					lin[(i*Nx+x)+z*N+y*N*Kx]=lin2[z+x*Kx+y*Nx*Kx+i*Nx*Kx*My];
				}
			}
		}
	}
	time_local[m]+=MPI_Wtime()-time;

#ifdef DEBUG2
	printf("%d %d: transposed x-z\n",coor[0],coor[1]);
	for (z=0;z<Kx;z++) {
			for (y=0;y<My;y++) {
				printf("%d %d: ",coor[0],coor[1]);
				for(x=0;x<N;x++) {
					printf("%f ",((rtype*)lin)[(x+z*N+y*N*Kx)*2]);
				}
				printf("\n");
			}
	}
#endif	
	time=MPI_Wtime();
	FFTW(execute)(p13);
	time_fft[m]+=MPI_Wtime()-time;

#ifdef DEBUG2
	printf("%d %d: FFT in x\n",coor[0],coor[1]);
	for (z=0;z<Kx;z++) {
				for (y=0;y<My;y++) {
					printf("%d %d: ",coor[0],coor[1]);
					for(x=0;x<N;x++) {
						printf("%f ",((rtype*)lin2)[(x+z*N+y*N*Kx)*2]);
					}
					printf("\n");
				}
		}
#endif
	
	if (m==0) {
		if (N>128) {
			if (prank==0) printf("No correctness check above 128\n");
		} else {
#ifdef DEBUG2
			if (prank==0) {
				for(z=0;z<K;z++) {
					for (y=0;y<M;y++) {
						for (x=0;x<N;x++) {				
							printf("%f+%fi ",((rtype*)in)[(x+y*N+z*N*M)*2],((rtype*)in)[(x+y*N+z*N*M)*2+1]);
						}
						printf("\n");
					}
				}
			}
#endif
			FFTW(execute)(p2);
#ifdef DEBUG2
			if (prank==0) {
				for(z=0;z<K;z++) {
					for (y=0;y<M;y++) {
						for (x=0;x<N;x++) {				
							printf("%f+%fi ",((rtype*)in)[(x+y*N+z*N*M)*2],((rtype*)in)[(x+y*N+z*N*M)*2+1]);
						}
						printf("\n");
					}
				}
			}
#endif
			//print("in",in,N,2,ld);
			//print("tmp",tmp,N,2,ld);
			//assert(test_equal(in,tmp,N*N*N,N,2,N));
			
			for (x=0;x<N;x++) {//x i
				for (z=0;z<min(Kx,K-Kx*coor[0]);z++) {//z j
					for (y=0;y<min(My,M-My*coor[1]);y++) {//y k
						for (l=0;l<2;l++) {
//							printf("%f %f ",((rtype*)lin2)[(x+z*N+y*N*Kx)*sizeof(type)/sizeof(rtype)+l],
//									((rtype*)in)[(x+(coor[1]*My+y)*N+(coor[0]*Kx+z)*N*M)*sizeof(type)/sizeof(rtype)+l]);
							assert(fabs(((rtype*)lin2)[x+l+z*N*sizeof(type)/sizeof(rtype)+y*N*Kx*sizeof(type)/sizeof(rtype)]-
									((rtype*)in)[x+l+(coor[1]*My+y)*N*sizeof(type)/sizeof(rtype)+(coor[0]*Kx+z)*N*M*sizeof(type)/sizeof(rtype)])<N*M*K*EPS);
						}
//						printf("\n");
					}
				}
			}
		
			if (prank==0) printf("OK\n");
//			return 0;

		}
	}
	} // end measure
	
	avg(time_local,N_measure);
	avg(time_fft,N_measure);
	avg(time_mpi1,N_measure);
	avg(time_mpi2,N_measure);
	
	//printf("avg: %lf\n",time_mpi1[0]);
	
	double times[]={time_local[0],time_fft[0],time_mpi1[0],time_mpi2[0],
					time_local[1],time_fft[1],time_mpi1[1],time_mpi2[1]},otimes[sizeof(times)/sizeof(double)];
	
	MPI_Reduce(times,otimes,sizeof(times)/sizeof(double),MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
	
	if(prank==0) {
		printf("Timing (ms): local: %lf+/-%lf, fft: %lf+/-%lf, mpi1: %lf+/-%lf, mpi2:%lf+/-%lf\n",
				otimes[0]/size,sqrt(otimes[4]/size),
				otimes[1]/size,sqrt(otimes[5]/size),
				otimes[2]/size,sqrt(otimes[6]/size),
				otimes[3]/size,sqrt(otimes[7]/size));
	}
	

	FFTW(destroy_plan)(p11);
	FFTW(destroy_plan)(p12);
	FFTW(destroy_plan)(p13);
	
	FFTW(destroy_plan)(p2);
#ifdef FFT_MPI_TRANSPOSE
	FFTW(destroy_plan)(mpip1);
	FFTW(destroy_plan)(mpip2);
#endif
	FFTW(free)(in);
	FFTW(free)(lin);
	FFTW(free)(lin2);

	MPI_Finalize();
	
//	int ld=2*(N/2+1);
//
//	in = (type*) FFTW(malloc)(sizeof(type) * ld*N*N);
//#ifndef NDEBUG
//	type* tmp = (type*) FFTW(malloc)(sizeof(type) * ld*N*N);
//#endif
////	    fftw_plan_many_dft_r2c(rank, n, howmany, *in, *inembed, istride, idist,
////	                   *out, *onembed, ostride, odist, flags);
//
//	 
//	p11 = FFTW(plan_many_dft_r2c)(1, &N, N*N,                in, &N, 1,      ld, 
//							  				 (FFTW(complex)*)in, &N, 1,      ld/2, FFTW_ESTIMATE);//prod: FFTW_MEASURE
//	p12 = FFTW(plan_many_dft)(1, &N, ld/2,   (FFTW(complex)*)in, &N, ld/2,   1, 
//											 (FFTW(complex)*)in, &N, ld/2,   1, FFTW_FORWARD, FFTW_ESTIMATE);//prod: FFTW_MEASURE
//	p13 = FFTW(plan_many_dft)(1, &N, ld/2*N, (FFTW(complex)*)in, &N, ld/2*N, 1, 
//											 (FFTW(complex)*)in, &N, ld/2*N, 1, FFTW_FORWARD, FFTW_ESTIMATE);//prod: FFTW_MEASURE
//		
//	p2 = FFTW(plan_dft_c2r_3d)(N, N, N, (FFTW(complex)*)in, in, FFTW_ESTIMATE);
//
//	init_random(in,N,2,ld);
//	memcpy(tmp,in,sizeof(type)*ld*N*N);
//	FFTW(execute)(p11);
//	for (i=0;i<N;i++)
//		FFTW(execute_dft)(p12,((FFTW(complex)*)in)+i*ld/2*N,((FFTW(complex)*)in)+i*ld/2*N);
//	FFTW(execute)(p13); 
//	FFTW(execute)(p2);
//
//	//print("in",in,N,2,ld);
//	//print("tmp",tmp,N,2,ld);
//	assert(test_equal(in,tmp,N*N*N,N,2,ld));
//	printf("OK\n");
//
//	FFTW(destroy_plan)(p11);
//	FFTW(destroy_plan)(p12);
//	FFTW(destroy_plan)(p2);
//	FFTW(free)(in);
	
//2d from 1d
//	in = (type*) FFTW(malloc)(sizeof(type) * ld*N);
//#ifndef NDEBUG
//	type* tmp = (type*) FFTW(malloc)(sizeof(type) * ld*N);
//#endif
////	    fftw_plan_many_dft_r2c(rank, n, howmany, *in, *inembed, istride, idist,
////	                   *out, *onembed, ostride, odist, flags);
//
//
//	p11 = FFTW(plan_many_dft_r2c)(1, &N, N, in, &N, 1, ld, 
//							(FFTW(complex)*)in, &N, 1, ld/2, FFTW_ESTIMATE);//prod: FFTW_MEASURE
//	p12 = FFTW(plan_many_dft)(1, &N, ld/2, (FFTW(complex)*)in, &N, ld/2, 1, 
//										(FFTW(complex)*)in, &N, ld/2, 1, FFTW_FORWARD, FFTW_ESTIMATE);//prod: FFTW_MEASURE
//	p2 = FFTW(plan_dft_c2r_2d)(N, N, (FFTW(complex)*)in, in, FFTW_ESTIMATE);
//
//	init_random(in,N,2,ld);
//	memcpy(tmp,in,sizeof(type)*ld*N);
//	FFTW(execute)(p11);
//	FFTW(execute)(p12); 
//	FFTW(execute)(p2);
//	
//	//print("in",in,N,2,ld);
//	//print("tmp",tmp,N,2,ld);
//	assert(test_equal(in,tmp,N*N,N,2,ld));
//	printf("OK\n");
//	
//	FFTW(destroy_plan)(p11);
//	FFTW(destroy_plan)(p12);
//	FFTW(destroy_plan)(p2);
//	FFTW(free)(in);
	
//2d	
//	in = (type*) FFTW(malloc)(sizeof(type) * ld*N);
//	p = FFTW(plan_dft_r2c_2d)(N, N, in, (FFTW(complex)*)in, FFTW_ESTIMATE);
//	p2 = FFTW(plan_dft_c2r_2d)(N, N, (FFTW(complex)*)in, in, FFTW_ESTIMATE);
//
//	init_random(in,N,2,ld);
//	print("in",in,N,2,ld);
//	FFTW(execute)(p); 
//	printc("out",(FFTW(complex)*)in,N,N/2+1);
//	FFTW(execute)(p2);
//	print("back",in,N,2,ld);
//	
//	FFTW(destroy_plan)(p);
//	FFTW(destroy_plan)(p2);
//	FFTW(free)(in);
	
// 1d 	
//	in = (type*) FFTW(malloc)(sizeof(type) * 2*(N/2+1));
//	p = FFTW(plan_dft_r2c_1d)(N, in, (FFTW(complex)*)in, FFTW_ESTIMATE);
//	p2 = FFTW(plan_dft_c2r_1d)(N, (FFTW(complex)*)in, in, FFTW_ESTIMATE);
//
//	init_random(in,N,1,ld);
//	print("in",in,N,1,ld);
//	FFTW(execute)(p); 
//	printc("out",(FFTW(complex)*)in,1,N/2+1);
//	FFTW(execute)(p2);
//	print("back",in,N,1,ld);
//	
//	FFTW(destroy_plan)(p);
//	FFTW(destroy_plan)(p2);
//	FFTW(free)(in); 
	
	return 0;
}


