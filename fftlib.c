#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "fftlib.h"
#include <float.h>
#include <math.h>

#ifndef __USE_ISOC99
inline double fmax(double a, double b){
	return (a>b)?a:b;
}
inline double fmin(double a, double b){
	return (a<b)?a:b;
}
#endif

//largest factor smaller than sqrt
int lfactor(int z) {  
	int i;
	for (i=sqrt(z);;i--)
		if (z%i==0) return i;
}

//largest factor 
int l2factor(int z) {  
	int i;
	if (z==1) return 1;
	for (i=z/2;;i--)
		if (z%i==0) return i;
}

//largest prime factor: WARNING: slow recursion
int lpfactor(int z) {
	int f = l2factor(z);
	if (f==1) return z;
	return fmax(lpfactor(f),lpfactor(z/f));
}


//NxMxK the size of the data
//comm communicator to use for PFFT
//P0 number of processor in 1st axes (can be null for automatic)
//lin is allocated by pfft because size of array is only known after planning phase

pfft_plan pfft_plan_3d(int N, int M, int K, MPI_Comm comm, int P0, int direction, int realcomplex, type** rlin, type** rlin2) {
	int size,prank;
	MPI_Comm_size(comm,&size);
	MPI_Comm_rank(comm,&prank);
	if (P0==0) P0 = lfactor(size);
	if (size%P0!=0) {
		if (prank==0) printf("WARNING: Number of processors %d not evenly dividable by %d\n",size,P0);
		P0 = lfactor(size);
	}
	
	if (prank==0) printf("Using %dx%d processor grid\n",P0,size/P0);
	
	int P[] = {P0,size/P0}; //per proc 4*4*4 or 8*4*2


	
	if (prank==0) {
		printf("N: %d, M: %d, K: %d, P: %dx%d, real to complex: %d, direction: %d\n",N,M,K,P[0],P[1],realcomplex,direction);
		if (fmax(fmax(lpfactor(N),lpfactor(M)),lpfactor(K))>7) {
			printf("WARNING: FFT very slow with prime factors larger 7\n");
			printf("Change FFT size or in case you cannot change it look at\n");
			printf("http://www.fftw.org/fftw3_doc/Generating-your-own-code.html\n");
		}
	}
	
	if (N==0 || M==0 || K==0) {
		if (prank==0) printf("FATAL: Datasize cannot be zero in any dimension\n");
		MPI_Finalize();
		return 0;
	}

	int rN=N;
	if (realcomplex) {
		N = N/2+1;
	}
	int N1=ceil((double)N/P[1]);
	int M0=ceil((double)M/P[0]);
	int K0=ceil((double)K/P[0]),K1=ceil((double)K/P[1]);
	
	//Difference between x-y-z regarding 2d docmposition is whether they are distributed 
	//along axis 1, 2 or both
	
	int coor[2];
	
	int wrap[]={0,0};
	MPI_Comm cart;
	MPI_Cart_create(comm,2,P,wrap,1,&cart); //parameter 4: value 1: reorder
	MPI_Cart_get(cart,2,P,wrap,coor);
	int rdim1[] = {0,1}, rdim2[] = {1,0};
	MPI_Comm cart1, cart2;
	MPI_Cart_sub(cart, rdim1 , &cart1);
	MPI_Cart_sub(cart, rdim2 , &cart2);
	

	
	int lsize = fmax(N1*M0*K1*P[1],N1*M0*K0*P[0]);
	type* lin = (type*)FFTW(malloc)(sizeof(type) * lsize); //local in	
	type* lin2 = (type*)FFTW(malloc)(sizeof(type) * lsize); //local in
	
	FFTW(plan) p11,p12,p13;
	if (realcomplex) {
		if (direction==-1) {
			p11 = FFTW(plan_many_dft_r2c)(1, &rN, M0*K1,   (rtype*)lin, &rN, 1,   N*2, 
					(FFTW(complex)*)lin2, &N, 1,   N, FFTW_MEASURE|FFTW_DESTROY_INPUT);//prod: FFTW_MEASURE
		} else {
			p11 = FFTW(plan_many_dft_c2r)(1, &rN, M0*K1,   (FFTW(complex)*)lin, &rN, 1,   N*2, 
								(rtype*)lin2, &N, 1,   N, FFTW_MEASURE|FFTW_DESTROY_INPUT);//prod: FFTW_MEASURE
		}
	} else {
		p11 = FFTW(plan_many_dft)(1, &N, M0*K1,   (FFTW(complex)*)lin, &N, 1,   N, 
				(FFTW(complex)*)lin2, &N, 1,   N, direction, FFTW_MEASURE|FFTW_DESTROY_INPUT);//prod: FFTW_MEASURE
	}
	p12 = FFTW(plan_many_dft)(1, &K, N1*M0,   (FFTW(complex)*)lin, &K, 1,   K, 
			(FFTW(complex)*)lin2, &K, 1,   K, direction, FFTW_MEASURE|FFTW_DESTROY_INPUT);//prod: FFTW_MEASURE
	p13 = FFTW(plan_many_dft)(1, &M, K0*N1,   (FFTW(complex)*)lin, &M, 1,   M, 
			(FFTW(complex)*)lin2, &M, 1,   M, direction, FFTW_MEASURE|FFTW_DESTROY_INPUT);//prod: FFTW_MEASURE
			
	
#ifdef FFT_LOCAL_TRANSPOSE
	FFTW(plan) p12 = FFTW(plan_many_dft)(1, &N, N1*M0,   (FFTW(complex)*)lin, &N, N1*M0,   1, 
														 (FFTW(complex)*)lin2, &N, 1,   N, direction, FFTW_MEASURE|FFTW_DESTROY_INPUT);//prod: FFTW_MEASURE
#endif
	
#ifdef FFT_MPI_TRANSPOSE
	FFTW(plan) mpip1 = FFTW(mpi_plan_many_transpose)(P[1], P[1], N1*K1*M0*2, 1, 1, (rtype*)lin, (rtype*)lin2, cart1, FFTW_MEASURE);
	FFTW(plan) mpip2 = FFTW(mpi_plan_many_transpose)(P[0], P[0], N1*M0*K0*2, 1, 1, (rtype*)lin, (rtype*)lin2, cart2, FFTW_MEASURE);
//	FFTW(plan) mpip1 = FFTW(mpi_plan_many_transpose)(P[1], N, N1*M0*2, 1, M0, (rtype*)lin, (rtype*)lin2, cart1, FFTW_MEASURE|FFTW_MPI_TRANSPOSED_OUT);
//	FFTW(plan) mpip2 = FFTW(mpi_plan_many_transpose)(P[0], N, N1*M0*2, 1, N1, (rtype*)lin, (rtype*)lin2, cart2, FFTW_MEASURE|FFTW_MPI_TRANSPOSED_OUT);//|FFTW_MPI_TRANSPOSED_IN);
#endif
	pfft_plan plan = (pfft_plan)malloc(sizeof(struct pfft_plan_t));
	plan->lin=lin;
	plan->lin2=lin2;
	plan->p11=p11;plan->p12=p12;plan->p13=p13;
	#ifdef FFT_MPI_TRANSPOSE
	plan->mpip1=mpip1;plan->mpip2=mpip2;
	#else
	plan->cart1=cart1; plan->cart2=cart2;
	#endif
	
	plan->N1=N1;plan->M0=M0;plan->K0=K0;plan->K1=K1;
	plan->N=N;plan->M=M;plan->K=K;
	plan->P[0]=P[0];
	plan->P[1]=P[1];
	plan->coor[0]=coor[0];	
	plan->coor[1]=coor[1];
	*rlin=lin;
	*rlin2=lin2;
	return plan;
}


void pfft_execute(pfft_plan plan,pfft_time times) {
	type *lin = plan->lin;
	type *lin2 = plan->lin2;
	FFTW(plan) p11=plan->p11,p12=plan->p12,p13=plan->p13;
#ifdef FFT_MPI_TRANSPOSE
	FFTW(plan) mpip1=plan->mpip1,mpip2=plan->mpip2;
#else
	MPI_Comm cart1=plan->cart1, cart2=plan->cart2;
#endif
	int N1=plan->N1,M0=plan->M0,K0=plan->K0,K1=plan->K1;
	int N=plan->N,M=plan->M,K=plan->K;
	int *P = plan->P;
	
	double time_fft=0,time_local=0,time_mpi1=0,time_mpi2=0,time;
	int i,z,y,x;
	
#ifdef DEBUG2
	int *coor = plan->coor;
	printf("%d %d: copy in lin\n",coor[0],coor[1]);
	for (z=0;z<K1;z++) {
		for(y=0;y<M0;y++) {
			printf("%d %d: ",coor[0],coor[1]);
			for (x=0;x<N;x++) {
				printf("%f+%fi ",((rtype*)lin)[(x+y*N+z*M0*N)*2],((rtype*)lin)[(x+y*N+z*M0*N)*2+1]);
			}
			printf("\n");
		}
	}
#endif

	//lin: x,y,z
	
	time=MPI_Wtime();
	FFTW(execute)(p11); //in:lin out:lin2
	time_fft=MPI_Wtime()-time;

#ifdef DEBUG2
	printf("%d %d: FFT in x\n",coor[0],coor[1]);
	for (z=0;z<K1;z++) {
		for(y=0;y<M0;y++) {
			printf("%d %d: ",coor[0],coor[1]);
			for (x=0;x<N;x++) {
				printf("%f+%fi ",((rtype*)lin2)[(x+y*N+z*M0*N)*2],((rtype*)lin2)[(x+y*N+z*M0*N)*2+1]);
			}
			printf("\n");
		}
	}
#endif
	
	time=MPI_Wtime(); 
	//prepare for AllToAll
	//1. (most outer) axes (x) is split into P[1] parts of size M0 for sending 
	for (i=0;i<P[1];i++) { //index cube along long axis
		for (z=0;z<K1;z++) { //3. z l
			for (y=0;y<M0;y++) { //2. y k
				for (x=0;x<fmin(N1,N-N1*i);x++) { //1. x j
					lin[x+y*N1+z*M0*N1+i*M0*N1*K1]=lin2[(i*N1+x)+y*N+z*N*M0];
				}
			}
		}
	}
	time_local=MPI_Wtime()-time;
	
	

	//send, recv
	time=MPI_Wtime();

#ifdef FFT_MPI_TRANSPOSE
    FFTW(execute)(mpip1);
#else
    MPI_Alltoall(lin,M0*N1*K1*sizeof(type)/sizeof(rtype),MPI_RTYPE,lin2,M0*N1*K1*sizeof(type)/sizeof(rtype),MPI_RTYPE,cart1);
#endif
	time_mpi1=MPI_Wtime()-time;


	
	time=MPI_Wtime();
#ifdef FFT_LOCAL_TRANSPOSE
	for (i=0;i<P[1];i++) { //index cube along long axis
		for (l=0;l<M0;l++) { //3.
			for (j=0;j<M0;j++) { //1.
				for (k=0;k<M0;k++) { //2.
					lin[k+j*M0+l*M0*M0+i*M0*M0*M0]=lin2[j+k*M0+l*M0*M0+i*M0*M0*M0];
				}
			}
		}
	}	
#else
	
	//bring back in matrix form (could be avoided by storing blocks as eleftheriou)
	//thus make z ( 1. axes) again contiguos
	//also local transpose 1 and 3 
	//then z,y,x
	for (i=0;i<P[1];i++) { //index cube along long axis
		for (x=0;x<N1;x++) { //1.j
			for (y=0;y<M0;y++) { //2.k
				for (z=0;z<fmin(K1,K-K1*i);z++) { //3.l
					lin[(i*K1+z)+y*K+x*K*M0]=lin2[x+y*N1+z*M0*N1+i*M0*N1*K1];
				}
			}
		}
	}	
#endif
	time_local+=MPI_Wtime()-time;

#ifdef DEBUG2
	printf("%d %d: transposed x-z\n",coor[0],coor[1]);
	for (z=0;z<K;z++) {
		for(y=0;y<M0;y++) {
			printf("%d %d: ",coor[0],coor[1]);
			for (x=0;x<N1;x++) {
				printf("%f ",((rtype*)lin)[(z+y*K+x*M0*K)*2]);
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
	time_fft+=MPI_Wtime()-time;
	
#ifdef DEBUG2
	printf("%d %d: FFT in z\n",coor[0],coor[1]);
	for (z=0;z<K;z++) {
		for(y=0;y<M0;y++) {
				printf("%d %d: ",coor[0],coor[1]);
				for (x=0;x<N1;x++) {
					printf("%f ",((rtype*)lin2)[(z+y*K+x*M0*K)*2]);
				}
				printf("\n");
			}
	}
#endif
	//prepare alltoall. split 1 axes (z) into P[0] parts with size M0 
	time=MPI_Wtime();
	for (i=0;i<P[0];i++) { //index cube along long axis
		for (x=0;x<N1;x++) { //3.x l 
			for (y=0;y<M0;y++) { //2.y k
				for (z=0;z<fmin(K0,K-K0*i);z++) { //1.z j
					lin[z+y*K0+x*M0*K0+i*M0*K0*N1]=lin2[(i*K0+z)+y*K+x*K*M0];
				}
			}
		}
	}
	
	time_local+=MPI_Wtime()-time;

	time=MPI_Wtime();
#ifdef FFT_MPI_TRANSPOSE
    FFTW(execute)(mpip2);
#else
	MPI_Alltoall(lin,M0*K0*N1*sizeof(type)/sizeof(rtype),MPI_RTYPE,lin2,M0*K0*N1*sizeof(type)/sizeof(rtype),MPI_RTYPE,cart2);
#endif
	time_mpi2=MPI_Wtime()-time;

	time=MPI_Wtime();
	//make y contiguous  and also transpose 1 and 2
	//now y,z,x

	for (i=0;i<P[0];i++) { //index cube along long axis
		for (x=0;x<N1;x++) { //3. x l
			for (z=0;z<K0;z++) { //1. z j
				for (y=0;y<fmin(M0,M-M0*i);y++) { //2. y k
					lin[(i*M0+y)+z*M+x*M*K0]=lin2[z+y*K0+x*M0*K0+i*M0*K0*N1];
				}
			}
		}
	}
	time_local+=MPI_Wtime()-time;

#ifdef DEBUG2
	printf("%d %d: transposed y-z\n",coor[0],coor[1]);
	for (z=0;z<K0;z++) {
		for(y=0;y<M;y++) {
				printf("%d %d: ",coor[0],coor[1]);
				for (x=0;x<N1;x++) {
					printf("%f ",((rtype*)lin)[(y+z*M+x*M*K0)*2]);
				}
				printf("\n");
			}
	}
#endif	
	time=MPI_Wtime();
	FFTW(execute)(p13);
	time_fft+=MPI_Wtime()-time;

#ifdef DEBUG2
	printf("%d %d: FFT in y\n",coor[0],coor[1]);
	for (z=0;z<K0;z++) {
		for(y=0;y<M;y++) {
					printf("%d %d: ",coor[0],coor[1]);
					for (x=0;x<N1;x++) {
						printf("%f ",((rtype*)lin2)[(y+z*M+x*M*K0)*2]);
					}
					printf("\n");
				}
		}
#endif
	times->fft=time_fft;
	times->local=time_local;
	times->mpi2=time_mpi1;
	times->mpi1=time_mpi2;
}

void pfft_destroy(pfft_plan plan) {
	FFTW(destroy_plan)(plan->p11);
	FFTW(destroy_plan)(plan->p12);
	FFTW(destroy_plan)(plan->p13);
	
	
#ifdef FFT_MPI_TRANSPOSE
	FFTW(plan) mpip1=plan->mpip1;
	FFTW(plan) mpip2=plan->mpip2;
		
	FFTW(destroy_plan)(mpip1);
	FFTW(destroy_plan)(mpip2);
#endif
	FFTW(free)(plan->lin);
	FFTW(free)(plan->lin2);

	free(plan);
	
}

void pfft_local_size(pfft_plan plan,int* N1,int* M0,int* K0,int* K1,int** coor) {
	*N1=plan->N1;
	*M0=plan->M0;
	*K0=plan->K0;
	*K1=plan->K1;
	*coor=plan->coor;
}
