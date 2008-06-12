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

enum fftorder {
	ZY,
	YZ
};

//NxMxK the size of the data
//comm communicator to use for PFFT
//P0 number of processor in 1st axes (can be null for automatic)
//lin is allocated by pfft because size of array is only known after planning phase

pfft_plan pfft_plan_3d(int NG, int MG, int KG, MPI_Comm comm, int P0, int direction, int realcomplex, int inplace, type** rlin, type** rlout) {
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
		printf("N: %d, M: %d, K: %d, P: %dx%d, real to complex: %d, direction: %d\n",NG,MG,KG,P[0],P[1],realcomplex,direction);
		if (fmax(fmax(lpfactor(NG),lpfactor(MG)),lpfactor(KG))>7) {
			printf("WARNING: FFT very slow with prime factors larger 7\n");
			printf("Change FFT size or in case you cannot change it look at\n");
			printf("http://www.fftw.org/fftw3_doc/Generating-your-own-code.html\n");
		}
	}
	
	if (NG==0 || MG==0 || KG==0) {
		if (prank==0) printf("FATAL: Datasize cannot be zero in any dimension\n");
		MPI_Finalize();
		return 0;
	}

	int rNG=NG,rMG=MG,rKG=KG;
	int fftorder = ZY; //TODO
	if (realcomplex) {
		if (direction==-1) NG = NG/2+1;
		else {
			if (fftorder==ZY) MG=MG/2+1;
			else KG=KG/2+1;
		}
	}
	
	
	int N0=ceil((double)NG/P[0]),N1=ceil((double)NG/P[1]);
	int M0=ceil((double)MG/P[0]),M1=ceil((double)MG/P[1]);
	int K0=ceil((double)KG/P[0]),K1=ceil((double)KG/P[1]);

	//for step 1-3 the local N,M,K sizes of the transposed system
	//C: contiguous dimension, and nP: number of processor in subcommunicator
	int N[3],M[3],K[3],C[3],rC[3],nP[2];
	
	
	M[0] = M0;
	K[0] = K1;
	C[0] = NG;
	rC[0] = rNG;
	if (fftorder==ZY) {
		N[0] = N1;
		nP[0] = P[1];
		C[1] = KG;
		N[1] = K0;
		M[1] = M0;
		K[1] = N1;
		nP[1] = P[0];
		C[2] = MG;
		rC[2] = rMG;
		M[2] = K0;
		K[2] = N1;
	} else {
		N[0] = N0;
		nP[0] = P[0];
		C[1] = MG;
		N[1] = M1;
		M[1] = N0;
		K[1] = K1;
		nP[1] = P[1];
		C[2] = KG;
		rC[2] = rKG;
		M[2] = N0;
		K[2] = M1;
	}
	
	//Difference between x-y-z regarding 2d docmposition is whether they are distributed 
	//along axis 1, 2 or both
	
	int coor[2];
	
	int wrap[]={0,0};
	MPI_Comm gcart;
	MPI_Cart_create(comm,2,P,wrap,1,&gcart); //parameter 4: value 1: reorder
	MPI_Cart_get(gcart,2,P,wrap,coor); 
	int rdim1[] = {0,1}, rdim2[] = {1,0};
	MPI_Comm cart[2];
	if (fftorder==ZY) {
		MPI_Cart_sub(gcart, rdim1 , &cart[0]);
		MPI_Cart_sub(gcart, rdim2 , &cart[1]);
	} else {
		MPI_Cart_sub(gcart, rdim1 , &cart[1]);
		MPI_Cart_sub(gcart, rdim2 , &cart[0]);
	}

	
	int lsize = fmax(N[0]*M[0]*K[0]*nP[0],N[1]*M[1]*K[1]*nP[1]);
	type* lin = (type*)FFTW(malloc)(sizeof(type) * lsize); //localin	
	type* lout = (type*)FFTW(malloc)(sizeof(type) * lsize); //local output
	
	int flags=FFTW_MEASURE|FFTW_DESTROY_INPUT;
	type* output=lout;
	pfft_plan plan = (pfft_plan)malloc(sizeof(struct pfft_plan_t));
	int s;
	for (s=0;s<3;s++) {
		if (inplace && s==2) {
			output=lin;
			flags^=FFTW_DESTROY_INPUT;
		}
		if (realcomplex && direction==-1 && s==0) {
			plan->p1d[s] = FFTW(plan_many_dft_r2c)(1, &rC[s], M[s]*K[s],   
					(rtype*)lin, &rC[s], 1,   C[s]*2, 
					(FFTW(complex)*)output, &C[s], 1,   C[s], flags);
		} else if (realcomplex && direction==1 && s==2) {
			plan->p1d[s] = FFTW(plan_many_dft_c2r)(1, &rC[s], M[s]*K[s],   
					(FFTW(complex)*)lin, &C[s], 1,   C[s], 
					(rtype*)output, &rC[s], 1,   C[s]*2, flags);
		} else {
			plan->p1d[s] = FFTW(plan_many_dft)(1, &C[s], M[s]*K[s],   
					(FFTW(complex)*)lin, &C[s], 1,   C[s], 
					(FFTW(complex)*)output, &C[s], 1,   C[s], direction, flags);
		}
	}
		
#ifdef FFT_MPI_TRANSPOSE
	//TODO XY
	FFTW(plan) mpip1 = FFTW(mpi_plan_many_transpose)(nP[1], nP[1], N1*K1*M0*2, 1, 1, (rtype*)lin, (rtype*)lout, cart[0], FFTW_MEASURE);
	FFTW(plan) mpip2 = FFTW(mpi_plan_many_transpose)(nP[0], nP[0], N1*M0*K0*2, 1, 1, (rtype*)lin, (rtype*)lout, cart[1], FFTW_MEASURE);
//	FFTW(plan) mpip1 = FFTW(mpi_plan_many_transpose)(P[1], N, N1*M0*2, 1, M0, (rtype*)lin, (rtype*)lout, cart[0], FFTW_MEASURE|FFTW_MPI_TRANSPOSED_OUT);
//	FFTW(plan) mpip2 = FFTW(mpi_plan_many_transpose)(P[0], N, N1*M0*2, 1, N1, (rtype*)lin, (rtype*)lout, cart[1], FFTW_MEASURE|FFTW_MPI_TRANSPOSED_OUT);//|FFTW_MPI_TRANSPOSED_IN);
#endif
	
	plan->lin=lin;
	plan->lout=lout;
	#ifdef FFT_MPI_TRANSPOSE
	plan->mpip1=mpip1;plan->mpip2=mpip2;
	#else
	plan->cart[0]=cart[0]; plan->cart[1]=cart[1];
	#endif
	
//	plan->N0=N0;plan->N1=N1;plan->M0=M0;plan->M1=M1;plan->K0=K0;plan->K1=K1;
//	plan->N=N;plan->M=M;plan->K=K;
//	plan->P[0]=P[0];
//	plan->P[1]=P[1];
	
	for (s=0;s<3;s++) {
		plan->N[s]=N[s];plan->M[s]=M[s];plan->K[s]=K[s];plan->C[s]=C[s];
	}
	for (s=0;s<2;s++) {
		plan->P[s]=nP[s];plan->coor[s]=coor[s];
	}
	
	plan->fftorder=fftorder;
		
	*rlin=lin;
	*rlout=lout;
	return plan;
}

enum order {
	XYZ,
	XZY,
	YXZ,
	YZX,
	ZXY,
	ZYX
};



//here x,y,z and N,M,K is in rotated coordinate system!!
//x (and N) is mayor (consecutive) dimension, y (M) middle and z (K) major
//N,M,K is size of local data!
//NG, MG, KG is size of global data
void splitaxes(type* lin,const type* lout,int N,int M,int K,int P,int NG) {
	int x,y,z,i;
	for (i=0;i<P;i++) { //index cube along long axis
		for (z=0;z<K;z++) { //3. z l
			for (y=0;y<M;y++) { //2. y k
				for (x=0;x<fmin(N,NG-N*i);x++) { //1. x j
					lin[x+y*N+z*M*N+i*M*N*K]=lout[(i*N+x)+y*NG+z*NG*M];
				}
			}
		}
	}
}

//make axis contiguous again (after AllToAll) and also do local transpose
//transpose mayor and major dimension
//variables see above
//the major, middle, minor order is only correct for x,y,z (N,M,K) for the input
void joinAxesTrans13(type* lin,const type* lout,int N,int M,int K,int P,int KG) {
	int i,x,y,z;
	for (i=0;i<P;i++) { //index cube along long axis
		for (x=0;x<N;x++) { //1.j
			for (z=0;z<fmin(K,KG-K*i);z++) { //3.l
				for (y=0;y<M;y++) { //2.k
					lin[(i*K+z)+y*KG+x*KG*M]=lout[x+y*N+z*M*N+i*M*N*K];
				}
			}
		}
	}
}

//make axis contiguous again (after AllToAll) and also do local transpose
//tranpose mayor and middle dimension
//variables see above
//the minor, middle, major order is only correct for x,y,z (N,M,K) for the input
void joinAxesTrans12(type* lin,const type* lout,int N,int M,int K,int P,int MG) {
	int i,z,y,x;
	for (i=0;i<P;i++) { //index cube along long axis
		for (z=0;z<K;z++) { 
			for (x=0;x<N;x++) { 
				for (y=0;y<fmin(M,MG-M*i);y++) { 
					lin[(i*M+y)+x*MG+z*MG*N]=lout[x+y*N+z*M*N+i*M*N*K];
				}
			}
		}
	}
}

void print_localdata(const type* lin, const char* txt, int N,int M,int K, int s, int fftorder, const int* coor) {
#ifdef DEBUG2
	int xo,yo,zo,x,y,z,o;
	if (fftorder==ZY) {
		switch (s) {
		case 0: o=XYZ; break;
		case 1: o=ZYX; break;
		case 2: o=YZX; break;
		}
	} else {
		switch (s) {
		case 0: o=ZXY; break;
		case 1: o=XZY; break;
		case 2: o=YZX; break;
		}
	}
	switch (o) {
	case XYZ:xo=1  ;yo=N  ;zo=N*M;break;
	case XZY:xo=1  ;yo=N*K;zo=N  ;break;
	case YXZ:xo=M  ;yo=1  ;zo=N*M;break;
	case YZX:xo=M*K;yo=1  ;zo=M  ;break;
	case ZXY:xo=K  ;yo=N*K;zo=1  ;break;
	case ZYX:xo=M*K;yo=K  ;zo=1  ;break;
	}
	printf(txt,coor[0],coor[1],s);
	for (z=0;z<K;z++) {
		for(y=0;y<M;y++) {
			printf("%d %d: ",coor[0],coor[1]);
			for (x=0;x<N;x++) {
				printf("%f+%fi ",((rtype*)lin)[(z*zo+y*yo+x*xo)*2],((rtype*)lin)[(z*zo+y*yo+x*xo)*2+1]);
			}
			printf("\n");
		}
	}
#endif
}

void pfft_execute(pfft_plan plan,pfft_time times) {
	type *lin = plan->lin;
	type *lout = plan->lout;
	FFTW(plan) *p1d=plan->p1d;
#ifdef FFT_MPI_TRANSPOSE
	FFTW(plan) *mpip=plan->mpip;
#else
	MPI_Comm *cart=plan->cart;
#endif
	//int N0=plan->N0,N1=plan->N1,M0=plan->M0,M1=plan->M1,K0=plan->K0,K1=plan->K1;
	//int N=plan->N,M=plan->M,K=plan->K;
	//int *P = plan->P;
	double time_fft=0,time_local=0,time_mpi[2]={0},time;	
	int *coor = plan->coor;
	int fftorder = plan->fftorder; 
	
	int *N=plan->N,*M=plan->M,*K=plan->K,*C=plan->C,*P=plan->P;
	
	

	//lin: x,y,z
	int s=0;
	print_localdata(lin, "%d %d: copy in lin\n", C[0], M[0], K[0], s, fftorder, coor);
	for (s=0;s<2;s++) {
		time=MPI_Wtime();
		FFTW(execute)(p1d[s]); //in:lin out:lout
		time_fft+=MPI_Wtime()-time;
	
		print_localdata(lout, "%d %d: FFT %d\n", C[s], M[s], K[s], s, fftorder, coor);
		
		time=MPI_Wtime(); 
		//prepare for AllToAll
		//1. (most outer) axes (x) is split into P[1] parts of size M0 for sending
		splitaxes(lin,lout,N[s],M[s],K[s],P[s],C[s]);
		time_local+=MPI_Wtime()-time;
		
		//send, recv
		time=MPI_Wtime();
	#ifdef FFT_MPI_TRANSPOSE
			FFTW(execute)(mpip[s]);
	#else
	   	MPI_Alltoall(lin,N[s]*M[s]*K[s]*sizeof(type)/sizeof(rtype),MPI_RTYPE,lout,N[s]*M[s]*K[s]*sizeof(type)/sizeof(rtype),MPI_RTYPE,cart[s]);
	#endif
		time_mpi[s]=MPI_Wtime()-time;
	
		time=MPI_Wtime();
		//bring back in matrix form (could be avoided by storing blocks as eleftheriou, really?)
		//thus make z ( 1. axes) again contiguos
		//also local transpose 1 and 3 
		//then z,y,x
		if ((s==0 && fftorder==ZY) || (s==1 && fftorder==YZ)) 
			joinAxesTrans13(lin,lout,N[s],M[s],K[s],P[s],C[s+1]);
		else 
			joinAxesTrans12(lin,lout,N[s],M[s],K[s],P[s],C[s+1]);	
		time_local+=MPI_Wtime()-time;
	
		print_localdata(lin, "%d %d: tranposed %d\n", C[s+1], M[s+1], K[s+1], s+1, fftorder, coor);
				
		//print_localdata(lin, "%d %d: transposed x-z\n", N1, M0, K, ZYX, coor);
	}	
	
	time=MPI_Wtime();
	FFTW(execute)(p1d[2]);
	time_fft+=MPI_Wtime()-time;
	print_localdata(lout, "%d %d: FFT %d\n", C[s], M[s], K[s], s, fftorder, coor);
	//print_localdata(lout, "%d %d: FFT in y\n", N1, M, K0, YZX, coor);
	
	if (times!=0) {
		times->fft=time_fft;
		times->local=time_local;
		times->mpi2=time_mpi[1];
		times->mpi1=time_mpi[0];
	}
}

void pfft_destroy(pfft_plan plan) {
	int s;
	for (s=0;s<3;s++)
		FFTW(destroy_plan)(plan->p1d[s]);
	
#ifdef FFT_MPI_TRANSPOSE
	for (s=0;s<2;s++)	
		FFTW(destroy_plan)(mpip[s]);
#endif
	FFTW(free)(plan->lin);
	FFTW(free)(plan->lout);

	free(plan);
	
}

void pfft_local_size(pfft_plan plan,int* N1,int* M0,int* K0,int* K1,int** coor) {
	*N1=plan->N[0];
	*M0=plan->M[0];
	*K1=plan->K[0];
	*K0=plan->N[1];
	
	*coor=plan->coor;
}
