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
#include <mpi.h>
#include <float.h>
#include <math.h>

#define FFTW(x) fftwf_##x
typedef FFTW(complex) type;
#define EPS __FLT_EPSILON__

void init_random(float* x, int l, int dim, int ld) {
	int i,j;
	for (i=0;i<((dim==1)?1:l);i++) {
		for (j=0;j<l;j++) {
			x[j+i*ld]=((type)rand())/RAND_MAX;
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

int main(int argc,char** argv)
{
	MPI_Init( &argc, &argv );
	int size,prank;
	MPI_Comm_size(MPI_COMM_WORLD,&size);
	MPI_Comm_rank(MPI_COMM_WORLD,&prank);
	int N = 10;
	int P0 = lfactor(size);
	if (argc>1) {
		N=atoi(argv[1]);
		if (argc>2) {
			P0=atoi(argv[2]);
		}
	}
	int P[] = {P0,size/P0}; //per proc 4*4*4 or 8*4*2
	int Nx=N/P[0],Ny=N/P[1];
	int i,j,k,l,coor[2];
	type *in;
	
	if (prank==0) printf("N: %d, P: %dx%d\n",N,P[0],P[1]);
	
	if (N%P0!=0 || N%P[1]!=0) {
		if (prank==0) printf("N needs to be dividible by the processor grid dimensions\n");
		MPI_Finalize();
		return 1;
	}
	
	FFTW(plan) p11,p12,p2;
	 
	//MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	int wrap[]={0,0};
	MPI_Comm cart;
	MPI_Cart_create(MPI_COMM_WORLD,2,P,wrap,1,&cart); //true: reorder
	MPI_Cart_get(cart,2,P,wrap,coor);
	
	int N2=N;//for real:2*(N/2+1)
	in = (type*) FFTW(malloc)(sizeof(type) * N2*N*N);


	
	type *lin = (type*)FFTW(malloc)(sizeof(type) * N2*Nx*Ny); //local in
	type *lin2 = (type*)FFTW(malloc)(sizeof(type) * N2*Nx*Ny); //local in

	p2 = FFTW(plan_dft_3d)(N, N, N, (FFTW(complex)*)in, (FFTW(complex)*)in, FFTW_FORWARD, FFTW_ESTIMATE);
	p11 = FFTW(plan_many_dft)(1, &N, Nx*Ny,   (FFTW(complex)*)lin, &N, 1,   N, 
													 (FFTW(complex)*)lin2, &N, 1,   N, FFTW_FORWARD, FFTW_MEASURE|FFTW_DESTROY_INPUT);//prod: FFTW_MEASURE
	p12 = FFTW(plan_many_dft)(1, &N, Nx*Ny,   (FFTW(complex)*)lin, &N, Nx*Ny,   1, 
														 (FFTW(complex)*)lin2, &N, 1,   N, FFTW_FORWARD, FFTW_MEASURE|FFTW_DESTROY_INPUT);//prod: FFTW_MEASURE
				
	init_random((float*)in,N2*N*N*sizeof(type)/sizeof(float),1,1);
		

	
	for(i=0;i<N;i++) {
		for (j=0;j<Nx;j++) {
			for (k=0;k<Ny;k++) {
				//lin[i]=in[i+coor[0]*N2+coor[1]*N2*N];  //in[Px][i][Pz]
				lin[i+j*N+k*N*Nx]=in[(coor[0]*Nx+j)+i*N2+(coor[1]*Ny+k)*N2*N];  //in[Px][i][Pz]
				//lin[i*2+1]=in[coor[0]*2+1+i*N2+coor[1]*N2*N]; //imaginary part
			}
		}
	}
	
	

	
	int rdim1[] = {0,1}, rdim2[] = {1,0};
	MPI_Comm cart1, cart2;
	MPI_Cart_sub(cart, rdim1 , &cart1);
	MPI_Cart_sub(cart, rdim2 , &cart2);

	//lin: y,x,z
	
#define N_measure 100
	int m;
	double time_fft[N_measure]={0},time_local[N_measure]={0},time_mpi1[N_measure]={0},time_mpi2[N_measure]={0},time;
	
	for (m=0;m<N_measure;m++) {
	time=MPI_Wtime(); 
	FFTW(execute)(p11); //in:lin out:lin2
	time_fft[m]=MPI_Wtime()-time;

	time=MPI_Wtime(); 
	for (i=0;i<P[1];i++) { //index cube along long axis
		for (l=0;l<Ny;l++) { //3.
			for (k=0;k<Nx;k++) { //2.
				for (j=0;j<Ny;j++) { //1.
					lin[j+k*Ny+l*Nx*Ny+i*Nx*Ny*Ny]=lin2[(i*Ny+j)+k*N+l*N*Nx];
				}
			}
		}
	}
	time_local[m]=MPI_Wtime()-time;
	
//	for (k=0;k<Nx;k++) {
//		for (j=0;j<Nx;j++) {
//			printf("%d %d: ",coor[0],coor[1]);
//			for(i=0;i<N;i++) {
//				printf("%f ",((float*)lin)[(i+j*N+k*Nx*N)*2]);
//			}
//			printf("\n");
//		}
//	}

	//send, recv
	time=MPI_Wtime();
	MPI_Alltoall(lin,Nx*Ny*Ny*sizeof(type)/sizeof(float),MPI_FLOAT,lin2,Nx*Ny*Ny*sizeof(type)/sizeof(float),MPI_FLOAT,cart1);
	time_mpi1[m]=MPI_Wtime()-time;
	
	//printf("%lf\n",time_mpi1[m]);
	
	time=MPI_Wtime();
#ifdef FFT_TRANSPOSE
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
	for (i=0;i<P[1];i++) { //index cube along long axis
		for (j=0;j<Ny;j++) { //1.
			for (k=0;k<Nx;k++) { //2.
				for (l=0;l<Ny;l++) { //3.
					lin[(i*Ny+l)+k*N+j*N*Nx]=lin2[j+k*Ny+l*Nx*Ny+i*Nx*Ny*Ny];
				}
			}
		}
	}	
#endif
	time_local[m]+=MPI_Wtime()-time;
	
	time=MPI_Wtime();
#ifdef FFT_TRANSPOSE
	FFTW(execute)(p12);
#else
	FFTW(execute)(p11);
#endif
	time_fft[m]+=MPI_Wtime()-time;
	
	time=MPI_Wtime();
	for (i=0;i<P[0];i++) { //index cube along long axis
		for (l=0;l<Ny;l++) { //3.
			for (k=0;k<Nx;k++) { //2.
				for (j=0;j<Nx;j++) { //1.
					lin[j+k*Nx+l*Nx*Nx+i*Nx*Nx*Ny]=lin2[(i*Nx+j)+k*N+l*N*Nx];
				}
			}
		}
	}
	time_local[m]+=MPI_Wtime()-time;

	time=MPI_Wtime();
	MPI_Alltoall(lin,Nx*Nx*Ny*sizeof(type)/sizeof(float),MPI_FLOAT,lin2,Nx*Nx*Ny*sizeof(type)/sizeof(float),MPI_FLOAT,cart2);
	time_mpi2[m]=MPI_Wtime()-time;

	time=MPI_Wtime();
	for (i=0;i<P[0];i++) { //index cube along long axis
		for (l=0;l<Ny;l++) { //3.
			for (j=0;j<Nx;j++) { //1.
				for (k=0;k<Nx;k++) { //2.
					lin[(i*Nx+k)+j*N+l*N*Nx]=lin2[j+k*Nx+l*Nx*Nx+i*Nx*Nx*Ny];
				}
			}
		}
	}
	time_local[m]+=MPI_Wtime()-time;

	time=MPI_Wtime();
	FFTW(execute)(p11);
	time_fft[m]+=MPI_Wtime()-time;
	
	if (m==0) {
		if (N>128) {
			if (prank==0) printf("No correctness check above 128\n");
		} else {
			FFTW(execute)(p2);
		
			//print("in",in,N,2,ld);
			//print("tmp",tmp,N,2,ld);
			//assert(test_equal(in,tmp,N*N*N,N,2,N2));
			
			for (i=0;i<N2*sizeof(type)/sizeof(float);i+=2) {//x
				for (j=0;j<Nx;j++) {//z
					for (k=0;k<Ny;k++) {//y
						for (l=0;l<2;l++) {
	//						printf("%f %f ",((float*)lin2)[i+l+j*N*sizeof(type)/sizeof(float)+k*N*Nx*sizeof(type)/sizeof(float)],((float*)in)[i+l+(coor[1]*Nx+k)*N2*sizeof(type)/sizeof(float)+(coor[0]*Nx+j)*N*N2*sizeof(type)/sizeof(float)]);
							assert(((float*)lin2)[i+l+j*N*sizeof(type)/sizeof(float)+k*N*Nx*sizeof(type)/sizeof(float)]-((float*)in)[i+l+(coor[1]*Ny+k)*N2*sizeof(type)/sizeof(float)+(coor[0]*Nx+j)*N*N2*sizeof(type)/sizeof(float)]<N*N*N*EPS);
						}
	//					printf("\n");
					}
				}
			}
		
			if (prank==0) printf("OK\n");
		}
	}
	} // end measure
	
	avg(time_local,N_measure);
	avg(time_fft,N_measure);
	avg(time_mpi1,N_measure);
	avg(time_mpi2,N_measure);
	
	//printf("avg: %lf\n",time_mpi1[0]);
	
	double times[]={time_local[0],time_fft[0],time_mpi1[0],time_mpi2[0],
					time_local[1],time_fft[1],time_mpi1[1],time_mpi2[1]},otimes[8];
	
	MPI_Reduce(times,otimes,8,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
	
	if(prank==0) {
		printf("Timing (ms): local: %lf+/-%lf, fft: %lf+/-%lf, mpi1: %lf+/-%lf, mpi2:%lf+/-%lf\n",
				otimes[0]/size,sqrt(otimes[4]/size),
				otimes[1]/size,sqrt(otimes[5]/size),
				otimes[2]/size,sqrt(otimes[6]/size),
				otimes[3]/size,sqrt(otimes[7]/size));
	}
	

	FFTW(destroy_plan)(p11);
	//FFTW(destroy_plan)(p12);
	FFTW(destroy_plan)(p2);
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


