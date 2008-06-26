//============================================================================
// Name        : fft.cpp
// Author      : Roland Schulz
// Version     :
// Copyright   : GPL
// Description : Hello World in C, Ansi-style
//============================================================================

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>



#include "fft5d.h"

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
	int size,prank;
	MPI_Init( &argc, &argv );
	MPI_Comm_size(MPI_COMM_WORLD,&size);
	MPI_Comm_rank(MPI_COMM_WORLD,&prank);


	int N=10,K=10,M=10,P0=0;
	int x,y,z;
	int direction = -1;
	int realcomplex = 0;
	int debug = 0;
	int fftorder = ZY;
	switch (argc) { //falltrough!
	case 9:
		debug=atoi(argv[8]);		
	case 8:
		fftorder=atoi(argv[7]);
	case 7:
		direction=atoi(argv[6]);
	case 6:
		realcomplex=atoi(argv[5]);
	case 5:
		P0=atoi(argv[4]);
	case 4:
		K=atoi(argv[3]);
	case 3:
		M=atoi(argv[2]);
	case 2:
		N=atoi(argv[1]);
	}

	type* in = (type*) FFTW(malloc)(sizeof(type) * N*M*K);
	FFTW(plan) p2;

	int rN=N;
	if (realcomplex) {
		N = N/2+1;
	}
	if (realcomplex) {
		if (direction==-1) {
			p2 = FFTW(plan_dft_r2c_3d)(K, M, rN, (rtype*)in, (FFTW(complex)*)in, FFTW_ESTIMATE);
		} else {
			p2 = FFTW(plan_dft_c2r_3d)(K, M, rN, in, (rtype*)in, FFTW_ESTIMATE);
		}
	} else {
		p2 = FFTW(plan_dft_3d)(K, M, N, (FFTW(complex)*)in, (FFTW(complex)*)in, direction, FFTW_ESTIMATE);
	}

	init_random((rtype*)in,N*M*K*sizeof(type)/sizeof(rtype),1,1);

	if (direction==1 && realcomplex==1) { //in[0][y][z] needs to be real (otherwise data not conjugate complex)
		for (y=0;y<M;y++) {
			for (z=0;z<K;z++) {
				((rtype*)in)[(y*N+z*M*N)*2+1]=0;
			}
		}
	}
	if (debug) {
		printf("Input data\n");
		if (prank==0) {
			for(z=0;z<K;z++) {
				for (y=0;y<M;y++) {
					for (x=0;x<N;x++) {				
						printf("%f+%fi ",((rtype*)in)[(x+y*N+z*M*N)*2],((rtype*)in)[(x+y*N+z*M*N)*2+1]);
					}
					printf("\n");
				}
			}
		}
	}

	type *lin,*lout;
	int N1,N0,M1,M0,K0,K1,*coor;
	fft5d_plan plan;
	if (direction==-1) { //write in as standard X,Y,Z
		plan = fft5d_plan_3d(rN,M,K,MPI_COMM_WORLD,P0, direction, realcomplex,0,fftorder, &lin,&lout);

		fft5d_local_size(plan,&N1,&M0,&K0,&K1,&coor);
		for(x=0;x<N;x++) {  //x i
			for (y=0;y<fmin(M0,M-coor[0]*M0);y++) { //y j  2nd of fmin for non dividable
				for (z=0;z<fmin(K1,K-coor[1]*K1);z++) { //z k
					//lin[i]=in[i+coor[0]*M+coor[1]*M*M];  //in[Px][i][Pz]
					lin[x+y*N+z*N*M0]=in[x+(coor[0]*M0+y)*N+(coor[1]*K1+z)*N*M];  //in[Px][i][Pz]
					//lin[i*2+1]=in[coor[0]*2+1+i*M+coor[1]*M*M]; //imaginary part
				}
			}
		}
	} else { //write in as tranposed Z,X,Y so that it is X,Y,Z as result
		//neccessary for realcomplex to have X as real axes
		if (fftorder==ZY) {
			plan = fft5d_plan_3d(K,rN,M,MPI_COMM_WORLD,P0, direction, realcomplex,0,fftorder,&lin,&lout);
	
			fft5d_local_size(plan,&K1,&N0,&M0,&M1,&coor);
			for(z=0;z<K;z++) {  //x i
				for (x=0;x<fmin(N0,N-coor[0]*N0);x++) { //y j  2nd of fmin for non dividable
					for (y=0;y<fmin(M1,M-coor[1]*M1);y++) { //z k
						//lin[i]=in[i+coor[0]*M+coor[1]*M*M];  //in[Px][i][Pz]
						lin[z+x*K+y*K*N0]=in[(x+coor[0]*N0)+(y+coor[1]*M1)*N+z*N*M];  //in[Px][i][Pz]
						//lin[i*2+1]=in[coor[0]*2+1+i*M+coor[1]*M*M]; //imaginary part
					}
				}
			}
		} else {
			plan = fft5d_plan_3d(M,K,rN,MPI_COMM_WORLD,P0, direction, realcomplex,0,fftorder,&lin,&lout);
				
			fft5d_local_size(plan,&M1,&K0,&N0,&N1,&coor);
			for(y=0;y<M;y++) {  //x i
				for (z=0;z<fmin(K0,K-coor[0]*K0);z++) { //y j  2nd of fmin for non dividable
					for (x=0;x<fmin(N1,N-coor[1]*N1);x++) { //z k
						//lin[i]=in[i+coor[0]*M+coor[1]*M*M];  //in[Px][i][Pz]
						lin[y+z*M+x*M*K0]=in[(x+coor[1]*N1)+y*N+(z+coor[0]*K0)*N*M];  //in[Px][i][Pz]
						//lin[i*2+1]=in[coor[0]*2+1+i*M+coor[1]*M*M]; //imaginary part
					}
				}
			}
		}
	}


#define N_measure 10
	int m;
	double time_fft[N_measure]={0},time_local[N_measure]={0},time_mpi1[N_measure]={0},time_mpi2[N_measure]={0};
	fft5d_time ptimes=(fft5d_time)malloc(sizeof(struct fft5d_time_t));
	for (m=0;m<N_measure;m++) {

		fft5d_execute(plan, ptimes,debug);
		time_fft[m]=ptimes->fft;
		time_local[m]=ptimes->local;
		time_mpi1[m]=ptimes->mpi1;
		time_mpi2[m]=ptimes->mpi2;
		if (m==0) {
			if (M>128) {
				if (prank==0) printf("No correctness check above 128\n");
			} else {
				if (debug) {
				printf("Input\n");
					if (prank==0) {
						for(z=0;z<K;z++) {
							for (y=0;y<M;y++) {
								for (x=0;x<N;x++) {				
									printf("%f+%fi ",((rtype*)in)[(x+y*N+z*M*N)*2],((rtype*)in)[(x+y*N+z*M*N)*2+1]);
								}
								printf("\n");
							}
						}
					}
				}
				FFTW(execute)(p2);
				if (debug) {
	
					if (prank==0) {
						printf("Result from FFTW\n");
						for(z=0;z<K;z++) {
						for (y=0;y<M;y++) {
							for (x=0;x<N;x++) {				
								printf("%f+%fi ",((rtype*)in)[(x+y*N+z*M*N)*2],((rtype*)in)[(x+y*N+z*M*N)*2+1]);
							}
							printf("\n");
						}
						}
					}
				}
				//print("in",in,N,2,ld);
				//print("tmp",tmp,N,2,ld);
				//assert(test_equal(in,tmp,N*N*N,N,2,N));

				if (prank==0) printf("Comparison\n");
				compare_data(lout, in, plan,debug);
				if (debug) { 
					return 0;
				} else {
					if (prank==0) printf("OK\n");
				}
//				if (direction==-1) {
//					for (x=0;x<fmin(N1,N-N1*coor[1]);x++) {//x i
//						for (z=0;z<fmin(K0,K-K0*coor[0]);z++) {//z j
//							for (y=0;y<M;y++) {//y k
//								for (l=0;l<2;l++) {
//#ifdef DEBUG2
//									printf("%f %f ",((rtype*)lout)[(y+z*M+x*M*K0)*sizeof(type)/sizeof(rtype)+l],
//											((rtype*)in)[(x+coor[1]*N1+y*N+(coor[0]*K0+z)*N*M)*sizeof(type)/sizeof(rtype)+l]);
//#else
//									assert(fabs(((rtype*)lout)[(y+z*M+x*M*K0)*sizeof(type)/sizeof(rtype)+l]-
//											((rtype*)in)[(coor[1]*N1+x+y*N+(coor[0]*K0+z)*N*M)*sizeof(type)/sizeof(rtype)+l])<2*N*M*K*EPS);
//#endif
//								}
//#ifdef DEBUG2
//								printf("\n");
//#endif										
//							}
//						}
//					}
//				} else {
//					for (z=0;z<fmin(K1,K-K1*coor[1]);z++) {//x i
//						for (y=0;y<fmin(M0,M-M0*coor[0]);y++) {//z j
//							for (x=0;x<rN;x++) {//y k
//								int ls = realcomplex?1:2;
//								for (l=0;l<ls;l++) { //don't check padding field
//#ifdef DEBUG2
//									printf("%f,%f ",((rtype*)lout)[x*ls+(y*N+z*N*M0)*2+l],
//											((rtype*)in)[x*ls+((coor[0]*M0+y)*N+(coor[1]*K1+z)*N*M)*2+l]);
//
//#else
//									assert(fabs(((rtype*)lout)[x*ls+(y*N+z*N*M0)*sizeof(type)/sizeof(rtype)+l]-
//											((rtype*)in)[x*ls+((coor[0]*M0+y)*N+(coor[1]*K1+z)*N*M)*sizeof(type)/sizeof(rtype)+l])<2*N*M*K*EPS);
//									//									rtype diff = fabs(((rtype*)lout)[x*ls+(y*N+z*N*M0)*sizeof(type)/sizeof(rtype)+l]-
//									//											((rtype*)in)[x*ls+((coor[0]*M0+y)*N+(coor[1]*K1+z)*N*M)*sizeof(type)/sizeof(rtype)+l]);
//									//									if (diff>N*M*K*EPS) printf("ERROR: LARGE DIFFERENCE: %g\n",diff/(N*M*K*EPS));
//#endif
//								}
//							}
//#ifdef DEBUG2
//							printf("\n");	
//#endif
//
//						}
//					}
//				}
//				if (prank==0) printf("OK\n");
//#ifdef DEBUG2
//				return 0;
//#endif
			}
		}
	} // end measure
	free(ptimes);	
	avg(time_local,N_measure);
	avg(time_fft,N_measure);
	avg(time_mpi1,N_measure);
	avg(time_mpi2,N_measure);

	//printf("avg: %lf\n",time_mpi1[0]);

	double times[]={time_local[0],time_fft[0],time_mpi1[0],time_mpi2[0],
			time_local[1],time_fft[1],time_mpi1[1],time_mpi2[1]},otimes[sizeof(times)/sizeof(double)];

	MPI_Reduce(times,otimes,sizeof(times)/sizeof(double),MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);

	if(prank==0) {
		printf("Timing (ms): local: %lf+/-%lf, fft: %lf+/-%lf, mpi1: %lf+/-%lf, mpi2: %lf+/-%lf\n",
				otimes[0]/size,sqrt(otimes[4]/size),
				otimes[1]/size,sqrt(otimes[5]/size),
				otimes[2]/size,sqrt(otimes[6]/size),
				otimes[3]/size,sqrt(otimes[7]/size));
	}
	FFTW(destroy_plan)(p2);
	FFTW(free)(in);
	fft5d_destroy(plan);

	return 0;	
}
