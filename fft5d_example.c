//============================================================================
// Name        : fft.cpp
// Author      : Roland Schulz
// Version     :
// Copyright   : GPL
// Description : Test and Timing for FFT5D
//============================================================================

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <unistd.h>



#include "fft5d.h"

void init_random(rtype* x, int l);
void avg(double* d, int n);



int main(int argc,char** argv)
{
	int size,prank;
	MPI_Init( &argc, &argv );
	MPI_Comm_size(MPI_COMM_WORLD,&size);
	MPI_Comm_rank(MPI_COMM_WORLD,&prank);


	int N=0,K=0,M=0,P0=0,N_measure=10;
	int flags = 0;
	int c;

	const char* helpmsg = "\
Usage: %s [OPTION] N\n\
   or: %s [OPTION] N M K \n\
\n\
Compute forward and backward parallel FFT on NxNxN or NxMxK size data on 2D prozessor grid\n\
Correctness is checked by comparison forward and backward transform equal to identity\n\
\n\
Options:\n\
  -P CPUS   CPUS number of processors used in 1st dimension\n\
            default is to use most squared CPU grid\n\
  -R        use real input (forward) or output (backward) data\n\
            default is complex to complex\n\
  -D        activate debuging\n\
  -N RUNS	RUNS number of measurement runs for timing (default: 10)\n\
  -O        use YZ order first (default: ZY order)\n\
            see manual\n";
		
	while ((c = getopt (argc, argv, "BORhDP:N:")) != -1)

		switch (c) { 
		case 'D':flags|=FFT5D_DEBUG;break;
		case 'R':flags|=FFT5D_REALCOMPLEX;break;
		case 'O':flags|=FFT5D_ORDER_YZ;break;
		case 'P':P0=atoi(optarg);break;
		case 'N':N_measure=atoi(optarg);break;
		default:
		        printf("%s: invalid option -- %c\n",argv[0],c);
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
                printf("%s: incorrect number of parameters\n",argv[0],c);
		printf(helpmsg,argv[0],argv[0]);
		abort();
	} 


	int rN=N;
	if (flags&FFT5D_REALCOMPLEX) {
		N = N/2+1;
	}

	type *lin,*lout,*initial;
	int N1,M0,K0,K1,*coor;
	fft5d_plan p1,p2;
	
	p1 = fft5d_plan_3d(rN,M,K,MPI_COMM_WORLD,P0, flags, &lin,&lout);
	fft5d_local_size(p1,&N1,&M0,&K0,&K1,&coor);
	int lsize = N*M0*K1; 
	initial=malloc(lsize*sizeof(type)); 
	
	int Nb,Mb,Kb; //dimension for backtransform (in starting order)
	if (!(flags&FFT5D_ORDER_YZ)) {Nb=M;Mb=K;Kb=rN;}		
	else {Nb=K;Mb=rN;Kb=M;}
	
	p2 = fft5d_plan_3d(Nb,Mb,Kb,MPI_COMM_WORLD,P0,  
		(flags|FFT5D_BACKWARD|FFT5D_NOMALLOC)^FFT5D_ORDER_YZ,&lout,&lin);

	init_random((rtype*)lin,lsize*sizeof(type)/sizeof(rtype));
	memcpy(initial,lin,lsize*sizeof(type));
	
	int m;
	struct fft5d_time_t ptimes={0};
	double ttime=0;
	for (m=0;m<N_measure;m++) {
	        ttime-=MPI_Wtime();
		fft5d_execute(p1, &ptimes);
		fft5d_execute(p2, &ptimes);
		ttime+=MPI_Wtime();
		if (m==0) {

			if (prank==0) printf("Comparison\n");
			
			fft5d_compare_data(lin, initial, p2, 1, 1); 
			if (flags&FFT5D_DEBUG) { 
				return 0;
			} else {
				if (prank==0) printf("OK\n");
			}
		

		}
	} // end measure
	struct fft5d_time_t otimes={0};
	double ottime;
	ptimes.fft/=N_measure;ptimes.local/=N_measure;ptimes.mpi1/=N_measure;ptimes.mpi2/=N_measure;
	ttime/=N_measure;
	//printf("avg: %lf\n",time_mpi1[0]);

	MPI_Reduce(&ptimes,&otimes,sizeof(ptimes)/sizeof(double),MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
	MPI_Reduce(&ttime,&ottime,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);

	if(prank==0) {
		printf("Timing (ms): local: %lf, fft: %lf, mpi1: %lf, mpi2: %lf, total: %lf\n",
				otimes.local*1000,otimes.fft*1000,otimes.mpi1*1000,otimes.mpi2*1000,ottime*1000);
	}
	
	fft5d_destroy(p2);
	fft5d_destroy(p1);
	
	MPI_Finalize();

	return 0;	
}

//initialize vector x of length l with random values
void init_random(rtype* x, int l) {
	int i;
	for (i=0;i<l;i++) {
		x[i]=((rtype)rand())/RAND_MAX;
	}
}


