/*============================================================================
  Name        : fft.cpp
  Author      : Roland Schulz
  Version     :
  Copyright   : GPL
  Description : Test and Timing for FFT5D
  ============================================================================*/

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <unistd.h>
#include <time.h>


#include "fft5d.h"

void init_random(fft5d_rtype* x, int l);
void avg(double* d, int n);



int main(int argc,char** argv)
{
    int size,prank;
    MPI_Init( &argc, &argv );
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Comm_rank(MPI_COMM_WORLD,&prank);


    int N=0,K=0,M=0,P0=0,N_measure=10;
    int x,y,z;
    int flags = 0;
    int c;
	
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
  -O        use YZ order (default: ZY order)\n\
            see manual\n";		
		
    while ((c = getopt (argc, argv, "BORhDP:N:")) != -1)

	switch (c) { 
	    case 'D':flags|=FFT5D_DEBUG;break;
	    case 'O':flags|=FFT5D_ORDER_YZ;break;
	    case 'B':flags|=FFT5D_BACKWARD;break;
	    case 'R':flags|=FFT5D_REALCOMPLEX;break;
	    case 'P':P0=atoi(optarg);break;
	    case 'N':N_measure=atoi(optarg);break;
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

    int ccheck=1;
    if (N*M*K>128*128*128) {
	if (prank==0) printf("No correctness check above 128\n");
	ccheck=0;
    }
    fft5d_type* in=0;
    if (ccheck) in = (fft5d_type*) FFTW(malloc)(sizeof(fft5d_type) * N*M*K);

    FFTW(plan) p2=0;

    int rN=N;
    if (flags&FFT5D_REALCOMPLEX) {
	N = N/2+1;
    }
    /*srand(time(0+prank);*/
    srand(prank);

    if (ccheck) {
	if (flags&FFT5D_REALCOMPLEX) {
	    if (!(flags&FFT5D_BACKWARD)) {
		p2 = FFTW(plan_dft_r2c_3d)(K, M, rN, (fft5d_rtype*)in, (FFTW(complex)*)in, FFTW_ESTIMATE);
	    } else {
		p2 = FFTW(plan_dft_c2r_3d)(K, M, rN, in, (fft5d_rtype*)in, FFTW_ESTIMATE);
	    }
	} else {
	    p2 = FFTW(plan_dft_3d)(K, M, N, (FFTW(complex)*)in, (FFTW(complex)*)in, (flags&FFT5D_BACKWARD)?1:-1, FFTW_ESTIMATE);
	}

	init_random((fft5d_rtype*)in,N*M*K*sizeof(fft5d_type)/sizeof(fft5d_rtype));

	if (flags&FFT5D_BACKWARD && flags&FFT5D_REALCOMPLEX) { /*in[0][y][z] needs to be real (otherwise data not conjugate complex)*/
	    for (y=0;y<M;y++) {
		for (z=0;z<K;z++) {
		    ((fft5d_rtype*)in)[(y*N+z*M*N)*2+1]=0;
		}
	    }
	}
	if (flags&FFT5D_DEBUG) {
	    printf("Input data\n");
	    if (prank==0) {
		for(z=0;z<K;z++) {
		    for (y=0;y<M;y++) {
			for (x=0;x<N;x++) {				
			    printf("%f+%fi ",((fft5d_rtype*)in)[(x+y*N+z*M*N)*2],((fft5d_rtype*)in)[(x+y*N+z*M*N)*2+1]);
			}
			printf("\n");
		    }
		}
	    }
	}
    }
    fft5d_type *lin,*lout;
    int N1,N0,M1,M0,K0,K1,*coor;
    fft5d_plan plan;
    if (!(flags&FFT5D_BACKWARD)) { /*write in as standard X,Y,Z*/
	plan = fft5d_plan_3d_cart(rN,M,K,MPI_COMM_WORLD,P0, flags, &lin,&lout);

	fft5d_local_size(plan,&N1,&M0,&K0,&K1,&coor);
	if (ccheck) {
	    for(x=0;x<N;x++) {  
		for (y=0;y<plan->pM[0];y++) { 
		    for (z=0;z<plan->pK[0];z++) { 
			lin[x+y*N+z*N*M0]=in[x+(plan->oM[0]+y)*N+(plan->oK[0]+z)*N*M];  
			/*fprintf(stderr,"%f+%fi ", ((fft5d_rtype*)in)[(x+(plan->oM[0]+y)*N+(plan->oK[0]+z)*N*M)*2],((fft5d_rtype*)in)[(x+(plan->oM[0]+y)*N+(plan->oK[0]+z)*N*M)*2+1]);*/
		    }
		}
	    }
	} else {
	    init_random((fft5d_rtype*)lin,N*M0*K1*sizeof(fft5d_type)/sizeof(fft5d_rtype));
	}
    } else { /*write in as tranposed Z,X,Y so that it is X,Y,Z as result*/
	/*neccessary for realcomplex to have X as real axes*/
	if (!(flags&FFT5D_ORDER_YZ)) {
	    plan = fft5d_plan_3d_cart(K,rN,M,MPI_COMM_WORLD,P0, flags,&lin,&lout);
		    
	    fft5d_local_size(plan,&K1,&N0,&M0,&M1,&coor);
	    if (ccheck) {
		for(z=0;z<K;z++) {  
		    for (x=0;x<plan->pM[0];x++) { 
			for (y=0;y<plan->pK[0];y++) { 
			    lin[z+x*K+y*K*N0]=in[(x+plan->oM[0])+(y+plan->oK[0])*N+z*N*M]; 
			}
		    }
		}
	    } else {
		init_random((fft5d_rtype*)lin,K*N0*M1*sizeof(fft5d_type)/sizeof(fft5d_rtype));
	    }
	} else {
	    plan = fft5d_plan_3d_cart(M,K,rN,MPI_COMM_WORLD,P0, flags,&lin,&lout);

	    fft5d_local_size(plan,&M1,&K0,&N0,&N1,&coor);
	    if (ccheck) {
		for(y=0;y<M;y++) {  
		    for (z=0;z<plan->pM[0];z++) { 
			for (x=0;x<plan->pK[0];x++) { 
			    lin[y+z*M+x*M*K0]=in[(x+plan->oK[0])+y*N+(z+plan->oM[0])*N*M];  
			}
		    }
		}
	    } else {
		init_random((fft5d_rtype*)lin,M*K0*N1*sizeof(fft5d_type)/sizeof(fft5d_rtype));
	    }
	}
    }


    int m;
    double *time_fft,*time_local,*time_mpi1,*time_mpi2;
    time_fft=calloc(sizeof(double),N_measure);
    time_local=calloc(sizeof(double),N_measure);
    time_mpi1=calloc(sizeof(double),N_measure);
    time_mpi2=calloc(sizeof(double),N_measure);
    fft5d_time ptimes=(fft5d_time)malloc(sizeof(struct fft5d_time_t));
    for (m=0;m<N_measure;m++) {
	ptimes->fft=ptimes->local=ptimes->mpi1=ptimes->mpi2=0;
	fft5d_execute(plan, ptimes);
	time_fft[m]=ptimes->fft;
	time_local[m]=ptimes->local;
	time_mpi1[m]=ptimes->mpi1;
	time_mpi2[m]=ptimes->mpi2;
	if (m==0) {
	    if (ccheck) {
		if (flags&FFT5D_DEBUG) {
		    printf("Input\n");
		    if (prank==0) {
			for(z=0;z<K;z++) {
			    for (y=0;y<M;y++) {
				for (x=0;x<N;x++) {				
				    printf("%f+%fi ",((fft5d_rtype*)in)[(x+y*N+z*M*N)*2],((fft5d_rtype*)in)[(x+y*N+z*M*N)*2+1]);
				}
				printf("\n");
			    }
			}
		    }
		}
		FFTW(execute)(p2);
		if (flags&FFT5D_DEBUG) {

		    if (prank==0) {
			printf("Result from FFTW\n");
			for(z=0;z<K;z++) {
			    for (y=0;y<M;y++) {
				for (x=0;x<N;x++) {				
				    printf("%f+%fi ",((fft5d_rtype*)in)[(x+y*N+z*M*N)*2],((fft5d_rtype*)in)[(x+y*N+z*M*N)*2+1]);
				}
				printf("\n");
			    }
			}
		    }
		}
		/*print("in",in,N,2,ld);
		  print("tmp",tmp,N,2,ld);
		  assert(test_equal(in,tmp,N*N*N,N,2,N));*/

		if (prank==0) printf("Comparison\n");
		fft5d_compare_data(lout, in, plan,0,0);
		if (flags&FFT5D_DEBUG) { 
		    return 0;
		} else {
		    if (prank==0) printf("OK\n");
		}
	    }

	}
    } /* end measure */
    free(ptimes);	
    avg(time_local,N_measure);
    avg(time_fft,N_measure);
    avg(time_mpi1,N_measure);
    avg(time_mpi2,N_measure);

    /*printf("avg: %lf\n",time_mpi1[0]);*/

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
    if (ccheck) { 
	FFTW(destroy_plan)(p2);
	FFTW(free)(in);
    }
    fft5d_destroy(plan);
    free(time_fft);free(time_local);free(time_mpi1);free(time_mpi2);
    MPI_Finalize();

    return 0;	
}

/*initialize vector x of length l with random values*/
void init_random(fft5d_rtype* x, int l) {
    int i;
    for (i=0;i<l;i++) {
	x[i]=((fft5d_rtype)rand())/RAND_MAX;
    }
}


/*average d of length n excluding first element, writing result in d[0]*/
void avg(double* d, int n) { 
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
