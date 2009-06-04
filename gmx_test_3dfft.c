#include "gmx_parallel_3dfft.h"
#include "smalloc.h"

void split_communicator(MPI_Comm comm, MPI_Comm cart[], int P[]) {
    int wrap[]={0,0};
    int coor[2];
    MPI_Comm gcart;
    MPI_Cart_create(comm,2,P,wrap,1,&gcart); //parameter 4: value 1: reorder
    MPI_Cart_get(gcart,2,P,wrap,coor);
    int rdim1[] = {0,1}, rdim2[] = {1,0};
    MPI_Cart_sub(gcart, rdim1 , &cart[0]);
    MPI_Cart_sub(gcart, rdim2 , &cart[1]);
}

//check eriks mail for one thing left to do
int main(int argc,char** argv) {
    int prank,i,j,k;

    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&prank);
    debug=stderr;
    if (argc!=6) {
	if (prank==0) printf("Usage: gmx_test_3dfft N M K P1 P2\n");
	return 1;
    }

    gmx_parallel_3dfft_t      pfft_setup;
    ivec                      ndata = {atoi(argv[1]),atoi(argv[2]),atoi(argv[3])};
    real *                    real_data;
    t_complex *               complex_data;
    MPI_Comm                  comm[2];
    ivec                      local_ndata;
    ivec                      local_offset;
    ivec                      local_size;    

    int P[]={atoi(argv[4]),atoi(argv[5])};
    real* compare;

    split_communicator(MPI_COMM_WORLD,comm,P);
    gmx_parallel_3dfft_init   (&pfft_setup, ndata, &real_data, &complex_data, comm, 0, 0); //last two: slab2index, bReprodusible
    gmx_parallel_3dfft_real_limits(pfft_setup,local_ndata,local_offset,local_size);
    snew(compare,ndata[0]*ndata[1]*ndata[2]); 
    srand(time(0)+prank);
    if (debug) {
	fprintf(debug,"local_ndata: %d %d %d\n",local_ndata[0],local_ndata[1],local_ndata[2]);
	fprintf(debug,"local_size: %d %d %d\n",local_size[0],local_size[1],local_size[2]);
	fprintf(debug,"local_offset: %d %d %d\n",local_offset[0],local_offset[1],local_offset[2]);
    }
    for (i=0;i<local_ndata[0];i++)  {
	for (j=0;j<local_ndata[1];j++)  {
	    for (k=0;k<local_ndata[2];k++)  {
		compare[i*local_ndata[1]*local_ndata[2]+j*local_ndata[2]+k]=real_data[i*local_size[1]*local_size[2]+j*local_size[2]+k]=((real)rand())/RAND_MAX;
	    }
	}
    }
    gmx_parallel_3dfft_execute(pfft_setup,GMX_FFT_REAL_TO_COMPLEX,0,0);
    gmx_parallel_3dfft_execute(pfft_setup,GMX_FFT_COMPLEX_TO_REAL,0,0);
    for (i=0;i<local_ndata[0];i++)  {
	for (j=0;j<local_ndata[1];j++)  {
	    for (k=0;k<local_ndata[2];k++)  {
		if (fabs(compare[i*local_ndata[1]*local_ndata[2]+j*local_ndata[2]+k] - 
			 real_data[i*local_size[1]*local_size[2]+j*local_size[2]+k]/(ndata[0]*ndata[1]*ndata[2]))>2*ndata[0]*ndata[1]*ndata[2]*FFT5D_EPS) {
		    printf("error: %d %d %d: %f %f\n",i,j,k,compare[i*local_ndata[1]*local_ndata[2]+j*local_ndata[2]+k],
			   real_data[i*local_size[1]*local_size[2]+j*local_size[2]+k]/(ndata[0]*ndata[1]*ndata[2]));
		}
	    }
	}
    }    
    gmx_parallel_3dfft_destroy(pfft_setup);
    MPI_Finalize();
    return 0;
}
