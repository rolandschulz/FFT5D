#include "smalloc.h"
#include "gmx_fatal.h"
#include "gmx_parallel_3dfft.h"

int
gmx_parallel_3dfft_init   (gmx_parallel_3dfft_t *    pfft_setup,
                           ivec                      ndata,
			   real **                   real_data,
			   t_complex **              complex_data,
                           MPI_Comm                  comm[2],
			   int                      *slab2index[2],
                           bool                      bReproducible) {
    	
    int rN=ndata[2],M=ndata[1],K=ndata[0];
    int flags = FFT5D_REALCOMPLEX; //|FFT5D_DEBUG; 
    snew(*pfft_setup,1);
    if (bReproducible) flags |= FFT5D_NOMEASURE; 
    
    MPI_Comm rcomm[]={comm[1],comm[0]};
    int Nb,Mb,Kb; //dimension for backtransform (in starting order)
    
    if (!(flags&FFT5D_ORDER_YZ)) { //currently always true because ORDER_YZ never set
	Nb=M;Mb=K;Kb=rN;		
    } else {
	Nb=K;Mb=rN;Kb=M;
    }
    
    (*pfft_setup)->p1 = fft5d_plan_3d(rN,M,K,rcomm, flags, (fft5d_type**)real_data, (fft5d_type**)complex_data,debug);
    
    (*pfft_setup)->p2 = fft5d_plan_3d(Nb,Mb,Kb,rcomm,
				      (flags|FFT5D_BACKWARD|FFT5D_NOMALLOC)^FFT5D_ORDER_YZ, (fft5d_type**)complex_data, (fft5d_type**)real_data,debug);
    
    return (*pfft_setup)->p1 != 0 && (*pfft_setup)->p2 !=0;
}
                           


int
gmx_parallel_3dfft_limits(fft5d_plan p, 
			       ivec                      local_offset,
			       ivec                      local_ndata,
			       ivec                      local_size) {
    int N1,M0,K0,K1,*coor;
    fft5d_local_size(p,&N1,&M0,&K0,&K1,&coor);  //M0=MG/P[0], K1=KG/P[1], NG,MG,KG global sizes
    
    local_offset[2]=0;
    local_offset[1]=p->coor[0]*M0;
    local_offset[0]=p->coor[1]*K1;
    
    local_ndata[2]=p->rC[0];
    local_ndata[1]=fmin(M0,p->MG-local_offset[1]);
    local_ndata[0]=fmin(K1,p->KG-local_offset[0]);
    
    if ((!(p->flags&FFT5D_BACKWARD)) && (p->flags&FFT5D_REALCOMPLEX)) {
	local_size[2]=p->C[0]*2;
    } else {
	local_size[2]=p->C[0];
    }
    local_size[1]=M0;
    local_size[0]=K1;
    return 0;
}

int
gmx_parallel_3dfft_real_limits(gmx_parallel_3dfft_t      pfft_setup,
			       ivec                      local_offset,
			       ivec                      local_ndata,
			       ivec                      local_size) {
    return gmx_parallel_3dfft_limits(pfft_setup->p1,local_ndata,local_offset,local_size);
}


int
gmx_parallel_3dfft_complex_limits(gmx_parallel_3dfft_t      pfft_setup,
				  ivec                      local_ndata,
				  ivec                      local_offset,
				  ivec                      local_size) {
    return gmx_parallel_3dfft_limits(pfft_setup->p2,local_ndata,local_offset,local_size);
}


int
gmx_parallel_3dfft_execute(gmx_parallel_3dfft_t    pfft_setup,
			   enum gmx_fft_direction  dir,
			   void *                  in_data,
			   void *                  out_data) {
    if ((!(pfft_setup->p1->flags&FFT5D_REALCOMPLEX)) ^ (dir==GMX_FFT_FORWARD ||dir==GMX_FFT_BACKWARD)) { 
	gmx_fatal(FARGS,"Invalid transform. Plan and execution don't match regarding reel/complex");
    }
    if (dir==GMX_FFT_FORWARD || dir==GMX_FFT_REAL_TO_COMPLEX) {
	fft5d_execute(pfft_setup->p1,0);
    } else {
	fft5d_execute(pfft_setup->p2,0);
    }
    return 0;
}

int
gmx_parallel_3dfft_destroy(gmx_parallel_3dfft_t    pfft_setup) {
    fft5d_destroy(pfft_setup->p2);
    fft5d_destroy(pfft_setup->p1);
    sfree(pfft_setup);
    return 0;
}
