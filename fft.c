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

#define FFTW(x) fftwf_##x
typedef float type;
#define EPS __FLT_EPSILON__

void init_random(type* x, int l, int dim, int ld) {
	int i,j;
	for (i=0;i<((dim==1)?1:l);i++) {
		for (j=0;j<l;j++) {
			x[j+i*ld]=((type)rand())/RAND_MAX;
		}
	}
}

int test_equal(type* x, type* y, type f, int l, int dim, int ld) {
	int i,j;
	for (i=0;i<((dim==1)?1:l);i++) {
		for (j=0;j<l;j++) {
			printf("%g %g %g %g\n",x[j+i*ld],y[j+i*ld],(x[j+i*ld]-y[j+i*ld]*f)/x[j+i*ld],__FLT_EPSILON__);
			if ((x[j+i*ld]-y[j+i*ld]*f)/x[j+i*ld]>EPS*f) return 0;
		}
	}
	return 1;
}

void print(const char* fn,const type* M,int l,int dim, int ld) {
	int i,j;
	FILE* f = fopen(fn,"w");
	for (j=0;j<((dim==1)?1:l);j++) {
		for (i=0;i<l;i++) {
			fprintf(f,"%f",M[i+j*ld]);
			if (i<l-1) fprintf(f,",");
		}
		fprintf(f,"\n");
	}
	fclose(f);
}

void printc(const char* fn,const FFTW(complex)* M,int n,int m) {
	int i,j;
	FILE* f = fopen(fn,"w");
	for (j=0;j<n;j++) {
		for (i=0;i<m;i++) {
			fprintf(f,"%f%+fi",creal(M[i+j*m]),cimag(M[i+j*m]));
			if (i<m-1) fprintf(f,",");
		}
		fprintf(f,"\n");
	}
	fclose(f);
}



int main()
{
	const int N = 8;
	type *in;
	FFTW(plan) p11,p12,p13,p2;
	
	int ld=2*(N/2+1);

	in = (type*) FFTW(malloc)(sizeof(type) * ld*N*N);
#ifndef NDEBUG
	type* tmp = (type*) FFTW(malloc)(sizeof(type) * ld*N*N);
#endif
//	    fftw_plan_many_dft_r2c(rank, n, howmany, *in, *inembed, istride, idist,
//	                   *out, *onembed, ostride, odist, flags);

	
	p11 = FFTW(plan_many_dft_r2c)(1, &N, N*N,                in, &N, 1,      ld, 
							  				 (FFTW(complex)*)in, &N, 1,      ld/2, FFTW_ESTIMATE);//prod: FFTW_MEASURE
	p12 = FFTW(plan_many_dft)(1, &N, ld/2*N, (FFTW(complex)*)in, &N, ld/2,   1, 
											 (FFTW(complex)*)in, &N, ld/2,   1, FFTW_FORWARD, FFTW_ESTIMATE);//prod: FFTW_MEASURE
	p13 = FFTW(plan_many_dft)(1, &N, ld/2*N, (FFTW(complex)*)in, &N, ld/2*N, 1, 
											 (FFTW(complex)*)in, &N, ld/2*N, 1, FFTW_FORWARD, FFTW_ESTIMATE);//prod: FFTW_MEASURE
		
	p2 = FFTW(plan_dft_c2r_3d)(N, N, N, (FFTW(complex)*)in, in, FFTW_ESTIMATE);

	init_random(in,N,2,ld);
	memcpy(tmp,in,sizeof(type)*ld*N);
	FFTW(execute)(p11);
	FFTW(execute)(p12);
	FFTW(execute)(p13); 
	FFTW(execute)(p2);

	//print("in",in,N,2,ld);
	//print("tmp",tmp,N,2,ld);
	assert(test_equal(in,tmp,N*N*N,N,2,ld));
	printf("OK\n");

	FFTW(destroy_plan)(p11);
	FFTW(destroy_plan)(p12);
	FFTW(destroy_plan)(p2);
	FFTW(free)(in);
	
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


