/* MPI-parallel FFT
 * Global vector has N unknowns, each processor works with its
 * part, which has lN = N/p unknowns.
 * Author: Huan Zhang/ Cai Maitland-Davies
 */

//mpic++ -fopenmp fft_v1.cpp -std=c++14
//mpirun -np 8 ./a.out 20 1

#include <iostream>
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <string.h>
#include <cstdlib>
#include <complex>
#include <cmath>
#include <cassert>
#include "utils.h"
#include <fftw3.h>

using namespace std::complex_literals;

double printError(std::complex<double> *fk,fftw_complex *result, int N){
	double err=0.0;
	#pragma omp parallel for default(none) shared(N,fk,result) reduction(+:err)
	for (int j=0;j<N;j++){
		err+=(real(fk[j])-result[j][0])* (real(fk[j])-result[j][0])+(imag(fk[j])-result[j][1])* (imag(fk[j])-result[j][1]);
	}
	return sqrt(err);
}

//DFT by matrix product
void naiveFFT( std::complex<double> *fx,std::complex<double> *fk, int N){
	std::complex<double> * FN    = (std::complex<double> *) calloc(2*sizeof(double), N*N	);
	for (int j=0;j<N;j++){
		for (int k=0;k<N;k++){
			std::complex<double> z = 1i*(2.0*M_PI/N*j*k);
			FN[j+k*N]=exp(z);
		}
	}
	for(int j=0;j<N;j++){
		fk[j]=0.0+0.0*1i;
		for(int k=0;k<N;k++){
			fk[j]+=FN[j+k*N]*fx[k];	
		}
	}
	free(FN);
}


//FFT by inducive formula. do not use this for computation. This is just an application of inducive formula.
void fastFT_inducive(std::complex<double> *fx, std::complex<double> *fk,int N){
	if (N>1){
		assert(("N%2!=0",N%2==0));
		std::complex<double> * fxeven = (std::complex<double> *) calloc(2*sizeof(double), N/2);
		std::complex<double> * fxodd = (std::complex<double> *) calloc(2*sizeof(double), N/2);
		std::complex<double> * fkeven = (std::complex<double> *) calloc(2*sizeof(double), N/2);
		std::complex<double> * fkodd = (std::complex<double> *) calloc(2*sizeof(double), N/2);

#pragma omp parallel for default(none) shared(N,fx,fxeven,fxodd)
		for (int k=0;k<N/2;k++){
			fxeven[k]=fx[2*k];
			fxodd[k]=fx[2*k+1];
		}
		fastFT_inducive(fxeven,fkeven,N/2);
		fastFT_inducive(fxodd,fkodd,N/2);
		
	#pragma omp parallel for default(none) shared(N,fk,fkeven,fkodd)	
		for (int k=0;k<N/2;k++){
			std::complex<double> z=-1i*(2.0*M_PI/N*k);
			fk[k]=fkeven[k]+exp(z)*fkodd[k];
			fk[k+N/2]=fkeven[k]-exp(z)*fkodd[k];
		}


		free(fxeven);
		free(fxodd);
		free(fkeven);
		free(fkodd);
	}else{fk[0]=fx[0];}

}


//compute base^exp and return an integer value
int ipow(int base, int exp)
{
    int result = 1;
    for (;;)
    {
        if (exp & 1)
            result *= base;
        exp >>= 1;
        if (!exp)
            break;
        base *= base;
    }

    return result;
}


//get the index of t0 t1 and k in the algorithm on page http://www.cs.berkeley.edu/~demmel/cs267/lecture24/lecture24.html.
void getind(int m,int s,int j, int &ind1,int &ind2,int &indw, int *binm1, int *vec2pow){
	//m is the decimal number, binm is the binary number corresponding to m, s is the length of vec binm
	int k=0; int res=m;int power;
	for(int ss=0;ss<s;ss++){
		binm1[ss]=0;
	}

	while (res>=1){
		power=ipow(2,s-k-2);
		if(k<j){
			binm1[k]=res/power;
		}else{binm1[k+1]=res/power;}

		res%= power;
		k++;
	}
	binm1[j]=0;
	for(int k=0;k<s;k++){
		ind1+=vec2pow[k]*binm1[k];
	}
	ind2=ind1+ipow(2,s-j-1);
	for (int jj=0;jj<j+1;jj++){
		indw+=binm1[j-jj]*vec2pow[jj];
	}

}

//FFT on single core without use of MPI
void fastFT_iter(std::complex<double> *fx, int s,int N,int *binm1,int *vec2pow){
	std::complex<double> logomega =2*M_PI/N*(1i),x,w,t0,t1;
	for(int j=0;j<s;j++){
		for(int k=0;k<N/2;k++){
			int indt0=0, indt1=0,indw=0;
			getind(k,s,j, indt0,indt1,indw, binm1, vec2pow);
			w=exp(indw*logomega);
			t0=fx[indt0];
			t1=fx[indt1];
			x=w*t1;
			fx[indt0]=t0+x;
			fx[indt1]=t0-x;
		}
	}
	
}


//from decimal to binary number with length len.
void dec2bin(int *bin, int len, int dec){
	for(int j=0;j<len;j++){bin[j]=0;}
	int k=0;int power;int res=dec;
	while (res>=1){
		power=ipow(2,len-k-1);
		bin[k]=res/power;
		res%=power;
		k++;
	}
}


//inverse procedure of dec2bin. from binary number with length len to a decimal number and return it.
int bin2dec(int *bin, int len){
	int dec=0;
	for (int j=0;j<len;j++){
		dec+=bin[j]*ipow(2,len-j-1);
	}
	return dec;	
}

void Reorder_fk(std::complex<double> *fk,std::complex<double> *fx,int N,int s){
	int *bin=(int*)calloc(sizeof(int),s);
	int *bin_inv=(int*)calloc(sizeof(int),s);
	for(int ind=0;ind<N;ind++){
		dec2bin(bin,s,ind);
		for(int ss=0;ss<s;ss++){bin_inv[ss]=bin[s-1-ss];}
		fk[bin2dec(bin_inv,s)]=fx[ind];
	}
}


int main(int argc, char * argv[]) {
 	int s=4;
	int NT=4;
	int mpirank, p;
	MPI_Status status, status1;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
	MPI_Comm_size(MPI_COMM_WORLD, &p);
	

	/* get name of host running MPI process */
	char processor_name[MPI_MAX_PROCESSOR_NAME];
	int name_len;
	MPI_Get_processor_name(processor_name, &name_len);
//	printf("Rank %d/%d running on %s.\n", mpirank, p, processor_name);
 

	sscanf(argv[1], "%d", &s);
	#ifdef _OPENMP
		sscanf(argv[2], "%d", &NT);
        omp_set_num_threads(NT);
	#endif

	# pragma omp parallel
	  {
	#ifdef _OPENMP
	    int my_threadnum = omp_get_thread_num();
	    int numthreads = omp_get_num_threads();
	#else
	    int my_threadnum = 0;
	    int numthreads = 1;
	#endif
//	    printf("Hello, I'm thread %d out of %d\n", my_threadnum, numthreads);
	  }


	/*most naive FFT by matrix * vector*/
	int N=ipow(2,s);
	int lN=N/p;
	if ((N % p)!=0 && mpirank==0){
     	printf("N: %d, local N: %d\n", N, lN);
     	printf("Exiting. N must be a multiple of p\n");
     	MPI_Abort(MPI_COMM_WORLD, 0);
	}

	int logp=(int)(log(p)/log(2)+0.5);
	printf("N=%d, p=%d, logp=%d\n",N,p,logp);
	std::complex<double> * fx    = (std::complex<double> *) calloc(2*sizeof(double), N	);
	std::complex<double> * fk    = (std::complex<double> *) calloc(2*sizeof(double), N	);
	std::complex<double> * fxcopy    = (std::complex<double> *) calloc(2*sizeof(double), N	);
	std::complex<double> * fkcopy    = (std::complex<double> *) calloc(2*sizeof(double), N	);
	std::complex<double> * sendbuffer   = (std::complex<double> *) calloc(2*sizeof(double), lN	);
	std::complex<double> * recvbuffer   = (std::complex<double> *) calloc(2*sizeof(double), lN	);

	int *bin_minind=(int*)calloc(sizeof(int),s);
	int *bin_maxind=(int*)calloc(sizeof(int),s);
	int *bin_mink=(int*)calloc(sizeof(int),s-1);
	int *bin_maxk=(int*)calloc(sizeof(int),s-1);
	int *binm1=(int *) calloc(sizeof(int),s);
	int *vec2pow=(int *) calloc(sizeof(int),s);
	std::complex<double> logomega =2*M_PI/N*(1i),x,w,t0,t1;
	int *bin_commrank=(int*)calloc(sizeof(int),logp);
	int *bin_ind=(int*)calloc(sizeof(int),s);//[k(0)k(1)...k(j-1)(0/1)k(j)...k(s-2)]

	for (int k=0;k<s;k++){
		vec2pow[k]=pow(2,s-k-1);
	}

	//make a copy of fx to apply two FFT on each(one is not parallel) to check the correctness of the result
	for(int j=0;j<N;j++){
		fx[j]=sin(2*j)+cos(3*j)*1i;
		fxcopy[j]=fx[j];
	}
	double t;
	MPI_Barrier(MPI_COMM_WORLD);
	//FFT using multiple core MPI 
	t=omp_get_wtime();
	for(int j=0;j<s;j++){
			if(j<logp){
				//communicate
				dec2bin(bin_commrank, logp, mpirank);
				bin_commrank[j]=1-bin_commrank[j];
				int comm_rank=bin2dec(bin_commrank,logp);
//				printf("j=%d, rank=%d, comm_rank=%d\n",j, mpirank,comm_rank);
				for (int ind=0;ind<lN;ind++){
					sendbuffer[ind]=fx[lN*mpirank+ind];
				}
				MPI_Sendrecv(sendbuffer,lN,MPI_C_DOUBLE_COMPLEX,comm_rank,123,recvbuffer,lN,MPI_C_DOUBLE_COMPLEX,comm_rank,123,MPI_COMM_WORLD,&status );
				for (int ind=0;ind<lN;ind++){
					fx[lN*comm_rank+ind]=recvbuffer[ind];
				}

				for (int ind=lN*mpirank;ind<lN*(mpirank+1);ind++){
					dec2bin(bin_ind, s, ind);
					//calculate w
					std::complex<double> w,t0,t1,x; int indw=0,indt0,indt1;
					for (int m=0;m<j;m++){
						indw+=bin_ind[j-1-m]*ipow(2,s-2-m);
					}
					w=exp(indw*logomega);
					//calculate t0 and t1;
					if (bin_ind[j]==0){indt0=ind;indt1=ind+ipow(2,s-1-j);}
					else{indt1=ind;indt0=ind-ipow(2,s-1-j);}
					t0=fx[indt0];t1=fx[indt1];
					x=w*t1;
					fx[indt0]=t0+x;fx[indt1]=t0-x;
				}
				
			}else{
			//after logp steps, no need to communicate
					dec2bin(bin_minind,s,lN*mpirank);
					dec2bin(bin_maxind,s,lN*(mpirank+1)-1);
					/*
					std::cout<<"rank="<<mpirank<<"bin_minind="<<lN*mpirank<<"bin_maxind="<<lN*(mpirank+1)-1<<std::endl;
					for(int m=0;m<s;m++){std::cout<<bin_minind[m]<<" ";}
					printf("\n");
					for(int m=0;m<s;m++){std::cout<<bin_maxind[m]<<" ";}
					printf("\n");
					*/
					for(int kk=0;kk<s;kk++){
						if(kk<j){bin_mink[kk]=bin_minind[kk];}else if(kk>j){bin_mink[kk-1]=bin_minind[kk];}
						if(kk<j){bin_maxk[kk]=bin_maxind[kk];}else if(kk>j){bin_maxk[kk-1]=bin_maxind[kk];}
					}
					int mink=bin2dec(bin_mink,s-1);
					int maxk=bin2dec(bin_maxk,s-1);
					for(int k=mink;k<=maxk;k++){
						int indt0=0, indt1=0,indw=0;
						getind(k,s,j, indt0,indt1,indw, binm1, vec2pow);
						w=exp(indw*logomega);
						t0=fx[indt0];
						t1=fx[indt1];
						x=w*t1;
						fx[indt0]=t0+x;
						fx[indt1]=t0-x;
					}
						
			}
	
	}
	for (int ind=0;ind<lN;ind++){
		sendbuffer[ind]=fx[lN*mpirank+ind];
	}
	
//	printf("rank=%d ",mpirank);for(int k=mpirank*lN;k<(mpirank+1)*lN;k++) printf("%+.2f%+.2fi ",real(fx[k]),imag(fx[k]));printf("\n");
	MPI_Gather(sendbuffer,lN, MPI_C_DOUBLE_COMPLEX, fx,lN,MPI_C_DOUBLE_COMPLEX,0,MPI_COMM_WORLD);
	
	double tmpi=omp_get_wtime() - t;
	if(mpirank==0)
	{
		//standard fftw3 library
		fftw_complex signal[N];
		fftw_complex result[N];
		fftw_plan plan=fftw_plan_dft_1d(N,signal,result,FFTW_BACKWARD,FFTW_ESTIMATE);
		for(int j=0;j<N;j++){
			signal[j][0]=sin(2*j);
			signal[j][1]=cos(3*j);
		}
		t = omp_get_wtime();
		fftw_execute(plan);
		printf("Standard FFTW time=%6.4fs\n",omp_get_wtime() - t);
		/*
		printf("fftw result=");
		for(int j=0;j<N;j++){
			printf("%+.2f%+.2fi ",result[j][0],result[j][1]);}printf("\n");
		/**/
		fftw_destroy_plan(plan);

		//multiple core FFT time implemented before
		printf("Multiple core FFT time: %6.4fs, ",tmpi );
		Reorder_fk(fk,fx,N,s);
		/*
		printf("fx=");
		for(int j=0;j<N;j++){
			printf("%+.2f%+.2fi ",real(fk[j]),imag(fk[j]));
		}
		printf("\n");
		/**/
		printf("Error(multiplecore)=%6.4f\n",printError(fk,result,N));	
	    
		
		//implement non-parallel FFT on single core 0
		t = omp_get_wtime();
		fastFT_iter(fxcopy,s,N,binm1,vec2pow);
		printf("Single core FFT time: %6.4fs, ",omp_get_wtime() - t );
		Reorder_fk(fkcopy,fxcopy,N,s);
		/*
		printf("single core fxcopy=");
		for(int j=0;j<N;j++){
			printf("%+.2f%+.2fi ",real(fkcopy[j]),imag(fkcopy[j]));
		}
		printf("\n");
		/**/
		printf("Error(fksinglecore)=%6.4f\n",printError(fkcopy,result,N));	
	
		//naive matrix product
		for(int j=0;j<N;j++){
			fx[j]=sin(2*j)+cos(3*j)*1i;
		}
		t = omp_get_wtime();
		naiveFFT(fx,fk,N);
		printf("NaiveFFTtime=%6.4fs  ",omp_get_wtime() - t);
		/*
		printf("naivefk=");
		for(int j=0;j<N;j++){
			printf("%+.2f%+.2fi ",real(fk[j]),imag(fk[j]));
		}
		printf("\n");
		/**/
		printf("Error(fknaive)=%6.4f\n",printError(fk,result,N));	
	}
	free(binm1);
	free(vec2pow);
	free(bin_minind);
	free(bin_maxind);
	free(bin_mink);
	free(bin_maxk);
	free(bin_commrank);
	free(bin_ind);
			/*	
	printf("rank=%d   ",mpirank);
	for(int k=lN*mpirank;k<lN*(mpirank+1);k++){
		printf("%+.2f%+.2fi ",real(fx[k]),imag(fx[k]));
	}
	printf("\n");
	*/
	//	Timer tt;
//	tt.tic();
//	naiveFFT(fx,fk,N);
//    printf("Reference time: %6.4fs\n", tt.toc());

	free(sendbuffer);
	free(recvbuffer);
	free(fx);
	free(fxcopy);
	free(fk);
	free(fkcopy);

}
