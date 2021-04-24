/* MPI-parallel Jacobi smoothing to solve -u''=f
 * Global vector has N unknowns, each processor works with its
 * part, which has lN = N/p unknowns.
 * Author: Georg Stadler
 */
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
using namespace std::complex_literals;

double printError(std::complex<double> *fkref,std::complex<double> *fk , int N){
	double err=0.0;
	#pragma omp parallel for default(none) shared(N,fkref,fk) reduction(+:err)
	for (int j=0;j<N;j++){
		err+=abs(fkref[j]-fk[j])* abs(fkref[j]-fk[j]);
	}
	return sqrt(err);
}


void naiveFFT( std::complex<double> *fx,std::complex<double> *fk, int N){
	std::complex<double> * FN    = (std::complex<double> *) calloc(2*sizeof(double), N*N	);
	for (int j=0;j<N;j++){
		for (int k=0;k<N;k++){
			std::complex<double> z = -1i*(2.0*M_PI/N*j*k);
			FN[j+k*N]=exp(z);
		}
	}
/*	
	for(int j=0;j<N;j++){
		printf("np.array([");
		for(int k=0;k<N;k++){
			printf("%+.2f%+.2f*1j, ",real(FN[j+N*k]),imag(FN[j+N*k]));	
		}
		printf("]),\n");
	}
*/
	for(int j=0;j<N;j++){
		fk[j]=0.0+0.0*1i;
		for(int k=0;k<N;k++){
			fk[j]+=FN[j+k*N]*fx[k];	
		}
	}
	free(FN);
}

void fastFT_inducive(std::complex<double> *fx, std::complex<double> *fk,int N){
		//do not use this for computation. This is just an application of inducive formula.
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

void fastFT_iter(std::complex<double> *fx,int s,int N,int *binm1,
                                               int *vec2pow){
//we In regards touse this iterative method that is equivalent to the reducive formula.
	std::complex<double> logomega =2*M_PI/N*(1i),x,w,t0,t1;
 
	for(int j=0;j<s;j++){
		for(int k=0;k<N/2;k++){
			int indt0=0, indt1=0,indw=0;
			getind(k,s,j, indt0,indt1,indw, binm1, vec2pow);
	//		for (int ss=0;ss<s;ss++){printf("%d ",binm1[ss]);}

			w=exp(indw*logomega);
			t0=fx[indt0];
			t1=fx[indt1];
			x=w*t1;
			fx[indt0]=t0+x;
			fx[indt1]=t0-x;
	//		printf("j=%d,k=%d,ind1=%d,ind2=%d,indw=%d,logomega=%+f%+fi\n",j,k,indt0,indt1,indw,real(logomega),imag(logomega));
		}
	
		printf("j=%d ",j);for(int k=0;k<N;k++) printf("%+.2f%+.2fi ",real(fx[j]),imag(fx[j]));printf("\n");
	}

}
void dec2bin(int *bin, int len, int dec){
	for(int j=0;j<len;j++){bin[j]=0.0;}
	int k=0;int power;int res=dec;
	while (res>=1){
		power=ipow(2,len-k-1);
		bin[k]=res/power;
		res%=power;
		k++;
	}
}
int bin2dec(int *bin, int len){
	int dec=0;
	for (int j=0;j<len;j++){
		dec+=bin[j]*ipow(2,len-j-1);
	}
	return dec;	
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
	std::complex<double> * fxcopy    = (std::complex<double> *) calloc(2*sizeof(double), N	);
	std::complex<double> * sendbuffer   = (std::complex<double> *) calloc(2*sizeof(double), lN	);
	std::complex<double> * recvbuffer   = (std::complex<double> *) calloc(2*sizeof(double), lN	);

//	std::complex<double> * fx2   = (std::complex<double> *) calloc(2*sizeof(double), N	);
//	std::complex<double> * fk    = (std::complex<double> *) calloc(2*sizeof(double), N	);
//	std::complex<double> * fkf   = (std::complex<double> *) calloc(2*sizeof(double), N	);
	for(int j=0;j<N;j++){
		fx[j]=sin(2*j)+cos(3*j)*1i;
		fxcopy[j]=fx[j];
	}
	std::complex<double> logomega =2*M_PI/N*(1i),x,w,t0,t1;
	
	for(int j=0;j<s;j++){
				int *bin_ind=(int*)calloc(sizeof(int),s);//[k(0)k(1)...k(j-1)(0/1)k(j)...k(s-2)]

			if(j<logp){
				//communicate
				int *bin_commrank=(int*)calloc(sizeof(int),logp);
				dec2bin(bin_commrank, logp, mpirank);
				bin_commrank[j]=1-bin_commrank[j];
				int comm_rank=bin2dec(bin_commrank,logp);
				printf("j=%d, rank=%d, comm_rank=%d\n",j, mpirank,comm_rank);
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
			//no need to communicate
				for (int ind=lN*mpirank;ind<lN*mpirank+lN/2;ind++){
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

			}
	
//		printf("j=%d ",j);for(int k=0;k<N;k++) printf("%+.2f%+.2fi ",real(fx[j]),imag(fx[j]));printf("\n");
	}
		printf("rank=%d ",mpirank);for(int k=mpirank*lN;k<(mpirank+1)*lN;k++) printf("%+.2f%+.2fi ",real(fx[k]),imag(fx[k]));printf("\n");
	
		if (mpirank==0){
			int *binm1=(int *) calloc(sizeof(int),s);
			int *vec2pow=(int *) calloc(sizeof(int),s);
			for (int k=0;k<s;k++){
				vec2pow[k]=pow(2,s-k-1);
			}
			fastFT_iter(fxcopy,s,N,binm1,vec2pow);
			printf("fxFFT=");
			for(int j=0;j<N;j++){
				printf("%+.2f%+.2fi ",real(fxcopy[j]),imag(fxcopy[j]));
			}
			printf("\n");
			free(binm1);
			free(vec2pow);
		}

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


/*
	double t = omp_get_wtime();
	fastFT_iter(fx,s,N,binm1,vec2pow);
 	t=omp_get_wtime() - t;
	
	printf("fx=");
	for(int j=0;j<N;j++){
		printf("%+.2f%+.2fi ",real(fx[j]),imag(fx[j]));
	}
	printf("\n");
	printf("fastFT time: %6.4fs\n",t );
*/

	//printf("norm(fk-fkf)=%6.4f\n",printError(fk,fkf,N));
	
	/*
	printf("fkfast=");
	for(int j=0;j<N;j++){
		printf("%+.2f%+.2fi ",real(fk[j]),imag(fk[j]));
	}
	printf("\n");
	*/


	free(sendbuffer);
	free(recvbuffer);
	free(fx);
	free(fxcopy);
//	free(fk);
//	free(fkf);

}
