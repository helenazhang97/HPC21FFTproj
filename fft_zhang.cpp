/* MPI-parallel Jacobi smoothing to solve -u''=f
 * Global vector has N unknowns, each processor works with its
 * part, which has lN = N/p unknowns.
 * Author: Georg Stadler
 */
#include <iostream>
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <string.h>
#include <cstdlib>
#include <complex>
#include <cmath>
#include <cassert>
#include "utils.h"
using namespace std::complex_literals;

void naiveFFT( std::complex<double> *fx,std::complex<double> *fk, int N){
	std::complex<double> * FN    = (std::complex<double> *) calloc(2*sizeof(double), N*N	);
	for (int j=0;j<N;j++){
		for (int k=0;k<N;k++){
			std::complex<double> z = -1i*2*M_PI/N*j*k;
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

void fastFT(std::complex<double> *fx, std::complex<double> *fk,int N){
	if (N>1){
		assert(("N%2!=0",N%2==0));
		std::complex<double> * fxeven = (std::complex<double> *) calloc(2*sizeof(double), N/2);
		std::complex<double> * fxodd = (std::complex<double> *) calloc(2*sizeof(double), N/2);
		std::complex<double> * fkeven = (std::complex<double> *) calloc(2*sizeof(double), N/2);
		std::complex<double> * fkodd = (std::complex<double> *) calloc(2*sizeof(double), N/2);

		for (int k=0;k<N/2;k++){
			fxeven[k]=fx[2*k];
			fxodd[k]=fx[2*k+1];
		}
		fastFT(fxeven,fkeven,N/2);
		fastFT(fxodd,fkodd,N/2);
		
		for (int k=0;k<N/2;k++){
			std::complex<double> z=-1i*2*M_PI/N*k;
			fk[k]=fkeven[k]+exp(z)*fkodd[k];
			fk[k+N/2]=fkeven[k]-exp(z)*fkodd[k];
		}

		free(fxeven);
		free(fxodd);
		free(fkeven);
		free(fkodd);
	}else{fk[0]=fx[0];}

}

int main(){
	/*most naive FFT by matrix * vector*/
	int N=10;
	N=pow(2,N);
	std::complex<double> * fx    = (std::complex<double> *) calloc(2*sizeof(double), N	);
	std::complex<double> * fk    = (std::complex<double> *) calloc(2*sizeof(double), N	);
	for(int j=0;j<N;j++){
		fx[j]=drand48()+drand48()*1i;
	}
	Timer tt;
	tt.tic();
	naiveFFT(fx,fk,N);
    printf("Reference time: %6.4fs\n", tt.toc());

	/*
	printf("fk=");
	for(int j=0;j<N;j++){
		printf("%+.2f%+.2fi ",real(fk[j]),imag(fk[j]));
	}
	printf("\n");
	*/
	tt.tic();
	fastFT(fx,fk,N);
    printf("fastFT time: %6.4fs\n", tt.toc());

	/*
	printf("fkfast=");
	for(int j=0;j<N;j++){
		printf("%+.2f%+.2fi ",real(fk[j]),imag(fk[j]));
	}
	printf("\n");
	*/

	free(fx);
	free(fk);


}
