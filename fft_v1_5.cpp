#include <iostream>
#include <stdio.h>
#include <math.h> // to use M_PI
// #include <mpi.h>
#include <complex>
#include <fftw3.h>

// g++ -std=c++14 fft_v1_5.cpp -lfftw3

// mpic++ -std=c++14 fftCai.cpp -lfftw3   >?-lm
// mpiexec -np 2 ./a.out

using namespace std::complex_literals;

void dft_naive(std::complex<double> *fx, std::complex<double> *fk, int N);
void fft_serial(std::complex<double> *fx, std::complex<double> *fk, int N, std::complex<double> *omega_);
double get_error(std::complex<double> *fk, fftw_complex *out, int N);
int my_log2(int N);
int reverse_bits(int x, int s);



// void fft_mpi(std::complex<double> *fx, std::complex<double> *fk, int N, std::complex<double> *omega_, int rank, int size) {
//     int s = my_log2(N), p = my_log2(size), q = my_log2(N/size);
//     int mask, index0, index1, Nlocal = N/size;
//     std::complex<double> t0, t1, w, x;
//
//     std::complex<double> *fxlocal = (std::complex<double> *) malloc(Nlocal* sizeof(std::complex<double>));
//     std::complex<double> *fxin = (std::complex<double> *) malloc(Nlocal* sizeof(std::complex<double>));
//
//     for (int j=0; j<Nlocal; j++) {
//         fxlocal[j] = fx[rank*Nlocal + j];
//     }
//
//     for (int j=0; j<p; j++) {
//         MPI_Sendrecv(fxlocal, Nlocal, MPI_DOUBLE_COMPLEX, (rank+2-j)%size, 310,
//                fxin, Nlocal, MPI_DOUBLE_COMPLEX, (rank-2+j)%size, 310, MPI_COMM_WORLD, status);
//
//         for (int k=rank*Nlocal; k<(rank+1)*Nlocal; k++) {
//             if (k<N/2) {
//                 t0 = fxlocal[k%Nlocal];
//                 t1 = fx1[k%Nlocal];
//
//                 w = omega_[reverse_bits(k >> s-1-j, s)];
//                 x = w*t1;
//
//                 fxlocal[k%Nlocal] = t0 + x;
//             }
//             else {
//                 t0 = fx1[k%Nlocal];
//                 t1 = fxlocal[k%Nlocal];
//
//                 w = omega_[reverse_bits((k-N/2) >> s-1-j, s)]; // need to change this
//                 x = w*t1;
//
//                 fxlocal[k%Nlocal] = t0 - x;
//             }
//         }
//     }
//     for (int j=p; j<s; j++) {
//         for (int k=rank*Nlocal; k<(rank+1)*Nlocal; k++) {
//
//         }
//     }
//
//
//
//
//
//     MPI_Gather(fxlocal, Nlocal, MPI_DOUBLE_COMPLEX, fk, Nlocal,
//                MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
// }



int main(int argc, char const *argv[]) {
    // MPI_Init(&argc, &argv);
    //
    // int rank, size;
    // MPI_Status status;
    // MPI_Comm_rank(comm, &rank);
    // MPI_Comm_size(comm, &size);

    int N = 16;

    std::complex<double> *omega_ = (std::complex<double> *) malloc(N* sizeof(std::complex<double>));

    // array of exp(-1i*j*k*2pi/N)
    for (int j=0; j<N; j++) {
        omega_[j] = exp(-1i*(2.*M_PI*j/N));
    }

    std::complex<double> *fx = (std::complex<double> *) malloc(N* sizeof(std::complex<double>));
    std::complex<double> *fx_copy = (std::complex<double> *) malloc(N* sizeof(std::complex<double>));
    std::complex<double> *fk = (std::complex<double> *) calloc(N, sizeof(std::complex<double>));
    std::complex<double> *fk2 = (std::complex<double> *) calloc(N, sizeof(std::complex<double>));
    std::complex<double> *fk3 = (std::complex<double> *) calloc(N, sizeof(std::complex<double>));

    // initialize vector to take fft of
    for (int j=0; j<N; j++) {
        fx[j] = sin(2*j)+cos(3*j)*1i;
        fx_copy[j] = fx[j];
    }

    dft_naive(fx, fk, N);
    fft_serial(fx_copy, fk2, N, omega_);

    // do fftw version, page 3 of http://www.fftw.org/fftw3.pdf
    fftw_complex *in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
    fftw_complex *out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);;
    fftw_plan p = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);;

    for (int j=0; j<N; j++) {
        in[j][0] = sin(2*j);
        in[j][1] = cos(3*j);
    }

    fftw_execute(p);

    printf("error for dft_naive = %f \n", get_error(fk,out,N));
    printf("error for fft_serial = %f \n", get_error(fk2,out,N));
    printf("error for fft_mpi = %f \n", get_error(fk3,out,N));

    fftw_destroy_plan(p);
    fftw_free(in);
    fftw_free(out);
    free(omega_);
    free(fx);
    free(fx_copy);
    free(fk);
    free(fk2);
    free(fk3);

    return 0;
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

void dft_naive(std::complex<double> *fx, std::complex<double> *fk, int N) {
    for (int j=0; j<N; j++) {  // put these exponential values in an input matrix to avoid calculation
        for (int k=0; k<N; k++) {
            fk[j] += exp(-1i*(2.*M_PI*j*k/N)) * fx[k];
        }
    }
}

//------------------------------------------------------------------------------

double get_error(std::complex<double> *fk, fftw_complex *out, int N) {
    double err{};

    for (int j=0; j<N; j++) {
        err += (real(fk[j])-out[j][0]) * (real(fk[j])-out[j][0]) +
                (imag(fk[j])-out[j][1])* (imag(fk[j])-out[j][1]);
    }

    return sqrt(err);
}

//------------------------------------------------------------------------------

int my_log2(int N) {
    int l{0}, N_temp{N};
    while (N_temp > 1) {
        N_temp /= 2;
        l += 1;
    }
    return l;
}

//------------------------------------------------------------------------------

int reverse_bits(int x, int s) {
    int x_reverse{}, temp;

    for (int j=0; j<s; j++) {
        temp = (x>>j) & 1;
        x_reverse += temp * (1<<(s-j-1));
    }
    return x_reverse;
}

//------------------------------------------------------------------------------

void fft_serial(std::complex<double> *fx, std::complex<double> *fk, int N, std::complex<double> *omega_) {
    int s = my_log2(N), index0, index1;
    int mask; // for inserting 0 or 1 in the bitstring
    std::complex<double> t0, t1, w, x;

    for (int j=0; j<s; j++) {
        // https://en.wikipedia.org/wiki/Bitwise_operations_in_C
        mask = (1 << (s-1-j)) - 1;
        for (int k=0; k<N/2; k++) {
            index0 = ((k & ~mask) << 1) | (k & mask);
            index1 = ((k & ~mask) << 1) | (k & mask) + (1 << (s-1-j));

            t0 = fx[index0];
            t1 = fx[index1];

            w = omega_[reverse_bits(index0 >> s-1-j, s)];
            x = w*t1;

            fx[index0] = t0 + x;
            fx[index1] = t0 - x;
        }
    }

    // rearrange output to give correct answer
    for (int k=0; k<N; k++) {
        fk[k] = fx[reverse_bits(k,s)];
    }
}
