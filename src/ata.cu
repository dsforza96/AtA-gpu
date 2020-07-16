#include "ata.h"
#include "strassen.cu"

void GPU_T(double *A, double *C,
    int lda, int ldc,
    int XA, int YA) {
  double one = 1.0;
  double zero = 0.0;
  cublasGeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, XA, YA, &one, A, lda, &zero, C, ldc, C, ldc);
}

void GPU_AtB(double *A, double *B, double *C,
    int lda, int ldb, int ldc,
    int XA, int XB, int XC,
    int YA, int YB, int YC,
    double alpha, double beta) {
  cublasGemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, XB, YA, XA, &alpha, B, ldb, A, lda, &beta, C, ldc);
}

/*
  lda, ldc is the width in actual memory.
  XA, XC is the width for computation.
  Returns the lower triangular part of C.
  A = XA x YA
  C = XC x YC
*/
void ata(double *A, double *C,
    int lda, int ldc,
    int XA, int XC,
    int YA, int YC,
    int depth) {

  int XA2 = XA / 2;
  int XC2 = XC / 2;

  int YA2 = YA / 2;
  int YC2 = YC / 2;

  double *W_1, *W_2;
  int ldw = XC2;
  cudaMalloc((void **)&W_1, ldw * YC2 * sizeof(double));
  cudaMalloc((void **)&W_2, ldw * YC2 * sizeof(double));

  int dXA = XA2;
  int dYA = YA2 * lda;

  int dXC = XC2;
  int dYC = YC2 * ldc;

  double *A11, *A12, *A21, *A22;
  double *C11, *C21, *C22;

  A11 = A;
  A12 = A + dXA;
  A21 = A + dYA;
  A22 = A + dXA + dYA;

  C11 = C;
  // C12 = C + dXC;
  C21 = C + dYC;
  C22 = C + dXC + dYC;

  /* cutoff criteria */
  bool stop = false;

  int cutoff = 2048;
  float mm = cutoff / XA2;
  float nn = cutoff / YA2;
  if ((mm + nn) >= 2) {
      stop = true;
  }

  if (depth <= 1 || stop) {
    GPU_AtB(A11, A11, W_1, lda, lda, ldw, YA2, XA2, XC2, XA2, YA2, YC2, 1.0, 0.0);  // S1 = ata(A11)
    GPU_AtB(A21, A21, W_2, lda, lda, ldw, YA2, XA2, XC2, XA2, YA2, YC2, 1.0, 0.0);  // S2 = ata(A21)
    GPU_add(W_1, W_2, C11, ldw, ldw, ldc, XC2, YC2, 1.0, 1.0);                      // C11 = S1 + S2
    GPU_AtB(A12, A12, W_1, lda, lda, ldw, YA2, XA2, XC2, XA2, YA2, YC2, 1.0, 0.0);  // S3 = ata(A12)
    GPU_AtB(A22, A22, W_2, lda, lda, ldw, YA2, XA2, XC2, XA2, YA2, YC2, 1.0, 0.0);  // S4 = ata(A22)
    GPU_add(W_1, W_2, C22, ldw, ldw, ldc, XC2, YC2, 1.0,  1.0);                     // C22 = S3 + S4
    GPU_AtB(A12, A11, W_1, lda, lda, ldw, YA2, XA2, XC2, XA2, YA2, YC2, 1.0, 0.0);  // S5 = strassen(A12_t, A11)
    GPU_AtB(A22, A21, W_2, lda, lda, ldw, YA2, XA2, XC2, XA2, YA2, YC2, 1.0, 0.0);  // S6 = strassen(A22_t, A21)
    GPU_add(W_1, W_2, C21, ldw, ldw, ldc, XC2, YC2, 1.0,  1.0);                     // C21 = S5 + S6
  }
  else {
    double *A2t;
    int ldt = YA2;
    cudaMalloc((void **)&A2t, ldt * XA2 * sizeof(double));

    ata(A11, W_1, lda, ldw, XA2, XC2, YA2, YC2, depth - 1);                           // S1 = ata(A11)
    ata(A21, W_2, lda, ldw, XA2, XC2, YA2, YC2, depth - 1);                           // S2 = ata(A21)
    GPU_add(W_1, W_2, C11, ldw, ldw, ldc, XC2, YC2, 1.0, 1.0);                        // C11 = S1 + S2
    ata(A12, W_1, lda, ldw, XA2, XC2, YA2, YC2, depth - 1);                           // S3 = ata(A12)
    ata(A22, W_2, lda, ldw, XA2, XC2, YA2, YC2, depth - 1);                           // S4 = ata(A22)
    GPU_add(W_1, W_2, C22, ldw, ldw, ldc, XC2, YC2, 1.0,  1.0);                       // C22 = S3 + S4
    GPU_T(A12, A2t, lda, ldt, YA2, XA2);                                              // A12t
    strassen(A2t, A11, W_1, ldt, lda, ldw, YA2, XA2, XC2, XA2, YA2, YC2, depth - 1);  // S5 = strassen(A12t, A11)
    GPU_T(A22, A2t, lda, ldt, YA2, XA2);                                              // A22t
    strassen(A2t, A21, W_2, ldt, lda, ldw, YA2, XA2, XC2, XA2, YA2, YC2, depth - 1);  // S6 = strassen(A22t, A21)
    GPU_add(W_1, W_2, C21, ldw, ldw, ldc, XC2, YC2, 1.0,  1.0);                       // C21 = S5 + S6

    cudaFree(A2t);
    cudaFree(A2t);
  }
  cudaFree(W_1);
  cudaFree(W_2);

  /* dynamic peeling fix-up */
  int pxa = XA % 2;
  int pya = YA % 2;
  int pxc = XC % 2;
  int pyc = YC % 2;

  int nxa = XA - pxa;
  int nya = YA - pya;
  int nxc = XC - pxc;
  int nyc = YC - pyc;

  double *a12, *a21;
  double *c21;
  int dxa = nxa;
  int dya = nya * lda;
  // int dxc = nxc;
  int dyc = nyc * ldc;

  a12 = A + dxa;
  a21 = A + dya;
  // a22 = A + dxa + dya;
  // c12 = C + dxc;
  c21 = C + dyc;
  // c22 = C + dxc + dyc;

  /*
    A11 = nxa x nya
    a12 = pxa x nya
    a21 = nxa x pya
    a22 = pxa x pya
   */
  GPU_AtB(a12, A, c21, lda, lda, ldc, YA, XA, XC, pxa, YA, pyc, 1.0, 0.0);
  GPU_AtB(a21, a21, C11, lda, lda, ldc, pya, nxa, nxc, nxa, pya, nyc, 1.0, 1.0);
}

// void printm(double* arr, int m, int n) {
//   for (int i = 0; i < m; i++) {
//    for (int j = 0; j < n; j++) {
//       printf("%f ", arr[j + i * n]);
//    }
//    printf("\n");
//   }
//   printf("\n");
// }


int main (int argc, char **argv) {
  if(argc != 6) {
    printf("Usage: %s <M> <N> <iter> <check> <depth>\n", argv[0]);
    return -1;
  }

  int M = atoi(argv[1]);
  int N = atoi(argv[2]);
  int iter = atoi(argv[3]);
  int check = atoi(argv[4]);
  int depth = atoi(argv[5]);

  int sizeA = M * N;
  int sizeC = N * N;
  int memSizeA = sizeA * sizeof(double);
  int memSizeC = sizeC * sizeof(double);

  // double *h_A = (double *)malloc(memSizeA);
  double *h_C = (double *)malloc(memSizeC);
  double *v_C = (double *)malloc(memSizeC);

  double *d_A, *d_C;
  cudaMalloc((void**)&d_A, memSizeA);
  cudaMalloc((void**)&d_C, memSizeC);

  curandGenerator_t rng;

  if (curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT) != CURAND_STATUS_SUCCESS) {
    fprintf(stderr, "!!!! cuRAND initialization error\n");
    fflush(NULL);
    return EXIT_FAILURE;
  }

  curandSetPseudoRandomGeneratorSeed(rng, rand());
  curandGenerateUniform(rng, d_A, sizeA);
  // cudaMemcpy(h_A, d_A, memSizeA, cudaMemcpyDeviceToHost);
  // printm(h_A, M, N);

  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "!!!! cuBLAS initialization error\n");
    fflush(NULL);
    return EXIT_FAILURE;
  }

  CudaTimer ct;
  ct.start();
  for (int i = 0; i < iter; i++) {
    ata(d_A, d_C, N, N, N, N, M, N, depth);
  }
  ct.stop();

  double ataTime = ct.value() / iter;
  cudaMemcpy(h_C, d_C, memSizeC, cudaMemcpyDeviceToHost);
  // printm(h_C, N, N);

  ct.start();
  for (int i = 0; i < iter; i++) {
    GPU_AtB(d_A, d_A, d_C, N, N, N, M, N, N, N, M, N, 1.0, 0.0);
  }
  ct.stop();

  double classicTime = ct.value() / iter;
  cudaMemcpy(v_C, d_C, memSizeC, cudaMemcpyDeviceToHost);
  // printm(v_C, N, N);

  double speedup = classicTime / ataTime;
  printf ("M: %d; N: %d; AtA time: %.2f; classic time %.2f; speedup: %.2f\n", M, N, ataTime, classicTime, speedup);

  if (check) {
    double absErr = 0.0;
    for (int i = 0; i < M; i++) {
      for (int j = 0; j <= i; j++) {
        absErr += abs(h_C[i * N + j] - v_C[i * N + j]);
      }
    }
    if (absErr > 1.0) {
      printf("CHECK: Absolute error: %lf\n", absErr);
    }
  }

  // free(h_A);
  free(h_C);
  free(v_C);
  cudaFree(d_A);
  cudaFree(d_C);

  if (curandDestroyGenerator(rng) != CURAND_STATUS_SUCCESS) {
    fprintf(stderr, "!!!! cuRAND shutdown error\n");
    fflush(NULL);
    return EXIT_FAILURE;
  }

  if (cublasDestroy(handle) != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "!!!! cuBLAS shutdown error\n");
    fflush(NULL);
    return EXIT_FAILURE;
  }
}
