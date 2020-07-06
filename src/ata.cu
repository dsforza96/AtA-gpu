#include "strassen.cu"

void GPU_trans(double *A, double *C,
    int lda, int ldc,
    int XA, int YA) {
  double one = 1.0;
  double zero = 0.0;
  cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, XA, YA, &one, A, lda, &zero, A, lda, C, ldc);
}

void GPU_ata(double *A, double *C, int M, int N) {
  double one = 1.0;
  double zero = 0.0;
#if CMAJOR
  cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, M, &one, A, M, A, M, &zero, C, M);
#else
  cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, M, &one, A, N, A, M, &zero, C, N);
#endif
}

/*
  lda, ldc is the width in actual memory.
  XA is the width for computation.
  A = XA x YA
  C = XA x XA
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
  double *A12_t, *A22_t;
  int lw1 = XA2;
  int lw2 = XA2;
  cudaMalloc((void **)&W_1, lw1 * YA2 * sizeof(double));
  cudaMalloc((void **)&W_2, lw2 * YA2 * sizeof(double));
  cudaMalloc((void **)&A12_t, YA2 * XA2 * sizeof(double));
  cudaMalloc((void **)&A22_t, YA2 * XA2 * sizeof(double));

  int dXA = XA2;
  int dYA = YA2 * lda;

  int dXC = XC2;
  int dYC = YC2 * ldc;

  double *A11, *A12, *A21, *A22;
  double *C11, *C12, *C21, *C22;
  
  A11 = A;
  A12 = A + dXA;
  A21 = A + dYA;
  A22 = A + dXA + dYA;

  C11 = C;
  C12 = C + dXC;
  C21 = C + dYC;
  C22 = C + dXC + dYC;

  /* cutoff criteria */
  bool stop = false;
  
#if 0
  int cutoff = 2048;
  float mm = cutoff / XB2;
  float nn = cutoff / YA2;
  float kk = cutoff / XA2;
  if ((mm + nn + kk) >= 3) {
      stop = true;
  }
#endif

  if (depth <= 1 || stop) {
    GPU_ata(A, C, XA, YA);
  }
  else {
    ata(A11, W_1, lda, lw1, XA2, XA2, YA2, YA2, depth - 1);  // S1 = ata(A11)
    ata(A21, W_2, lda, lw2, XA2, XA2, YA2, YA2, depth - 1);  // S2 = ata(A21)
    GPU_add(W_1, W_2, C11, lw1, lw2, ldc, XA2, YA2, 1.0, 1.0);  // C11 = S1 + S2
    ata(A12, W_1, lda, lw1, XA2, XA2, YA2, YA2, depth - 1);  // S3 = ata(A12)
    ata(A22, W_2, lda, lw2, XA2, XA2, YA2, YA2, depth - 1);  // S4 = ata(A22)
    GPU_add(W_1, W_2, C22, lw1, lw2, ldc, XA2, YA2, 1.0,  1.0);  // C22 = S3 + S4
    GPU_trans(A12, A12_t, C22, lda, lda, XA2, YA2);  // A12_t
    strassen(A12_t, A11, W_1, lda, lda, lw1, YA2, XA2, XA2, XA2, YA2, YA2, depth - 1);  // S5 = strassen(A12_t, A11)
    GPU_trans(A22, A22_t, C22, lda, lda, XA2, YA2);  // A22_t
    strassen(A22_t, A21, W_2, lda, lda, lw2, YA2, XA2, XA2, XA2, YA2, YA2, depth - 1);  // S6 = strassen(A22_t, A21)
    GPU_add(W_1, W_2, C21, lw1, lw2, ldc, XA2, YA2, 1.0,  1.0);  // C21 = S5 + S6
  }
  cudaFree(W_1);
  cudaFree(W_2);
  cudaFree(A12_t);
  cudaFree(A22_t);

  /* dynamic peeling fix-up */
  int pxa = XA % 2;
  int pya = YA % 2;
  int pxb = XB % 2;
  int pyb = YB % 2;
  int pxc = XC % 2;
  int pyc = YC % 2;
  
  int nxa = XA - pxa;
  int nya = YA - pya;
  int nxb = XB - pxb;
  int nyb = YB - pyb;
  int nxc = XC - pxc;
  int nyc = YC - pyc;

  double *a12, *a21;
  double *b12, *b21;
  double *c12, *c21;
  int dxa = nxa;
  int dya = nya * lda;
  int dxb = nxb;
  int dyb = nyb * ldb;
  int dxc = nxc;
  int dyc = nyc * ldc;
  
  a12 = A + dxa;
  a21 = A + dya;
  // a22 = A + dxa + dya;
  b12 = B + dxb;
  b21 = B + dyb;
  // b22 = B + dxb + dyb;
  c12 = C + dxc;
  c21 = C + dyc;
  // c22 = C + dxc + dyc;

  /* 
    A11 = nxa x nya
    a12 = pxa x nya
    a21 = nxa x pya
    a22 = pxa x pya
   */
  GPU_mul(a21, B11, c21, lda, ldb, ldc, nxa,  XB,  XC, pya, nyb, pyc, 1.0, 0.0);
  GPU_mul(A11, b12, c12, lda, ldb, ldc, nxa, pxb, pxc,  YA, nyb,  YC, 1.0, 0.0);
  GPU_mul(a12, b21, C11, lda, ldb, ldc, pxa,  XB,  XC,  YA, pyb,  YC, 1.0, 1.0);
}


int main (int argc, char **argv) {
  if(argc != 6) {
    printf("Usage: strassen <M> <N> <iter> <check> <depth>\n");
    return -1;
  }

  int M = atoi(argv[1]);
  int N = atoi(argv[2]);
  int iter = atoi(argv[3]);
  int check = atoi(argv[4]);
  int depth = atoi(argv[5]);

  int sizeA = M * N;
  int sizeC = M * M;
  int memSizeA = sizeA * sizeof(double);
  int memSizeC = sizeC * sizeof(double);

  double *h_A = (double *)malloc(memSizeA);
  double *h_C = (double *)malloc(memSizeC);
  double *v_C = (double *)malloc(memSizeC);
  
  for (int i = 0; i < sizeA; i++) {
    h_A[i] = i % 3;
  }
  for (int i = 0; i < sizeC; i++) {
    h_C[i] = 0.0;
    v_C[i] = 0.0;
  }

  double *d_A, *d_C;
  cudaMalloc((void**)&d_A, memSizeA);
  cudaMalloc((void**)&d_C, memSizeC);
  cudaMemcpy(d_A, h_A, memSizeA, cudaMemcpyHostToDevice);

  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "!!!! CUBLAS initialization error\n");
    fflush(NULL);
    return EXIT_FAILURE;
  }

  CudaTimer ct;
  ct.start();
  for (int i = 0; i < iter; i++) {
    ata(d_A, d_C, M, M, M, N, M, M, depth);
  }
  ct.stop();

  double strassenTime = ct.value() / iter;
  cudaMemcpy(h_C, d_C, memSizeC, cudaMemcpyDeviceToHost);

#if 1
  ct.start();
  for (int i = 0; i < iter; i++) {
    GPU_ata(d_A, d_C, M, N);
  }
  ct.stop();
  
  double classicTime = ct.value() / iter;
  cudaMemcpy(v_C, d_C, memSizeC, cudaMemcpyDeviceToHost);

  double speedup = classicTime / strassenTime;
  printf ("%d %d %.2f %.2f %.2f\n", M, N, strassenTime, classicTime, speedup);
#endif

  if (check) {
    double absErr = 0.0;
    for (int i = 0; i < sizeC; i++) {
      absErr += abs(h_C[i] - v_C[i]);
    }
    if (absErr > 1) {
      printf("CHECK: Absolute error: %lf\n", absErr);
    }
  }

  free(h_A);
  free(h_B);
  free(h_C);
  free(v_C);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  if (cublasDestroy(handle) != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "!!!! CUBLAS shutdown error\n");
    fflush(NULL);
    return EXIT_FAILURE;
  }
}
