// This license applies to strassen-dp, Strassen-Winograd’s Matrix Multiplication
// Algorithm on GPU

// Copyright (c) 2013, The Ohio State University
// All rights reserved.

// Contact: P. Sadayappan <saday@cse.ohio-state.edu>

// Written by: Pai-Wei Lai, Humayun Arafat and Venmugil Elango

// Reference: Pai-Wei Lai; Arafat, H.; Elango, V.; Sadayappan, P., "Accelerating
// Strassen-Winograd's matrix multiplication algorithm on GPUs", 20th
// International Conference on High Performance Computing (HiPC) 2013, pp.
// 139-148, Dec. 2013
// doi: 10.1109/HiPC.2013.6799109

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// - Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
// - Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
// - Neither the name of The Ohio State University nor the names of its
// contributors may be used to endorse or promote products derived from this
// software without specific prior written permission.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cassert>

#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cuda.h>

#include <cmath>
#include "ata.h"


extern cublasHandle_t handle;


class CudaTimer
{
private:
    cudaEvent_t _begEvent;
    cudaEvent_t _endEvent;

public:
    CudaTimer();
    ~CudaTimer();
    void start();
    void stop();
    float value();
};

#define SafeTimerCall(err) __safeTimerCall(err, __FILE__, __LINE__)

inline void __safeTimerCall(cudaError err, const char *file, const int line) {
#pragma warning(push)
#pragma warning(disable: 4127) Prevent warning on do-while(0);
  do {
    if (cudaSuccess != err) {
      fprintf(stderr, "CudaTimer failed at %s:%i : %s\n", file, line, cudaGetErrorString(err));
      exit(-1);
    }
  } while (0);
#pragma warning(pop)
  return;
}

CudaTimer::CudaTimer() {
  SafeTimerCall(cudaEventCreate(&_begEvent));
  SafeTimerCall(cudaEventCreate(&_endEvent));
  return;
}

CudaTimer::~CudaTimer() {
  SafeTimerCall(cudaEventDestroy(_begEvent));
  SafeTimerCall(cudaEventDestroy(_endEvent));
  return;
}

void CudaTimer::start() {
  SafeTimerCall(cudaEventRecord(_begEvent, 0));
  return;
}

void CudaTimer::stop() {
  SafeTimerCall(cudaEventRecord(_endEvent, 0));
  return;
}

float CudaTimer::value() {
  SafeTimerCall(cudaEventSynchronize(_endEvent));
  float timeVal;
  SafeTimerCall(cudaEventElapsedTime(&timeVal, _begEvent, _endEvent));
  return timeVal;
}

void GPU_mul(Float *A, Float *B, Float *C,
    int lda, int ldb, int ldc,
    int XA, int XB, int XC,
    int YA, int YB, int YC,
    Float alpha, Float beta) {
  cublasGemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, XB, YA, XA, &alpha, B, ldb, A, lda, &beta, C, ldc);
}

void GPU_add(Float *A, Float *B, Float *C,
    int lda, int ldb, int ldc,
    int XA, int YA,
    Float alpha, Float beta) {
  cublasGeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, XA, YA, &alpha, A, lda, &beta, B, ldb, C, ldc);
}

void verifyByCUBLAS(Float *d_A, Float *d_B, Float *d_C, int M, int N, int K) {
  Float one = 1.0;
  Float zero = 0.0;
#if CMAJOR
  cublasGemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &one, d_A, M, d_B, K, &zero, d_C, M);
#else
  cublasGemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &one, d_B, N, d_A, K, &zero, d_C, N);
#endif
}

/*
  lda, ldb, ldc is the width in actual memory.
  XA, XB, XC is the width for computation.
  A = XA x YA
  B = XB x YB
  C = XC x YC
*/
void strassen(Float *A, Float *B, Float *C,
    int lda, int ldb, int ldc,
    int XA, int XB, int XC,
    int YA, int YB, int YC,
    int depth) {

  int XA2 = XA / 2;
  int XB2 = XB / 2;
  int XC2 = XC / 2;
  
  int YA2 = YA / 2;
  int YB2 = YB / 2;
  int YC2 = YC / 2;

  Float *W_1, *W_2;
  int lw1 = (XA2 > XC2 ? XA2 : XC2);
  int lw2 = XB2;
  cudaMalloc((void **)&W_1, lw1 * YA2 * sizeof(Float));
  cudaMalloc((void **)&W_2, lw2 * YB2 * sizeof(Float));

  int dXA = XA2;
  int dYA = YA2 * lda;
  int dXB = XB2;
  
  int dYB = YB2 * ldb;
  int dXC = XC2;
  int dYC = YC2 * ldc;

  Float *A11, *A12, *A21, *A22;
  Float *B11, *B12, *B21, *B22;
  Float *C11, *C12, *C21, *C22;
  
  A11 = A;
  A12 = A + dXA;
  A21 = A + dYA;
  A22 = A + dXA + dYA;
  
  B11 = B;
  B12 = B + dXB;
  B21 = B + dYB;
  B22 = B + dXB + dYB;
  
  C11 = C;
  C12 = C + dXC;
  C21 = C + dYC;
  C22 = C + dXC + dYC;

  /* cutoff criteria */
  bool stop = false;
  
#if 1
  float mm = CUTOFF / XB2;
  float nn = CUTOFF / YA2;
  float kk = CUTOFF / XA2;
  if ((mm + nn + kk) >= 3) {
      stop = true;
  }
#endif

  if (depth <= 1 || stop) {
    GPU_add(A11, A21, W_1, lda, lda, lw1, XA2, YA2, 1.0, -1.0); // W_1 = A11 - A21
    GPU_add(B22, B12, W_2, ldb, ldb, lw2, XB2, YB2, 1.0, -1.0); // W_2 = B22 - B12
    GPU_mul(W_1, W_2, C21, lw1, lw2, ldc, XA2, XB2, XC2, YA2, YB2, YC2, 1.0, 0.0); // C21 = W_1 * W_2
    GPU_add(A21, A22, W_1, lda, lda, lw1, XA2, YA2, 1.0,  1.0); // W_1 = A21 + A22
    GPU_add(B12, B11, W_2, ldb, ldb, lw2, XB2, YB2, 1.0, -1.0); // W_2 = B12 - B11
    GPU_mul(W_1, W_2, C22, lw1, lw2, ldc, XA2, XB2, XC2, YA2, YB2, YC2, 1.0, 0.0); // C22 = W_1 * W_2
    GPU_add(W_1, A11, W_1, lw1, lda, lw1, XA2, YA2, 1.0, -1.0); // W_1 = W_1- A11
    GPU_add(B22, W_2, W_2, ldb, lw2, lw2, XB2, YB2, 1.0, -1.0); // W_2 = B22 - W_2
    GPU_mul(W_1, W_2, C11, lw1, lw2, ldc, XA2, XB2, XC2, YA2, YB2, YC2, 1.0, 0.0); // C11 = W_1 * W_2
    GPU_add(A12, W_1, W_1, lda, lw1, lw1, XA2, YA2, 1.0, -1.0); // W_1 = A12 - W_1
    GPU_mul(W_1, B22, C12, lw1, ldb, ldc, XA2, XB2, XC2, YA2, YB2, YC2, 1.0, 0.0); // C12 = W_1 * B22
    GPU_add(C22, C12, C12, ldc, ldc, ldc, XC2, YC2, 1.0,  1.0); // C12 = C22 + C12
    GPU_mul(A11, B11, W_1, lda, ldb, lw1, XA2, XB2, XC2, YA2, YB2, YC2, 1.0, 0.0); // W_1= A11 * B11
    GPU_add(W_1, C11, C11, lw1, ldc, ldc, XC2, YC2, 1.0,  1.0); // C11 = W_1 + C11
    GPU_add(C11, C12, C12, ldc, ldc, ldc, XC2, YC2, 1.0,  1.0); // C12 = C11 + C12
    GPU_add(C11, C21, C11, ldc, ldc, ldc, XC2, YC2, 1.0,  1.0); // C11 = C11 + C21
    GPU_add(W_2, B21, W_2, lw2, ldb, lw2, XB2, YB2, 1.0, -1.0); // W_2 = W_2- B21
    GPU_mul(A22, W_2, C21, lda, lw2, ldc, XA2, XB2, XC2, YA2, YB2, YC2, 1.0, 0.0); // C21 = A22 * W_2
    GPU_add(C11, C21, C21, ldc, ldc, ldc, XC2, YC2, 1.0, -1.0); // C11 = C11 - C21
    GPU_add(C11, C22, C22, ldc, ldc, ldc, XC2, YC2, 1.0,  1.0); // C22 = C11 + C22
    GPU_mul(A12, B21, C11, lda, ldb, ldc, XA2, XB2, XC2, YA2, YB2, YC2, 1.0, 0.0); // C11 = A12 * B21
    GPU_add(W_1, C11, C11, lw1, ldc, ldc, XC2, YC2, 1.0,  1.0); // C11 = W_1+ C11
  }
  else {
    GPU_add(A11, A21, W_1, lda, lda, lw1, XA2, YA2, 1.0, -1.0); // W_1 = A11 - A21
    GPU_add(B22, B12, W_2, ldb, ldb, lw2, XB2, YB2, 1.0, -1.0); // W_2 = B22 - B12
    strassen(W_1, W_2, C21, lw1, lw2, ldc, XA2, XB2, XC2, YA2, YB2, YC2, depth - 1);
    GPU_add(A21, A22, W_1, lda, lda, lw1, XA2, YA2, 1.0,  1.0); // W_1 = A21 + A22
    GPU_add(B12, B11, W_2, ldb, ldb, lw2, XB2, YB2, 1.0, -1.0); // W_2 = B12 - B11
    strassen(W_1, W_2, C22, lw1, lw2, ldc, XA2, XB2, XC2, YA2, YB2, YC2, depth - 1);
    GPU_add(W_1, A11, W_1, lw1, lda, lw1, XA2, YA2, 1.0, -1.0); // W_1 = W_1- A11
    GPU_add(B22, W_2, W_2, ldb, lw2, lw2, XB2, YB2, 1.0, -1.0); // W_2 = B22 - W_2
    strassen(W_1, W_2, C11, lw1, lw2, ldc, XA2, XB2, XC2, YA2, YB2, YC2, depth - 1);
    GPU_add(A12, W_1, W_1, lda, lw1, lw1, XA2, YA2, 1.0, -1.0); // W_1 = A12 - W_1
    strassen(W_1, B22, C12, lw1, ldb, ldc, XA2, XB2, XC2, YA2, YB2, YC2, depth - 1);
    GPU_add(C22, C12, C12, ldc, ldc, ldc, XC2, YC2, 1.0,  1.0); // C12 = C22 + C12
    strassen(A11, B11, W_1, lda, ldb, lw1, XA2, XB2, XC2, YA2, YB2, YC2, depth - 1);
    GPU_add(W_1, C11, C11, lw1, ldc, ldc, XC2, YC2, 1.0,  1.0); // C11 = W_1 + C11
    GPU_add(C11, C12, C12, ldc, ldc, ldc, XC2, YC2, 1.0,  1.0); // C12 = C11 + C12
    GPU_add(C11, C21, C11, ldc, ldc, ldc, XC2, YC2, 1.0,  1.0); // C11 = C11 + C21
    GPU_add(W_2, B21, W_2, lw2, ldb, lw2, XB2, YB2, 1.0, -1.0); // W_2 = W_2- B21
    strassen(A22, W_2, C21, lda, lw2, ldc, XA2, XB2, XC2, YA2, YB2, YC2, depth - 1);
    GPU_add(C11, C21, C21, ldc, ldc, ldc, XC2, YC2, 1.0, -1.0); // C11 = C11 - C21
    GPU_add(C11, C22, C22, ldc, ldc, ldc, XC2, YC2, 1.0,  1.0); // C22 = C11 + C22
    strassen(A12, B21, C11, lda, ldb, ldc, XA2, XB2, XC2, YA2, YB2, YC2, depth - 1);
    GPU_add(W_1, C11, C11, lw1, ldc, ldc, XC2, YC2, 1.0,  1.0); // C11 = W_1+ C11
  }
  cudaFree(W_1);
  cudaFree(W_2);

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

  Float *a12, *a21;
  Float *b12, *b21;
  Float *c12, *c21;
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
